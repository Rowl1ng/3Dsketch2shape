from flow_sdf_trainer import Trainer
import torch
import torch.nn as nn
import math
import importlib
import os
import numpy as np
from einops import rearrange
import wandb
from utils.model_utils import get_spec_with_default, loss_NCE, loss_NCE_adapted
import torch.nn.functional as F
from utils.pc_utils import sample_farthest_points
EXP_DIR = os.getenv('EXP_DIR')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Disentangle_Net(nn.Module):
    def __init__(self, code_length, num_basis):
        super().__init__()
        self.num_basis = num_basis
        self.code_length = code_length
        self.conv31 = torch.nn.Linear(self.code_length, 256)
        self.conv32 = torch.nn.Linear(256, 512)
        self.conv33 = torch.nn.Linear(512, self.num_basis * self.code_length)
        self.bn31 = nn.BatchNorm1d(256)
        self.bn32 = nn.BatchNorm1d(512)

        self.conv71 = torch.nn.Linear(self.code_length, 32)
        self.conv72 = torch.nn.Linear(32, 16)
        self.conv73 = torch.nn.Linear(16, self.num_basis * 2)
        self.bn71 = nn.BatchNorm1d(32)
        self.bn72 = nn.BatchNorm1d(16)

        self.sigmoid = nn.Sigmoid()

    def forward(self, emb):
        B, _ = emb.shape
        net = F.relu(self.bn31(self.conv31(emb)))
        net = F.relu(self.bn32(self.conv32(net)))
        basis = self.conv33(net).view(B, self.num_basis, self.code_length)#.transpose(1, 2)
        basis = basis / basis.norm(p=2, dim=-1, keepdim=True) 

        coef_range = F.relu(self.bn71(self.conv71(emb)))
        coef_range = F.relu(self.bn72(self.conv72(coef_range)))
        coef_range = self.conv73(coef_range).view(B, -1, 2)

        # the predicted range for each axis $[-A_i,+B_i]$ : ensure it covers the origin point
        coef_range = F.relu(coef_range).view(B, self.num_basis, 2) #* 0.01
        coef_range[:, :, 0] = coef_range[:, :, 0] * -1

        return basis, coef_range
    
    def get_loss(self, basis):
        dot = torch.bmm(basis.abs(), basis.transpose(1, 2).abs())
        dot[:, range(self.num_basis), range(self.num_basis)] = 0
        ortho_loss = dot.norm(p=2, dim=(1, 2)).mean()
        return ortho_loss

class Disentangle_Net_v2(Disentangle_Net):
    def __init__(self, code_length, num_basis):
        super().__init__(code_length, num_basis)
        self.conv73 = torch.nn.Linear(16, self.num_basis)

    def forward(self, emb):
        B, _ = emb.shape
        net = F.relu(self.bn31(self.conv31(emb)))
        net = F.relu(self.bn32(self.conv32(net)))
        basis = self.conv33(net).view(B, self.num_basis, self.code_length)#.transpose(1, 2)
        basis = basis / basis.norm(p=2, dim=-1, keepdim=True) 

        coef_range = F.relu(self.bn71(self.conv71(emb)))
        coef_range = F.relu(self.bn72(self.conv72(coef_range)))
        coef_range = self.conv73(coef_range).view(B, -1, 2)

        # the predicted range for each axis $[-A_i,+B_i]$ : ensure it covers the origin point
        coef_range = F.relu(coef_range).view(B, self.num_basis, 1) #* 0.01
        coef_range = torch.cat((coef_range * -1, coef_range), dim=-1)

        return basis, coef_range

class Disentangle_Net_v3(Disentangle_Net):
    def __init__(self, code_length, num_basis):
        super().__init__(code_length, num_basis)
        self.value_layer = nn.Sequential(
                            torch.nn.Conv1d(self.code_length * 2, self.code_length, 1),
                            nn.ReLU(),
                            torch.nn.Conv1d(self.code_length, 1, 1),
                            nn.ReLU()
                            )

    def get_loss(self, basis, z_g, z_f, loss_func):
        dot = torch.bmm(basis.abs(), basis.transpose(1, 2).abs())
        dot[:, range(self.num_basis), range(self.num_basis)] = 0
        ortho_loss = dot.norm(p=2, dim=(1, 2)).mean()

        # z_g+A*H->z_f
        sketch_num = z_g.shape[0]
        shape_num = z_f.shape[0]
        z_g = z_g.repeat(shape_num, 1)
        z_f = torch.repeat_interleave(z_f, sketch_num, dim = 0)
        basis = basis.repeat(shape_num, 1, 1)
        input = torch.cat(((z_f - z_g).unsqueeze(1).expand(-1, self.num_basis, -1), basis), dim=-1)
        coeff = self.value_layer(input.transpose(1, 2))
        offset = torch.bmm(coeff, basis).view(-1, self.code_length)
        coeff_loss = loss_func(z_g + offset, z_f)
        return ortho_loss, coeff_loss

class Disentangle_Trainer(Trainer):
    def init_network(self):
        super().init_network()
        if hasattr(self.config, 'disentangle_net'):
            if self.config.disentangle_net == 'v2':
                self.disentangle_net = Disentangle_Net_v2(self.config.CodeLength, self.config.num_basis).to(device)
            elif self.config.disentangle_net == 'v3':
                self.disentangle_net = Disentangle_Net_v3(self.config.CodeLength, self.config.num_basis).to(device)
        else:
            self.disentangle_net = Disentangle_Net(self.config.CodeLength, self.config.num_basis).to(device)

    def train_network(self, epochs, saving_intervals, resume=False, use_testing=False):
        self.eval_epoch_fun = get_spec_with_default(self.config, "eval_epoch_fun", "eval_one_epoch")
        self.starting_epoch = 0
        # Prepare optimizer
        self.flow_optimizer = torch.optim.Adam(self.latent_flow_network.parameters(), lr=0.00003)
        self.optimizer = torch.optim.Adam(
            [
            {"params": self.disentangle_net.parameters(), "lr": 0.001},
            {"params": self.encoder.parameters(), "lr": 0.001},
            ]
            )

        if hasattr(self.config, "resume_ckpt"):
            if hasattr(self.config, 'conditional_NF') and self.config.conditional_NF:
                self.load_model(os.path.join(EXP_DIR, self.config.resume_ckpt), load_optimizer=False, load_NF=False)
            else:
                self.load_model(os.path.join(EXP_DIR, self.config.resume_ckpt), load_optimizer=False)
        if resume:
            # Resume training
            self.resume()
        # Feed point cloud into encoder and get feat vectors
        self.EPS = 1e-6 
        self.loss_sdf_fn = torch.nn.L1Loss(reduction="sum")
        self.loss_l1 = torch.nn.L1Loss(reduction="sum")
        self.loss_l1_mean = torch.nn.L1Loss(reduction="mean")

        from pytorch_metric_learning import losses
        self.contrastive_loss = losses.ContrastiveLoss()
        # triplet loss
        from utils.triplet_loss import OnlineTripletLoss, AllNegativeTripletSelector
        self.triplet_loss = OnlineTripletLoss(self.config.margin, AllNegativeTripletSelector())

        # latent loss
        if hasattr(self.config, 'latent_loss') and self.config.latent_loss == 'contrastive':
            self.loss_latent_fn = losses.ContrastiveLoss()
        else:
            self.loss_latent_fn = torch.nn.MSELoss()

        eval_one_epoch = getattr(self, self.eval_epoch_fun)
        for epoch in range(self.starting_epoch+1, epochs+1):
            if epoch == 1 :
                self.save_model(0)
            if self.config.load_Pinaki_model and self.config.train_shape_encoder and epoch <= self.config.encoder_epoch :
                # align embedding space
                loss_dict = self.eval_encoder_epoch(epoch, self.train_shape_data_loader)
                loss_dict.update({"epoch": epoch})
                wandb.log(loss_dict)
            else:
                loss_dict = eval_one_epoch(epoch, self.train_data_loader, is_training=True)
                new_dict = {f'train_{key}': loss_dict[key] for key in loss_dict.keys()}
                new_dict.update({"epoch": epoch})
                wandb.log(new_dict)

                # Save model
                if epoch % saving_intervals == 0:
                    self.save_model(epoch)
                    if use_testing:
                        # validation function
                        loss_dict = eval_one_epoch(epoch, self.test_data_loader, is_training=False)
                        new_dict = {f'test_{key}': loss_dict[key] for key in loss_dict.keys()}
                        new_dict.update({"epoch": epoch})
                        wandb.log(new_dict)
                    # if epoch % 50 == 0:
                    #     selected = [134, 196, 34, 163, 187]# np.random.choice(len(trainer.train_samples), args.num_samples, replace=False)
                    #     self.inference(epoch=epoch, samples=self.test_samples, selected=selected, split='test', num_samples=5)

    def load_model(self, last_resume_ckpt, load_optimizer=True, load_NF=True):
        checkpoint = torch.load(last_resume_ckpt, map_location=device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if 'disentangle_net_state_dict' in checkpoint.keys():
            self.disentangle_net.load_state_dict(checkpoint['disentangle_net_state_dict']) 
        if load_NF:
            self.latent_flow_network.load_state_dict(checkpoint['latent_flow_state_dict'])

        if hasattr(self, "optimizer") and load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        if hasattr(self, "flow_optimizer") and 'flow_optimizer_state_dict' in checkpoint.keys() and load_optimizer:
            self.flow_optimizer.load_state_dict(checkpoint['flow_optimizer_state_dict']) 

    def save_model(self, epoch):
        save_model_path = os.path.join(self.exp_dir, f'model_epoch_{epoch}.pth')
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'disentangle_net_state_dict': self.disentangle_net.state_dict(),
            'latent_flow_state_dict': self.latent_flow_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'flow_optimizer_state_dict': self.flow_optimizer.state_dict(),
        }, save_model_path)
        print(f"Epoch {epoch} model saved at {save_model_path}")



    def eval_one_epoch_v3(self, epoch, data_loader, is_training=True):
        # Decomposing sampling space
        if is_training:
            self.disentangle_net.train()
            self.latent_flow_network.train()
            # train encoder with shape SDF loss
            if hasattr(self.config, "freeze_encoder") and self.config.freeze_encoder:
                self.encoder.eval()
                for param in self.encoder.module.parameters():
                    param.requires_grad = False
            else:
                self.encoder.train()
            sketch_index = self.sketch_index
            shape_index = self.shape_index
            sdf_index = self.sdf_index
        else:
            self.disentangle_net.eval()
            self.latent_flow_network.eval()
            self.encoder.eval()
            sketch_index = self.test_sketch_index
            shape_index = self.test_shape_index
            sdf_index = self.test_sdf_index

        loss_list = ["L1_loss", "NCE_loss", "latent_NCE_loss", "Triplet_loss", "Contrastive_loss", "prob_loss","sketch_sdf_loss","shape_sdf_loss","sample_emb_loss","sample_sketch_sdf_loss","sigma_loss", "ortho_loss", "coeff_loss"]
        loss_dict = {name: [] for name in loss_list}

        for pc_data, sdf_data, indices, all_shape_index, hull_point in data_loader:
            pc_data = rearrange(pc_data, 'b h w c -> (b h) w c')
            if pc_data.shape[1] != self.config.num_points:
                points = sample_farthest_points(pc_data.transpose(1, 2), self.config.num_points).to(device)
            else:
                points = pc_data.transpose(2, 1).to(device)

            sdf_data = rearrange(sdf_data, 'b h w c -> (b h) w c')[sdf_index]
            all_shape_index = all_shape_index.reshape(-1)
            hull_point = hull_point.to(device)


            #################################### AE+NF stage ###################################

            loss = 0.0

            if self.config.train_shape_encoder or not is_training:
                train_feat = self.encoder(points)
                sketch_emb = train_feat[sketch_index]
                shape_emb = train_feat[shape_index]
            else:
                train_feat = torch.zeros([points.shape[0], self.config.CodeLength]).to(device)
                sketch_emb = self.encoder(points[sketch_index])
                # If don't train encoder for shape, then use latent code from DeepSDF decoder instead
                shape_emb = self.gt_emb[all_shape_index][sdf_index]
                train_feat[sketch_index] = sketch_emb
                train_feat[shape_index] = shape_emb

            if self.config.train_with_emb_loss is not None:
                gt_batch_vecs = self.gt_emb[all_shape_index][sdf_index]
                sketch_labels = torch.arange(0, sketch_emb.shape[0]).to(device)
                shape_labels = torch.arange(0, shape_emb.shape[0]).to(device)
                if 'L1' in self.config.train_with_emb_loss:
                    # L1(e_f, e_g')
                    l1_reg_loss = self.loss_l1(sketch_emb, gt_batch_vecs[:sketch_emb.shape[0]])
                    if self.config.train_shape_encoder:
                    # L1(e_f, e_f')
                        l1_reg_loss += self.loss_l1(shape_emb, gt_batch_vecs)
                        l1_reg_loss = l1_reg_loss/2
                    loss += l1_reg_loss
                    loss_dict['L1_loss'].append(l1_reg_loss.item())
                if 'NCE' in self.config.train_with_emb_loss:
                    # L(e_f, e_g')
                    nce_reg_loss = loss_NCE(sketch_emb, sketch_labels, gt_batch_vecs)
                    if self.config.train_shape_encoder:
                        nce_reg_loss += loss_NCE(shape_emb, shape_labels, gt_batch_vecs)
                        nce_reg_loss = nce_reg_loss/2
                    loss += nce_reg_loss
                    loss_dict['NCE_loss'].append(nce_reg_loss.item())
                if 'Contrastive' in self.config.train_with_emb_loss:
                    embeds = torch.cat([sketch_emb, gt_batch_vecs], dim=0)
                    labels = torch.cat([sketch_labels, shape_labels])
                    contra_loss = self.contrastive_loss(embeds, labels) 
                    if self.config.train_shape_encoder:
                        embeds = torch.cat([shape_emb, gt_batch_vecs], dim=0)
                        labels = torch.cat([shape_labels, shape_labels])
                        contra_loss += self.contrastive_loss(embeds, labels) 
                        contra_loss = contra_loss/2
                    loss += contra_loss
                    loss_dict['Contrastive_loss'].append(contra_loss.item())

                if 'Triplet' in self.config.train_with_emb_loss:
                    loss_emb = self.triplet_loss(sketch_emb, shape_emb)    
                    loss += loss_emb
                    loss_dict['Triplet_loss'].append(loss_emb.item())

                if 'sketch_sdf' in self.config.train_with_emb_loss:
                    sketch_points = points[sketch_index].transpose(2, 1)
                    loss_sketch_sdf = self.get_sketch_SDF_loss(sketch_points, sketch_emb, hull_point)
                    loss += loss_sketch_sdf
                    loss_dict["sketch_sdf_loss"].append(loss_sketch_sdf.item())

            # 3. SDF loss for shape reconstruction for finetuning encoder
            xyz = sdf_data[:, :, 0:3].reshape(-1, 3).to(device)
            shape_batch_vecs = torch.repeat_interleave(shape_emb, self.SamplesPerScene, dim = 0)
            pred_sdf = self.decoder(shape_batch_vecs, xyz)
            pred_sdf = torch.clamp(pred_sdf, - self.ClampingDistance, self.ClampingDistance)
            sdf_gt = sdf_data[:, :, 3].reshape(-1, 1).to(device)
            sdf_gt = torch.clamp(sdf_gt, - self.ClampingDistance, self.ClampingDistance)
            loss_shape_sdf = self.loss_sdf_fn(pred_sdf, sdf_gt) / sdf_gt.shape[0]
            if self.config.train_shape_encoder:
                loss += loss_shape_sdf * self.config.lambda_shape_sdf_loss
            loss_dict["shape_sdf_loss"].append(loss_shape_sdf.item())

            if self.config.train_with_nf_loss:
                # Feed feat vectors to normalzing flow and get latent code and loss
                # TODO: why choose 0.1?
                train_embs = train_feat + 0.1 * torch.randn(train_feat.size(0), self.config.CodeLength).to(device)
                u, log_jacob = self.latent_flow_network(train_embs, None)
                log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi + self.EPS)).sum(-1, keepdim=True)
                loss_log_prob = - (log_probs + log_jacob).sum(-1, keepdim=True).mean() 
                if is_training:
                    self.flow_optimizer.zero_grad()
                    loss_log_prob.backward(retain_graph=True)
                    self.flow_optimizer.step()

                loss_dict["prob_loss"].append(loss_log_prob.item())

            if is_training and loss > 0:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            #################################### Sampling stage ###################################
            # 1. inverse flow: latent code-> feat, sampled from sketch latent code's neighborhood(decided by std sigma) N(sketch_emb, sigma)
            new_train_feat = self.encoder(points)
            new_sketch_emb = new_train_feat[sketch_index]
            if not self.config.train_shape_encoder:
                # use GT latent code of decoder instead
                new_train_feat[shape_index] = self.gt_emb[all_shape_index][sdf_index]

            new_u, log_jacob = self.latent_flow_network(new_train_feat, None)
            new_sketch_latent = new_u[sketch_index]
            # gt_shape_latent = new_u[self.shape_gt_index] if is_training else new_u[self.test_shape_index]

            B = new_sketch_emb.shape[0]
            basis, coef_range = self.disentangle_net(new_sketch_emb)
            # the sampled coefficient vector $a$: sampling different coefficients $ a_i $ for each axis
            coef = torch.rand(B * self.config.num_samples, self.config.num_basis).to(device)
            coef_range = coef_range.repeat(self.config.num_samples, 1, 1)
            coef = (coef * coef_range[:, :, 0] + (1 - coef) * coef_range[:, :, 1]).view(B * self.config.num_samples, 1, self.config.num_basis)
            sampled_shape_latent = new_sketch_latent.repeat(self.config.num_samples, 1) + torch.bmm(coef, basis.repeat(self.config.num_samples, 1, 1)).view(B * self.config.num_samples, self.config.CodeLength)
            
            loss = 0.0
            if hasattr(self.config, 'disentangle_net') and self.config.disentangle_net == 'v3':
                ortho_loss, coeff_loss = self.disentangle_net.get_loss(basis, new_sketch_latent, new_u[shape_index], self.loss_l1_mean)
                loss += ortho_loss + coeff_loss
                loss_dict["ortho_loss"].append(ortho_loss.item())
                loss_dict["coeff_loss"].append(coeff_loss.item())

            else:
                ortho_loss = self.disentangle_net.get_loss(basis)
                loss += ortho_loss
                loss_dict["ortho_loss"].append(ortho_loss.item())
            # TODO: think about encouraging the sparsity of $a$ vector during training to only focus on single axis
            if self.config.train_with_sketch_sdf_loss:
                # 2. sampled feat -> SDF
                sampled_shape_embs = self.latent_flow_network.sample(noise=sampled_shape_latent, cond_inputs=None)
                ##################################### Losses ###########################################
                # 1.SDF Loss for sampled shapes
                # a.sample points from sketch. then normalize sketch_points to align with shape SDF points scale
                sketch_points = points[sketch_index].transpose(2, 1)
                loss_sketch_sdf = self.get_sketch_SDF_loss(sketch_points.repeat(self.config.num_samples, 1, 1), sampled_shape_embs, hull_point.repeat(self.config.num_samples, 1, 1))
                loss += loss_sketch_sdf
                loss_dict["sample_sketch_sdf_loss"].append(loss_sketch_sdf.item())

            if hasattr(self.config, "train_with_sample_emb_loss") and self.config.train_with_sample_emb_loss:
                # 2.latent loss: d(shape_latent, sampled_shape_latent)
                p = self.config.NCE_p if hasattr(self.config, 'NCE_p') else 2
                sample_labels = torch.arange(0, self.config.num_samples).to(device)
                loss_latent = loss_NCE_adapted(sampled_shape_embs.view(B, self.config.num_samples, -1), sample_labels, new_train_feat[shape_index][:B], p=p)
                loss += loss_latent * (self.config.lambda_sample_emb_loss if hasattr(self.config, 'lambda_sample_emb_loss') else 1)
                loss_dict["sample_emb_loss"].append(loss_latent.item())
            
            if hasattr(self.config, "train_with_latent_loss") and self.config.train_with_latent_loss is not None:
                if 'NCE' in self.config.train_with_latent_loss:
                    # L(e_f, e_g')
                    nce_reg_loss = loss_NCE(new_sketch_latent, sketch_labels, new_u[shape_index])
                    loss += nce_reg_loss
                    loss_dict['latent_NCE_loss'].append(nce_reg_loss.item())

            # if hasattr(self.config, 'train_with_sigma_norm_loss') and self.config.train_with_sigma_norm_loss:
            #     l2_size_loss = torch.mean(torch.norm(coef_range[:, :, 1] - coef_range[:, :, 0], dim=1))
            #     loss += F.relu(self.config.sigma_min - l2_size_loss) 
            #     loss_dict["sigma_loss"].append(l2_size_loss.item())

            if is_training and (self.config.train_with_sketch_sdf_loss or self.config.train_with_sample_emb_loss):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        loss_mean = {name: np.mean(np.asarray(loss_dict[name])) for name in loss_dict.keys() if len(loss_dict[name])>0}
        
        print("[{}] Epoch {} ".format('train' if is_training else 'test', epoch) + ' | '.join(['{}: {:.4f}'.format(name, loss_mean[name]) for name in loss_mean.keys()])) 
        return loss_mean

    def inference(self, epoch, samples, selected, split, num_samples = 2, overwrite=False, eval=False, sketch_SDF=False):
        from utils.sdf_utils import create_mesh
        resume_ckpt = os.path.join(self.exp_dir, f'model_epoch_{epoch}.pth')
        self.load_model(resume_ckpt, load_optimizer=False)

        testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}')
        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)

        self.disentangle_net.eval()
        self.latent_flow_network.eval()
        self.encoder.eval()
        
        std_array = []
        dist_array = []
        if sketch_SDF:
            sketch_SDF_loss = []
            self.loss_sdf_fn = torch.nn.L1Loss(reduction="sum")

        B = 4 #num_samples - 2
        coef = (torch.range(0, B-1)/(B-1)).view(B, 1).to(device) #torch.rand(B, 1).to(device) 
        samples_total = B * self.config.num_basis + 2
        with torch.no_grad():
            for index in selected:
                ply_paths = [os.path.join(testing_folder, f'{split}_{index}_sample_{s_index}.ply') for s_index in range(samples_total)]
                exist = 0 
                for item in ply_paths:
                    if os.path.exists(item):
                        exist += 1
                if exist == samples_total and not overwrite:
                    continue
                # inference for single sketch: [sketch, GT, shape_emb, sketch_emb, sampled]
                pc_data = samples[index][0]
                points = torch.tensor(pc_data[:2]).transpose(2, 1).to(device)
                embs = self.encoder(points)
                sketch_emb, shape_emb = embs[0:1], embs[1:2]
                # dist_array.append(torch.dist(sketch_emb, shape_emb, p=2).cpu())

                if hasattr(self.config, 'conditional_NF') and self.config.conditional_NF:
                    noise = torch.Tensor(num_samples - 2, self.config.CodeLength).normal_().to(device) 
                    sampled_shape_embs = self.latent_flow_network.sample(noise=noise, cond_inputs=sketch_emb.repeat(num_samples - 2, 1))
                else:
                    latent, log_jacob = self.latent_flow_network(embs, None)
                    sketch_latent = latent[0:1]
                    basis, coef_range = self.disentangle_net(sketch_emb) # basis:[1,num_basis,256], coef_range[1,num_basis,2]
                    # control different axis
                    offset = []
                    for i in range(self.config.num_basis):
                        # offsetA = torch.matmul(coef_range[:, :, 0], basis).view(1, -1) - basis[:, i]

                        scale = (coef * coef_range[:, i, 0] + (1 - coef) * coef_range[:, i, 1]).view(B, 1)
                        offset.append(torch.matmul(scale, basis[:, i]))
                    
                    sampled_shape_latent = sketch_latent + torch.cat(offset, dim=0)
                    # feed latent to NF and get feature emb
                    sampled_shape_embs = self.latent_flow_network.sample(noise=sampled_shape_latent, cond_inputs=None)

                # decode from shape GT_emb and sketch_emb
                sampled_all_embs = torch.cat([shape_emb, sketch_emb, sampled_shape_embs])
                for id, sampled_shape_emb in enumerate(sampled_all_embs):
                    # feed feature emb to decoder to obtain final shape SDF 
                    mesh_filename = os.path.join(testing_folder, f'{split}_{index}_sample_{id}')
                    success = create_mesh(
                        self.decoder, sampled_shape_emb, mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                if sketch_SDF:
                    sketch_points = points[0:1].transpose(2, 1)
                    # b.feed points to decoder 
                    point_num = sketch_points.shape[1]
                    batch_vecs = torch.repeat_interleave(sketch_emb, point_num, dim = 0)
                    sketch_sdf_values = self.decoder(batch_vecs, sketch_points.reshape(-1, 3))
                    # c.compute SDF L1 loss at sketch points
                    num_sdf_samples = sketch_sdf_values.shape[0]
                    loss_sketch_sdf = self.loss_sdf_fn(sketch_sdf_values, torch.zeros_like(sketch_sdf_values)) / num_sdf_samples
                    sketch_SDF_loss.append(loss_sketch_sdf.cpu()) 
        
        if sketch_SDF:
            wandb.log({f"{split}_sketch_emb_sketch_SDF":np.array(sketch_SDF_loss).mean()})
        if eval:
            np.save(os.path.join(testing_folder, f'{split}_sigma_{len(selected)}.npy'), np.array(std_array))
            print(f'Save sigma array for {split}')
            np.save(os.path.join(testing_folder, f'{split}_dist_{len(selected)}.npy'), np.array(dist_array))
            print(f'Save dist array for {split}')

            self.evaluation(epoch=epoch,  selected=selected, samples=samples,  split=split, num_samples=num_samples, vis=False)

    def vis_generation(self, epoch, samples, selected, split='test', num_samples = 2, id= 0):
        # Visualize generated shapes and input sketch query: 1 sketch + 1 GT + n generated shapes

        testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}')
        assert os.path.exists(testing_folder)

        from utils.pc_utils import vis_pc
        import matplotlib.pyplot as plt
        from utils.ply_utils import render
        from utils.pc_utils import normalize_to_box

        B = 4 
        samples_total = B * self.config.num_basis + 2

        scale_factor = 0.6
        ply_paths = [os.path.join(testing_folder, f'{split}_{sketch_index}_sample_{index}.ply') for sketch_index in selected for index in range(samples_total)]


        nrows = len(selected)
        ncols = 2 + samples_total 
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols * 2, nrows * 2])

        for sketch_index, item in enumerate(selected):
            # inference for single sketch: [sketch, GT, shape_emb, sketch_emb, sampled]
            pc_data = samples[item][0]
            sketch_pc, shape_pc = pc_data[0], pc_data[1]
            axs[sketch_index, 0].imshow(vis_pc(normalize_to_box(sketch_pc)[0]*scale_factor))
            axs[sketch_index, 1].imshow(vis_pc(normalize_to_box(shape_pc)[0]*scale_factor))
            axs[sketch_index, 0].set_title(item)
            for index in range(samples_total):
                axs[sketch_index, 2+index].imshow(render(ply_paths[sketch_index * samples_total + index]))

        samples_list = [f'axis{i}_{j}' for i in range(self.config.num_basis) for j in range(B)]
        [axs[0, i + 1].set_title(title) for i, title in enumerate(['GT', 'shape_emb', 'sketch_emb'] + samples_list)]
        [axi.set_axis_off() for axi in axs.ravel()]
        fig.suptitle(f"Epoch{epoch}_{split}_{id}_Shapes conditioned on sketch")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        img = wandb.Image(fig, caption=f"Epoch{epoch}_{split}_{id}_Shapes conditioned on sketch")
        wandb.log({"img":img, "epoch": epoch})

        plt.savefig(os.path.join(testing_folder, f'Epoch{epoch}_{split}_{id}_generation.png'))
        plt.cla()