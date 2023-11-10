import torch
import torch.nn as nn
import math
import importlib
import os
import numpy as np
from einops import rearrange
import wandb
from utils.model_utils import dict_from_module, get_spec_with_default, loss_NCE, loss_NCE_adapted
import torch.nn.functional as F
from utils.pc_utils import sample_farthest_points

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EXP_DIR = os.getenv('EXP_DIR')

import signal, sys

import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

torch.autograd.set_detect_anomaly(True)
def sigterm_handler(signal, frame):
    # save the state here or do whatever you want
    print('caught sigint!')
    # fname = fpath + 'sigint_caught.txt'
    # open(fname,'w').close()
    wandb.finish()
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)

class Trainer(object):
    def __init__(self, exp_name, config, debug, batch_size = 4, sample_extra = 1):
        self.config = config
        self.debug = debug
    
        self.batch_size = batch_size
        self.sample_extra =  sample_extra
        self.exp_dir = os.path.join(EXP_DIR, exp_name)
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)

        self.init_network()
        wandb.login()
        wandb_id_file = os.path.join(self.exp_dir, 'wandb_id.txt')
        if os.path.exists(wandb_id_file):
            with open(wandb_id_file, 'r') as f:
                id = f.readlines()[0]
        else:
            id = wandb.util.generate_id()
            with open(wandb_id_file, 'w') as f:
                f.write(id)

        wandb.init(
            id=id,
            # Set the project where this run will be logged
            project="SDF-Flow", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=exp_name, 
            # Track hyperparameters and run metadata
            dir=os.getenv('CACHE_DIR'),
            config=dict_from_module(config),
            resume=True
        )

    def init_network(self):

        # Prepare encoder and decoder

        ## import config here
        self.SamplesPerScene = get_spec_with_default(self.config, "SamplesPerScene", 8192)
        self.ClampingDistance = get_spec_with_default(self.config, "ClampingDistance", 0.1)

        from models.encoder.pointnet2_cls_msg import PointNetEncoder
        from models.decoder.sdf import SDFDecoder
        # Prepare sketch-shape pairs
        from data.SDF_datasets import Sketch_SDFSamples
        from torch.utils.data import DataLoader

        encoder = PointNetEncoder(k=self.config.CodeLength).to(device)
        self.encoder = nn.DataParallel(encoder)
        decoder = SDFDecoder(config=self.config).to(device)
        self.decoder = nn.DataParallel(decoder)
        DATA_DIR = os.getenv('DATA_DIR')
        if not hasattr(self.config, 'encoder_from_scratch'):
            encoder_path = os.path.join(DATA_DIR, 'models/stage1_decoder/encoder_latest.pth')
            checkpoint = torch.load(encoder_path, map_location=device)
            self.encoder.load_state_dict(checkpoint['model_state_dict'])

        sdf_decoder_path = os.path.join(DATA_DIR, 'models/stage1_decoder/decoder_latest.pth')
        checkpoint = torch.load(sdf_decoder_path, map_location=device)
        self.decoder.load_state_dict(checkpoint['model_state_dict'])
        latent_code_path = os.path.join(DATA_DIR, 'models/stage1_decoder/latent_code.pth')
        self.gt_emb = torch.load(latent_code_path, map_location=device)['latent_codes']['weight']
        from data.SDF_datasets import SDFSamples
        self.train_shape_samples = SDFSamples('train', self.SamplesPerScene, load_ram=False, list_file='sdf_{}.txt', debug=self.debug)
        self.train_shape_data_loader = DataLoader(
                                                dataset=self.train_shape_samples, 
                                                batch_size=self.batch_size * 2,
                                                shuffle=True,
                                                num_workers=4,
                                                drop_last=True
                                                )
        self.decoder.eval()
        self.sigma_layer = nn.Linear(self.config.CodeLength, self.config.CodeLength).to(device)

        # Prepare normalzing flow
        from models.decoder import latent_flows
        if hasattr(self.config, 'conditional_NF') and self.config.conditional_NF:
            self.latent_flow_network = latent_flows.get_generator(num_inputs=self.config.CodeLength, num_cond_inputs=self.config.CodeLength, device=device, flow_type='realnvp_half', num_blocks=5, num_hidden=1024)
        else:
            self.latent_flow_network = latent_flows.get_generator(num_inputs=self.config.CodeLength, num_cond_inputs=None, device=device, flow_type='realnvp_half', num_blocks=5, num_hidden=1024)

        # add extra shapes and define index
        self.train_samples = Sketch_SDFSamples('train', 8192, sample_extra=self.sample_extra, load_ram=False, debug=self.debug)
        # Prepare test dataloader
        self.test_samples = Sketch_SDFSamples('test', 8192, sample_extra=0, load_ram=False, debug=self.debug)

        self.train_data_loader = DataLoader(
                                                dataset=self.train_samples,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                drop_last=True
                                                )
        self.test_data_loader = DataLoader(
                                                dataset=self.test_samples,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                drop_last=True
                                            )
        self.sketch_index = [i*(2+self.sample_extra) for i in range(self.batch_size)]
        self.shape_gt_index = [i+1 for i in self.sketch_index]
        extra_shape_index = [i for i in range(self.batch_size*(2 + self.sample_extra)) if i not in self.sketch_index + self.shape_gt_index]
        self.shape_index = self.shape_gt_index + extra_shape_index
        sdf_gt_index = [i*(1+self.sample_extra) for i in range(self.batch_size)]
        sdf_extra_shape_index = [i for i in range(self.batch_size*(1 + self.sample_extra)) if i not in sdf_gt_index]
        self.sdf_index = sdf_gt_index + sdf_extra_shape_index

        self.test_sketch_index = [i*2 for i in range(self.batch_size)]
        self.test_shape_index = [i+1 for i in self.test_sketch_index]
        self.test_sdf_index = [i for i in range(self.batch_size)]

    def load_model(self, last_resume_ckpt, load_optimizer=True, load_NF=True):
        checkpoint = torch.load(last_resume_ckpt, map_location=device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if hasattr(self, "sigma_layer") and load_optimizer:
            self.sigma_layer.load_state_dict(checkpoint['sigma_layer_state_dict'])
        if hasattr(self, "latent_flow_network") and load_NF:
            self.latent_flow_network.load_state_dict(checkpoint['latent_flow_state_dict'])

        if hasattr(self, "optimizer") and load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        if hasattr(self, "flow_optimizer") and 'flow_optimizer_state_dict' in checkpoint.keys() and load_optimizer:
            self.flow_optimizer.load_state_dict(checkpoint['flow_optimizer_state_dict']) 

    def train_network(self, epochs, saving_intervals, resume=False, use_testing=False):
        self.eval_epoch_fun = get_spec_with_default(self.config, "eval_epoch_fun", "eval_one_epoch")
        self.starting_epoch = 0
        # Prepare optimizer
        self.flow_optimizer = torch.optim.Adam(self.latent_flow_network.parameters(), lr=0.00003)
        self.optimizer = torch.optim.Adam(
            [
            {"params": self.sigma_layer.parameters(), "lr": 0.001},
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
        self.loss_l1_sum = torch.nn.L1Loss(reduction="sum")
        self.loss_l1 = torch.nn.L1Loss(reduction="mean")
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

            if self.config.load_AE_model and self.config.train_shape_encoder and epoch <= self.config.encoder_epoch :
                # stage1: train Autoencoder: align embedding space
                loss_dict = self.eval_encoder_epoch(epoch, self.train_shape_data_loader)
                loss_dict.update({"epoch": epoch})
                wandb.log(loss_dict)
            else:
                # stage2: train Flow model
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

    def get_sketch_SDF_loss(self, sketch_points, sketch_emb):
        # b.feed points to decoder 
        point_num = sketch_points.shape[1]
        batch_vecs = torch.repeat_interleave(sketch_emb, point_num, dim = 0)
        sketch_sdf_values = self.decoder(batch_vecs, sketch_points.reshape(-1, 3))
        # c.compute SDF L1 loss at sketch points
        loss_sketch_sdf = self.loss_l1(sketch_sdf_values, torch.zeros_like(sketch_sdf_values)) 
        # if hasattr(self.config, 'hull_point') and self.config.hull_point:
        #     hull_sdf_values = self.decoder(batch_vecs, hull_point[:, :, :3].reshape(-1, 3))
        #     if hasattr(self.config, 'hull_point_max'):
        #         hull_sdf_loss = F.relu(self.config.hull_point_max - hull_sdf_values)
        #     else:
        #         hull_sdf_loss = F.relu(hull_point[:, :, -1].reshape(-1, 1) - hull_sdf_values)
        #     loss_sketch_sdf += hull_sdf_loss.mean()

        return loss_sketch_sdf

    def save_model(self, epoch):
        save_model_path = os.path.join(self.exp_dir, f'model_epoch_{epoch}.pth')
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'sigma_layer_state_dict': self.sigma_layer.state_dict(),
            'latent_flow_state_dict': self.latent_flow_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'flow_optimizer_state_dict': self.flow_optimizer.state_dict(),
        }, save_model_path)
        print(f"Epoch {epoch} model saved at {save_model_path}")

    def eval_encoder_epoch(self, epoch, data_loader):
        emb_loss_fun = nn.MSELoss()
        loss_emb_array = []
        self.encoder.train()

        for pc_data, indices in data_loader:
            shape_points = pc_data.transpose(2, 1).to(device)
            shape_gt_emb = self.gt_emb[indices]

            shape_feat = self.encoder(shape_points)
            loss = emb_loss_fun(shape_feat, shape_gt_emb)
            loss_emb_array.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print("[{}] Epoch {} | Emb loss: {:.4f}".format('Train', epoch, np.mean(loss_emb_array)) )

        return {
                "train_emb_loss": np.mean(np.array(loss_emb_array))
                }

    def eval_one_epoch_AE(self, epoch, data_loader, is_training=True):
        if is_training:
            self.sigma_layer.train()
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
            self.sigma_layer.eval()
            self.latent_flow_network.eval()
            self.encoder.eval()
            sketch_index = self.test_sketch_index
            shape_index = self.test_shape_index
            sdf_index = self.test_sdf_index

        loss_list = ["L1_loss", "NCE_loss", "Triplet_loss", "Contrastive_loss", "prob_loss","sketch_sdf_loss","shape_sdf_loss","latent_loss","sample_sketch_sdf_loss","sigma_loss", "sketch2shape_sdf_loss"]
        loss_dict = {name: [] for name in loss_list}

        for pc_data, sdf_data, indices, all_shape_index in data_loader:
            pc_data = rearrange(pc_data, 'b h w c -> (b h) w c')
            if pc_data.shape[1] != self.config.num_points:
                points = sample_farthest_points(pc_data.transpose(1, 2), self.config.num_points).to(device)
            else:
                points = pc_data.transpose(2, 1).to(device)

            sdf_data = rearrange(sdf_data, 'b h w c -> (b h) w c')[sdf_index]
            all_shape_index = all_shape_index.reshape(-1)
            

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
                    l1_reg_loss = self.loss_l1_sum(sketch_emb, gt_batch_vecs[:sketch_emb.shape[0]])
                    if self.config.train_shape_encoder:
                    # L1(e_f, e_f')
                        l1_reg_loss += self.loss_l1_sum(shape_emb, gt_batch_vecs)
                        l1_reg_loss = l1_reg_loss/2
                    loss += l1_reg_loss * (self.config.lambda_l1_loss if hasattr(self.config, 'lambda_l1_loss') else 1.0)
                    loss_dict['L1_loss'].append(l1_reg_loss.item())
                if 'NCE' in self.config.train_with_emb_loss:
                    # L(e_f, e_g')
                    nce_reg_loss = loss_NCE(sketch_emb, sketch_labels, gt_batch_vecs, 'sum')
                    if self.config.train_shape_encoder:
                        nce_reg_loss += loss_NCE(shape_emb, shape_labels, gt_batch_vecs, 'sum')
                        nce_reg_loss = nce_reg_loss/2
                    loss += nce_reg_loss * (self.config.lambda_NCE_loss if hasattr(self.config, 'lambda_NCE_loss') else 1.0)
                    loss_dict['NCE_loss'].append(nce_reg_loss.item())
                if 'sketch_sdf' in self.config.train_with_emb_loss:
                    sketch_points = points[sketch_index].transpose(2, 1)
                    loss_sketch_sdf = self.get_sketch_SDF_loss(sketch_points, sketch_emb)
                    loss += loss_sketch_sdf
                    loss_dict["sketch_sdf_loss"].append(loss_sketch_sdf.item())
                if 'shape_sdf' in self.config.train_with_emb_loss:
                    # 3. SDF loss for shape reconstruction for finetuning encoder
                    xyz = sdf_data[:, :, 0:3].reshape(-1, 3).to(device)
                    shape_batch_vecs = torch.repeat_interleave(shape_emb, self.SamplesPerScene, dim = 0)
                    pred_sdf = self.decoder(shape_batch_vecs, xyz)
                    pred_sdf = torch.clamp(pred_sdf, - self.ClampingDistance, self.ClampingDistance)
                    sdf_gt = sdf_data[:, :, 3].reshape(-1, 1).to(device)
                    sdf_gt = torch.clamp(sdf_gt, - self.ClampingDistance, self.ClampingDistance)
                    loss_shape_sdf = self.loss_l1(pred_sdf, sdf_gt) # / sdf_gt.shape[0]
                    if self.config.train_shape_encoder:
                        loss += loss_shape_sdf * self.config.lambda_shape_sdf_loss
                    loss_dict["shape_sdf_loss"].append(loss_shape_sdf.item())
                if 'shape_sdf_sketch' in self.config.train_with_emb_loss:
                    B = sketch_emb.shape[0]
                    # 3. SDF loss for shape reconstruction for finetuning encoder
                    xyz = sdf_data[:B, :, 0:3].reshape(-1, 3).to(device)
                    sketch_batch_vecs = torch.repeat_interleave(sketch_emb, self.SamplesPerScene, dim = 0)
                    pred_sdf = self.decoder(sketch_batch_vecs, xyz)
                    pred_sdf = torch.clamp(pred_sdf, - self.ClampingDistance, self.ClampingDistance)
                    sdf_gt = sdf_data[:B, :, 3].reshape(-1, 1).to(device)
                    sdf_gt = torch.clamp(sdf_gt, - self.ClampingDistance, self.ClampingDistance)
                    loss_shape_sdf = self.loss_l1(pred_sdf, sdf_gt) # / sdf_gt.shape[0]
                    loss += loss_shape_sdf
                    loss_dict["sketch2shape_sdf_loss"].append(loss_shape_sdf.item())

            if is_training and loss > 0:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        loss_mean = {name: np.mean(np.asarray(loss_dict[name])) for name in loss_dict.keys() if len(loss_dict[name])>0}
        
        print("[{}] Epoch {} ".format('train' if is_training else 'test', epoch) + ' | '.join(['{}: {:.4f}'.format(name, loss_mean[name]) for name in loss_mean.keys()])) 
        return loss_mean

    def eval_one_epoch_Flow(self, epoch, data_loader, is_training=True):
        # conditional NF
        if is_training:
            self.sigma_layer.train()
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
            self.sigma_layer.eval()
            self.latent_flow_network.eval()
            self.encoder.eval()
            sketch_index = self.test_sketch_index
            shape_index = self.test_shape_index
            sdf_index = self.test_sdf_index

        loss_list = ["L1_loss", "NCE_loss", "Triplet_loss", "Contrastive_loss", "prob_loss","sketch_sdf_loss","shape_sdf_loss","latent_loss","sample_sketch_sdf_loss","sample_emb_div", "sample_emb_rec"]
        loss_dict = {name: [] for name in loss_list}

        for pc_data, sdf_data, indices, all_shape_index in data_loader:
            pc_data = rearrange(pc_data, 'b h w c -> (b h) w c')
            if pc_data.shape[1] != self.config.num_points:
                points = sample_farthest_points(pc_data.transpose(1, 2), self.config.num_points).to(device)
            else:
                points = pc_data.transpose(2, 1).to(device)

            sdf_data = rearrange(sdf_data, 'b h w c -> (b h) w c')[sdf_index]
            all_shape_index = all_shape_index.reshape(-1)
            # hull_point = hull_point.to(device)

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
                if 'shape_sdf' in self.config.train_with_emb_loss:
                    # 3. SDF loss for shape reconstruction for finetuning encoder
                    xyz = sdf_data[:, :, 0:3].reshape(-1, 3).to(device)
                    shape_batch_vecs = torch.repeat_interleave(shape_emb, self.SamplesPerScene, dim = 0)
                    pred_sdf = self.decoder(shape_batch_vecs, xyz)
                    pred_sdf = torch.clamp(pred_sdf, - self.ClampingDistance, self.ClampingDistance)
                    sdf_gt = sdf_data[:, :, 3].reshape(-1, 1).to(device)
                    sdf_gt = torch.clamp(sdf_gt, - self.ClampingDistance, self.ClampingDistance)
                    loss_shape_sdf = self.loss_l1(pred_sdf, sdf_gt)# / sdf_gt.shape[0]
                    if self.config.train_shape_encoder:
                        loss += loss_shape_sdf * self.config.lambda_shape_sdf_loss
                    loss_dict["shape_sdf_loss"].append(loss_shape_sdf.item())

            if self.config.train_with_nf_loss:
                # Feed feat vectors to normalzing flow and get latent code and loss
                # TODO: why choose 0.1?
                # train_embs = shape_emb + 0.1 * torch.randn(shape_emb.size(0), self.config.CodeLength).to(device)
                u, log_jacob = self.latent_flow_network(shape_emb[:sketch_emb.shape[0]], sketch_emb)
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
            # 1. inverse flow: latent code -> feat, sampled from sketch latent code's neighborhood(decided by std sigma) N(sketch_emb, sigma)
            new_train_feat = self.encoder(points)
            new_shape_emb = new_train_feat[shape_index]
            new_sketch_emb = new_train_feat[sketch_index]
            B = new_sketch_emb.shape[0]
            if not self.config.train_shape_encoder:
                # use GT latent code of decoder instead
                new_train_feat[shape_index] = self.gt_emb[all_shape_index][sdf_index]

            # log_var = self.sigma_layer(new_sketch_emb)
            new_u, log_jacob = self.latent_flow_network(new_shape_emb[:new_sketch_emb.shape[0]], new_sketch_emb)

            loss = 0.0
            # 2. sampled feat -> SDF
            noise = torch.Tensor(self.config.num_samples * B, self.config.CodeLength).normal_().to(device) 
            sampled_shape_embs = self.latent_flow_network.sample(noise=noise, cond_inputs=new_sketch_emb.repeat(self.config.num_samples,1))
            # sample N (self.config.num_samples) samples for each sketch cond
            if hasattr(self.config, "train_with_sketch_sdf_loss") and self.config.train_with_sketch_sdf_loss:
                ##################################### Losses ###########################################
                # 1.SDF Loss for sampled shapes
                # a.sample points from sketch. then normalize sketch_points to align with shape SDF points scale
                sketch_points = points[sketch_index].transpose(2, 1)
                loss_sketch_sdf = self.get_sketch_SDF_loss(sketch_points.repeat(self.config.num_samples, 1, 1), sampled_shape_embs)
                loss += loss_sketch_sdf * (self.config.lambda_sketch_sdf_loss if hasattr(self.config, 'lambda_sketch_sdf_loss') else 1.0)
                loss_dict["sample_sketch_sdf_loss"].append(loss_sketch_sdf.item())

            if is_training and loss > 0:
                # Optimize NF by sketch sdf loss
                self.flow_optimizer.zero_grad()
                if hasattr(self.config, 'optimize_encoder') and self.config.optimize_encoder:
                    self.optimizer.zero_grad()
                loss.backward()
                if not hasattr(self.config, 'not_optimize_NF'):
                    self.flow_optimizer.step()
                if hasattr(self.config, 'optimize_encoder') and self.config.optimize_encoder:
                    self.optimizer.step()


        loss_mean = {name: np.mean(np.asarray(loss_dict[name])) for name in loss_dict.keys() if len(loss_dict[name])>0}
        
        print("[{}] Epoch {} ".format('train' if is_training else 'test', epoch) + ' | '.join(['{}: {:.4f}'.format(name, loss_mean[name]) for name in loss_mean.keys()])) 
        return loss_mean


    def resume(self):
        from glob import glob
        files = glob(os.path.join(self.exp_dir, "model_epoch_" + '*.pth'))
        if len(files) > 0:
            last_ckpt = sorted(files, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split('_')[-1]))[-1]
        else:
            last_ckpt = None
        
        if last_ckpt is not None:
            self.load_model(last_ckpt)
            epoch = os.path.basename(last_ckpt).split('_')[-1][:-4]
            self.starting_epoch = int(epoch) + 1


    def debug_model(self, epoch, samples, selected, split='train', num_samples=2):
        # Load ckpt
        # if epoch > 0:
        resume_ckpt = os.path.join(self.exp_dir, f'model_epoch_{epoch}.pth')
        self.load_model(resume_ckpt)

        self.sigma_layer.eval()
        self.latent_flow_network.eval()
        self.encoder.eval()

        emb_array = []
        latent_array = []
        indices_list = []
        std_array = []
        with torch.no_grad():
            for index in selected:
                # inference for single sketch and GT shape:
                # TODO: get top-10 closest shapes as extra shape samples
                pc_data, sdf_data, indices, _ = samples[index]
                indices_list.append(indices)
                points = torch.tensor(pc_data[:2]).transpose(2, 1).to(device)
                # get feature
                embs = self.encoder(points)
                emb_array.append(embs.cpu().numpy())

                # get latent
                latent, log_jacob = self.latent_flow_network(embs, None)
                latent_array.append(latent.cpu().numpy())

                # save sigma
                sketch_embs = embs[0:1]
                log_var = self.sigma_layer(sketch_embs)
                std = torch.exp(0.5 * log_var)
                std_array.append(torch.norm(std))
                std_repeat = std.repeat(num_samples, 1)
                sketch_latent = latent[0:1]
                sampled_shape_latent = sketch_latent + torch.randn_like(std).to(device) * std_repeat
                latent_array.append(sampled_shape_latent.cpu().numpy())

        emb_array = np.concatenate(emb_array)
        latent_array = np.concatenate(latent_array)
        def compute_mean_dist(feat_array):

            feat_array = feat_array / np.linalg.norm(feat_array, axis=1, keepdims=True)
            N = len(indices_list)#int(feat_array.shape[0] // 2)
            batch_size = int(feat_array.shape[0] // N)
            indexes = [i * batch_size for i in range(N)]
            GT_indexes = [i + 1 for i in indexes]
            dist = (feat_array[indexes] - feat_array[GT_indexes])**2
            dist = np.sum(dist, axis=1)
            pos_dist_mean = np.sqrt(dist).mean()

            neg_dist = []
            for index in indexes:
                GT_index = index + 1
                neg = [item  for item in GT_indexes if item != GT_index]
                dist = (feat_array[index] - feat_array[neg])**2
                dist = np.sum(dist, axis=1)
                dist_mean = np.sqrt(dist).mean()
                neg_dist.append(dist_mean)
            neg_dist_mean = np.array(neg_dist).mean()
            return pos_dist_mean, neg_dist_mean
        # Plot T-sne
        pos_dist_mean, neg_dist_mean = compute_mean_dist(emb_array)
        print('Epoch{} split_{}: {:.4f} {:.4f} {:.4f}'.format(epoch, split, pos_dist_mean, neg_dist_mean, neg_dist_mean - pos_dist_mean))
        self.plot_T_sne(epoch, emb_array, indices_list, save_name='emb_space_epoch{}_{}_{:.4f}_{:.4f}_{:.4f}'.format(epoch, split, pos_dist_mean, neg_dist_mean, neg_dist_mean - pos_dist_mean))
        pos_dist_mean, neg_dist_mean = compute_mean_dist(latent_array)
        self.plot_T_sne(epoch, latent_array, indices_list, save_name='latent_space_epoch{}_{}_{:.4f}_{:.4f}_{:.4f}'.format(epoch, split, pos_dist_mean, neg_dist_mean, neg_dist_mean - pos_dist_mean), num_samples=num_samples, std_array=std_array)

    def plot_T_sne(self, epoch, features, labels, save_name, num_samples=0, std_array=None):
        testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}')
        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)

        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2).fit_transform(features)
        import matplotlib.cm as cm
        colors = cm.rainbow(np.linspace(0, 1, len(labels)))
        bias = 0.02
        # scale and move the coordinates so they fit [0; 1] range
        def scale_to_01_range(x):
            # compute the distribution range
            value_range = (np.max(x) - np.min(x))
        
            # move the distribution so that it starts from zero
            # by extracting the minimal value from all its values
            starts_from_zero = x - np.min(x)
        
            # make the distribution fit [0; 1] by dividing by its range
            return starts_from_zero / value_range
        
        # extract x and y coordinates representing the positions of the images on T-SNE plot
        tx = tsne[:, 0]
        ty = tsne[:, 1]
        
        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # for every class, we'll add a scatter plot separately
        for index, label in enumerate(labels):
            # find the samples of the current class in the data
            indices = range(index * (2 + num_samples), (index+1) * (2 + num_samples))
        
            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
                
            # add a scatter plot with the corresponding color and label
            ax.scatter(current_tx[1], current_ty[1], c=colors[index].reshape(1,-1), label=label)
            ax.scatter(current_tx[0], current_ty[0], marker='*', c=colors[index], label=label)
            for id in range(2, len(current_tx)):
                ax.scatter(current_tx[id], current_ty[id], alpha = 0.3, c=colors[index], label=label)

            if std_array is not None:
                ax.annotate("{:.0f}".format(std_array[index] * 100), (current_tx[0]+bias, current_ty[0]+bias))

        # build a legend using the labels we set previously
        # ax.legend(loc='best')
        ax.title.set_text(save_name)
        # finally, show the plot
        img = wandb.Image(fig, caption=save_name)
        wandb.log({"img":img, "epoch": epoch})

        plt.savefig(os.path.join(testing_folder, f'{save_name}.png') )
        plt.cla()


    def inference(self, epoch, samples, selected, split, num_samples = 2, overwrite=False, eval=False, to_mesh=True, save_embed=False, dataset='original'):
        from utils.sdf_utils import create_mesh
        resume_ckpt = os.path.join(self.exp_dir, f'model_epoch_{epoch}.pth')
        if os.path.exists(resume_ckpt):
            self.load_model(resume_ckpt, load_optimizer=False)
        else:
            epoch = 0

        testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}_{dataset}')
        # testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}')

        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)

        self.sigma_layer.eval()
        self.latent_flow_network.eval()
        self.encoder.eval()
        
        torch.manual_seed(0)
        all_embeds = []
        with torch.no_grad():
            for index in selected:
                ply_paths = [os.path.join(testing_folder, f'{split}_{index}_sample_{s_index}.ply') for s_index in range(num_samples)]
                exist = 0 
                for item in ply_paths:
                    if os.path.exists(item):
                        exist += 1
                if exist == num_samples and not overwrite:
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
                    sketch_emb = self.latent_flow_network.sample(noise=torch.zeros(1, self.config.CodeLength).to(device), cond_inputs=sketch_emb)
                else:
                    latent, log_jacob = self.latent_flow_network(embs, None)
                    log_var = self.sigma_layer(sketch_emb) # sketch_emb
                    std = torch.exp(0.5 * log_var)
                    # std_array.append(torch.norm(std).cpu())

                    std_repeat = std.repeat(num_samples - 2, 1) 
                    sketch_latent = latent[0:1]
                    sampled_shape_latent = sketch_latent + torch.randn_like(std_repeat).to(device) * std_repeat
                    # feed latent to NF and get feature emb
                    sampled_shape_embs = self.latent_flow_network.sample(noise=sampled_shape_latent, cond_inputs=None)

                # decode from shape GT_emb and sketch_emb
                sampled_all_embs = torch.cat([shape_emb, sketch_emb, sampled_shape_embs])
                all_embeds.append(sampled_all_embs)
                if to_mesh:
                    for id, sampled_shape_emb in enumerate(sampled_all_embs):
                        # feed feature emb to decoder to obtain final shape SDF 
                        mesh_filename = os.path.join(testing_folder, f'{split}_{index}_sample_{id}')
                        success = create_mesh(
                            self.decoder, sampled_shape_emb, mesh_filename, N=256, max_batch=int(2 ** 18)
                        )
        if eval:
            self.evaluation(epoch=epoch,  selected=selected, samples=samples,  split=split, num_samples=num_samples, vis=False)

        if save_embed:
            # testing_folder = '/mnt/disk1/ling/SketchGen'
            np.save(os.path.join(testing_folder, 'generation_embed.npy'), np.array(torch.cat(all_embeds, 0).cpu().data))
            print(f'Save shape embed array for {split}')

    def inference_shape(self, epoch, num_samples = 2):
        from utils.sdf_utils import create_mesh
        resume_ckpt = os.path.join(self.exp_dir, f'model_epoch_{epoch}.pth')
        self.load_model(resume_ckpt, load_optimizer=False)

        testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}')
        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)

        self.sigma_layer.eval()
        self.latent_flow_network.eval()
        self.encoder.eval()
        
        with torch.no_grad():
            mean_shape = torch.zeros(1, self.config.CodeLength).to(device)  
            sampled_shape_latent = torch.Tensor(num_samples-1, self.config.CodeLength).normal_().to(device) 
            noise = torch.cat([mean_shape, sampled_shape_latent], dim=0)
            # feed latent to NF and get feature emb
            sampled_shape_embs = self.latent_flow_network.sample(noise=noise, cond_inputs=None)            
            for id, sampled_shape_emb in enumerate(sampled_shape_embs):
                # feed feature emb to decoder to obtain final shape SDF 
                mesh_filename = os.path.join(testing_folder, f'shape_sample_{id}')
                success = create_mesh(
                    self.decoder, sampled_shape_emb, mesh_filename, N=256, max_batch=int(2 ** 18)
                )
                k = 0
                # resampling in case fails to decode the mesh
                while not success and k<100 and id>0:
                    sampled_shape_latent = torch.Tensor(1, self.config.CodeLength).normal_().to(device) 
                    new_sampled_shape_emb = self.latent_flow_network.sample(noise=sampled_shape_latent, cond_inputs=None)
                    success = create_mesh(
                        self.decoder, new_sampled_shape_emb, mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                    k += 1

        self.vis_generation_shape(epoch=epoch, num_samples=num_samples)


    def vis_generation(self, epoch, samples, selected, split='test', num_samples = 2, id= 0):
        # Visualize generated shapes and input sketch query: 1 sketch + 1 GT + n generated shapes

        testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}')
        assert os.path.exists(testing_folder)

        from utils.pc_utils import normalize_to_box
        from utils.vis_utils import vis_pc, vis_mesh
        import matplotlib.pyplot as plt
        from utils.ply_utils import render

        scale_factor = 0.6
        ply_paths = [os.path.join(testing_folder, f'{split}_{sketch_index}_sample_{index}.ply') for sketch_index in selected for index in range(num_samples)]


        nrows = len(selected)
        ncols = 2 + num_samples # + 2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols * 2, nrows * 2])

        for sketch_index, item in enumerate(selected):
            # inference for single sketch: [sketch, GT, shape_emb, sketch_emb, sampled]
            pc_data = samples[item][0]
            sketch_pc, shape_pc = pc_data[0], pc_data[1]
            axs[sketch_index, 0].imshow(vis_pc(normalize_to_box(sketch_pc)[0]*scale_factor))
            axs[sketch_index, 1].imshow(vis_pc(normalize_to_box(shape_pc)[0]*scale_factor))
            axs[sketch_index, 0].set_title(item)
            for index in range(num_samples):
                # axs[sketch_index, 2+index].imshow(render(ply_paths[sketch_index * num_samples + index]))
                axs[sketch_index, 2+index].imshow(vis_mesh(ply_paths[sketch_index * num_samples + index]))

        # plot sigma and d(sketch_emb, shape_emb)
        # std_path = os.path.join(testing_folder, f'{split}_sigma_{len(self.test_samples)}.npy')
        # std_array = np.load(std_path) if os.path.exists(std_path) else None
        # if std_array is not None:
        #     [axs[sketch_index, 2].text(0, 0, '{:.2f}'.format(std_array[item]), ha='center') for sketch_index in range(len(selected))]

        # dist_path = os.path.join(testing_folder, f'{split}_dist_{len(self.test_samples)}.npy')
        # dist_array = np.load(dist_path) if os.path.exists(dist_path) else None
        # if dist_array is not None:
        #     [axs[sketch_index, 3].text(0, 0, '{:.2f}'.format(dist_array[item]), ha='center') for sketch_index in range(len(selected))]


        # axs[0, 0].set_title(item)
        axs[0, 1].set_title('GT')
        # axs[0, 2].set_title('sigma')
        # axs[0, 3].set_title('dist(e_f,e_g)')

        axs[0, 2].set_title('shape_emb')
        axs[0, 3].set_title('sketch_emb')

        [axi.set_axis_off() for axi in axs.ravel()]
        fig.suptitle(f"Epoch{epoch}_{split}_{id}_Shapes conditioned on sketch")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        img = wandb.Image(fig, caption=f"Epoch{epoch}_{split}_{id}_Shapes conditioned on sketch")
        wandb.log({"img":img, "epoch": epoch})

        plt.savefig(os.path.join(testing_folder, f'Epoch{epoch}_{split}_{id}_generation.png'))
        plt.cla()

    def vis_interpolation(self, epoch, samples, selected, split='test', id=0):
        from utils.sdf_utils import create_mesh
        resume_ckpt = os.path.join(self.exp_dir, f'model_epoch_{epoch}.pth')
        self.load_model(resume_ckpt, load_optimizer=False)

        testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}')
        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)

        self.sigma_layer.eval()
        self.latent_flow_network.eval()
        self.encoder.eval()

        ratio_list = [0.0, 0.25, 0.5, 0.75, 1.0]
        import random
        import matplotlib.pyplot as plt
        from utils.pc_utils import vis_pc
        from utils.ply_utils import render
        from utils.pc_utils import normalize_to_box
        scale_factor = 0.6

        nrows = len(selected) * 2
        ncols = 2 + len(ratio_list) 
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols * 2, nrows * 2])

        with torch.no_grad():
            for ax_index, index in enumerate(selected):
                pc_data, sdf_data, indices, _ = samples[index]
                b_index = random.randrange(len(samples))

                points = torch.tensor(pc_data[:2]).transpose(2, 1).to(device)
                pc_data_B = samples[b_index][0]
                points_B = torch.tensor(pc_data_B[:2]).transpose(2, 1).to(device)

                embs = self.encoder(torch.cat([points, points_B]))
                sketch_emb, shape_emb, sketch_B_emb, shape_B_emb = embs[0:1], embs[1:2], embs[3:4], embs[-1:]
                latent, log_jacob = self.latent_flow_network(embs, None)
                sketch_latent, shape_latent, sketch_B_latent, shape_B_latent = latent[0:1], latent[1:2], latent[3:4], latent[-1:]

                shape_pc, shape_pc_B = pc_data[1], pc_data_B[1]
                axs[ax_index * 2, 0].imshow(vis_pc(normalize_to_box(shape_pc)[0]*scale_factor))
                axs[ax_index * 2, ncols-1].imshow(vis_pc(normalize_to_box(shape_pc_B)[0]*scale_factor))


                # interpolation between shapes in emb space and latent space
                for col_index, ratio in enumerate(ratio_list):
                    interpolated_emb = torch.lerp(shape_emb, shape_B_emb, ratio)
                    interpolated_latent = torch.lerp(shape_latent, shape_B_latent, ratio)
                    sampled_shape_embs = self.latent_flow_network.sample(noise=interpolated_latent, cond_inputs=None)

                    # feed feature emb to decoder to obtain final shape SDF 
                    mesh_filename = os.path.join(testing_folder, f'{split}_{index}_interpolate_emb_{col_index}')
                    success = create_mesh(
                        self.decoder, interpolated_emb[0], mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                    axs[ax_index * 2, 1 + col_index].imshow(render(mesh_filename + '.ply'))

                    mesh_filename = os.path.join(testing_folder, f'{split}_{index}_interpolate_latent_{col_index}')
                    success = create_mesh(
                        self.decoder, sampled_shape_embs[0], mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                    axs[ax_index * 2 + 1, 1 + col_index].imshow(render(mesh_filename + '.ply'))

        [axs[0, i].set_title(title) for i, title in enumerate(['A'] + ratio_list + ['B'])]
        [axi.set_axis_off() for axi in axs.ravel()]
        fig.suptitle(f"Epoch{epoch}_{split}_{id}_shape_interpolation")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        img = wandb.Image(fig, caption=f"Epoch{epoch}_{split}_{id}_shape_interpolation")
        wandb.log({"img":img, "epoch": epoch})

        plt.savefig(os.path.join(testing_folder, f'Epoch{epoch}_{split}_{id}_shape_interpolation.png'))
        plt.cla()


    def vis_sigma_variants(self, epoch, samples, selected, split='test', num_samples = 2, id= 0):
        from utils.sdf_utils import create_mesh
        resume_ckpt = os.path.join(self.exp_dir, f'model_epoch_{epoch}.pth')
        self.load_model(resume_ckpt, load_optimizer=False)

        testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}')
        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)

        self.sigma_layer.eval()
        self.latent_flow_network.eval()
        self.encoder.eval()

        import matplotlib.pyplot as plt
        from utils.pc_utils import vis_pc
        from utils.ply_utils import render
        from utils.pc_utils import normalize_to_box
        scale_factor = 0.6
        scale_list = [0, 1, 2, 4, 8, 10]
        nrows = len(selected) 
        ncols = 2 + len(scale_list) 
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols * 2, nrows * 2])


        with torch.no_grad():
            for ax_index, index in enumerate(selected):
                pc_data, sdf_data, indices, _ = samples[index]
                points = torch.tensor(pc_data[:2]).transpose(2, 1).to(device)
                embs = self.encoder(points)
                sketch_emb, shape_emb = embs[0:1], embs[1:2]
                latent, log_jacob = self.latent_flow_network(embs, None)

                sketch_pc, shape_pc = pc_data[0], pc_data[1]
                axs[ax_index, 0].imshow(vis_pc(normalize_to_box(sketch_pc)[0]*scale_factor))
                axs[ax_index, 1].imshow(vis_pc(normalize_to_box(shape_pc)[0]*scale_factor))

                log_var = self.sigma_layer(sketch_emb) # sketch_emb
                std = torch.exp(0.5 * log_var)

                # std_repeat = std.repeat(len(scale_list), 1) 

                sketch_latent = latent[0:1]
                noise = torch.randn_like(std).to(device) 
                
                # TODO: scaling the std
                for col_index, scale in enumerate(scale_list):
                    sampled_shape_latent = sketch_latent + noise * std * scale
                    
                    # feed latent to NF and get feature emb
                    sampled_shape_embs = self.latent_flow_network.sample(noise=sampled_shape_latent, cond_inputs=None)
                    # feed feature emb to decoder to obtain final shape SDF 
                    mesh_filename = os.path.join(testing_folder, f'{split}_{index}_sigma_variant_{col_index}')
                    success = create_mesh(
                        self.decoder, sampled_shape_embs, mesh_filename, N=256, max_batch=int(2 ** 18)
                    )

                    axs[ax_index, 2 + col_index].imshow(render(mesh_filename + '.ply'))
        
                [axs[ax_index, i].set_title(title) for i, title in enumerate(['sketch', 'GT'] + ['%.2f' % torch.norm(elem * std).cpu() for elem in torch.tensor(scale_list) ])]

        [axi.set_axis_off() for axi in axs.ravel()]
        fig.suptitle(f"Epoch{epoch}_{split}_{id}_sigma_variant")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        img = wandb.Image(fig, caption=f"Epoch{epoch}_{split}_{id}_sigma_variant")
        wandb.log({"img":img, "epoch": epoch})

        plt.savefig(os.path.join(testing_folder, f'Epoch{epoch}_{split}_{id}_sigma_variant.png'))
        plt.cla()


    def vis_generation_shape(self, epoch, num_samples=2, ncols=10):
        # Visualize generated shapes and input sketch query: 1 sketch + 1 GT + n generated shapes

        testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}')
        assert os.path.exists(testing_folder)

        from utils.pc_utils import vis_pc
        import matplotlib.pyplot as plt
        from utils.ply_utils import sample_ply
        from utils.pc_utils import normalize_to_box

        scale_factor = 0.6

        ply_paths = [os.path.join(testing_folder, f'shape_sample_{index}.ply') for index in range(num_samples)]
        gen_pcs = sample_ply(ply_paths)
        gen_pcs, _, _ = normalize_to_box(gen_pcs)

        nrows = num_samples // ncols + 1
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols, nrows])

        for index in range(num_samples):
            x, y = index // ncols, index % ncols
            axs[x, y].imshow(vis_pc(gen_pcs[index]*scale_factor))
            axs[x, y].set_title('mean' if index==0 else f's{index}')
        axs[0, 0].set_title('mean')
        [axi.set_axis_off() for axi in axs.ravel()]
        fig.suptitle(f"Epoch{epoch} shapes sampled from latent space")
        plt.tight_layout(True)

        img = wandb.Image(fig, caption=f"Epoch{epoch} shapes sampled from latent space")
        wandb.log({"img":img, "epoch": epoch})

        plt.savefig(os.path.join(testing_folder, f'sampled_shapes.png'))
        plt.cla()

    def evaluation(self, epoch, selected, samples,  split='test', num_samples = 2, vis=False, save=False, dataset='original'):
        import pytorch3d.loss
        from utils.ply_utils import sample_ply, compute_SDF_loss

        testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}_{dataset}')
        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)
        
        ply_paths = [os.path.join(testing_folder, f'{split}_{sketch_index}_sample_{index}.ply') for sketch_index in selected for index in range(num_samples)]
        meshes, gen_pcs_all = sample_ply(ply_paths)

        cd_list = []
        div_cd_list = []
        sketch_SDF_list = []
        # hull_SDF_list = []
        sample_num = num_samples - 2

        for id, index in enumerate(selected):
            # inference for single sketch: [sketch, GT, shape_emb, sketch_emb, sampled]
            data = samples[index]
            pc_data = data[0]
            points = torch.tensor(pc_data[:2])
            gen_pcs = gen_pcs_all[id * num_samples: (id + 1) * num_samples]
            cd_dist = pytorch3d.loss.chamfer_distance(gen_pcs, points[1:2].repeat(num_samples, 1, 1), batch_reduction=None)[0]
            cd_list.append(cd_dist)
            # diversity of sampled shapes
            target = torch.repeat_interleave(gen_pcs[2:], sample_num, dim = 0)
            div_cd_dist = pytorch3d.loss.chamfer_distance(gen_pcs[2:].repeat(sample_num, 1, 1), target, batch_reduction=None)[0]
            div_cd_list.append(div_cd_dist.sum()/(div_cd_dist.shape[0] - sample_num))
            # SDF loss towards sketch and hull
            # index_list = [*range(id * num_samples+2, (id + 1) * num_samples)]
            index_list = [*range(id * num_samples+1, (id + 1) * num_samples)]

            sketch_SDF = compute_SDF_loss(points[0], meshes, index_list)
            sketch_SDF_list.append(sketch_SDF)
            # hull_SDF_list.append(hull_SDF)
        cd_list = torch.cat(cd_list).reshape(-1, num_samples)
        if save:
            np.save(os.path.join(testing_folder, 'cd.npy'), np.array(cd_list.cpu().data))
            print(f'Save shape embed array for {split}')

            np.save(os.path.join(testing_folder, 'sketch_SDF.npy'), np.array(sketch_SDF_list))
            print(f'Save shape embed array for {split}')

        wandb.log({f"{split}_sketch_CD":cd_list[:, 1].mean() * 100,f"{split}_shape_CD":cd_list[:, 0].mean() * 100, f"{split}_samples_CD":cd_list[:, 2:].mean() * 100, 
                    f"{split}_samples_div":torch.tensor(div_cd_list).mean() * 100, 
                    f"{split}_samples_sketch_SDF_mean":torch.tensor(sketch_SDF_list).mean(), 
                    f"{split}_samples_sketch_SDF_std":torch.tensor(sketch_SDF_list).std(), 
                    # f"{split}_samples_hull_SDF":((torch.tensor(hull_SDF_list) > 0) * 1.0).mean(),              
                    "epoch": epoch})
        print({f"{split}_sketch_CD":cd_list[:, 1].mean() * 100,f"{split}_shape_CD":cd_list[:, 0].mean() * 100, f"{split}_samples_CD":cd_list[:, 2:].mean() * 100, 
                    f"{split}_samples_div":torch.tensor(div_cd_list).mean() * 100, 
                    f"{split}_samples_sketch_SDF_mean":torch.tensor(sketch_SDF_list).mean(), 
                    f"{split}_samples_sketch_SDF_std":torch.tensor(sketch_SDF_list).std(), 
                    # f"{split}_samples_hull_SDF":((torch.tensor(hull_SDF_list) > 0) * 1.0).mean(),              
                    "epoch": epoch})
    
    def retrieve(self, split='test', data='shape'):
        self.load_model(os.path.join(EXP_DIR, self.config.resume_ckpt), load_optimizer=False, load_NF=False)
        if data == 'shape':
            from data.SDF_datasets import shape_samples
            shape_dataset = shape_samples(split=split)
        elif data == 'sketch':
            from data.SDF_datasets import sketch_samples
            shape_dataset = sketch_samples(split=split)
        from torch.utils.data import DataLoader
        test_data_loader = DataLoader(
                                                dataset=shape_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                drop_last=False
                                            )
        shape_embeds = []

        self.encoder.eval()

        with torch.no_grad():
            for shape_pc, index in test_data_loader:
                points = torch.tensor(shape_pc).transpose(2, 1).to(device)
                embs = self.encoder(points)
                shape_embeds.append(embs)
        
        testing_folder = '/mnt/disk1/ling/SketchGen'
        np.save(os.path.join(testing_folder, f'{split}_embed_{len(shape_dataset)}_{data}.npy'), np.array(torch.cat(shape_embeds, 0).cpu().data))
        print(f'Save shape embed array for {split}')
    
    def recon(self, split='test', data='sketch'):
        from utils.sdf_utils import create_mesh

        # testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}')
        from data.SDF_datasets import sketch_samples
        shape_dataset = sketch_samples(split=split)

        testing_folder = '/mnt/disk1/ling/SketchGen'
        sketch_emb = torch.tensor(np.load(os.path.join(testing_folder, f'{split}_embed_{len(shape_dataset)}_{data}.npy'))).to(device)

        for id, sampled_shape_emb in enumerate(sketch_emb):
            # feed feature emb to decoder to obtain final shape SDF 
            mesh_filename = os.path.join(testing_folder, 'recon_sketch', f'{split}_{id}')
            success = create_mesh(
                self.decoder, sampled_shape_emb, mesh_filename, N=256, max_batch=int(2 ** 18))

    def interp_z_space(self, epoch, samples):
        # selected = {
        #     '95': [1, 5],
        #     '98': [3, 6],
        #     '101': [2,5]
        #     '71': [1,6],
        #     '124': [1,4],
        #     '167': [1,3],
        #     '180': [1,6],
        #     '157': [4, 3]
        # }
        selected = [
            [95, 2, 3],
            [95, 3, 4],
            [95, 4, 5],
            [95, 5, 6],
            [95, 6, 2],
            [98, 2, 3],
            [98, 3, 4],
            [98, 4, 5],
            [98, 5, 6],
            [98, 6, 2],
        ]
        from utils.sdf_utils import create_mesh
        resume_ckpt = os.path.join(self.exp_dir, f'model_epoch_{epoch}.pth')
        self.load_model(resume_ckpt, load_optimizer=False)

        testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}_original')
        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)

        embed_array = np.load(os.path.join(testing_folder, 'generation_embed.npy'))
        saving_folder = os.path.join(testing_folder, 'interp')
        if not os.path.exists(saving_folder):
            os.mkdir(saving_folder)

        self.sigma_layer.eval()
        self.latent_flow_network.eval()
        self.encoder.eval()

        # ratio_list = [0.0, 0.25, 0.5, 0.75, 1.0]
        ratio_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        with torch.no_grad():
            for indexes in selected:
                index, a_index, b_index = indexes

                index = int(index)
                pc_data = samples[index][0]
                points = torch.tensor(pc_data[:1]).transpose(2, 1).to(device)
                sketch_emb = self.encoder(points)
            
                a_embed, b_embed = embed_array[index * 7 + a_index], embed_array[index * 7 + b_index]
                embs = np.concatenate([[a_embed], [b_embed]], axis=0)
                latent, log_jacob = self.latent_flow_network(torch.tensor(embs).to(device), cond_inputs=sketch_emb.repeat(2, 1))
                A_latent, B_latent = latent[0], latent[1]

                for col_index, ratio in enumerate(ratio_list):
                    interpolated_latent = torch.lerp(A_latent, B_latent, ratio)
                    sampled_shape_embs = self.latent_flow_network.sample(noise=interpolated_latent.view(1, -1), cond_inputs=sketch_emb)

                    # feed feature emb to decoder to obtain final shape SDF 
                    mesh_filename = os.path.join(saving_folder, f'{index}_{a_index}_{b_index}_interpolate_z_{col_index}')
                    success = create_mesh(
                        self.decoder, sampled_shape_embs[0], mesh_filename, N=256, max_batch=int(2 ** 18)
                    )


if __name__ == '__main__':
    ## additional args for parsing
    import argparse
    optional_args = [("mode", str, "eval"),
                    # ("run_id", int, 0), 
                    ("resume_path", str, 'configs/stage2_GenNF.py'), 
                    ("batch_size", int, 6),
                    ("sample_extra", int, 2), 
                    ("saving_intervals", int, 5), 
                    ("epoch", int, 20),
                    ("resume_epoch", int, 5), 
                    ("num_samples", int, 5),
                    ("num_gen", int, 7), 
                    ("margin", float, 0.3), 
                    ('lambda_latent_loss', float, 1.0), 
                    ('lambda_shape_sdf_loss', float, 1.),
                    ('sigma_min', float, 0.), 
                    ('encoder_epoch', int, 0)
                    ]
    parser = argparse.ArgumentParser()
    for optional_arg, arg_type, default in optional_args:
        parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", type=arg_type, default=default)
    parser.add_argument('--resume_training', default=False, action='store_true', help='whether to resume')
    parser.add_argument('--use_testing', default=True, action='store_true', help='whether to test on test set')
    parser.add_argument('--debug', default=False, action='store_true', help='whether to resume')
    parser.add_argument('--overwrite', default=False, action='store_true', help='whether to resume')
    parser.add_argument('--train_shape_encoder',choices=('True','False'), default='False')
    parser.add_argument('--unseen', default=False, action='store_true', help='whether to resume')
    parser.add_argument('--seen', default=False, action='store_true', help='whether to resume')

    args = parser.parse_args()

    ## import config here
    spec = importlib.util.spec_from_file_location('*', args.resume_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    config.margin = args.margin
    config.lambda_latent_loss = args.lambda_latent_loss
    config.lambda_shape_sdf_loss = args.lambda_shape_sdf_loss
    if not hasattr(config, 'sigma_min') :
        config.sigma_min = args.sigma_min
    config.encoder_epoch = args.encoder_epoch
    if not hasattr(config, 'load_AE_model') :
        config.load_AE_model = False
    if not hasattr(config, 'train_shape_encoder') :
        config.train_shape_encoder = eval(args.train_shape_encoder)

    if not hasattr(config, 'num_points') :
        config.num_points = 4096

    # if not hasattr(config, 'num_basis'):
    #     config.num_basis = args.num_basis

    # exp_name = os.path.basename(args.resume_path).split('.')[0] + f'_run{args.run_id}'
    exp_name = os.path.basename(args.resume_path).split('.')[0]

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        args.batch_size = torch.cuda.device_count() * args.batch_size

    if hasattr(config, 'disentangle'):
        from train.others.disentangled_flow_sdf_trainer import Disentangle_Trainer
        trainer = Disentangle_Trainer(exp_name, config, debug=args.debug, batch_size=args.batch_size,  sample_extra=args.sample_extra)
    else:
        trainer = Trainer(exp_name, config, debug=args.debug, batch_size=args.batch_size,  sample_extra=args.sample_extra)

    def inference():
        if not hasattr(config, 'conditional_NF'):
            trainer.inference_shape(epoch=args.resume_epoch, num_samples=30)
        # selected = [134, 196, 34, 163, 187]# np.random.choice(len(trainer.train_samples), args.num_samples, replace=False)
        # trainer.inference(epoch=args.resume_epoch, samples=trainer.train_samples, selected=selected, split='train', num_samples=args.num_gen)
        selected = [29, 82] # np.random.choice(len(trainer.test_samples), args.num_samples, replace=False)
        trainer.inference(epoch=args.resume_epoch, samples=trainer.test_samples, selected=selected, split='test', num_samples=args.num_gen)

    def debug():
        train_selected = np.random.choice(len(trainer.train_samples), 100, replace=False)
        test_selected = np.random.choice(len(trainer.test_samples), 100, replace=False)

        # Hint: must load epoch 0 before later ones
        trainer.debug_model(epoch=0, samples=trainer.train_samples, selected=train_selected, split='train')
        trainer.debug_model(epoch=0, samples=trainer.test_samples, selected=test_selected, split='test')

        trainer.debug_model(epoch=args.resume_epoch, samples=trainer.train_samples, selected=train_selected, split='train')
        trainer.debug_model(epoch=args.resume_epoch, samples=trainer.test_samples, selected=test_selected, split='test')

    def evaluation():
        # name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}_sketch.txt')
        # sdf_name_list = [line.rstrip() for line in open(name_list_path)]
        name_path = os.path.join(os.getenv('DATA_DIR'), 'split/test_5participants.txt')
        set_names = [line.rstrip().split('_')[1] for line in open(name_path)]
        unseen_list = [trainer.test_samples.name_list.index(name) for name in set_names]
        if args.debug:
            selected = [134, 196, 34, 163, 187]# np.random.choice(len(trainer.train_samples), args.num_samples, replace=False)
        elif args.unseen:
            # load unseen 5 participants id
            selected = unseen_list
        elif args.seen:
            all = [*range(0, len(trainer.test_samples))]
            selected = [index for index in all if index not in unseen_list]
        else:
            selected = [*range(len(trainer.test_samples))]
        trainer.inference(epoch=args.resume_epoch, samples=trainer.test_samples, selected=selected, split='test', num_samples=args.num_gen, overwrite=args.overwrite, eval=True)

    def vis():
        if args.debug:
            subset_list = [np.array([134, 196, 34, 163, 187])]
        else:
            selected = [*range(len(trainer.test_samples))]
            # groupA, groupB = trainer.analysis_sigma()
            # arr = groupA + groupB
            arr = selected # range(len(samples))
            subset_list = np.array_split(arr, 20)
        for id, subset in enumerate(subset_list):
            trainer.vis_generation(epoch=args.resume_epoch,  samples=trainer.test_samples, selected = subset.tolist(), split='test', num_samples = args.num_gen, id=id)

    def infer_other(dataset):
        from data.SDF_datasets import sketch_samples
        samples = sketch_samples(split='test', dataset=dataset)
        # selected = [1]
        selected = [*range(len(samples))]
        trainer.inference(args.resume_epoch, samples, selected, 'test', num_samples = args.num_gen, overwrite=False, eval=False, to_mesh=True, save_embed=False, dataset=dataset)

    def eval_other(dataset):
        from data.SDF_datasets import sketch_samples
        samples = sketch_samples(split='test', dataset=dataset)
        selected = [*range(len(samples))]
        trainer.inference(args.resume_epoch, samples, selected, 'test', num_samples = args.num_gen, overwrite=False, eval=True, to_mesh=True, save_embed=False, dataset=dataset)
        # trainer.evaluation(epoch=args.resume_epoch,  selected=selected, samples=samples,  split='test', num_samples=args.num_gen, vis=False, dataset=dataset)

    if args.mode == 'train':
        trainer.train_network(epochs=args.epoch, saving_intervals=args.saving_intervals, resume=args.resume_training, use_testing=args.use_testing)
        inference()
        evaluation()
    elif args.mode=='debug':
        debug()
    elif args.mode == 'inference':
        inference()
    elif args.mode == 'eval':
        evaluation()
        # vis()
    elif args.mode == 'vis':
        vis()
    elif args.mode == 'vis_interp':
        selected = [134, 196, 34, 163, 187]
        trainer.vis_interpolation(epoch=args.resume_epoch, samples=trainer.test_samples, selected=selected, split='test', id=0)
    elif args.mode == 'vis_sigma_variants':
        selected = [134, 196, 34, 163, 187]
        trainer.vis_sigma_variants(epoch=args.resume_epoch, samples=trainer.test_samples, selected=selected, split='test', id=0)
    elif args.mode == 'retrieve':
        trainer.retrieve(data='sketch')
    elif args.mode == 'save_embed':
        if args.debug:
            selected = [134, 196, 34, 163, 187] # np.random.choice(len(trainer.train_samples), args.num_samples, replace=False)
        else:
            selected = [*range(len(trainer.test_samples))]
        trainer.inference(epoch=args.resume_epoch, samples=trainer.test_samples, selected=selected, split='test', num_samples=args.num_gen, overwrite=args.overwrite, eval=False, to_mesh=False, save_embed=True)
    elif args.mode == 'save_metric':
        selected = [*range(len(trainer.test_samples))]
        # trainer.inference(epoch=args.resume_epoch, samples=trainer.test_samples, selected=selected, split='test', num_samples=args.num_gen, overwrite=args.overwrite, eval=True, to_mesh=False, save_embed=False)
        trainer.evaluation(epoch=args.resume_epoch,  selected=selected, samples=trainer.test_samples,  split='test', num_samples=args.num_gen, vis=False, save=True)
    elif args.mode == 'recon':
        trainer.recon(data='sketch')
    elif args.mode =='infer_other':
        # infer_other('human_sketch')
        # infer_other('network')
        infer_other('FVRS_M')
    elif args.mode =='eval_other':
        eval_other('FVRS_M')
    elif args.mode =='interp_z':
        trainer.interp_z_space(epoch=args.resume_epoch, samples=trainer.test_samples)
    else:
        debug()
        inference()
    wandb.finish()
