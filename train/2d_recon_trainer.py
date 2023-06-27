import torch
import torch.nn as nn
import math
import importlib
import os
import numpy as np
import wandb
from utils.model_utils import dict_from_module, get_spec_with_default, loss_NCE, loss_NCE_adapted
import torch.nn.functional as F
from train.flow_sdf_trainer import Trainer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EXP_DIR = os.getenv('EXP_DIR')

class Trainer_2D(Trainer):
    def init_network(self):
        ## import config here
        self.SamplesPerScene = get_spec_with_default(self.config, "SamplesPerScene", 8192)
        self.ClampingDistance = get_spec_with_default(self.config, "ClampingDistance", 0.1)

        from models.encoder import ngvnn
        encoder = ngvnn.Net_Prev(pretraining=True, num_views=3, ngram_filter_sizes=[3], num_filters=256).to(device)
        self.encoder = nn.DataParallel(encoder)
        from models.decoder.sdf import SDFDecoder

        decoder = SDFDecoder(config=self.config).to(device)
        self.decoder = nn.DataParallel(decoder)

        DATA_DIR = os.getenv('DATA_DIR')

        sdf_decoder_path = os.path.join(DATA_DIR, 'datasets/shape/decoder_latest.pth')
        checkpoint = torch.load(sdf_decoder_path, map_location=device)
        self.decoder.load_state_dict(checkpoint['model_state_dict'])
        latent_code_path = os.path.join(DATA_DIR, 'datasets/shape/latent_code.pth')
        self.gt_emb = torch.load(latent_code_path, map_location=device)['latent_codes']['weight']

        self.decoder.eval()

        from data.SDF_datasets import MultiviewImgDataset
        from torch.utils.data import DataLoader

        self.train_samples = MultiviewImgDataset('train', self.SamplesPerScene, num_views=3, debug=self.debug)
        self.train_data_loader = DataLoader(
                                                dataset=self.train_samples,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                drop_last=True
                                                )

        self.test_samples = MultiviewImgDataset('test', self.SamplesPerScene, num_views=3, debug=self.debug)
        self.test_data_loader = DataLoader(
                                        dataset=self.test_samples,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        drop_last=False
                                    )


    def train_network(self, epochs, saving_intervals, resume=False, use_testing=False):
        self.eval_epoch_fun = get_spec_with_default(self.config, "eval_epoch_fun", "eval_one_epoch")
        self.starting_epoch = 0
        # Prepare optimizer
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.001)

        if hasattr(self.config, "resume_ckpt"):
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

        # latent loss
        if hasattr(self.config, 'latent_loss') and self.config.latent_loss == 'contrastive':
            self.loss_latent_fn = losses.ContrastiveLoss()
        else:
            self.loss_latent_fn = torch.nn.MSELoss()

        eval_one_epoch = getattr(self, self.eval_epoch_fun)
        for epoch in range(self.starting_epoch+1, epochs+1):
            if epoch == 1 :
                self.save_model(0)
            # if self.config.load_Pinaki_model and self.config.train_shape_encoder and epoch <= self.config.encoder_epoch :
                # align embedding space
            # loss_dict = self.eval_encoder_epoch(epoch, self.train_data_loader)
            # loss_dict.update({"epoch": epoch})
            # wandb.log(loss_dict)
            # else:
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

    def eval_one_epoch(self, epoch, data_loader, is_training=True):
        if is_training:
            # self.latent_flow_network.train()
            # train encoder with shape SDF loss
            self.encoder.train()
        else:
            # self.latent_flow_network.eval()
            self.encoder.eval()

        loss_list = ["L1_loss", "NCE_loss", "Triplet_loss", "Contrastive_loss", "prob_loss","sketch_sdf_loss","shape_sdf_loss","latent_loss","sample_sketch_sdf_loss","sigma_loss", "sketch2shape_sdf_loss"]
        loss_dict = {name: [] for name in loss_list}

        for sketches, sdf_data, all_shape_index in data_loader:
            sketches = sketches.view(sketches.size(0)* sketches.size(1), sketches.size(2), sketches.size(3), sketches.size(4))

            loss = 0.0

            sketch_emb = self.encoder(sketches) #torch.Size([6, 512])
            # If don't train encoder for shape, then use latent code from DeepSDF decoder instead
            gt_shape_emb = self.gt_emb[all_shape_index]
            sketch_labels = torch.arange(0, sketch_emb.shape[0]).to(device)
            if 'L1' in self.config.train_with_emb_loss:
                # L1(e_f, e_g')
                l1_reg_loss = self.loss_l1_sum(sketch_emb, gt_shape_emb)
                loss += l1_reg_loss * (self.config.lambda_l1_loss if hasattr(self.config, 'lambda_l1_loss') else 1.0)
                loss_dict['L1_loss'].append(l1_reg_loss.item())
            if 'NCE' in self.config.train_with_emb_loss:
                # L(e_f, e_g')
                nce_reg_loss = loss_NCE(sketch_emb, sketch_labels, gt_shape_emb, 'sum')
                loss += nce_reg_loss
                loss_dict['NCE_loss'].append(nce_reg_loss.item())

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


    def save_model(self, epoch):
        save_model_path = os.path.join(self.exp_dir, f'model_epoch_{epoch}.pth')
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            # 'sigma_layer_state_dict': self.sigma_layer.state_dict(),
            # 'latent_flow_state_dict': self.latent_flow_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'flow_optimizer_state_dict': self.flow_optimizer.state_dict(),
        }, save_model_path)
        print(f"Epoch {epoch} model saved at {save_model_path}")

    def inference(self, epoch, selected, split,  overwrite=False, eval=False, to_mesh=True, save_embed=False, dataset='original'):
        from utils.sdf_utils import create_mesh
        resume_ckpt = os.path.join(self.exp_dir, f'model_epoch_{epoch}.pth')
        if os.path.exists(resume_ckpt):
            self.load_model(resume_ckpt, load_optimizer=False)
        else:
            epoch = 0

        testing_folder = os.path.join(self.exp_dir, f'test_epoch{epoch}_{dataset}')

        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)

        self.encoder.eval()
        torch.manual_seed(0)

        with torch.no_grad():
            for index in selected:
                mesh_filename = os.path.join(testing_folder, f'{split}_{index}_recon')
                if os.path.exists(mesh_filename + '.ply') and not args.overwrite:
                    continue
                sketches = self.test_samples[index][0]
                # sketches = sketches.view(sketches.size(0)* sketches.size(1), sketches.size(2), sketches.size(3), sketches.size(4))

                sketch_emb = self.encoder(sketches) #torch.Size([6, 512])

                success = create_mesh(
                    self.decoder, sketch_emb, mesh_filename, N=256, max_batch=int(2 ** 18)
                )
        
        from utils.ply_utils import sample_ply, compute_SDF_loss
        import pytorch3d.loss

        ply_paths = [os.path.join(testing_folder, f'{split}_{sketch_index}_recon.ply') for sketch_index in selected]
        meshes, gen_pcs_all = sample_ply(ply_paths)
        gen_pcs_all = torch.tensor(gen_pcs_all)
        cd_list = []
        for id, index in enumerate(selected):
            shape_pc = torch.tensor(self.test_samples[index][1])

            cd_dist = pytorch3d.loss.chamfer_distance(gen_pcs_all[index].unsqueeze(0), shape_pc.unsqueeze(0), batch_reduction=None)[0]
            cd_list.append(cd_dist)

        cd_list = torch.cat(cd_list)
        print({f"{split}_sketch_CD": cd_list.mean() * 100}) 
        wandb.log({f"{split}_sketch_CD": cd_list.mean() * 100}) 

if __name__ == '__main__':
    ## additional args for parsing
    import argparse
    optional_args = [("mode", str, "eval"),
                    ("run_id", int, 1), ("resume_path", str, 'configs/flow_sdf/2d_recon.py'),  ("batch_size", int, 6),
                    ("sample_extra", int, 2), ("saving_intervals", int, 5), 
                    ("epoch", int, 30) ,("resume_epoch", int, 20), ("num_samples", int, 5),
                    ("num_gen", int, 7) ,("margin", float, 0.3), #("eval_epoch_fun", str, 'eval_one_epoch_v3'), 
                    ('lambda_latent_loss', float, 1.0), ('lambda_shape_sdf_loss', float, 1.),
                     ('encoder_epoch', int, 0)
                    ]
    parser = argparse.ArgumentParser()
    for optional_arg, arg_type, default in optional_args:
        parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", type=arg_type, default=default)

    parser.add_argument('--overwrite', default=False, action='store_true', help='whether to resume')
    parser.add_argument('--use_testing', default=True, action='store_true', help='whether to test on test set')
    parser.add_argument('--resume_training', default=False, action='store_true', help='whether to resume')
    parser.add_argument('--debug', default=False, action='store_true', help='whether to resume')
    args = parser.parse_args()
    exp_name = os.path.basename(args.resume_path).split('.')[0] + f'_run{args.run_id}'
    ## import config here
    spec = importlib.util.spec_from_file_location('*', args.resume_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    trainer = Trainer_2D(exp_name, config, debug=args.debug, batch_size=args.batch_size,  sample_extra=args.sample_extra)

    if args.mode == 'train':
        trainer.train_network(epochs=args.epoch, saving_intervals=args.saving_intervals, resume=args.resume_training, use_testing=args.use_testing)

    elif args.mode == 'eval':
        if args.debug:
            selected = [*range(10)]
        else:
            selected = [*range(len(trainer.test_samples))]
        trainer.inference(epoch=args.resume_epoch, selected=selected, split='test', overwrite=args.overwrite, eval=True)
