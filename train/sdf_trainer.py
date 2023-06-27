from utils.model_utils import get_latest_ckpt, load_model, adjust_learning_rate, get_learning_rate_schedules, dict_from_module, get_spec_with_default
from models.network import SDFAutoEncoder
import torch
import torch.nn as nn
import os
import argparse
from utils.optutil import generate_meta_info
import wandb
from tqdm import tqdm
from data.SDF_datasets import SDFSamples, Sketch_SDFSamples
from torch.utils.data import DataLoader
import numpy as np
from utils.debugger import MyDebugger
import utils.provider as provider
from utils.sdf_utils import create_mesh
from utils.ply_utils import sample_ply
from utils.pc_utils import normalize_to_box
from einops import rearrange

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(FILE_DIR, os.path.pardir)

class Trainer(object):

    def __init__(self, config, debugger, minfo, args):
        self.debugger = debugger
        self.config = config
        self.minfo = minfo
        self.args = args
        if args.debug:
            self.config.batch_size = 10
            # self.config.saving_intervals = 5
            self.config.vis_interval = 1
        self.init_network()

    def init_network(self):
        clamp_dist = self.config.ClampingDistance
        self.minT = -clamp_dist
        self.maxT = clamp_dist
        self.enforce_minmax = True
        self.grad_clip = get_spec_with_default(self.config, "GradientClipNorm", None)
        self.code_reg_lambda = get_spec_with_default(self.config, "CodeRegularizationLambda", 1e-4)
        self.do_code_regularization = get_spec_with_default(self.config, "CodeRegularization", False)
        self.config.norm_latent=get_spec_with_default(self.config, "norm_latent", None)
        self.train_with_sketch=get_spec_with_default(self.config, "train_with_sketch", False)
        self.freeze_epoch=get_spec_with_default(self.config, "freeze_epoch", 0)
        self.scale_range=get_spec_with_default(self.config, "scale_range", None)
        self.encoder_type = get_spec_with_default(self.config, "encoder_type", 'PointNet')
        list_file = get_spec_with_default(self.config, "list_file", 'sdf_{}.txt')
        if self.grad_clip is not None:
            print("clipping gradients to max norm {}".format(self.grad_clip))

        ## create model
        from models.encoder.pointnet2_cls_msg import get_model
        from models.decoder.sdf import SDFDecoder
        if self.encoder_type == 'PointNet':
            self.encoder = get_model(config=self.config)
        elif config.encoder_type == '3DCNN':
            from models.encoder.cnn_3d import CNN3D
            self.encoder = CNN3D(config=config)
        else:
            assert NotImplementedError

        self.decoder = SDFDecoder(config=self.config)

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Use {torch.cuda.device_count()} GPUS!")
            self.decoder = nn.DataParallel(self.decoder)
            self.encoder = nn.DataParallel(self.encoder)
            self.config.batch_size = self.config.batch_size * torch.cuda.device_count()
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)

        
        ### create dataset

        if self.train_with_sketch:
            from utils.triplet_loss import OnlineTripletLoss, AllNegativeTripletSelector
            self.train_samples = Sketch_SDFSamples('train', self.config.SamplesPerScene, sample_extra=self.config.sample_extra, load_ram=False, debug=self.args.debug)
            self.triplet_loss = OnlineTripletLoss(self.config.margin, AllNegativeTripletSelector())
            self.sketch_index = [i*(2+self.config.sample_extra) for i in range(self.config.batch_size)]
            shape_gt_index = [i+1 for i in self.sketch_index]
            extra_shape_index = [i for i in range(self.config.batch_size*(2 + self.config.sample_extra)) if i not in self.sketch_index + shape_gt_index]
            self.shape_index = shape_gt_index + extra_shape_index

            self.test_sketch_index = [i*2 for i in range(self.config.batch_size)]
            self.test_shape_index = [i+1 for i in self.test_sketch_index]


        else:
            self.train_samples = SDFSamples('train', self.config.SamplesPerScene, load_ram=False, list_file=list_file, debug=self.args.debug)
        self.train_data_loader = DataLoader(
                                        dataset=self.train_samples,
                                        batch_size=self.config.batch_size,
                                        shuffle=True,
                                        num_workers=self.config.data_worker,
                                        drop_last=True
                                        )
        if hasattr(self.config, 'use_testing') and self.config.use_testing:
            if self.train_with_sketch:
                self.test_samples = Sketch_SDFSamples('test', self.config.SamplesPerScene, sample_extra=self.config.sample_extra, load_ram=False, debug=self.args.debug)
            else:
                self.test_samples = SDFSamples('test', self.config.SamplesPerScene, load_ram=False, list_file=list_file, debug=self.args.debug)
            self.test_data_loader = DataLoader(
                                            dataset=self.test_samples,
                                            batch_size=self.config.batch_size,
                                            shuffle=False,
                                            num_workers=self.config.data_worker,
                                            drop_last=True
                                        )



    def train_network(self):
        ### create optimizer and schedules
        lr_schedules = get_learning_rate_schedules(self.config)

        self.optimizer = torch.optim.Adam(
        [
            {
                "params": self.decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": self.encoder.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
        )

        starting_epoch = 0
        last_resume_ckpt, last_optimizer = get_latest_ckpt(exp_path=self.debugger._debug_dir_name, prefix='model_epoch_')
        # Resume from last time
        if self.args.resume_training and  last_resume_ckpt is not None: 
            self.load_model(last_resume_ckpt)
            epoch = os.path.basename(last_resume_ckpt).split('_')[-1][:-4]
            starting_epoch = int(epoch) + 1

        loss_l1 = torch.nn.L1Loss(reduction="sum")

        for epoch in range(starting_epoch, self.config.training_epochs + 1):


            with tqdm(self.train_data_loader, unit='batch') as tepoch:
                tepoch.set_description(f'Epoch {epoch}')
                adjust_learning_rate(lr_schedules, self.optimizer, epoch)
                losses = []
                losses = self.evaluate_one_epoch(loss_l1, losses, tepoch, epoch, is_training = True)
                # lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

                wandb.log({"train_loss": np.mean(losses),"epoch": epoch, "decoder_lr": lr_schedules[0].get_learning_rate(epoch),  "encoder_lr": lr_schedules[1].get_learning_rate(epoch)})
                print(f"Train Loss for epoch {epoch} : {np.mean(losses)}")

                ## saving the models
                if epoch % self.config.saving_intervals == 0:
                    save_model_path = self.debugger.file_path(f'model_epoch_{epoch}.pth')
                    torch.save({
                        'encoder_state_dict': self.encoder.state_dict(),
                        'decoder_state_dict': self.decoder.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, save_model_path)
                    print(f"Epoch {epoch} model saved at {save_model_path}")
                    # Testing
                    if hasattr(self.config, 'use_testing') and self.config.use_testing:
                        with tqdm(self.test_data_loader, unit='batch') as tepoch:
                            losses = []
                            losses = self.evaluate_one_epoch(loss_l1, losses, tepoch, epoch, is_training = False)
                            wandb.log({"test_loss": np.mean(losses),"epoch": epoch})

                    # Visualizing
                    if hasattr(self.config, 'vis_interval') and epoch % self.config.vis_interval == 0:
                        success = self.vis_epoch(epoch)


    def inference(self, epoch):
        # selected = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        resume_ckpt = os.path.join(self.debugger._debug_dir_name, f'model_epoch_{epoch}.pth')
        self.load_model(resume_ckpt)
        testing_folder = os.path.join(self.debugger._debug_dir_name, f'test_epoch{epoch}')
        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            selected = [*range(len(self.test_samples))]

            for idx in selected:
                pc_data, _, _, _ = self.test_samples[idx]
                
                pc_data = pc_data[1] if self.train_with_sketch else pc_data

                pc_data = torch.from_numpy(pc_data).float().to(device)
                latent = self.encoder(pc_data.unsqueeze(0).transpose(2, 1))
                mesh_filename = os.path.join(testing_folder, f'testing_{idx}')
                success = create_mesh(
                    self.decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
            )

            vis_whole_set(self.test_samples, testing_folder, split='test')

            selected = [*range(len(self.train_samples))]

            for idx in selected:
                pc_data, _, _, _ = self.train_samples[idx]
                
                pc_data = pc_data[1] if self.train_with_sketch else pc_data

                pc_data = torch.from_numpy(pc_data).float().to(device)
                latent = self.encoder(pc_data.unsqueeze(0).transpose(2, 1))
                mesh_filename = os.path.join(testing_folder, f'training_{idx}')
                success = create_mesh(
                    self.decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
            )

            vis_whole_set(self.train_samples, testing_folder, split='train')

    def vis_epoch(self, epoch):
        self.encoder.eval()
        self.decoder.eval()
        if self.args.debug:
            selected = [1, 2, 3]
        else:
            selected = [17, 30, 40]
        testing_folder = os.path.join(self.debugger._debug_dir_name, f'test_epoch{epoch}')
        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)

        with torch.no_grad():
            for idx in selected:
                pc_data, voxel_data, _, _ = self.test_samples[idx]
                
                if self.encoder_type == 'PointNet':
                    pc_data = pc_data[1] if self.train_with_sketch else pc_data
                    pc_data = torch.from_numpy(pc_data).float().to(device)
                    latent = self.encoder(pc_data.unsqueeze(0).transpose(2, 1))
                elif self.encoder_type == '3DCNN':
                    voxel_data = torch.from_numpy(voxel_data).to(device)
                    latent = self.encoder(voxel_data.unsqueeze(0))
                else:
                    NotImplementedError
                mesh_filename = os.path.join(testing_folder, f'testing_{idx}')
                success = create_mesh(
                    self.decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
            )
                if not success:
                    return False
            for idx in selected:
                pc_data, voxel_data, _, _ = self.train_samples[idx]
                if self.encoder_type == 'PointNet':
                    pc_data = pc_data[1] if self.train_with_sketch else pc_data
                    pc_data = torch.from_numpy(pc_data).float().to(device)
                    latent = self.encoder(pc_data.unsqueeze(0).transpose(2, 1))
                elif self.encoder_type == '3DCNN':
                    voxel_data = torch.from_numpy(voxel_data).to(device)
                    latent = self.encoder(voxel_data.unsqueeze(0))
                else:
                    NotImplementedError
                mesh_filename = os.path.join(testing_folder, f'training_{idx}')
                success = create_mesh(
                    self.decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
            )
                if not success:
                    return False
        fig = vis_samples(self.train_samples, testing_folder, selected, split='train', with_sketch=self.train_with_sketch)
        img = wandb.Image(fig, caption="Train set")
        wandb.log({"img":img, "epoch": epoch})

        fig = vis_samples(self.test_samples, testing_folder, selected, split='test', with_sketch=self.train_with_sketch)
        img = wandb.Image(fig, caption="Test set")
        wandb.log({"img":img, "epoch": epoch})

    def load_model(self, last_resume_ckpt):
        checkpoint = torch.load(last_resume_ckpt)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 

    def process_pc(self, points, sdf_xyz):
        n_points = points.shape[1]
        points = provider.random_point_dropout(points.numpy())
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        if self.scale_range is not None:
            pc_to_scale = torch.cat([points, sdf_xyz], dim=1)
            new_pc, _ = provider.apply_random_scale_xyz(pc_to_scale, scale=self.scale_range)
            points, sdf_xyz = torch.split(new_pc, [n_points, self.config.SamplesPerScene], dim = 1)
        points = points.transpose(2, 1)
        return points, sdf_xyz

    def process_pc_sketch(self, points, sdf_xyz):

        points = provider.random_point_dropout(points.numpy())
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)

        sketch_pc = points[self.sketch_index]
        shape_pc = points[self.shape_index]
        B, n_points, _ = shape_pc.shape

        if self.scale_range is not None:
            sketch_num = sketch_pc.shape[0]
            sketch_pc = torch.repeat_interleave(sketch_pc, B//sketch_num, dim = 0)

            pc_to_scale = torch.cat([shape_pc, sdf_xyz, sketch_pc], dim=1)
            new_pc, _ = provider.apply_random_scale_xyz(pc_to_scale, scale=self.scale_range)
            shape_pc, sdf_xyz, sketch_pc = torch.split(new_pc, [n_points, self.config.SamplesPerScene, n_points], dim = 1)
            # new_points = torch.zeros_like(points)
            sketch_index = [i*(B//sketch_num) for i in range(sketch_num)]
            points[self.sketch_index] = sketch_pc[sketch_index]
            points[self.shape_index] = shape_pc

        points = points.transpose(2, 1)

        return points, sdf_xyz

    def evaluate_one_epoch(self, loss, losses, tepoch, epoch, is_training):
        ## main training loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if is_training:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        for pc_data, voxel_data, sdf_data, indices in tepoch:
            if self.train_with_sketch:
                pc_data = rearrange(pc_data, 'b h w c -> (b h) w c')
                sdf_data = rearrange(sdf_data, 'b h w c -> (b h) w c')
            # Process the input data
            # sdf_data = sdf_data.reshape(-1, 4)

            num_sdf_samples = sdf_data.shape[0] * sdf_data.shape[1]
            sdf_data.requires_grad = False

            xyz = sdf_data[:, :, 0:3]
            sdf_gt = sdf_data[:, :, 3].reshape(-1, 1)
            
            if is_training and self.encoder_type == 'PointNet':
                if self.train_with_sketch:
                    pc_data, xyz = self.process_pc_sketch(pc_data, xyz)
                else:
                    pc_data, xyz = self.process_pc(pc_data, xyz)
            else:
                pc_data = torch.Tensor(pc_data).transpose(2, 1)

            xyz, sdf_gt =  xyz.to(device), sdf_gt.to(device)

            xyz = xyz.reshape(-1, 3)
            if self.enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, self.minT, self.maxT)

            batch_loss = 0.0

            if is_training:
                self.optimizer.zero_grad()

            if self.encoder_type == 'PointNet':
                pc_data = pc_data.to(device)
                batch_vecs = self.encoder(pc_data)
            elif self.encoder_type == '3DCNN':
                voxel_data = voxel_data.to(device)
                batch_vecs = self.encoder(voxel_data)
            else:
                NotImplementedError

            if self.train_with_sketch:
                # Triplet loss

                if is_training:
                    sketch_latent = batch_vecs[self.sketch_index]
                    shape_latent = batch_vecs[self.shape_index]
                else:
                    sketch_latent = batch_vecs[self.test_sketch_index]
                    shape_latent = batch_vecs[self.test_shape_index]
                
                triplet_loss = self.triplet_loss(sketch_latent, shape_latent) 
                batch_loss = batch_loss + triplet_loss.to(device)
                # repeat_num = int(xyz.shape[0] / latent.shape[0])
                batch_vecs = torch.repeat_interleave(shape_latent, self.config.SamplesPerScene, dim = 0)

                # pred_sdf = self.decoder(shape_latent, xyz)

            else:
                batch_vecs = torch.repeat_interleave(batch_vecs, self.config.SamplesPerScene, dim = 0)

            pred_sdf = self.decoder(batch_vecs, xyz)


            if self.enforce_minmax:
                pred_sdf = torch.clamp(pred_sdf, self.minT, self.maxT)

            sdf_loss = loss(pred_sdf, sdf_gt) / num_sdf_samples
            batch_loss = batch_loss + sdf_loss.to(device)
            losses.append(sdf_loss.detach().item())

            if self.do_code_regularization:
                l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                reg_loss = (
                    self.code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                ) / num_sdf_samples

                batch_loss = batch_loss + reg_loss.to(device)
            if is_training:
                batch_loss.backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)

                self.optimizer.step()
            
            # batch_loss += batch_loss.item()
            tepoch.set_postfix(loss=f'{np.mean(losses)}')

        return losses


def vis_samples(samples, testing_folder, selected, split='test', save='', with_sketch=False):
    from utils.pc_utils import vis_pc
    import matplotlib.pyplot as plt
    scale_factor = 0.6
    ply_paths = [os.path.join(testing_folder, f'{split}ing_{index}.ply') for index in selected]
    recon_pcs = sample_ply(ply_paths)
    recon_pcs, _, _ = normalize_to_box(recon_pcs)

    nrows = 2
    ncols = len(selected)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols, nrows])
    for index, item in enumerate(selected):

        axs[1, index].imshow(vis_pc(recon_pcs[index]*scale_factor))

        shape_pc = samples[item][0]
        if with_sketch:
            shape_pc = shape_pc[1]
        axs[0, index].imshow(vis_pc(normalize_to_box(shape_pc)[0]*scale_factor))

        axs[0, index].set_title(item)
    [axi.set_axis_off() for axi in axs.ravel()]
    plt.tight_layout(True)
    if save != '':
        plt.savefig(os.path.join(testing_folder, f'{save}.png'))
    return fig

def vis_whole_set(samples, testing_folder, split = 'train'):
    from utils.pc_utils import vis_pc
    import matplotlib.pyplot as plt

    scale_factor = 0.6
    # testing_folder = '/mnt/disk1/ling/SketchGen/experiments/sdf_autoencoder_codereg_8192_similar_gpu2_runs_100/test_epoch2000'
    batch_num = len(samples) // 10
    for batch_index in range(batch_num):
        selected = [*range(10 * batch_index, 10 * (batch_index + 1))]
        ply_paths = [os.path.join(testing_folder, f'{split}ing_{index}.ply') for index in selected]
        recon_pcs = sample_ply(ply_paths)
        recon_pcs, _, _ = normalize_to_box(recon_pcs)

        nrows = 2
        ncols = len(selected)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols, nrows])
        for index, item in enumerate(selected):

            axs[1, index].imshow(vis_pc(recon_pcs[index]*scale_factor))

            shape_pc = samples[item][0]
            axs[0, index].imshow(vis_pc(normalize_to_box(shape_pc)[0]*scale_factor))

            axs[0, index].set_title(item)
        [axi.set_axis_off() for axi in axs.ravel()]
        plt.tight_layout(True)

        plt.savefig(os.path.join(testing_folder, f'{split}_batch_{batch_index}.png'))

if __name__ == '__main__':
    import importlib

    ## additional args for parsing
    optional_args = [("run_id", str), ("network_resume_path", str), ("optimizer_resume_path", str), ("starting_epoch", int),
                     ("special_symbol", str), ("resume_path", str),  ("load_epoch", int)]
    parser = argparse.ArgumentParser()
    for optional_arg, arg_type in optional_args:
        parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", type=arg_type)
    parser.add_argument('--debug', default=False, action='store_true', help='whether to use debug mode')
    parser.add_argument('--resume_training', default=True, action='store_true', help='whether to resume')
    parser.add_argument('--inference', default=False, action='store_true', help='whether to do inference only')

    args = parser.parse_args()


    ## Resume setting
    resume_path = None

    ## resume from path if needed
    if args.resume_path is not None:
        resume_path = args.resume_path

    if resume_path is None:
        from configs import config
        resume_path = os.path.join('configs', 'config.py')
    else:
        ## import config here
        spec = importlib.util.spec_from_file_location('*', resume_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

    for optional_arg, arg_type in optional_args:
        if args.__dict__.get(optional_arg, None) is not None:
            locals()['config'].__setattr__(optional_arg, args.__dict__.get(optional_arg, None))

    minfo = generate_meta_info(DEFAULT_ROOT, 'SDF_sketch', src_name='src')

    subdir = os.path.basename(os.path.dirname(resume_path))
    model_type = os.path.basename(resume_path).split('.')[0] 
    # if args.resume_exp == "":
    exp_name = f'{model_type}_gpu{torch.cuda.device_count()}_runs_{args.run_id}'
    debugger = MyDebugger([subdir, exp_name], is_save_print_to_file = False if args.debug else True, config_path = resume_path, debug_dir=minfo['experiments_dir'])
    # if not args.debug:
    wandb.login()
    wandb_id_file = os.path.join(debugger._debug_dir_name, 'wandb_id.txt')
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
        project="SDF-Autoencoder", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=debugger.model_name, 
        # Track hyperparameters and run metadata
        dir=os.getenv('CACHE_DIR'),
        config=dict_from_module(config),
        resume="allow"
    )
    trainer = Trainer(config = config, debugger = debugger, minfo=minfo, args=args)
    if args.inference:
        trainer.inference(epoch=args.load_epoch)
    else:
        trainer.train_network()
    wandb.finish()