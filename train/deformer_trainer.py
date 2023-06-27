from utils.model_utils import get_latest_ckpt, load_model, adjust_learning_rate, get_learning_rate_schedules, dict_from_module, get_spec_with_default
from models.deformer.keypointdeformer import KP_Deformer
import torch
import torch.nn as nn
import os
import argparse
from utils.optutil import generate_meta_info
import wandb
from tqdm import tqdm
from data.SDF_datasets import PCSamples
from torch.utils.data import DataLoader
import numpy as np
from utils.debugger import MyDebugger
import utils.provider as provider
from utils.sdf_utils import create_mesh
from utils.ply_utils import sample_ply
from utils.pc_utils import normalize_to_box, sample_farthest_points
from einops import rearrange
import pytorch3d.loss

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
            # self.config.batch_size = 10
            self.config.saving_intervals = 5
            self.config.vis_interval = 1

    def train_network(self):
        self.num_points = get_spec_with_default(self.config, 'num_points', 1024)

        ## create model
        network = KP_Deformer(config=self.config)

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Use {torch.cuda.device_count()} GPUS!")
            network = nn.DataParallel(network)
            self.config.batch_size = self.config.batch_size * torch.cuda.device_count()
        network = network.to(device)
        
        ### create optimizer and schedules
        lr_schedules = get_learning_rate_schedules(self.config)

        params = [{"params": self.influence_predictor.parameters()}]
        optimizer = torch.optim.Adam(params, lr=self.config.lr)
        optimizer.add_param_group({'params': network.influence_param, 'lr': 10 * self.config.lr})
        params = [{"params": network.keypoint_predictor.parameters()}]
        keypoint_optimizer = torch.optim.Adam(params, lr=self.config.lr)

        optimizers = [optimizer, keypoint_optimizer]

        ### create dataset
        train_samples = PCSamples('train', debug=self.args.debug)

        train_data_loader = DataLoader(
                                        dataset=train_samples,
                                        batch_size=self.config.batch_size,
                                        shuffle=True,
                                        num_workers=self.config.data_worker,
                                        drop_last=True
                                        )
        if hasattr(self.config, 'use_testing') and self.config.use_testing:
            test_samples = PCSamples('test', debug=self.args.debug)
            test_data_loader = DataLoader(
                                            dataset=test_samples,
                                            batch_size=self.config.batch_size,
                                            shuffle=False,
                                            num_workers=self.config.data_worker,
                                            drop_last=True
                                        )

        starting_epoch = 0
        last_resume_ckpt, last_optimizer, last_keypoint_optimizer = self.get_latest_ckpt(exp_path=self.debugger._debug_dir_name, prefix='model_epoch_')
        # Resume from last time
        if self.args.resume_training and  last_resume_ckpt is not None: 
            network, optimizer, keypoint_optimizer = load_model(network, optimizer, keypoint_optimizer, last_resume_ckpt, last_optimizer, last_keypoint_optimizer)
            epoch = os.path.basename(last_resume_ckpt).split('_')[-1][:-4]
            starting_epoch = int(epoch) + 1


        for epoch in range(starting_epoch, self.config.training_epochs + 1):
            with tqdm(train_data_loader, unit='batch') as tepoch:
                tepoch.set_description(f'Epoch {epoch}')
                adjust_learning_rate(lr_schedules, optimizer, epoch)
                losses = ()
                losses = self.evaluate_one_epoch(losses, network, optimizers, tepoch, epoch, is_training = True)
                # lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

                wandb.log({"train_loss": np.mean(losses),"epoch": epoch})
                print(f"Train Loss for epoch {epoch} : {np.mean(losses)}")

                ## saving the models
                if epoch % self.config.saving_intervals == 0:
                    save_model_path = self.debugger.file_path(f'model_epoch_{epoch}.pth')
                    save_optimizer_path = self.debugger.file_path(f'optimizer_epoch_{epoch}.pth')
                    save_keypoint_optimizer_path = self.debugger.file_path(f'keypoint_optimizer_epoch_{epoch}.pth')
                    torch.save(network.state_dict(), save_model_path)
                    torch.save(optimizer.state_dict(), save_optimizer_path)
                    torch.save(keypoint_optimizer.state_dict(), save_keypoint_optimizer_path)

                    print(f"Epoch {epoch} model saved at {save_model_path}")
                    print(f"Epoch {epoch} optimizer saved at {save_optimizer_path}")
                    # Testing
                    if hasattr(self.config, 'use_testing') and self.config.use_testing:
                        with tqdm(test_data_loader, unit='batch') as tepoch:
                            losses = []
                            losses = self.evaluate_one_epoch(losses, network, optimizers, tepoch, epoch, is_training = False)
                            wandb.log({"test_loss": np.mean(losses),"epoch": epoch})

                    # Visualizing
                    # if hasattr(self.config, 'vis_interval') and epoch % self.config.vis_interval == 0:
                    #     success = self.vis_epoch(test_samples, network, epoch)

def load_model(self, network, optimizer, keypoint_optimizer, resume_ckpt, optimizer_path, keypoint_optimizer_path):
    from utils.model_utils import process_state_dict
    network_state_dict = torch.load(resume_ckpt)
    new_net_dict = network.state_dict()
    network_state_dict = process_state_dict(network_state_dict)
    pretrained_dict = {k: v for k, v in network_state_dict.items() if k in new_net_dict}
    new_net_dict.update(pretrained_dict)
    network.load_state_dict(new_net_dict)
    network.train()
    print(f"Reloaded the network from {resume_ckpt}")

    if optimizer_path is not None:
        optimizer_state_dict = torch.load(optimizer_path)
        optimizer.load_state_dict(optimizer_state_dict)
        print(f"Reloaded the optimizer from {optimizer_path}")
        optimizer_state_dict = torch.load(keypoint_optimizer_path)
        keypoint_optimizer.load_state_dict(optimizer_state_dict)
        print(f"Reloaded the keypoint optimizer from {keypoint_optimizer_path}")
    return network, optimizer, keypoint_optimizer

def get_latest_ckpt(self, exp_path, prefix):
    from glob import glob
    files = glob(os.path.join(exp_path, prefix + '*.pth'))
    if len(files) > 0:
        file = sorted(files, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split('_')[-1]))[-1]
    else:
        file = None
    optimizer_path = None
    keypoint_optimizer_path = None
    if file is not None:
        # file = str(file)
        optimizer_path = file.replace('model_epoch', 'optimizer_epoch') 
        keypoint_optimizer_path = file.replace('model_epoch', 'keypoint_optimizer_epoch') 
    return file, optimizer_path, keypoint_optimizer_path

    def vis_epoch(self, samples, network, epoch):
        network.eval()

        selected = [17, 30, 40]
        testing_folder = os.path.join(self.debugger._debug_dir_name, f'test_epoch{epoch}')
        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)

        with torch.no_grad():
            for idx in selected:
                pc_data, sdf_data, indices = samples[idx]
                
                pc_data = pc_data[0] if self.train_with_sketch else pc_data

                pc_data = torch.from_numpy(pc_data).float().to(device)
                if torch.cuda.device_count() > 1:               
                    latent = network.module.encoder(pc_data.unsqueeze(0).transpose(2, 1))
                else:
                    latent = network.encoder(pc_data.unsqueeze(0).transpose(2, 1))
                mesh_filename = os.path.join(testing_folder, f'testing_{idx}')
                if torch.cuda.device_count() > 1:               
                    success = create_mesh(
                        network.module.decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18))
                else:
                    success = create_mesh(
                        network.decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
                )
                if not success:
                    return False
        fig = vis_samples(samples, testing_folder, selected)
        wandb.log({"img":fig, "epoch": epoch})

        network.train()

    def process_pc(self, points):
        B, N, C = points.shape       
        if N != self.num_points: # N can be larger than M
            source_shape = sample_farthest_points(source_shape.transpose(1,2), self.hparams.num_points).transpose(1,2)

        return points
    def _sum_losses(self, losses, names):
        return sum(v for k, v in losses.items() if k in names)

    def evaluate_one_epoch(self, loss, losses, network, optimizers, tepoch, epoch, is_training):
        optimizer, keypoint_optimizer = optimizers
        ## main training loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if is_training:
            network.train()
        else:
            network.eval()



        for source_shape, target_shape, indices in tepoch:
            losses_batch = {}

            # Process the input data
            source_shape = self.process_pc(source_shape)
            target_shape = self.process_pc(target_shape)
            
            source_shape, target_shape = source_shape.to(device), target_shape.to(device)

            if is_training:
                optimizer.zero_grad()
            
            outputs = network(source_shape, target_shape)

            if self.config.lambda_init_points > 0:
                init_points_loss = pytorch3d.loss.chamfer_distance(
                    rearrange(network.keypoints, 'b d n -> b n d'), 
                    rearrange(network.init_keypoints, 'b d n -> b n d'))[0]
                losses_batch['KP_net'] = self.config.lambda_init_points * init_points_loss
            if self.config.lambda_chamfer > 0:
                chamfer_loss = pytorch3d.loss.chamfer_distance(
                    outputs["deformed"], target_shape)[0]
                losses_batch['PC_dist'] = self.config.lambda_chamfer * chamfer_loss
            if self.config.lambda_KP_dist > 0:
                KP_dist = self.KP_dist_fun(outputs["target_keypoints"], outputs["deformed_keypoints"])
                losses_batch['KP_dist'] = self.config.lambda_KP_dist * KP_dist
            if self.config.lambda_influence_predict_l2 > 0:
                losses_batch['influence_predict_l2'] = self.config.lambda_influence_predict_l2 * torch.mean(network.influence_offset ** 2)

            if is_training:
                if epoch < self.config.epochs_KP:
                    keypoints_loss = self._sum_losses(losses_batch, ['KP_net'])
                    keypoints_loss.backward(retain_graph=True) 
                    keypoint_optimizer.step()
                else:
                    loss = self._sum_losses(losses_batch, losses.keys())
                    loss.backward()
                    optimizer.step()
                    keypoint_optimizer.step()
            
            losses.append(self._sum_losses(losses_batch, ['PC_dist']))
            tepoch.set_postfix(loss=f'{np.mean(losses)}')

        return losses


def vis_samples(samples, testing_folder, selected, save=''):
    from utils.pc_utils import vis_pc
    import matplotlib.pyplot as plt
    scale_factor = 0.6
    ply_paths = [os.path.join(testing_folder, f'testing_{index}.ply') for index in selected]
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
    if save != '':
        plt.savefig(os.path.join(testing_folder, f'{save}.png'))
    return fig

if __name__ == '__main__':
    import importlib

    ## additional args for parsing
    optional_args = [("run_id", str), ("network_resume_path", str), ("optimizer_resume_path", str), ("starting_epoch", int),
                     ("special_symbol", str), ("resume_path", str), ("starting_phase", int), ("freeze_epoch", int)]
    parser = argparse.ArgumentParser()
    for optional_arg, arg_type in optional_args:
        parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", type=arg_type)
    parser.add_argument('--debug', default=False, action='store_true', help='whether to use debug mode')
    parser.add_argument('--resume_training', default=True, action='store_true', help='whether to resume')

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
        project="Keypoint-Deformer", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=exp_name, 
        # Track hyperparameters and run metadata
        dir=os.getenv('CACHE_DIR'),
        config=dict_from_module(config),
        resume="allow"
    )
    trainer = Trainer(config = config, debugger = debugger, minfo=minfo, args=args)
    trainer.train_network()
    wandb.finish()