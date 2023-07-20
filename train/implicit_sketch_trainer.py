import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.network import ShapeSketchAutoEncoder
from data.datasets import ImNetSketchSample
from torch.utils.data import DataLoader
from utils.debugger import MyDebugger
import os
import argparse
from utils.optutil import generate_meta_info
import utils.provider as provider
from contextlib import suppress
import wandb
from pathlib import Path
from utils.model_utils import get_latest_ckpt
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# torch.backends.cudnn.benchmark = True
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(FILE_DIR, os.path.pardir)


debug_batch = 100

import signal, sys

torch.autograd.set_detect_anomaly(True)
def sigterm_handler(signal, frame):
    # save the state here or do whatever you want
    print('caught sigint!')
    # fname = fpath + 'sigint_caught.txt'
    # open(fname,'w').close()
    wandb.finish()
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)
# signal.signal(signal.SIGQUIT, sigquit_handler)
# signal.signal(signal.SIGINT, sigint_handler)

class Trainer(object):

    def __init__(self, config, debugger, minfo, args):
        self.debugger = debugger
        self.config = config
        self.minfo = minfo
        self.args = args
        if args.debug:
            # self.config.batch_size = 4
            self.config.vis_interval = 20

    def train_network(self):
        phases = range(self.config.starting_phase, 3) if hasattr(self.config, 'starting_phase') else [
            int(np.log2(self.config.sample_voxel_size // 16))]

        latent_loss_fn = [torch.nn.MSELoss(), torch.nn.MSELoss()] 
        for phase in phases:
            print(f"Start Phase {phase}")
            sample_voxel_size = 16 * (2 ** (phase))

            if phase == 2:
                if not hasattr(self.config, 'half_batch_size_when_phase_2') or self.config.half_batch_size_when_phase_2:
                    self.config.batch_size = self.config.batch_size // 2
                self.config.training_epochs = self.config.training_epochs * 2

            if self.config.network_type == 'AutoEncoder':
                network = ShapeSketchAutoEncoder(config=self.config)
            else:
                raise Exception("Unknown Network type!")

            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                print(f"Use {torch.cuda.device_count()} GPUS!")
                network = nn.DataParallel(network)
                self.config.batch_size = self.config.batch_size * torch.cuda.device_count()
            network = network.to(device)

            ### create dataset
            train_samples = ImNetSketchSample(datasets_dir=self.minfo['datasets_dir'], phase='train',
                                         sample_voxel_size=sample_voxel_size, debug=self.args.debug)

            train_data_loader = DataLoader(dataset=train_samples,
                                     batch_size=self.config.batch_size,
                                     num_workers=self.config.data_worker,
                                     shuffle=True,
                                     drop_last=False)

            if hasattr(self.config, 'use_testing') and self.config.use_testing:
                test_samples = ImNetSketchSample(datasets_dir=self.minfo['datasets_dir'], phase='test',
                                            sample_voxel_size=sample_voxel_size,
                                            interval=self.config.testing_interval, debug=self.args.debug)

                test_data_loader = DataLoader(dataset=test_samples,
                                               batch_size=self.config.batch_size,
                                               num_workers=self.config.data_worker,
                                               shuffle=True,
                                               drop_last=False)

            ## reload the network if needed
            
            optimizer = torch.optim.Adam(params=network.parameters(), lr=self.config.lr,
                                betas=(self.config.beta1, 0.999))

            last_resume_ckpt, last_optimizer = get_latest_ckpt(exp_path=self.debugger._debug_dir_name, prefix = f'model_epoch_{phase}_')
            if self.args.resume_training and  last_resume_ckpt is not None: # Resume from last time
                network, optimizer = load_model(network, optimizer, last_resume_ckpt, last_optimizer)
                epoch = os.path.basename(last_resume_ckpt).split('_')[-1][:-4]
                self.config.starting_epoch = int(epoch) + 1

            elif self.config.network_resume_path is not None:
                #Resume from pretrained
                resume_ckpt = os.path.join(self.minfo['experiments_dir'], self.config.network_resume_path)
                network, optimizer = load_model(network, optimizer, resume_ckpt, self.config.optimizer_resume_path)
                self.config.network_resume_path = None
                self.config.optimizer_resume_path = None

    
            for idx in range(self.config.starting_epoch, self.config.training_epochs + 1):
                with tqdm(train_data_loader, unit='batch') as tepoch:
                    tepoch.set_description(f'Epoch {idx}')
                    losses = []
                    if self.config.train_encoder_only:
                        for param in network.encoder.parameters():
                            param.requires_grad = False 

                        losses = self.evaluate_one_epoch_latent(latent_loss_fn, losses, network, optimizer, tepoch, is_training=True)
                        wandb.log({"train_latent_loss": np.mean(losses),"epoch": idx})
                    else:
                        losses = self.evaluate_one_epoch(losses, network, optimizer, tepoch, is_training = True)
                        wandb.log({"train_loss": np.mean(losses),"epoch": idx})

                    print(f"Train Loss for epoch {idx} : {np.mean(losses)}")

                    ## saving the models
                    if idx % self.config.saving_intervals == 0:
                        # save
                        save_model_path = self.debugger.file_path(f'model_epoch_{phase}_{idx}.pth')
                        save_optimizer_path = self.debugger.file_path(f'optimizer_epoch_{phase}_{idx}.pth')
                        torch.save(network.state_dict(), save_model_path)
                        torch.save(optimizer.state_dict(), save_optimizer_path)
                        print(f"Epoch {idx} model saved at {save_model_path}")
                        print(f"Epoch {idx} optimizer saved at {save_optimizer_path}")
                        self.config.network_resume_path = save_model_path  ## add this resume after the whole things are compelete

                        if hasattr(self.config, 'use_testing') and self.config.use_testing:
                            with tqdm(test_data_loader, unit='batch') as tepoch:
                                losses = []
                                if self.config.train_encoder_only:
                                    losses = self.evaluate_one_epoch_latent(latent_loss_fn, losses, network, optimizer, tepoch, is_training=False)
                                    wandb.log({"test_latent_loss": np.mean(losses),"epoch": idx})
                                else:
                                    losses = self.evaluate_one_epoch(losses, network, optimizer = None, tepoch = tepoch, is_training=False)
                                    wandb.log({"test_loss": np.mean(losses),"epoch": idx})

                                print(f"Test Loss for epoch {idx} : {np.mean(losses)}")

                    if hasattr(self.config, 'vis_interval') and idx % self.config.vis_interval == 0:
                        self.vis_epoch(test_samples, network, idx)


            ## when done the phase
            self.config.starting_epoch = 0


    def vis_epoch(self, samples, network, epoch):
        network.eval()
        selected = [17, 110, 138]
        encode_func = config.encode_func if hasattr(config, 'encode_func') else 'encode'
        resolution = 64
        max_batch = 20000
        space_range = (-0.5, 0.5)
        thershold = 0.01
        testing_folder = os.path.join(self.debugger._debug_dir_name, f'test_epoch{epoch}')
        if not os.path.exists(testing_folder):
            os.mkdir(testing_folder)
        for index in selected:
            sketch_inputs, shape_inputs, latent_vector_gt, _, _ = samples[index][0]
            sketch_inputs = torch.from_numpy(sketch_inputs).float().to(device)
            shape_inputs = torch.from_numpy(shape_inputs).float().to(device)
            file_path = os.path.join(testing_folder, f'testing_{index}.off')

            if torch.cuda.device_count() > 1:
                _, embedding = getattr(network.module, encode_func)(sketch_inputs.unsqueeze(0).transpose(2, 1), shape_inputs.unsqueeze(0).transpose(2, 1), is_training=False)
                network.module.save_bsp_deform(inputs=None, file_path=file_path, resolution=resolution, max_batch=max_batch,
                                            space_range=space_range, thershold_1=thershold, embedding=embedding[:1, :])

            else:
                _, embedding = getattr(network, encode_func)(sketch_inputs.unsqueeze(0).transpose(2, 1), shape_inputs.unsqueeze(0).transpose(2, 1), is_training=False)
                network.save_bsp_deform(inputs=None, file_path=file_path, resolution=resolution, max_batch=max_batch,
                                                        space_range=space_range, thershold_1=thershold, embedding=embedding[:1, :])

        from utils.sketch_mesh_visualization import vis_samples
        fig = vis_samples(samples, testing_folder, selected)
        wandb.log({"img":fig, "epoch": epoch})
        network.train()

    def process_pc(self, sketch_points, shape_points):
        sketch_points, _ = provider.apply_random_scale_xyz(sketch_points)
        points = torch.cat([sketch_points, shape_points])
        points = provider.random_point_dropout(points.numpy())
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        sketch_points, shape_points = torch.split(points, [sketch_points.shape[0], shape_points.shape[0]])
        del points
        return sketch_points, shape_points

    def z_t_loss(self, loss_fn, pred_latent_vector, latent_vector_gt):
        latent_vector_gt = latent_vector_gt.repeat(2,1)

        z_t_loss = loss_fn(pred_latent_vector[:, self.config.decoder_input_embbeding_size:], latent_vector_gt[:, self.config.decoder_input_embbeding_size:])
        z_t_loss = (torch.exp(z_t_loss.mean(dim=1)) - 1).mean()
        return z_t_loss

    def latent_loss_v0(self, loss_fn, pred_latent_vector, latent_vector_gt):
        loss_shape_fn, loss_sketch_fn = loss_fn
        sketch_num = pred_latent_vector.shape[0] // 2
        sketch_z = pred_latent_vector[:sketch_num, :]
        loss = loss_shape_fn(sketch_z, latent_vector_gt) 
        return loss
    def latent_loss_v1(self, loss_fn, pred_latent_vector, latent_vector_gt):
        loss_shape_fn, loss_sketch_fn = loss_fn
        latent_vector_gt = latent_vector_gt.repeat(2,1)
        loss = loss_shape_fn(pred_latent_vector, latent_vector_gt) 
        return loss
    def latent_loss_v2(self, loss_fn, pred_latent_vector, latent_vector_gt):
        loss_shape_fn, loss_sketch_fn = loss_fn
        sketch_num = pred_latent_vector.shape[0] // 2
        shape_z = pred_latent_vector[sketch_num:, :]
        sketch_z_t = pred_latent_vector[:sketch_num, self.config.decoder_input_embbeding_size:]
        gt_z_t = latent_vector_gt[:, self.config.decoder_input_embbeding_size:]
        loss = loss_shape_fn(shape_z, latent_vector_gt) + loss_sketch_fn(sketch_z_t, gt_z_t)
        return loss
    def evaluate_one_epoch_latent(self, loss_fn, losses, network, optimizer, tepoch, is_training):
        ###
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ###
        encode_func = self.config.encode_func if hasattr(self.config, 'encode_func') else 'encode'
        gamma = 5
        z_t_loss_fun = nn.MSELoss(reduction='none')
        for i, [(sketch_points, shape_points, latent_vector_gt, _, _), samples_indices] in enumerate(tepoch):
            sketch_points, shape_points = self.process_pc(sketch_points, shape_points)
            sketch_points, shape_points, latent_vector_gt = sketch_points.to(device), shape_points.to(device), latent_vector_gt.to(device)

            if is_training:
                optimizer.zero_grad()

            ##

            if torch.cuda.device_count() > 1:

                handle, pred_latent_vector =  getattr(network.module, encode_func)(sketch_points, shape_points, is_training=is_training)
            else:
                handle, pred_latent_vector = getattr(network, encode_func)(sketch_points, shape_points, is_training=is_training)


            ## output results
            latent_loss = getattr(self, self.config.latent_loss)(loss_fn, pred_latent_vector, latent_vector_gt)
            loss = latent_loss

            

            if self.config.handle_network == 'Eigen' and is_training:

                z_t_loss = self.z_t_loss(z_t_loss_fun, pred_latent_vector, latent_vector_gt)
                loss = (loss + z_t_loss * gamma) / (1+gamma)

                if torch.cuda.device_count() > 1:               
                    reg_loss = network.module.orthogonal_regularizer()
                else:
                    reg_loss = network.orthogonal_regularizer()
            
                loss = loss + reg_loss * self.config.ortho_loss_factor

                if self.config.cov_loss:
                    if torch.cuda.device_count() > 1:               
                        cov_loss = network.module.cov_loss(handle) #* 100
                    else:
                        cov_loss = network.cov_loss(handle) #* 100
                    loss = loss + cov_loss #* 10

                # if self.config.sp_loss and is_training:
                #     if torch.cuda.device_count() > 1:               
                #         sp_loss = network.module.sp_loss(handle) #* 100
                #     else:
                #         sp_loss = network.sp_loss(handle) #* 100
                #     loss = loss + sp_loss #* 0.01
            
            if self.config.handle_network == 'VAE' and is_training:
                mu, log_var = handle
                vae_loss = network.vae_net.loss_function(mu, log_var)
                loss = loss + vae_loss * 0.01 #* 10

            losses.append(latent_loss.item())

            if is_training:
                loss.backward()
                if hasattr(self.config, 'clip_gradient'):
                    torch.nn.utils.clip_grad_norm_(network.parameters(), self.config.clip_gradient)
                optimizer.step()

            tepoch.set_postfix(loss=f'{np.mean(losses)}')
            if i > debug_batch and self.args.debug:
                return losses
        return losses

    def evaluate_one_epoch_shape(self, losses, network, optimizer, tepoch, forward_name='forward', is_training = True):
        ## main training loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for i, [inputs, samples_indices] in enumerate(tepoch):

            ## get voxel_inputs
            shape_points, _, coordinate_inputs, occupancy_ground_truth = inputs
            normals_gt = None
            shape_points = self.process_pc_shape(shape_points)

            ## remove gradient
            if is_training:
                optimizer.zero_grad()
            shape_points, coordinate_inputs, occupancy_ground_truth, samples_indices = shape_points.to(device), coordinate_inputs.to(device), occupancy_ground_truth.to(
                device), samples_indices.to(device)

            if self.config.network_type == 'AutoEncoder':
                prediction = network(shape_points, coordinate_inputs, is_training)

            else:
                raise Exception("Unknown Network Type....")

            _, prediction, _, convex_layer_weights = self.extract_prediction(prediction)

            occupancy_ground_truth = torch.cat([occupancy_ground_truth, occupancy_ground_truth])
            loss = self.config.loss_fn(torch.clamp(prediction, min=0, max=1), occupancy_ground_truth)

            ### loss function to be refactor
            if (self.config.decoder_type == 'Flow' and self.config.flow_use_bsp_field == True) or self.config.decoder_type == 'MVP':
                loss, losses = self.flow_bsp_loss(loss, losses, network,
                                                  occupancy_ground_truth,
                                                  prediction, convex_layer_weights)
            else:
                raise Exception("Unknown Network Type....")

            if is_training:
                loss.backward()
                optimizer.step()

            del loss
            del prediction, convex_layer_weights, occupancy_ground_truth
            tepoch.set_postfix(loss=f'{np.mean(losses)}')

            if i > debug_batch and self.args.debug:
                return losses
        return losses

    def evaluate_one_epoch(self, losses, network, optimizer, tepoch, forward_name='forward', is_training = True):
        ## main training loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for i, [inputs, samples_indices] in enumerate(tepoch):

            ## get voxel_inputs
            sketch_points, shape_points, _, coordinate_inputs, occupancy_ground_truth = inputs
            normals_gt = None
            sketch_points, shape_points = self.process_pc(sketch_points, shape_points)

            ## remove gradient
            if is_training:
                optimizer.zero_grad()
            sketch_points, shape_points, coordinate_inputs, occupancy_ground_truth, samples_indices = sketch_points.to(
                device), shape_points.to(device), coordinate_inputs.to(device), occupancy_ground_truth.to(
                device), samples_indices.to(device)

            if self.config.network_type == 'AutoEncoder':
                prediction = network(sketch_points, shape_points, coordinate_inputs, is_training)

            else:
                raise Exception("Unknown Network Type....")

            _, prediction, _, convex_layer_weights = self.extract_prediction(prediction)

            occupancy_ground_truth = torch.cat([occupancy_ground_truth, occupancy_ground_truth])
            loss = self.config.loss_fn(torch.clamp(prediction, min=0, max=1), occupancy_ground_truth)

            ### loss function to be refactor
            if (self.config.decoder_type == 'Flow' and self.config.flow_use_bsp_field == True) or self.config.decoder_type == 'MVP':
                loss, losses = self.flow_bsp_loss(loss, losses, network,
                                                  occupancy_ground_truth,
                                                  prediction, convex_layer_weights)
            else:
                raise Exception("Unknown Network Type....")
            if self.config.handle_network == 'Eigen':
                if torch.cuda.device_count() > 1:               
                    reg_loss = network.module.orthogonal_regularizer() * 100
                else:
                    reg_loss = network.orthogonal_regularizer() * 100
            
                loss = loss + reg_loss.detach().item()


            if is_training:
                loss.backward()
                optimizer.step()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache() 
            del loss
            del prediction, convex_layer_weights, occupancy_ground_truth
            tepoch.set_postfix(loss=f'{np.mean(losses)}')

            if i > debug_batch and self.args.debug:
                return losses
        return losses

    def flow_bsp_loss(self, loss, losses, network, occupancy_ground_truth, prediction, convex_layer_weights):

        bsp_thershold = self.config.bsp_thershold if hasattr(self.config, 'bsp_thershold') else 0.01
        if self.config.bsp_phase == 0:
            concave_layer_weights = network.decoder.bsp_field.concave_layer_weights if torch.cuda.device_count() <= 1 else network.module.decoder.bsp_field.concave_layer_weights
            losses.append(loss.detach().item())
            loss = loss + torch.sum(
                torch.abs(concave_layer_weights - 1))  ### convex layer weight close to 1
            loss = loss + torch.sum(
                torch.clamp(convex_layer_weights - 1, min=0) - torch.clamp(convex_layer_weights,
                                                                           max=0))
        elif self.config.bsp_phase == 1:
            loss = torch.mean((1 - occupancy_ground_truth) * (
                    1 - torch.clamp(prediction, max=1)) + occupancy_ground_truth * torch.clamp(
                prediction, min=0))
            losses.append(loss.detach().item())
            loss = loss + torch.sum(
                (convex_layer_weights < bsp_thershold).float() * torch.abs(
                    convex_layer_weights)) + torch.sum(
                (convex_layer_weights >= bsp_thershold).float() * torch.abs(convex_layer_weights - 1))
        else:
            raise Exception("Unknown Phase.....")


        return loss, losses


    def extract_prediction(self, prediction):

        assert self.config.decoder_type == 'Flow' and self.config.flow_use_bsp_field == True
        convex_prediction, prediction, exist, convex_layer_weights = prediction  # unpack the idead

        return convex_prediction, prediction, exist, convex_layer_weights




if __name__ == '__main__':
    import importlib

    ## additional args for parsing
    optional_args = [("run_id", str), ("network_resume_path", str), ("optimizer_resume_path", str), ("starting_epoch", int),
                     ("special_symbol", str), ("resume_path", str), ("starting_phase", int)]
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

    minfo = generate_meta_info(DEFAULT_ROOT, 'implicit_sketch', src_name='src')

    model_type = os.path.basename(resume_path).split('.')[0] if config.network_type == 'AutoEncoder' else f"AutoDecoder-{config.decoder_type}"
    exp_name = f'{model_type}_gpu{torch.cuda.device_count()}_runs_{args.run_id}'
    debugger = MyDebugger(exp_name, is_save_print_to_file = False if args.debug else True, config_path = resume_path, debug_dir=minfo['experiments_dir'])
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
        project="Neural-Template", 
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