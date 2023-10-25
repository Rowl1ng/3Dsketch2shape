from utils.model_utils import get_latest_ckpt, load_model, adjust_learning_rate, get_learning_rate_schedules, dict_from_module, get_spec_with_default
from models.network import SDFAutoEncoder
import torch
import torch.nn as nn
import os
import argparse
from utils.optutil import generate_meta_info
import wandb
from tqdm import tqdm
from data.SDF_datasets import SDFSamples
from torch.utils.data import DataLoader
import numpy as np
from utils.debugger import MyDebugger
import utils.provider as provider
from utils.sdf_utils import create_mesh, convert_sdf_samples_to_ply
from utils.ply_utils import sample_ply
from utils.pc_utils import normalize_to_box
from train.others.sdf_trainer import vis_samples
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(FILE_DIR, os.path.pardir)

split = 'train'
N = 256
def get_model(device_id, config, last_resume_ckpt):

    if config.network_type == 'AutoEncoder':
        network = SDFAutoEncoder(config=config)
    else:
        raise Exception("Unknown Network type!")
    network, _ = load_model(network, None, last_resume_ckpt, None)
    print(f'Load ckpt from {last_resume_ckpt}')
        # Load original decoder
    return network

def vis_gt(sdf_values, ply_filename):
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    offset=None
    scale=None
    convert_sdf_samples_to_ply(
        sdf_values.reshape(N, N, N).data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )
def inference(samples, network, device_id, testing_folder):
    from data.SDF_datasets import remove_nans
    selected = [17, 30, 40]

    with torch.no_grad():
        for idx in selected:
            pc_data, sdf_data, indices = samples[idx]

            sdf_gt = sdf_data[:, 3]
            ply_filename = os.path.join(testing_folder, f'{split}_{idx}_gt')
            vis_gt(sdf_gt, ply_filename)

            pc_data = torch.from_numpy(pc_data).float().to(device_id)
            if torch.cuda.device_count() > 1:               
                latent = network.module.encoder(pc_data.unsqueeze(0).transpose(2, 1), is_training=False)
            else:
                latent = network.encoder(pc_data.unsqueeze(0).transpose(2, 1), is_training=False)

            latent_path = os.path.join(testing_folder, f'{split}_{idx}_latent.npy')
            np.save(latent_path, latent.cpu().numpy())

            mesh_filename = os.path.join(testing_folder, f'{split}_{idx}')
            if torch.cuda.device_count() > 1:               
                success = create_mesh(
                    network.module.decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18))
            else:
                success = create_mesh(
                    network.decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
            )

    fig = vis_samples(samples, testing_folder, selected, save=f'{split}_recon')


if __name__ == '__main__':
    import importlib

    minfo = generate_meta_info(DEFAULT_ROOT, 'SDF_sketch', src_name='src')

    exp_name = 'sdf_autoencoder_gpu2_runs_1'
        ### debugger
    config_path = os.path.join(minfo['experiments_dir'], exp_name, 'config.py')

    debugger = MyDebugger(exp_name, is_save_print_to_file = False, config_path = config_path, debug_dir=minfo['experiments_dir'])
    phase = 2
    last_resume_ckpt, last_optimizer = get_latest_ckpt(exp_path=debugger._debug_dir_name, prefix='model_epoch_')
    epoch = os.path.basename(last_resume_ckpt).split('_')[-1][:-4]


    device_id = 0
    torch.cuda.set_device(device_id)

    spec = importlib.util.spec_from_file_location('*', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    samples = SDFSamples(split, N ** 3, load_ram=False, debug=True)

    #TODO: manipulate given dimension of z vector
    testing_folder = os.path.join(debugger._debug_dir_name, f'latent_epoch{epoch}')
    if not os.path.exists(testing_folder):
        os.mkdir(testing_folder)

    network = get_model(device_id, config, last_resume_ckpt).to(device_id)

    inference(samples, network, device_id, testing_folder)