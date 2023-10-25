import argparse
import os, importlib
import torch
from utils.debugger import MyDebugger
from data.datasets import ImNetSketchSample
from train.others.implicit_sketch_trainer import get_latest_ckpt, load_model, dict_from_module
from models.network import ShapeSketchAutoEncoder
from utils.optutil import generate_meta_info

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(FILE_DIR, os.path.pardir)
resolution = 64
max_batch = 20000
space_range = (-0.5, 0.5)
thershold = 0.01

load_decoder = False
def get_model(device_id, config, last_resume_ckpt):

    if config.network_type == 'AutoEncoder':
        network = ShapeSketchAutoEncoder(config=config)
    else:
        raise Exception("Unknown Network type!")
    network, _ = load_model(network, None, last_resume_ckpt, None)
    print(f'Load ckpt from {last_resume_ckpt}')
        # Load original decoder
    if load_decoder:
        EXP_DIR = os.getenv('EXP_DIR')
        last_resume_ckpt = os.path.join(EXP_DIR, 'phase_2_model/model_epoch_2_300.pth') 
        network, _ = load_model(network, None, last_resume_ckpt, None)
    return network

# def vis_recon(network, file_path, embedding):


def manipulate_z(config, samples, device_id, testing_folder):
    network = get_model(device_id, config, network_path)
    network.eval().cuda(device_id)

    # encode_func = config.encode_func if hasattr(config, 'encode_func') else 'encode'
    for index in [17, 110, 138]:
        sketch_inputs, shape_inputs, latent_code = samples[index][0]
        sketch_inputs = torch.from_numpy(sketch_inputs).float().cuda(device_id)
        shape_inputs = torch.from_numpy(shape_inputs).float().cuda(device_id)

        latent_code = torch.from_numpy(latent_code).cuda(device_id)
        
        assert config.handle_network == 'VAE'

        file_path = os.path.join(testing_folder, f'{index}_gt.off')

        # vis_recon(network, file_path, latent_code.unsqueeze(0))
        network.save_bsp_deform(inputs=None, file_path=file_path, resolution=resolution, max_batch=max_batch,
                                                space_range=space_range, thershold_1=thershold, embedding=latent_code.unsqueeze(0))
        embedding = network.encoder(sketch_inputs.unsqueeze(0).transpose(2, 1), is_training=False)
        z_s = embedding[:, :config.decoder_input_embbeding_size]
        z_t = embedding[:, config.decoder_input_embbeding_size:]
        mu = network.vae_net.convfeat1(z_t)
        log_var = network.vae_net.convfeat2(z_t)
        std = torch.exp(0.5 * log_var) 
        for sample_id in range(10):
            file_path = os.path.join(testing_folder, f'{index}_sample{sample_id}.off')
            # if not os.path.exists(file_path):
            eps = torch.randn_like(std)
            new_z_t = eps * std + mu
            new_emb = torch.cat([z_s, new_z_t], dim=1)
            network.save_bsp_deform(inputs=None, file_path=file_path, resolution=resolution, max_batch=max_batch,
                                            space_range=space_range, thershold_1=thershold, embedding=new_emb)


# def vis_variants(testing_folder):
    


if __name__ == '__main__':
    # optional_args = [("exp_name", str)]
    # parser = argparse.ArgumentParser()
    # for optional_arg, arg_type in optional_args:
    #     parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", type=arg_type)
    # parser.add_argument('--debug', default=False, action='store_true', help='whether to use debug mode')
    # parser.add_argument('--last_epoch', default=True, action='store_true', help='whether to use last epoch')   

    # args = parser.parse_args()
    minfo = generate_meta_info(DEFAULT_ROOT, 'implicit_sketch', src_name='src')

    exp_name = 'config_discrete_sketch_encoder_v1_eigen_v4_gpu1_runs_99'
        ### debugger
    config_path = os.path.join(minfo['experiments_dir'], exp_name, 'config.py')

    debugger = MyDebugger(exp_name, is_save_print_to_file = False, config_path = config_path, debug_dir=minfo['experiments_dir'])
    phase = 2
    network_path, _ = get_latest_ckpt(exp_path=debugger._debug_dir_name, phase=phase)
    epoch = os.path.basename(network_path).split('_')[-1][:-4]

    samples = ImNetSketchSample(datasets_dir=minfo['datasets_dir'], phase='test', sample_voxel_size=16 * (2 ** (phase)), inference=True)

    device_id = 0
    torch.cuda.set_device(device_id)

    spec = importlib.util.spec_from_file_location('*', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    
    #TODO: manipulate given dimension of z vector
    testing_folder = os.path.join(debugger._debug_dir_name, f'z_exp_epoch{epoch}')
    if not os.path.exists(testing_folder):
        os.mkdir(testing_folder)

    manipulate_z(config, samples, device_id, testing_folder)