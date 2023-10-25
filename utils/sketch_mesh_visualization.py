import numpy as np
import random
import torch
import importlib
import os
from utils.debugger import MyDebugger
from torch.multiprocessing import Pool, Process, set_start_method
from utils.optutil import generate_meta_info
import argparse
from data.datasets import ImNetSketchSample
from utils.model_utils import get_latest_ckpt, load_model, dict_from_module
from models.network import ShapeSketchAutoEncoder


from utils.pc_utils import normalize_to_box
from torch.utils.data import DataLoader
import wandb

from utils.ply_utils import sample_ply
from pytorch3d.loss import chamfer_distance

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(FILE_DIR, os.path.pardir)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load_decoder = True
def extract_one_input(args):
    args_list, config_path, last_resume_ckpt, device_id = args

    torch.cuda.set_device(device_id)
    spec = importlib.util.spec_from_file_location('*', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # network = SketchAutoEncoder(config=config)

    ### set autoencoder
    # assert hasattr(config, 'auto_encoder_config_path') and os.path.exists(config.auto_encoder_config_path)
    if config.network_type == 'AutoEncoder':
        network = ShapeSketchAutoEncoder(config=config)
    else:
        raise Exception("Unknown Network type!")

    network, _ = load_model(network, None, last_resume_ckpt, None)

    # Load original decoder
    # if load_decoder:
    #     EXP_DIR = os.getenv('EXP_DIR')
    #     last_resume_ckpt = os.path.join(EXP_DIR, 'phase_2_model/model_epoch_2_300.pth') 
    #     network, _ = load_model(network, None, last_resume_ckpt, None)


    network.eval().cuda(device_id)
    encode_func = config.encode_func if hasattr(config, 'encode_func') else 'encode'

    for args in args_list:
        inputs, file_path, resolution, max_batch, space_range, thershold, obj_path, with_surface_point, file_name = args
        sketch_inputs, shape_inputs, latent_code = inputs[0]
        if os.path.exists(os.path.join(os.path.dirname(file_path), file_path[:-4] + "_original.ply")):
            print(f"{os.path.join(os.path.dirname(file_path), file_path[:-4] + '_original.ply')} exists!")
            continue
        else:
            sketch_inputs = torch.from_numpy(sketch_inputs).float().cuda(device_id)
            shape_inputs = torch.from_numpy(shape_inputs).float().cuda(device_id)

            file = open(file_name, mode='a')
            file.write(f"mesh name for {obj_path}")
            print(f"Outputing {obj_path} for {device_id}")

            ## new embedding
            embedding = getattr(network, encode_func)(sketch_inputs.unsqueeze(0).transpose(2, 1), shape_inputs.unsqueeze(0).transpose(2, 1), is_training=False)

            # embedding = torch.from_numpy(latent_code).unsqueeze(0).cuda(device_id)
            network.save_bsp_deform(inputs=None, file_path=file_path, resolution=resolution, max_batch=max_batch,
                                                 space_range=space_range, thershold_1=thershold, embedding=embedding[:1, :])


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def evaluate(dataset, testing_folder, epoch):
    save_path = os.path.join(testing_folder, f'CD.npy')
    if not os.path.exists(save_path):
        shape_loader = DataLoader(dataset=dataset,
                                    batch_size=16,
                                    num_workers=4,
                                    shuffle=False,
                                    drop_last=False)
        cd_array = []

        for i, data in enumerate(shape_loader):
            _, shape_pc, indexes = data
            #Load ply file
            ply_paths = [os.path.join(testing_folder, f'testing_{index}_deformed.ply') for index in indexes]
            recon_pcs = sample_ply(ply_paths)
            recon_pcs, _, _ = normalize_to_box(recon_pcs)
            # load GT
            shape_pc, _, _ = normalize_to_box(shape_pc)
            cd = chamfer_distance(
                recon_pcs, shape_pc, batch_reduction=None)[0]
            cd_array.extend(cd.data.cpu().numpy())
        #Chamfer distance
        cd_array = np.array(cd_array)
        np.save(save_path, cd_array)
    else:
        cd_array = np.load(save_path)
    print(f'CD mean @ {epoch}: ', cd_array.mean())
    wandb.log({"test_CD": cd_array.mean(),"epoch": epoch})
    #normal consistencys

    #LFD: Light field distance

    #fitting gap


def run_inference(samples, testing_folder, selected):

    sample_interval = 1
    resolution = 64
    max_batch = 20000
    save_deformed = True
    thershold = 0.01
    with_surface_point = True

    device_count = torch.cuda.device_count()
    print(f"Use {device_count} GPUS!")

    device_ratio = 1
    worker_nums = int(device_count * device_ratio)
    testing_cnt = 180


    file_name = os.path.join(testing_folder, 'obj_list.txt')
    args = [(samples[i], os.path.join(testing_folder, f'testing_{i}.off'), resolution, max_batch, (-0.5, 0.5), thershold,
             samples.name_list[i], with_surface_point, file_name) for i in selected if
            i % sample_interval == 0]
    random.shuffle(args)
    args = args[:testing_cnt]
    splited_args = split(args, worker_nums)
    final_args = [(splited_args[i], config_path, network_path,  i % device_count) for i in range(worker_nums)]
    set_start_method('spawn')

    # for arg in args:
    #     extract_one_input(arg)

    if device_count > 1:
        pool = Pool(device_count)
        pool.map(extract_one_input, final_args)
    else:
        extract_one_input(final_args[0])

def vis_samples(samples, testing_folder, selected):
    from utils.pc_utils import vis_pc
    import matplotlib.pyplot as plt
    scale_factor = 0.6
    ply_paths = [os.path.join(testing_folder, f'testing_{index}_deformed.ply') for index in selected]
    recon_pcs = sample_ply(ply_paths)
    recon_pcs, _, _ = normalize_to_box(recon_pcs)

    nrows = 3 
    ncols = len(selected)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols, nrows])
    for index, item in enumerate(selected):
        # f.add_subplot(1,3, 1)
        # plt.axis('off')
        axs[2, index].imshow(vis_pc(recon_pcs[index]*scale_factor))

        sketch_pc, shape_pc = samples[item][0][0], samples[item][0][1]

        # f.add_subplot(1,3, 2)
        axs[1, index].imshow(vis_pc(normalize_to_box(shape_pc)[0]*scale_factor))
        # f.add_subplot(1,3, 3)
        axs[0, index].imshow(vis_pc(normalize_to_box(sketch_pc)[0]*scale_factor))
        axs[0, index].set_title(item)
    [axi.set_axis_off() for axi in axs.ravel()]
    plt.tight_layout(True)
    # plt.savefig(f'original_decoder_False_test_runs_96.png')
    return fig
if __name__ == '__main__':
    optional_args = [("exp_name", str)]
    parser = argparse.ArgumentParser()
    for optional_arg, arg_type in optional_args:
        parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", type=arg_type)
    parser.add_argument('--debug', default=False, action='store_true', help='whether to use debug mode')
    parser.add_argument('--last_epoch', default=True, action='store_true', help='whether to use last epoch')   

    args = parser.parse_args()
    minfo = generate_meta_info(DEFAULT_ROOT, 'implicit_sketch', src_name='src')
    ## folder for testing
    config_path = os.path.join(minfo['experiments_dir'], args.exp_name, 'config.py')

    ### debugger
    debugger = MyDebugger(args.exp_name, is_save_print_to_file = False, config_path = config_path, debug_dir=minfo['experiments_dir'])
    


    ## import config here
    spec = importlib.util.spec_from_file_location('*', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    ## dataload
    ### create dataset
    testing_flag = True
    # if os.path.exists(config.train_data_path) and not testing_flag:
    #     data_path = config.train_data_path

    ## loading index
    use_phase = True
    phase = 2

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
        name=args.exp_name, 
        # Track hyperparameters and run metadata
        dir=os.getenv('CACHE_DIR'),
        config=dict_from_module(config),
        resume="allow"
    )

    if args.last_epoch:
        network_path, _ = get_latest_ckpt(exp_path=debugger._debug_dir_name, phase=phase)
        epoch = os.path.basename(network_path).split('_')[-1][:-4]
    else:
        epoch = 200
        network_path = os.path.join(debugger._debug_dir_name, f'model_epoch{"_" + str(phase) if use_phase else ""}_{epoch}.pth')

    testing_folder = os.path.join(debugger._debug_dir_name, f'test_epoch{epoch}')
    if not os.path.exists(testing_folder):
        os.mkdir(testing_folder)

    samples = ImNetSketchSample(datasets_dir=minfo['datasets_dir'], phase='test', sample_voxel_size=16 * (2 ** (phase)), inference=True)

    selected = [17, 110, 138, 17, 110, 138, 178, 60, 86, 22, 81, 43, 16]

    run_inference(samples, testing_folder, selected)
    # evaluate(samples, testing_folder, epoch)
    fig = vis_samples(samples, testing_folder, selected)
    if not args.debug:
        wandb.log({"img":fig, "epoch": epoch})

    wandb.finish()
