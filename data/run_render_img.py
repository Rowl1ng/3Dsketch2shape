import os, sys
from glob import glob
from multiprocessing import Pool
import numpy as np
import torch

import sys
# sys.path.append('../')
# sys.path.append('/user/HS229/ll00931/projects/ShapeFormer/')


# import xgutils.vis.fresnelvis as fresnelvis
# from utils.pc_utils import rotate_point_cloud
# from utils.pc_utils import farthest_point_sample
# from utils.pc_utils import normalize_to_box

resolution=(512, 512)
samples=32
cloudR = 0.008
vis_camera = dict(camPos=np.array([2, 2, 2]), camLookat=np.array([0., 0., 0.]),
                        camUp=np.array([0, 1, 0]), camHeight=2, resolution=resolution, samples=samples)

def vis_pc(pc):
    if isinstance(pc, np.ndarray):
        pc = torch.from_numpy(pc)
    new_pc = rotate_point_cloud(pc, dim='y', angle=180)
    pc_view = fresnelvis.renderMeshCloud(
                cloud=new_pc, cloudR=cloudR, **vis_camera)
    return pc_view
#TODO: Replace with your blender path
blender_path = '/scratch/software/blender-3.4.1-linux-x64' # replace this with your Blender path

DATA_DIR = '/scratch/dataset/SketchGen'
split = 'test'
name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}_sketch.txt')
sdf_name_list = [line.rstrip() for line in open(name_list_path)]

def render(work_info):
    model_path, save_dir, color_id = work_info
    os.system('/scratch/software/blender-2.93.4-linux-x64/blender --background --python /user/HS229/ll00931/projects/Neural-Template/data/render_chair.py -- %s %s %s' % (model_path, save_dir, color_id))

def render360(work_info):
    model_path, save_dir, color_id = work_info
    os.system('/scratch/software/blender-2.93.4-linux-x64/blender --background --python /user/HS229/ll00931/projects/Neural-Template/data/render_chair_360.py -- %s %s %s' % (model_path, save_dir, color_id))

def render_gif():
    from PIL import Image

    selected = [29, 82]
    obj_folder = '/scratch/Remote_data/SketchGen/experiments/gen_conNF_v43_run1/test_epoch300_original/'

    for id in selected:
        for i in range(22):
            obj_file = os.path.join(obj_folder, 'test_{}_sample_{}.ply'.format(id, i))
            save_dir = os.path.join(obj_folder, 'render_gif')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            work_info = (obj_file, save_dir, 2)
            render360(work_info)
            azi_origins = np.linspace(0, 350, 36)
            image_path_list = [ os.path.basename(obj_file).split('.')[0]+'_azi{}.png'.format(i) for i in azi_origins]
            image_list = [Image.open(os.path.join(save_dir, file)) for file in image_path_list]
            image_list[0].save('data/gifs/{}_{}_360.gif'.format(id, i), format='GIF', save_all=True, disposal=2, append_images=image_list[1:], duration=150, loop=0)

def make_interpolation():
    sketch_dir = '/scratch/visualization/chair_1005_align_view'
    exp = 'gen_conNF_v43_run1'
    img_folder = '/scratch/Remote_data/SketchGen/experiments/gen_conNF_v43_run1/test_epoch300/interp'
    saving_folder = '/scratch/visualization/sketch_to_shape_gen'
    test_shape_file = os.path.join(DATA_DIR, f'split/{split}_shape.txt') #'/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test_shape.txt'
    test_shape_list = [line.rstrip() for line in open(test_shape_file)]
    import matplotlib.pyplot as plt
    from PIL import Image
    import matplotlib.image as mpimg
    selected = {
    '95': [1, 5],
    '98': [3, 6],
    '101': [2,5],
    '71': [1,6],
    '124': [1,4],
    '167': [1,3],
    '180': [1,6],
    '157': [4, 3]
}
    nrows = len(selected.keys())
    ncols = 2 + 5 # + 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols*2, nrows*2])

    def vis_img(p_imgs, row, axs):
        # f = plt.figure(figsize=(20, 6), dpi=80)
        for i in range(len(p_imgs)):
            # f.add_subplot(1, len(p_imgs), i+1)
            # img = Image.open(p_imgs[i])
            # small = img.crop((offset, offset, -offset, -offset)) 
            if os.path.exists(p_imgs[i]):
                img = mpimg.imread(p_imgs[i])
                if i == 0:
                    offset = int(img.shape[1] * 0.15)
                    axs[row, i].imshow(img[offset:-offset, offset:-offset])

                else:
                #     offset = int(img.shape[1] * 0.01)
                    axs[row, i].imshow(img)
    for i, index in enumerate(selected.keys()):
        a_index, b_index = selected[index]
        id = sdf_name_list[int(index)]
        sketch_img = os.path.join(sketch_dir, f'{id}.jpg')
        shape_img = f'/vol/vssp/datasets/multiview/3VS/datasets/ShapeNetCore.v2/03001627/{id}/models/model_normalized.png'
        mesh_filenames = [os.path.join(img_folder, f'{index}_{a_index}_{b_index}_interpolate_z_{col_index}.png') for col_index in range(5)]
        all_img = [sketch_img] + [shape_img] + mesh_filenames
        vis_img(all_img, i, axs)

    [axi.set_axis_off() for axi in axs.ravel()]
    plt.tight_layout()
    # plt.show()
    save_path = os.path.join(saving_folder, f'interp.pdf')
    plt.savefig(save_path)
    print(f'Save to: {save_path}')


def make_comparison_other():
    # network_dir = '/vol/vssp/datasets/multiview/3VS/visualization/images/3dv20/modelnet'
    network_dir = '/scratch/Remote_data/SketchGen/datasets/other_sketch_types/img/network'
    # human_sketch_dir = '/vol/vssp/datasets/multiview/3VS/visualization/images/3dv20/human_whiteshaded'
    human_sketch_dir = '/scratch/Remote_data/SketchGen/datasets/other_sketch_types/img/human_sketch'
    # shape_dir = 'smb://isilon01-az1.surrey.ac.uk/rs301$/VR_Sketch/mitsuba_view/shape_view/modelnet'
    shape_dir = '/scratch/visualization/sketch_to_shape_gen/modelnet_img'
    exp_dir = '/scratch/visualization/sketch_to_shape_gen/exps'
    saving_folder = f'/scratch/visualization/sketch_to_shape_gen/comparison_other' #/p={p}'

    exp = 'gen_conNF_v43_run1'
    name_list_path = '/vol/vssp/datasets/multiview/3VS/datasets/3DV_dataset/list/human_test.txt'
    test_name_list = [line.rstrip().split(' ')[0] for line in open(name_list_path)]

    import matplotlib.pyplot as plt
    from PIL import Image
    import matplotlib.image as mpimg
    nrows = 2
    ncols = 2 + 2 + 5 # + 2

    def vis_img(p_imgs, row, axs):
        # f = plt.figure(figsize=(20, 6), dpi=80)
        for i in range(len(p_imgs)):
            # f.add_subplot(1, len(p_imgs), i+1)
            # img = Image.open(p_imgs[i])
            # small = img.crop((offset, offset, -offset, -offset)) 
            if os.path.exists(p_imgs[i]):
                img = mpimg.imread(p_imgs[i])
                # if i == 0:
                #     offset = int(img.shape[1] * 0.15)
                #     axs[row, i].imshow(img[offset:-offset, offset:-offset])

                # else:
                #     offset = int(img.shape[1] * 0.01)
                axs[row, i].imshow(img)

    sketch_types = ['human_sketch', 'network']

    # from utils.vis_utils import vis_pc
    scale_factor = 1.5
    for index, id in enumerate(test_name_list):
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols*2, nrows*2])

        shape_img = os.path.join(shape_dir, f'{id}.png')

        for si, sketch_type in enumerate(sketch_types):
            if sketch_type == 'network':
                sketch_img = os.path.join(network_dir, f'{id}.png')
                # sketch_path = os.path.join(network_dir, id + '_opt_quad_network_20_aggredated.txt')
            elif sketch_type == 'human_sketch':
                sketch_img = os.path.join(human_sketch_dir, f'{id}.png')
                # sketch_path = os.path.join(human_sketch_dir, id + '.txt')
            # sketch_array = np.loadtxt(sketch_path, delimiter=",").astype('float32')
            # sketch_array = farthest_point_sample(sketch_array, 4096)
            # img = vis_pc(normalize_to_box(sketch_array)[0]*0.4*scale_factor)
            # axs[si, 0].imshow(img)
            # print(sketch_img)
            gen_imgs = [os.path.join(exp_dir, exp, f'test_epoch300_{sketch_type}/test_{index}_sample_{item}.png') for item in range(0,7)]
            all_img = [sketch_img] + [shape_img] + gen_imgs
            vis_img(all_img, si, axs)


        [axi.set_axis_off() for axi in axs.ravel()]
        plt.tight_layout()
        # plt.show()
        save_path = os.path.join(saving_folder, f'{index}.pdf')
        plt.savefig(save_path)
        print(f'Save to: {save_path}')
        # plt.cla()
        plt.close(fig)



def make_comparison(p = 1):
    sketch_dir = '/scratch/visualization/chair_1005_align_view'#'/vol/vssp/datasets/multiview/3VS/visualization/images/mitsuba_view/chair_1005_align_view'
    exps = [
        'gen_conNF_v24_run1', 
        'gen_conNF_v43_run1']
    saving_folder = f'/scratch/visualization/sketch_to_shape_gen/comparison_miss_absolute' #/p={p}'
    # for index, id in enumerate(sdf_name_list):

    exp_dir = '/scratch/visualization/sketch_to_shape_gen/exps'
    
    sketch_retrieve = np.load(os.path.join(exp_dir, f'sketch_retrieve_p={p}.npy'))
    gen_retrieve = np.load(os.path.join(exp_dir, f'gen_retrieve_p={p}.npy'))
    retrieve_previous = np.load(f'/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/multifold/adaptive_triplet_multickpt_sym_aug_1/inference/test_rank_aligned_sketch_last.npy')
    retrieve_structure = np.argsort(retrieve_previous, axis=1)
    CD_list = {exp_name: np.load(os.path.join(exp_dir, exp_name, 'test_epoch300/cd.npy')) for exp_name in exps}
    # sketch_SDF_list = {exp_name: np.load(os.path.join(exp_dir, exp_name, 'test_epoch300/sketch_SDF.npy')).mean(axis=-1) for exp_name in exps}
    sketch_SDF_list = {exp_name: np.absolute(np.load(os.path.join(exp_dir, exp_name, 'test_epoch300/sketch_SDF.npy'))).mean(axis=-1) for exp_name in exps}

    test_shape_file = os.path.join(DATA_DIR, f'split/{split}_shape.txt') #'/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test_shape.txt'
    test_shape_list = [line.rstrip() for line in open(test_shape_file)]
    import matplotlib.pyplot as plt
    from PIL import Image
    import matplotlib.image as mpimg
    nrows = len(exps) + 2 + 1
    ncols = 2 + 1 + 1 + 5 # + 2

    def vis_img(p_imgs, row, axs):
        # f = plt.figure(figsize=(20, 6), dpi=80)
        for i in range(len(p_imgs)):
            # f.add_subplot(1, len(p_imgs), i+1)
            # img = Image.open(p_imgs[i])
            # small = img.crop((offset, offset, -offset, -offset)) 
            if os.path.exists(p_imgs[i]):
                img = mpimg.imread(p_imgs[i])
                if i == 0:
                    offset = int(img.shape[1] * 0.15)
                    axs[row, i].imshow(img[offset:-offset, offset:-offset])

                else:
                #     offset = int(img.shape[1] * 0.01)
                    axs[row, i].imshow(img)
    
    for index, id in enumerate(sdf_name_list):
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols*2, nrows*2])

    # id = sdf_name_list[index]

        sketch_img = os.path.join(sketch_dir, f'{id}.jpg')
        shape_img = f'/vol/vssp/datasets/multiview/3VS/datasets/ShapeNetCore.v2/03001627/{id}/models/model_normalized.png'
        recon_img = f'/scratch/visualization/sketch_to_shape_gen/exps/recon_sketch/test_{index}.png'
        # Sketch retrieval
        topk = sketch_retrieve[index][:5]
        retrieve_imgs = [f'/vol/vssp/datasets/multiview/3VS/datasets/ShapeNetCore.v2/03001627/{test_shape_list[int(id)]}/models/model_normalized.png' for id in topk]
        all_img = [sketch_img] + [shape_img] + [''] + [''] + retrieve_imgs
        vis_img(all_img, 0, axs)

        for exp_index, exp_name in enumerate(exps):
            # load metric

            gen_imgs = [os.path.join(f'/scratch/visualization/sketch_to_shape_gen/exps', exp_name, f'test_{index}_sample_{item}.png') for item in range(1,7)]
            
            # sort generated shapes: 2~7
            order = np.argsort(CD_list[exp_name][index][2:]).tolist()
            # plot score
            new_order = [0]+list(np.array(order) + 1)
            ordered_sdf = sketch_SDF_list[exp_name][index][new_order]
            [axs[exp_index+1, i+3].set_title('{:.3f}'.format(ordered_sdf[i])) for i in range(sketch_SDF_list[exp_name].shape[1])]
            all_img = [sketch_img] + [shape_img] + [recon_img] + [gen_imgs[0]] + [gen_imgs[ :][i] for i in order]
            vis_img(all_img, exp_index+1, axs)
            if exp_name == 'gen_conNF_v43_run1':
                # Gen retrieval
                topk = gen_retrieve[index*7+1:(index+1)*7, 0][new_order]
                retrieve_imgs = [f'/vol/vssp/datasets/multiview/3VS/datasets/ShapeNetCore.v2/03001627/{test_shape_list[int(id)]}/models/model_normalized.png' for id in topk]
                all_img = [sketch_img] + [shape_img] + [''] + retrieve_imgs
                vis_img(all_img, exp_index+2, axs)

        # Sketch retrieval
        topk = retrieve_structure[index][:5]
        retrieve_imgs = [f'/vol/vssp/datasets/multiview/3VS/datasets/ShapeNetCore.v2/03001627/{test_shape_list[int(id)]}/models/model_normalized.png' for id in topk]
        all_img = [sketch_img] + [shape_img] + [''] + [''] + retrieve_imgs
        vis_img(all_img, nrows-1, axs)


        [axi.set_axis_off() for axi in axs.ravel()]
        plt.tight_layout()
        # plt.show()
        save_path = os.path.join(saving_folder, f'{index}.pdf')
        plt.savefig(save_path)
        print(f'Save to: {save_path}')
        # plt.cla()
        plt.close(fig)



if __name__ == '__main__':
    # argv = sys.argv
    # argv = argv[argv.index("--") + 1:]
    # obj_dir = argv[0]
    # save_dir = argv[1]
    # data_type = argv[2] #'sketch' or 'shape'
    # render_type = argv[3] #'Phong' or 'depth'

    # make_interpolation()
    render_gif()
    quit()

    render_target = ['interp_shape']

    exp_list = {
        'gen_conNF_v24_run1': 1,
        'gen_conNF_v43_run1': 2,
    }
    work_info = []

    if 'shape' in render_target:
        # render shape model
        obj_dir = '/vol/vssp/datasets/multiview/3VS/datasets/ShapeNetCore.v2/03001627'
        # model_files = [os.path.join(obj_dir, f'{id}/models/model_normalized.obj') for id in sdf_name_list]

        # obj_dir = '/vol/vssp/datasets/multiview/3VS/datasets/ShapeNet/original/obj/03001627'
        # model_files = [os.path.join(obj_dir, f'{id}/model.obj') for id in sdf_name_list]

        name_list_path = os.path.join(DATA_DIR, f'split/{split}_shape.txt')
        test_name_list = [line.rstrip() for line in open(name_list_path)]
        model_files = [os.path.join(obj_dir, f'{id}/models/model_normalized.obj') for id in test_name_list]

        # model_files = glob(os.path.join(obj_dir, '*/models/model_normalized.obj'))

        work_info.extend([(path, os.path.dirname(path), 0) for path in model_files])
    if 'results' in render_target:
        obj_dir = '/scratch/visualization/sketch_to_shape_gen/exps'
        for exp_name in exp_list.keys():
            model_files = glob(os.path.join(obj_dir, f'{exp_name}/*/*.ply'))
            save_dir = os.path.join(obj_dir, f'{exp_name}')
            work_info.extend([(path, save_dir, exp_list[exp_name]) for path in model_files])
    if 'recon_sketch' in render_target:
        obj_dir = '/scratch/visualization/sketch_to_shape_gen/exps/recon_sketch'
        model_files = glob(os.path.join(obj_dir, '*.ply'))
        work_info.extend([(path, obj_dir, 3) for path in model_files])
    if 'other_results' in render_target:
        obj_dir = '/scratch/visualization/sketch_to_shape_gen/exps'
        for exp_name in exp_list.keys():
            model_files = glob(os.path.join(obj_dir, f'{exp_name}/test_epoch300_*/ply/*.ply'))
            # save_dir = os.path.join(obj_dir, f'{exp_name}')
            work_info.extend([(path, os.path.dirname(path), exp_list[exp_name]) for path in model_files])
    if 'other_shape' in render_target:
        name_list_path = '/vol/vssp/datasets/multiview/3VS/datasets/3DV_dataset/list/human_test.txt'
        test_name_list = [line.rstrip().split(' ')[0] for line in open(name_list_path)]
        obj_dir = '/scratch/dataset/3DV_dataset/shape/obj'
        save_dir = '/scratch/visualization/sketch_to_shape_gen/modelnet_img'
        model_files = [os.path.join(obj_dir, f'{id}.obj') for id in test_name_list]
        work_info.extend([(path, save_dir, 0) for path in model_files])
    if 'interp_shape' in render_target:
        obj_dir = '/scratch/Remote_data/SketchGen/experiments/gen_conNF_v43_run1/test_epoch300_original/interp'
        model_files = glob(os.path.join(obj_dir, '*.ply'))
        new_model_files = [path for path in model_files if not os.path.exists(path[:-3]+'png')]
        work_info.extend([(path, obj_dir, 2) for path in new_model_files])

    # else:
    #     NotImplementedError

    # render([model_files[128], './', 3])
    # quit()
    with Pool(6) as p:
        p.map(render, work_info)

