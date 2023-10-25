import pytorch3d.loss
import torch
import os
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.ops import sample_points_from_meshes
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def sample_ply(obj_paths):
    mesh_list = []
    for obj_path in obj_paths:
        if not os.path.exists(obj_path):
            obj_path = os.path.join(os.path.dirname(__file__), 'template.ply')
        verts, faces = load_ply(obj_path)
        mesh = Meshes(verts=[verts], faces=[faces])
        mesh_list.append(mesh)
    meshes = join_meshes_as_batch(mesh_list)
    pcs = sample_points_from_meshes(
                meshes, num_samples=4096)
    return pcs

def normalize_to_box(input):
    """
    normalize point cloud to unit bounding box
    center = (max - min)/2
    scale = max(abs(x))
    input: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance

    From https://github.com/yifita/pytorch_points
    """
    if len(input.shape) == 2:
        axis = 0
        P = input.shape[0]
        D = input.shape[1]
    elif len(input.shape) == 3:
        axis = 1
        P = input.shape[1]
        D = input.shape[2]
    else:
        raise ValueError()
    
    if isinstance(input, np.ndarray):
        maxP = np.amax(input, axis=axis, keepdims=True)
        minP = np.amin(input, axis=axis, keepdims=True)
        centroid = (maxP+minP)/2
        input = input - centroid
        furthest_distance = np.amax(np.abs(input), axis=(axis, -1), keepdims=True)
        input = input / furthest_distance
    elif isinstance(input, torch.Tensor):
        maxP = torch.max(input, dim=axis, keepdim=True)[0]
        minP = torch.min(input, dim=axis, keepdim=True)[0]
        centroid = (maxP+minP)/2
        input = input - centroid
        in_shape = list(input.shape[:axis])+[P*D]
        furthest_distance = torch.max(torch.abs(input).reshape(in_shape), dim=axis, keepdim=True)[0]
        furthest_distance = furthest_distance.unsqueeze(-1)
        input = input / furthest_distance
    else:
        raise ValueError()

    return input, centroid, furthest_distance

def rotate_point_cloud(batch_data, dim='x', angle=-90): # torch.Size([1024, 3])
    rotation_angle = angle/360 * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if dim=='x':
        rotation_matrix = torch.tensor([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]]).float()
    elif dim=='y':
        rotation_matrix = torch.tensor([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]]).float()
    elif dim=='z':
        rotation_matrix = torch.tensor([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]]).float()
    else:
        NotImplementedError
        
    rotated_data = torch.mm(batch_data, rotation_matrix)
    return rotated_data # torch.Size([1024, 3])

def vis_pc(pc):
    import sys
    sys.path.append(TOOL_DIR) # where I put the 'xgutils'
    # this is a tool for visualize pointcloud , can send you later
    import xgutils.vis.fresnelvis as fresnelvis
    resolution=(512, 512)
    samples=32
    cloudR = 0.008
    vis_camera = dict(camPos=np.array([2, 2, 2]), camLookat=np.array([0., 0., 0.]),
                            camUp=np.array([0, 1, 0]), camHeight=2, resolution=resolution, samples=samples)
    if isinstance(pc, np.ndarray):
        pc = torch.from_numpy(pc)

    new_pc = rotate_point_cloud(pc, dim='y', angle=180)
    pc_view = fresnelvis.renderMeshCloud(
                cloud=new_pc, cloudR=cloudR, **vis_camera)
    return pc_view

def compute_CD(selected, epoch, samples, encoder, decoder, split, vis=False):
    """
    selected: index list of shape samples for evaluation
    epoch: the epoch of ckpt to be evaluated
    samples: dataset class
    encoder: loaded PointNet++ encoder
    decoder: loaded DeepSDF decoder
    split: 'train' or 'test'
    vis: whether to visualize
    """
    # TODO: change to your experiment dir
    testing_folder = os.path.join(exp_dir, f'test_epoch{epoch}')
    if not os.path.exists(testing_folder):
        os.mkdir(testing_folder)

    # for visualize the reconstruction results
    if vis:
        import matplotlib.pyplot as plt
        scale_factor = 0.6

        nrows = len(selected)
        ncols = 3
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols, nrows])

    cd_list = []
    encoder.eval()
    decoder.eval()
    with torch.no_grad():

        for id, index in enumerate(selected):
            # TODO: get point cloud sample by index, need to be replaced with yours function
            pc_data, voxel_data, sdf_data, indices = samples[index]
            points = torch.tensor(pc_data).unsqueeze(0).to(device)
            # points = torch.tensor(pc_data[:2]).transpose(2, 1).to(device)
            embs = encoder(points.transpose(2, 1))
            # sketch_emb, shape_emb = embs[0:1], embs[1:2]

            predicted_shape_emb = embs[0]
            
            # TODO: extract mesh from embedding, can be replaced with your func
            predict_filename = os.path.join(testing_folder, f'{split}_{index}_predict')
            if not os.path.exists(predict_filename + '.ply'):
                success = create_mesh(
                    decoder, predicted_shape_emb, predict_filename, N=256, max_batch=int(2 ** 18)
                )

            ply_paths = [predict_filename + '.ply']

            gen_pcs = sample_ply(ply_paths)
            gen_pcs = normalize_to_box(gen_pcs)[0]

            # Normalize reconstructed point cloud
            gt_shape = normalize_to_box(points)[0]
            cd_dist = pytorch3d.loss.chamfer_distance(gen_pcs, gt_shape.repeat(gen_pcs.shape[0], 1, 1), batch_reduction=None)[0]
            cd_list.append(cd_dist)

            if vis:
                axs[id, 0].imshow(vis_pc(gt_shape[0]*scale_factor))
                axs[id, 1].imshow(vis_pc(gen_pcs[0]*scale_factor))

                # inference for single sketch: [sketch, GT, shape_emb, sketch_emb, sampled]
                axs[id, 0].set_title(f'ID{index}')
                axs[id, 1].set_title("{:.2f}".format(cd_dist[0] * 100))

    if vis: 
        [axi.set_axis_off() for axi in axs.ravel()]
        # fig.suptitle(f"Epoch{epoch}_{split}_reconstruction")
        plt.tight_layout(True)

        plt.savefig(os.path.join(testing_folder, f'{split}_decoder.png'))
        plt.cla()

    return {
        'cd': cd_list
    }