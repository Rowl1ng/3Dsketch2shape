import trimesh
import numpy as np
import igl
import os
from igl import signed_distance
from skimage import measure
import sys
# from utils.pc_utils import normalize_to_box
import meshplot as mp # mesh display 
mp.offline()

obj_dir = '/vol/vssp/datasets/multiview/3VS/datasets/ShapeNetCore.v2/03001627/'
sdf_dir = '/vol/vssp/datasets/multiview/SDF_ShapeNet/chairs/samples'
sketch_dir = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/aligned_sketch'
DATA_DIR = '/vol/vssp/datasets/multiview/3VS/datasets/SketchGen'

save_dir = '/scratch/visualization/sketch_to_shape_gen/align'
# Variables
visualize = False #True


# Change the following path to data_dir of dataset:
data_dir = '/user/HS229/ll00931/Downloads' #'./data/watertight_obj/'
# Output folder:
output_dir = '/user/HS229/ll00931/Downloads' #'./recentered_meshes'
os.makedirs(output_dir, exist_ok=True)


import point_cloud_utils as pcu
def sample_pointcloud_mesh(obj_path, point_num):
    v, f = pcu.load_mesh_vf(obj_path)
    # if off_n is None or off_n.shape[0] != off_v.shape[0]:
    f_idx, bc = pcu.sample_mesh_random(v, f, num_samples=point_num)
    v_sampled = pcu.interpolate_barycentric_coords(f, f_idx, bc, v)
    return v_sampled

def sample_pc(point_num = 4096):
     # 15000
    name_list_path = os.path.join(DATA_DIR, f'split/all.txt')
    pc_name_list = [line.rstrip() for line in open(name_list_path)]
    pc_array = []
    for item in pc_name_list:
        obj_path = os.path.join(obj_dir, f'{item}/models/model_normalized.obj' )
        pc = sample_pointcloud_mesh(obj_path, point_num)
        pc_array.append(pc)
    pc_array = np.array(pc_array)
    np.save(os.path.join(save_dir, f'shape_pc_{point_num}.npy'), pc_array)

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def save_sketch_pc(split = 'train', point_num = 15000):
    name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}_sketch.txt')
    sdf_name_list = [line.rstrip() for line in open(name_list_path)]
    pc_array = []
    for name in sdf_name_list:
        mesh = trimesh.load(os.path.join(obj_dir, f'{name}/models/model_normalized.obj' ), force='mesh')
        # Get mesh vertices and faces:
        v = mesh.vertices
        # Bounding box check:
        m = v.min(axis=0)
        M = v.max(axis=0)
        shape_centroid = (m + M) / 2
        sketch_pc = np.load(os.path.join(sketch_dir, f'{name}.npy' ))
        mesh_max = np.amax(sketch_pc, axis=0)
        mesh_min = np.amin(sketch_pc, axis=0)
        sketch_centroid = (mesh_max + mesh_min) / 2
        translated_sketch = sketch_pc + shape_centroid - sketch_centroid
        if translated_sketch.shape[0] != point_num:
            translated_sketch = farthest_point_sample(translated_sketch, point_num)
        pc_array.append(translated_sketch)
    pc_array = np.array(pc_array)
    np.save(os.path.join(save_dir, f'sketch_pc_{point_num}_{split}.npy'), pc_array)

def check_shape_sketch_pc(index, split):
        # Mesh name:
    name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}_sketch.txt')
    sdf_name_list = [line.rstrip() for line in open(name_list_path)]
    id = sdf_name_list[index]

    # 1. Load mesh:
    mesh = trimesh.load(os.path.join(obj_dir, f'{id}/models/model_normalized.obj' ), force='mesh')
    v = mesh.vertices 
    f = mesh.faces

    # 2. Load SDF:
    sdf = np.load(os.path.join(sdf_dir, f'{id}.npz' ))['pos']
    # 3. Load shape pc:
    point_num = 15000
    shape_pc_all = np.load(os.path.join(save_dir, f'shape_pc_{point_num}.npy'))
    name_list_path = os.path.join(DATA_DIR, f'split/all.txt')
    pc_name_list = [line.rstrip() for line in open(name_list_path)]

    shape_pc = shape_pc_all[pc_name_list.index(id)]

    # 3. Load sketch pc:
    sketch_pc_all = np.load(os.path.join(save_dir, f'sketch_pc_{point_num}_{split}.npy'))
    sketch_pc = sketch_pc_all[index]
    # Visualization 
    if visualize:
        p = mp.plot(v, f)
        p = mp.plot(v, f, return_plot=True)

        value = 0.0001 # can be smaller    
        clamping = np.where(np.absolute(sdf[:,-1]) < value)[0]
        sdf_pc = sdf[clamping, :3]

        # TODO: vis SDF points
        p.add_points(sdf_pc, shading={"point_color": "red", "point_size": 0.03})
        # TODO: vis sketch points
        p.add_points(sketch_pc, shading={"point_color": "blue", "point_size": 0.03})

        p.add_points(shape_pc, shading={"point_color": "orange", "point_size": 0.03})

        p.save(os.path.join(save_dir, f"{id}_{split}_check.html"))

def sketch_npy():
    split = 'test'
    index = 98
    name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}_sketch.txt')
    sdf_name_list = [line.rstrip() for line in open(name_list_path)]
    id = sdf_name_list[index]
    # id = '8b005a01cf4ae50dab49dceef0d15b99'
    # index = sdf_name_list.index(id)
    target_obj_path = os.path.join(obj_dir, f'{id}/models/model_normalized.obj')
    gen_dir = '/scratch/Remote_data/SketchGen/experiments/gen_conNF_v43_run1/test_epoch300'
    mesh = trimesh.load(target_obj_path, force='mesh')
    v = mesh.vertices 
    f = mesh.faces

    point_num = 4096
    sketch_pc_all = np.load(os.path.join(save_dir, f'sketch_pc_{point_num}_{split}.npy'))
    sketch_pc = sketch_pc_all[index]
    if visualize:
        p = mp.plot(v, f)
        p = mp.plot(v, f, return_plot=True)

        # TODO: vis sketch points
        p.add_points(sketch_pc, shading={"point_color": "blue", "point_size": 0.03})
        p.save(os.path.join(save_dir, f"{id}_{split}_check.html"))

    import shutil 
    sample_dir = '/scratch/visualization/sketch_to_shape_gen/sample5'
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    # np.save(os.path.join(sample_dir, f'sketch_{point_num}_{id}.npy'), sketch_pc)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sketch_pc)
    o3d.io.write_point_cloud(os.path.join(sample_dir, f'sketch_{point_num}_{id}.ply'), pcd)
    shutil.copytree(os.path.join(obj_dir, id), os.path.join(sample_dir, 'GT'))
    for i in range(7):
        shutil.copy(os.path.join(gen_dir, f'test_{index}_sample_{i}.ply'), sample_dir)


def vis_SDF():
    split = 'test'
    index = 34
    name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}_sketch.txt')
    sdf_name_list = [line.rstrip() for line in open(name_list_path)]
    id = sdf_name_list[index]

    target_obj_path = os.path.join(obj_dir, f'{id}/models/model_normalized.obj')
    mesh = trimesh.load(target_obj_path, force='mesh')
    v = mesh.vertices 
    f = mesh.faces

    sdf = np.load(os.path.join(sdf_dir, f'{id}.npz'))['pos']

    sdf_pc = farthest_point_sample(sdf[:, :3], 2048)


    # p = mp.plot(v, f, c=v[:, 1], )
    p = mp.plot(v, f, c=v[:, 1], return_plot=True)
    # TODO: vis SDF points
    p.add_points(sdf_pc, shading={"point_color": "red", "point_size": 0.1})
    m = np.min(sdf_pc[:, :3], axis=0)
    ma = np.max(sdf_pc[:, :3], axis=0)

    # Corners of the bounding box
    v_box = np.array([[m[0], m[1], m[2]], [ma[0], m[1], m[2]], [ma[0], ma[1], m[2]], [m[0], ma[1], m[2]],
                    [m[0], m[1], ma[2]], [ma[0], m[1], ma[2]], [ma[0], ma[1], ma[2]], [m[0], ma[1], ma[2]]])

    # Edges of the bounding box
    f_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], 
     [7, 4], [0, 4], [1, 5], [2, 6], [7, 3]], dtype=np.int)

    p.add_edges(v_box, f_box, shading={"line_color": "black"})
    p.save(os.path.join(save_dir, f"{id}_{split}_vis_SDF.html"))


def align(index, split):
    # Mesh name:
    # id = sdf_name_list[index]
    id = '2a2d705d0238396488422a4c20c0b1e6'
    # 1. Load mesh:
    mesh = trimesh.load(os.path.join(obj_dir, f'{id}/models/model_normalized.obj' ), force='mesh')
    # Get mesh vertices and faces:
    v = mesh.vertices
    f = mesh.faces

    # Bounding box check:
    m = v.min(axis=0)
    M = v.max(axis=0)
    print("min = [%f,%f,%f]" % (m[0],m[1],m[2]) )
    print("max = [%f,%f,%f]" % (M[0],M[1],M[2]) )
    # _, centroid, furthest_distance = normalize_to_box(v)
    shape_centroid = (m + M) / 2

    print('mesh.centroid: ', mesh.centroid)
    print('shape_centroid: ', shape_centroid)

    # print('func mesh centroid: {}, scale: {}'.format(centroid, furthest_distance))

    # 2. Load SDF:
    sdf = np.load(os.path.join(sdf_dir, f'{id}.npz' ))['pos']

    # 3. Load shape pc:
    point_num = 15000
    shape_pc_all = np.load(os.path.join(save_dir, f'shape_pc_{point_num}.npy'))
    name_list_path = os.path.join(DATA_DIR, f'split/all.txt')
    pc_name_list = [line.rstrip() for line in open(name_list_path)]

    shape_pc = shape_pc_all[pc_name_list.index(id)]

    # Load sketch:
    syn_sketch_dir = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/network'
    sketch_pc = np.load(os.path.join(syn_sketch_dir, f'{id}.npy' ))
    # TODO: compute sketch centroid and recenter sketch
    # normalized_sketch, sketch_centroid, furthest_distance = normalize_to_box(sketch_pc)
    mesh_max = np.amax(sketch_pc, axis=0)
    mesh_min = np.amin(sketch_pc, axis=0)
    sketch_centroid = (mesh_max + mesh_min) / 2
    # print('old sketch centroid: {}, scale: {}'.format(sketch_centroid, furthest_distance))

    # new_normalized_sketch, centroid, furthest_distance = normalize_to_box(normalized_sketch)

    # print('sketch centroid: {}, scale: {}'.format(centroid, furthest_distance))

    # translated_sketch_v1 = sketch_pc + mesh.centroid - sketch_centroid
    translated_sketch_v2 = sketch_pc + shape_centroid - sketch_centroid
    # TODO: Might want to save recentered mesh:
    # outmesh_path = os.path.join(output_dir, meshname)
    # ret = igl.write_triangle_mesh(outmesh_path, v, f)



    # TODO: Sample points on the surface of a mesh:


    # Visualization 
    if visualize:
        p = mp.plot(v, f)
        p = mp.plot(v, f, return_plot=True)

        value = 0.0001 # can be smaller    
        clamping = np.where(np.absolute(sdf[:,-1]) < value)[0]
        sdf_pc = sdf[clamping, :3]

        # TODO: vis SDF points
        p.add_points(sdf_pc, shading={"point_color": "red", "point_size": 0.03})
        # TODO: vis sketch points
        # p.add_points(translated_sketch_v1, shading={"point_color": "green", "point_size": 0.03})

        p.add_points(translated_sketch_v2, shading={"point_color": "blue", "point_size": 0.03})

        p.add_points(shape_pc, shading={"point_color": "orange", "point_size": 0.03})

        p.save(os.path.join(save_dir, f"{id}_{split}_original.html"))

if __name__ == '__main__':
    # save_sketch_pc(split='train', point_num=4096)
    # save_sketch_pc(split='test', point_num=4096)
    # align(1, 1)
    sketch_npy()
    # vis_SDF()
    quit()
    # sample_pc(point_num=15000)
    # quit()
    # namelist
    split = 'train'
    name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}_sketch.txt')
    sdf_name_list = [line.rstrip() for line in open(name_list_path)]
    # np.random.seed(0) 
    selected = np.random.choice(len(sdf_name_list), 10, replace=False)
    for index in selected:
        # align(index, split)
        check_shape_sketch_pc(index, split)