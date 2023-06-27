import os
import torch
import numpy as np
from utils.pc_utils import rotate_point_cloud
from pytorch3d.io import load_ply, load_obj

TOOL_DIR = os.getenv('TOOL_DIR')
import sys
sys.path.append(TOOL_DIR)
import xgutils.vis.fresnelvis as fresnelvis

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

def vis_ply(obj_path):
    if not os.path.exists(obj_path):
        obj_path = os.path.join(os.path.dirname(__file__), 'template.ply')
    verts, faces = load_ply(obj_path)
    img = fresnelvis.renderMeshCloud({"vert":verts,"face":faces}, **vis_camera)
    return img

def vis_obj(obj_path):
    verts, faces = load_obj(obj_path)
    img = fresnelvis.renderMeshCloud({"vert":verts,"face":faces}, **vis_camera)
    return img