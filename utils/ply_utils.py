import numpy as np
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.ops import sample_points_from_meshes
import os
from igl import signed_distance

import torch
from pytorch3d.renderer import (
look_at_view_transform,
FoVPerspectiveCameras, 
PointLights,
RasterizationSettings,
MeshRasterizer
)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Pixel coordinates
IMG_SIZE = 224 #image.shape[-1]
X, Y = torch.meshgrid(torch.arange(0, IMG_SIZE), torch.arange(0, IMG_SIZE))
X = (2*(0.5 + X.unsqueeze(0).unsqueeze(-1))/IMG_SIZE - 1).float().to(device)
Y = (2*(0.5 + Y.unsqueeze(0).unsqueeze(-1))/IMG_SIZE - 1).float().to(device)
R, T = look_at_view_transform(1.2, 30, 225) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)   
raster_settings = RasterizationSettings(
    image_size=IMG_SIZE,
    blur_radius=0.000001,
    faces_per_pixel=1,
)     

# instantiate renderers
depth_renderer = MeshRasterizer(
    cameras=cameras,
    raster_settings=raster_settings
)
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])


EPS = 1e-4
def read_ply_point_normal(shape_name):
    file = open(shape_name,'r')
    lines = file.readlines()

    start = 0
    while True:
        line = lines[start].strip()
        if line == "end_header":
            start += 1
            break
        line = line.split()
        if line[0] == "element":
            if line[1] == "vertex":
                vertex_num = int(line[2])
        start += 1

    vertices = np.zeros([vertex_num,3], np.float32)
    normals = np.zeros([vertex_num,3], np.float32)
    for i in range(vertex_num):
        line = lines[i+start].split()
        vertices[i,0] = float(line[0]) #X
        vertices[i,1] = float(line[1]) #Y
        vertices[i,2] = float(line[2]) #Z
        normals[i,0] = float(line[3]) #normalX
        normals[i,1] = float(line[4]) #normalY
        normals[i,2] = float(line[5]) #normalZ
    return vertices, normals


def triangulate_mesh_with_subdivide(vertices, faces, triangle_subdivide_cnt=3):
    vertices = np.array(vertices)
    new_faces = split_ply(faces)

    ### new
    cnt = 0
    final_vertices, final_faces = [], []
    for face in new_faces:
        face_vertices = vertices[face]

        base_1, base_2, base_3 = [face_vertices[0], face_vertices[1]], [face_vertices[1], face_vertices[2]], [
            face_vertices[0], face_vertices[2]]

        base_1_lin, base_2_lin, base_3_lin = np.linspace(base_1[0], base_1[1],
                                                         triangle_subdivide_cnt + 2), np.linspace(base_2[0],
                                                                                                  base_2[1],
                                                                                                  triangle_subdivide_cnt + 2), np.linspace(
            base_3[0], base_3[1], triangle_subdivide_cnt + 2)
        vertices_lin = [base_1_lin]
        for i in range(1, triangle_subdivide_cnt + 1):
            new_lin = np.linspace(base_3_lin[i], base_2_lin[i], triangle_subdivide_cnt + 2 - i)
            vertices_lin.append(new_lin)
        vertices_lin.append([face_vertices[2]])

        ## push
        vertices_to_append = np.zeros((0, 3))
        for vertex_lin in vertices_lin:
            vertices_to_append = np.concatenate((vertices_to_append, vertex_lin), axis=0)

        faces_to_append = []
        current_cnt = 0
        for i in range(triangle_subdivide_cnt + 1):
            for j in range(triangle_subdivide_cnt + 1 - i):
                faces_to_append.append(
                    [current_cnt + j, current_cnt + j + 1, current_cnt + (triangle_subdivide_cnt + 2 - i) + j])
                if i > 0:
                    faces_to_append.append(
                        [current_cnt + j, current_cnt - (triangle_subdivide_cnt + 1 - i) + j - 1,
                         current_cnt + j + 1])
            current_cnt += (triangle_subdivide_cnt + 2 - i)

        final_vertices.append(vertices_to_append)
        final_faces.append(np.array(faces_to_append) + cnt)
        cnt += vertices_to_append.shape[0]

    final_vertices = np.concatenate(tuple(final_vertices), axis=0)
    final_faces = np.concatenate(tuple(final_faces), axis=0)

    return final_vertices, final_faces

def split_ply(faces):

    result_faces = []
    for face in faces:
        for i in range(2, len(face)):
            result_faces.append([face[0], face[i-1], face[i]])

    return np.array(result_faces)

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
    return meshes, pcs

def compute_SDF_loss(sketch_pc, meshes, index_list):
    sketch_dist = []
    # hull_dist = []
    for index in index_list:
        v = meshes.verts_list()[index]
        f = meshes.faces_list()[index]
        sketch_values = signed_distance(np.array(sketch_pc[:, :3]), np.array(v), np.array(f))[0]
        # hull_values = signed_distance(np.array(hull_pc[:, :3]), np.array(v), np.array(f))[0]
        sketch_dist.append(sketch_values)
        # hull_dist.append(hull_values)
    return np.array(sketch_dist) #, np.array(hull_dist)

def depth_2_normal(depth, depth_unvalid, cameras):

    B, H, W, C = depth.shape

    grad_out = torch.zeros(B, H, W, 3).to(device)
    # Pixel coordinates
    xy_depth = torch.cat([X, Y, depth], 3).to(device).reshape(B,-1, 3)
    xyz_unproj = cameras.unproject_points(xy_depth, world_coordinates=False)

    # compute tangent vectors
    XYZ_camera = xyz_unproj.reshape(B, H, W, 3)
    vx = XYZ_camera[:,1:-1,2:,:]-XYZ_camera[:,1:-1,1:-1,:]
    vy = XYZ_camera[:,2:,1:-1,:]-XYZ_camera[:,1:-1,1:-1,:]

    # finally compute cross product
    normal = torch.cross(vx.reshape(-1, 3),vy.reshape(-1, 3))
    normal_norm = normal.norm(p=2, dim=1, keepdim=True)

    normal_normalized = normal.div(normal_norm)
    # reshape to image
    normal_out = normal_normalized.reshape(B, H-2, W-2, 3)
    grad_out[:,1:-1,1:-1,:] = (0.5 - 0.5*normal_out)

    # zero out +Inf
    grad_out[depth_unvalid] = 0.0

    return grad_out

def render(obj_path):
    if not os.path.exists(obj_path):
        obj_path = os.path.join(os.path.dirname(__file__), 'template.ply')
    verts, faces = load_ply(obj_path)
    mesh = Meshes(verts=[verts], faces=[faces])
    depth = depth_renderer(meshes_world=mesh.to(device), cameras=cameras, lights=lights)
    depth_ref = depth.zbuf[...,0].unsqueeze(-1)
    depth_unvalid = depth_ref<0
    depth_ref[depth_unvalid] = 5
    depth_out = depth_ref[..., 0]
    normals_out = depth_2_normal(depth_ref, depth_unvalid.squeeze(-1), cameras)[0]
    return normals_out.cpu().numpy()