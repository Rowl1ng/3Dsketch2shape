import trimesh
import numpy as np
import igl
import os
from igl import signed_distance
from skimage import measure
import sys

import meshplot as mp # mesh display 
mp.offline()

# Variables
visualize = True

# Change the following path to data_dir of dataset:
data_dir = './data/watertight_obj/'
# Mesh name:
meshname = '1a74a83fa6d24b3cacd67ce2c72c02e.obj'


# Output folder:
output_dir = './recentered_meshes'
os.makedirs(output_dir, exist_ok=True)


# Keep only vertices and faces information in the obj file:
v, f = igl.read_triangle_mesh(os.path.join(data_dir, meshname))
igl.write_triangle_mesh(os.path.join(data_dir, meshname), v, f)


# # Bounding box check:
# m = v.min(axis=0)
# M = v.max(axis=0)
# print("min = [%f,%f,%f]" % (m[0],m[1],m[2]) )
# print("max = [%f,%f,%f]" % (M[0],M[1],M[2]) )


# Load mesh:
mesh = trimesh.load(os.path.join(data_dir, meshname))


# Resize mesh and center it:
scene = trimesh.Scene(mesh)
# print("scene.scale = " + str(scene.scale))
#scene = scene.scaled(1.0/scene.scale)
# print("scene.scale = " + str(scene.scale))
mesh = scene.geometry[meshname]
# print(mesh.centroid)
mesh.rezero()
mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)    

# Get mesh vertices and faces:
v = mesh.vertices
f = mesh.faces

# Might want to save recentered mesh:
outmesh_path = os.path.join(output_dir, meshname)
ret = igl.write_triangle_mesh(outmesh_path, v, f)



# Sample points on the surface of a mesh:
count = 250000
variance_1 = 0.0025/2.0 
variance_2 = 0.00024/2.0
sdt1 = np.sqrt(variance_1) #~0.035
sdt2 = np.sqrt(variance_2) #~0.012

samples, face_index = trimesh.sample.sample_surface(mesh, count, face_weight=None)


# Disturb the points around the surface:
samples_noisy = []
for s in samples:
    noise  = np.random.normal(0.0, sdt1, 3)
    samples_noisy.append(s + noise)    
    noise  = np.random.normal(0.0, sdt2, 3)
    samples_noisy.append(s + noise)
samples_noisy_np = np.array(samples_noisy)  # transformed to a numpy array

# Sample points on a sphere:
count = 25000
samples_free_space = trimesh.sample.volume_rectangular([[1.0, 1.0, 1.0]], count, transform=None)


# Visualization 
if visualize:
    p = mp.plot(v, f)

    m = np.min(v, axis=0)
    ma = np.max(v, axis=0)

    print(m)
    print(ma)

    p = mp.plot(v, f, return_plot=True)

    p.add_points(samples_noisy_np, shading={"point_color": "red"})
    p.add_points(samples_free_space, shading={"point_color": "green"})

    p.save("test.html")


samples = np.concatenate((samples_noisy, samples_free_space), axis=0)

# Calculate SDF using IGL package:
sdf = signed_distance(samples, v, f)[0][:, np.newaxis]
