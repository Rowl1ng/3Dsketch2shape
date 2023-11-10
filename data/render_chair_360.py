import os
import warnings
import numpy as np
import bpy
from render_freestyle_svg import register
from mathutils import Vector
import sys 
import math

#register()
warnings.filterwarnings('ignore')

color_set = ['41A2FF', '8DFF41', 'D651FD', 'FF912C']
color_set = [np.array(tuple(int(h[i:i+2], 16) for i in (0, 2, 4))) / 255 for h in color_set]

def normalize(context):
    # obj = context.active_object
    obj = bpy.context.selected_objects[0]
    v = obj.data.vertices
    highest = [v[0].co[0], v[0].co[1], v[0].co[2]]
    lowest = [v[0].co[0], v[0].co[1], v[0].co[2]]

    for v in obj.data.vertices:
        c = v.co
        
        if c[0] > highest[0]:
            highest[0] = c[0]
        
        if c[0] < lowest[0]:
            lowest[0] = c[0]
        
        if c[1] > highest[1]:
            highest[1] = c[1]
        
        if c[1] < lowest[1]:
            lowest[1] = c[1]
        
        if c[2] > highest[2]:
            highest[2] = c[2]
        
        if c[2] < lowest[2]:
            lowest[2] = c[2]
    
    bpy.ops.transform.translate(
        value=(-0.5*(highest[0]+lowest[0]), -0.5*(highest[1]+lowest[1]), -0.5*(highest[2]+lowest[2])),
        orient_type='GLOBAL',
        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        orient_matrix_type='GLOBAL',
        mirror=True,
        use_proportional_edit=False,
        proportional_edit_falloff='SMOOTH',
        proportional_size=1,
        use_proportional_connected=False,
        use_proportional_projected=False
    )

    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

    size = [highest[0]-lowest[0], highest[1]-lowest[1],highest[2]-lowest[2]]

    s = 2/sorted(size)[2]

    bpy.ops.transform.resize(
        value=(s, s, s),
        orient_type='GLOBAL',
        orient_matrix=((1, 0, 0),(0, 1, 0), (0, 0, 1)),
        orient_matrix_type='GLOBAL',
        mirror=True,
        use_proportional_edit=False,
        proportional_edit_falloff='SMOOTH',
        proportional_size=1,
        use_proportional_connected=False,
        use_proportional_projected=False
    )

def look_at(obj_camera, point):
    direction = point - obj_camera.location
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def spherical_to_euclidian(elev, azimuth, r):
    x_pos = r * np.cos(elev/180.0*np.pi) * np.cos(azimuth/180.0*np.pi)
    y_pos = r * np.cos(elev/180.0*np.pi) * np.sin(azimuth/180.0*np.pi)
    z_pos = r * np.sin(elev/180.0*np.pi)
    return x_pos, y_pos, z_pos


def iterateTillInsideBounds(val, bounds, origin, mean, std, random_state):
    while val < bounds[0] or val > bounds[1] or (val > bounds[2] and val < bounds[3]):
        val = round(origin + mean + std*random_state.randn())
    return val


def find_longest_diagonal_old(imported):
    local_bbox_center = 0.125 * sum((Vector(b) for b in imported.bound_box), Vector())
    ld = 0.0
    for v in imported.bound_box:
        lv = Vector(local_bbox_center) - Vector(v)
        ld = max(ld, lv.length)
    return ld


def find_longest_diagonal(imported):

    points = np.array([Vector(b) for b in imported.bound_box])
    print('point', points)
    points = points.max(axis=0) - points.min(axis=0)

    return points.max()


def compute_longest_diagonal(mesh_path):
    #try:
    bpy.ops.import_scene.obj(filepath=mesh_path, axis_forward='-X')
    obj_object = bpy.context.selected_objects[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    ld = find_longest_diagonal(obj_object)
    return ld
    #except:
        #return 1.0
    
def fill_in_camera_positions():
    num_base_viewpoints = 360 # TODO: change to 360
    num_add_viewpoints = 0

    random_state = np.random.RandomState()


    mean_azi = 0
    std_azi = 7
    mean_elev = 0
    std_elev = 7
    mean_r = 0
    std_r = 7

    delta_azi_max = 15
    delta_elev_max = 15
    delta_azi_min = 5
    delta_elev_min = 5
    delta_r = 0.1

    # azi_origins = np.linspace(0, 359, num_base_viewpoints)
    azi_origins = np.linspace(0, 350, 36)
    elev_origin = 30 #10
    r_origin = 9 #1.5

    bound_azi = [(azi - delta_azi_max, azi + delta_azi_max, azi - delta_azi_min, azi + delta_azi_min) for azi in azi_origins]
    bound_elev = (elev_origin - delta_elev_max, elev_origin + delta_elev_max, elev_origin - delta_elev_min, elev_origin + delta_elev_min)
    bound_r = (r_origin - delta_r, r_origin + delta_r)

    azis = []
    elevs = []
    for azi in azi_origins:
        azis.append(azi)
        elevs.append(elev_origin)

    x_pos = []
    y_pos = []
    z_pos = []
    for azi, elev in zip(azis, elevs):
        x_pos_, y_pos_, z_pos_ = spherical_to_euclidian(elev, azi, r_origin)
        x_pos.append(x_pos_)
        y_pos.append(y_pos_)
        z_pos.append(z_pos_)

    return azis, elevs, x_pos, y_pos, z_pos

def render(filepath, output_dir='', color=[0,1,0]):

    # Clean the default Blender Scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Import OBJ
    if filepath[-3:] == 'obj':
        bpy.ops.import_scene.obj(filepath=filepath, axis_forward='-X')
    else:
        bpy.ops.import_mesh.ply(filepath=filepath)
        bpy.context.object.rotation_euler[0] = 1.5708
        bpy.context.object.rotation_euler[2] = -1.5708
    obj = bpy.context.selected_objects[0]
    # if filepath[-3:] == 'obj':
        # bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME') 
        # bpy.ops.object.location_clear(clear_delta=False)
        # normalize(bpy.context)
    # ld = find_longest_diagonal(obj)
    # print('ld: ', ld)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        # bpy.ops.object.matrix_world.translation = (0, 0, 0)
    # else:
    #     bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    # assert ld == 1, 'Works'
    scale = 4.0 #2.5 #4.0 / (1.3*ld) #1.0
    obj.scale = (scale, scale, scale)
    center = Vector((0.0, 0.0, 0.0))
    obj.location = center
    
    # Set color
    mat = bpy.data.materials.new("Green")
    mat.use_nodes = True
    mat_nodes = mat.node_tree.nodes
    if filepath[-3:] == 'obj':
        objs = bpy.context.selected_objects[:]
        for obj in objs:
            print(obj.location)
            for i in range(len(obj.data.materials)):
                obj.data.materials[i] = mat
    else:
        obj.data.materials.append(mat)

    # mat_nodes["Principled BSDF"].inputs["Metallic"].default_value = 1.0
    mat_nodes["Principled BSDF"].inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    # tree = 
    # 
    # bsdf = nodes["Principled BSDF"]
    # bsdf.inputs["Base Color"].default_value = (0, 1, 0, 0.8)
    # matg.diffuse_color = (0, 1, 0, 0.8)
    # bpy.data.materials['Material'].node_tree.nodes['Principled BSDF'].inputs[0].default_value = (0, 1, 0, 1)
    # bpy.data.materials['Material'].diffuse_color = (0, 1, 0, 1)

    # Add a Point Light
    bpy.ops.object.light_add(type='POINT', align='WORLD', location=(3, 0, 3))
    obj_light = bpy.data.objects['Point']
    obj_light.data.energy = 400
    
    bpy.ops.object.light_add(type='POINT', align='WORLD', location=(-3, 0, 0))
    obj_light = bpy.data.objects['Point']
    obj_light.data.energy = 400
    
    # Add a camera
    bpy.ops.object.camera_add()
    obj_camera = bpy.data.objects['Camera']
    obj_camera.data.sensor_height = 32
    obj_camera.data.sensor_width = 32
    obj_camera.data.lens = 35
    obj_camera.data.type = 'PERSP'#'ORTHO'
    obj_camera.data.lens_unit = 'FOV'
    obj_camera.data.angle = math.radians(25)

    bpy.context.scene.camera = bpy.context.object
    
    # Camera parameters
    azimuths, elevations, x_pos, y_pos, z_pos = fill_in_camera_positions()
    
    # Set the canvas
    bpy.data.scenes['Scene'].render.resolution_x = 540
    bpy.data.scenes['Scene'].render.resolution_y = 540
    bpy.data.scenes['Scene'].render.resolution_percentage = 100
    
    # Render preferences
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    
    bpy.context.scene.cycles.samples = 4
    bpy.context.scene.cycles.preview_samples = 4
    bpy.context.scene.render.tile_x = 540
    bpy.context.scene.render.tile_y = 540
    bpy.context.scene.render.threads = 1024
    bpy.context.scene.render.threads_mode = 'AUTO'
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    
    center = Vector((0.0, 0.0, 0.0))
    
    for azi, elev, x_pos_, y_pos_, z_pos_ in zip(azimuths, elevations, x_pos, y_pos, z_pos):
        # x_pos_, y_pos_, z_pos_ = spherical_to_euclidian(30, 45, 2.0)
        obj_camera.location = (x_pos_, y_pos_, z_pos_)
        look_at(obj_camera, center)
        filename = os.path.basename(filepath).split('.')[0]+'_azi{}'.format(azi)
        bpy.context.scene.render.filepath = os.path.join(output_dir, filename)
        bpy.ops.render.render(write_still = True)
    
if __name__ == '__main__':
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    # print(argv[0], argv[1], argv[2])
    model_path = argv[0]
    export_dir = argv[1]
    color_id = argv[2]
    # obj_path = '/scratch/visualization/sketch_to_shape_gen/sample/GT/models/model_normalized.obj'
    # ply_path = '/scratch/visualization/sketch_to_shape_gen/sample/generated_shape.ply'
    
    # output_dir = './'
    
    render(model_path, export_dir, color_set[int(color_id)])
