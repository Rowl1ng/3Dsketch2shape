#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

# import deep_sdf.workspace as ws

SDF_DATA_DIR = os.getenv('SDF_DATA_DIR')
DATA_DIR = os.getenv('DATA_DIR')
def get_instance_filenames(name_list):
    npzfiles = []
    for instance_name in name_list:
        # instance_filename = os.path.join(
        #     dataset, class_name, instance_name + ".npz"
        # )
        instance_filename = os.path.join(
            SDF_DATA_DIR, instance_name + ".npz"
        )
        # if not os.path.isfile(
        #     os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
        # ):
        # if not os.path.isfile(instance_filename):
        #     # raise RuntimeError(
        #     #     'Requested non-existent file "' + instance_filename + "'"
        #     # )
        #     logging.warning(
        #         "Requested non-existent file '{}'".format(instance_filename)
        #     )
        npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"].astype(float)))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"].astype(float)))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0).float()

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0).float()

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        subsample,
        load_ram=False,
        list_file='sdf_{}.txt',
        debug=False
    ):
        super(SDFSamples, self).__init__()

        self.subsample = subsample

        name_list_path = os.path.join(DATA_DIR, 'split', list_file.format(split))
        self.name_list = [line.rstrip() for line in open(name_list_path)]
        if debug:
            self.name_list = self.name_list[:200]
        self.npyfiles = get_instance_filenames(self.name_list)

        # PC data
        name_list_path = os.path.join(DATA_DIR, 'split/all.txt')
        self.all_pc_names = [line.rstrip() for line in open(name_list_path)]

        self.shape_pcs = np.load(os.path.join(DATA_DIR, 'datasets/shape/pc_4096.npy'))
        # 'voxel':
        import h5py
        shape_data_path = os.path.join(DATA_DIR, 'datasets/all_vox256_img/chair_vox256.hdf5')
        name_list_path = os.path.join(DATA_DIR, 'datasets/all_vox256_img/all_chair.txt')
        if os.path.exists(shape_data_path):
            with h5py.File(shape_data_path, 'r') as f:
                self.shape_voxels = f['voxels'][:].astype(np.float32)
            self.shape_voxels = np.reshape(self.shape_voxels, [-1, 1, self.shape_voxels.shape[1], self.shape_voxels.shape[2],
                                                self.shape_voxels.shape[3]])


        self.all_voxel_names = [line.rstrip() for line in open(name_list_path)]

        print(
            "using "
            + str(len(self.npyfiles))
            + " shapes "
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(SDF_DATA_DIR, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        sdf_path = self.npyfiles[idx]
        filename = self.name_list[idx]
        shape_index = self.all_pc_names.index(filename)
        shape_pc = self.shape_pcs[shape_index]
        shape_index = self.all_voxel_names.index(filename)
        shape_voxel = self.shape_voxels[shape_index]
        if self.load_ram:
            return (
                shape_pc, shape_voxel, unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return shape_pc, shape_voxel, unpack_sdf_samples(sdf_path, self.subsample), idx


class Sketch_SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        subsample,
        sample_extra=0,
        load_ram=False,
        debug=False
    ):
        super(Sketch_SDFSamples, self).__init__()
        self.split = split
        self.subsample = subsample
        self.sample_extra = sample_extra
        name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}_sketch.txt')
        self.name_list = [line.rstrip() for line in open(name_list_path)]
        if debug:
            self.name_list = self.name_list[:202]

        name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}.txt')
        self.sdf_name_list = [line.rstrip() for line in open(name_list_path)]

        self.npyfiles = get_instance_filenames(self.sdf_name_list)
        name_list_path = os.path.join(DATA_DIR, 'split/all.txt')
        self.all_pc_names = [line.rstrip() for line in open(name_list_path)]
        self.shape_pc = np.load(os.path.join(DATA_DIR, 'datasets/shape/shape_pc_4096.npy')).astype('float32')
        self.sketch_pc = np.load(os.path.join(DATA_DIR, f'datasets/sketch/sketch_pc_4096_{split}.npy')).astype('float32')

        self.extra_shape_list = [item for item in self.sdf_name_list if item not in self.name_list]
        print(
            "using "
            + str(len(self.name_list))
            + " shapes "
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(SDF_DATA_DIR, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        # filename = self.npyfiles[idx]
        instance_name = self.name_list[idx]
        sketch_pc = self.sketch_pc[idx]
        # Sample extra shape from extra shape list
        shape_list = [instance_name]
        if len(self.extra_shape_list) > self.sample_extra:
            extra_shape = random.sample(self.extra_shape_list, self.sample_extra)
            shape_list.extend(extra_shape)
        shape_index = [self.all_pc_names.index(instance_name) for instance_name in shape_list]
        shape_pc = self.shape_pc[shape_index]
        sdf_list = shape_list
        filenames = [os.path.join(SDF_DATA_DIR, instance_name + ".npz") for instance_name in sdf_list]

        # get convex hull SDF points
        hull_point = np.load(os.path.join(DATA_DIR, 'datasets/sketch_SDF', instance_name +'.npy')).astype('float32')
        pc_data = np.concatenate((np.expand_dims(sketch_pc, axis=0), shape_pc), axis=0)
        if self.load_ram:
            return (
                pc_data, unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx
            )
        else:
            sdfs = [unpack_sdf_samples(filename, self.subsample) for filename in filenames]
            return pc_data, torch.stack(sdfs, 0), idx, np.array([self.sdf_name_list.index(instance_name) for instance_name in shape_list]), hull_point


class SketchSDF(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)

        # Get point cloud
        pc_4096 = np.load('/vol/research/datasets/still/adobe-wang/i3d/shape_pc_4096.npy') # 6778 x 4096 x 3
        all_files_names = open('/vol/research/datasets/still/adobe-wang/i3d/all.txt', 'r').read().split('\n')[:-1] # 6778 x 1
        self.all_point_set = {}
        for idx in range(len(all_files_names)):
            self.all_point_set[all_files_names[idx]] = pc_4096[idx]

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        try:
            mesh_name = self.npyfiles[idx].split(".npz")[0]
            mesh_name = os.path.split(mesh_name)[-1]

            # fetch sdf samples
            sdf_filename = os.path.join(
                self.data_source, self.npyfiles[idx]
            )
            sdf_samples = unpack_sdf_samples(sdf_filename,  self.subsample)

            point_set = self.all_point_set[mesh_name] # 4096 x 3
            point_set = torch.from_numpy(point_set).transpose(1, 0).type(torch.FloatTensor) # 3 x 4096

            return sdf_samples, point_set, mesh_name, idx

        except:
            print ('skipping...', mesh_name)
            return self.__getitem__((idx+1)%self.__len__())
        
        
from torchvision import transforms
from PIL import Image
class MultiviewImgDataset(torch.utils.data.Dataset):
    def __init__(self, 
                split, 
                subsample,
                 data_type = 'sketch_amateur',
                 num_views=12, 
                 view=1,
                debug=False
                ):

        self.subsample = subsample

        self.shape_id = []
        name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}_sketch.txt')
        self.name_list = [line.rstrip() for line in open(name_list_path)]

        if debug:
            self.name_list = self.name_list[:202]

        name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}.txt')
        self.sdf_name_list = [line.rstrip() for line in open(name_list_path)]

        name_list_path = os.path.join(DATA_DIR, 'split/all.txt')
        self.all_pc_names = [line.rstrip() for line in open(name_list_path)]
        self.shape_pc = np.load(os.path.join(DATA_DIR, 'datasets/shape/shape_pc_4096.npy')).astype('float32')

        self.file_paths = []

        all_files = []
        # if 'shape' in data_type:
        #     self.num_views = num_views
        #     if self.num_views == 12:
        #         for line in self.name_list:
        #             shape_paths = [(os.path.join(DATA_DIR, 'view_based', data_type, line + '_{}.png')).format(i) for i in range(12)]
        #             all_files.extend(shape_paths)  # Edge
        #     else:
        #         for line in self.name_list:
        #             shape_paths = [(os.path.join(data_dir, 'view_based', data_type, '{}_{}.png'.format(line, view)))]
        #             all_files.extend(shape_paths)  # Edge
        #     self.shape_id.append(line)
        # elif 'sketch' in data_type:
        if num_views == 3:
            self.num_views = 3
            for line in self.name_list:
                sketch_paths = [
                    os.path.join(DATA_DIR, data_type, '{}_{}.png'.format(line, view)) for i in range(3)]

                all_files.extend(sketch_paths)
        else:
            self.num_views = 1
            for line in self.name_list:
                sketch_paths = [(os.path.join(DATA_DIR, data_type, '{}_{}.png'.format(line, view)))]

                all_files.extend(sketch_paths)
        self.shape_id.append(line)

        ## Select subset for different number of views
        self.file_paths = all_files

        print('The size of data is %d' % (len(self.name_list)))
        if split in ['test']:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return int(len(self.file_paths) / self.num_views)

    def __getitem__(self, idx):
        instance_name = self.name_list[idx]
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.file_paths[idx * self.num_views + i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        shape_index = self.all_pc_names.index(instance_name)
        shape_pc = self.shape_pc[shape_index]
        sdf = unpack_sdf_samples(os.path.join(SDF_DATA_DIR, instance_name + ".npz"), self.subsample)
        # return (class_id, torch.stack(imgs), self.file_paths[idx * self.num_views:(idx + 1) * self.num_views])
        return (torch.stack(imgs), shape_pc, sdf, self.sdf_name_list.index(instance_name))
    
class PCSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        debug=False
    ):
        super(PCSamples, self).__init__()
        name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}.txt')
        self.name_list = [line.rstrip() for line in open(name_list_path)]
        if debug:
            self.name_list = self.name_list[:100]

        name_list_path = os.path.join(DATA_DIR, 'split/all.txt')
        self.all_pc_names = [line.rstrip() for line in open(name_list_path)]
        self.shape_pc = np.load(os.path.join(DATA_DIR, 'datasets/shape/pc_4096.npy'))

        print(
            "using "
            + str(len(self.name_list))
            + " shapes "
        )
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        shape_index = self.all_pc_names.index(self.name_list[idx])
        shape_pc = self.shape_pc[shape_index]

        index_2 = np.random.randint(self.__len__())
        shape_index2 = self.all_pc_names.index(self.name_list[index_2])
        shape_pc2 = self.shape_pc[shape_index2]

        return shape_pc, shape_pc2, idx

class shape_samples(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        debug=False
    ):
        super(shape_samples, self).__init__()
        name_list_path = os.path.join(DATA_DIR, f'split/{split}_shape.txt')
        self.name_list = [line.rstrip() for line in open(name_list_path)]
        if debug:
            self.name_list = self.name_list[:100]

        name_list_path = os.path.join(DATA_DIR, 'split/all.txt')
        self.all_pc_names = [line.rstrip() for line in open(name_list_path)]
        self.shape_pc = np.load(os.path.join(DATA_DIR, 'datasets/shape/shape_pc_4096.npy')).astype('float32')

        print(
            "using "
            + str(len(self.name_list))
            + " shapes "
        )
    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, idx):
        shape_index = self.all_pc_names.index(self.name_list[idx])
        shape_pc = self.shape_pc[shape_index]
        return shape_pc, idx 

from utils.pc_utils import normalize_to_box, farthest_point_sample

class sketch_samples(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        dataset='original',
        debug=False
    ):
        super(sketch_samples, self).__init__()
        self.dataset = dataset
        if dataset in ['original', 'FVRS_M']:
            name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}_sketch.txt')
            self.name_list = [line.rstrip() for line in open(name_list_path)]
            name_list_path = os.path.join(DATA_DIR, 'split/all.txt')
            self.all_pc_names = [line.rstrip() for line in open(name_list_path)]
            self.shape_pc = np.load(os.path.join(DATA_DIR, 'datasets/shape/shape_pc_4096.npy')).astype('float32')
            if dataset == 'original':
                self.sketch_pc = np.load(os.path.join(DATA_DIR, f'datasets/sketch/sketch_pc_4096_{split}.npy')).astype('float32')
            elif dataset == 'FVRS_M':
                self.sketch_pc = np.load(os.path.join(DATA_DIR, f'datasets/sketch/sketch_pc_4096_{split}_M.npy')).astype('float32')
            else:
                NotImplementedError
        elif dataset in ['human_sketch', 'network']:
            name_list_path = os.path.join(DATA_DIR, f'datasets/other_sketch_types/human_{split}.txt')
            self.name_list = [line.rstrip().split(' ')[0] for line in open(name_list_path)]
            # self.sketch_pc = np.load(os.path.join(DATA_DIR, f'datasets/other_sketch_types/human_{split}.npy')).astype('float32')
            self.sketch_dir = os.path.join(DATA_DIR, f'datasets/other_sketch_types/{dataset}')
            self.shape_dir = os.path.join(DATA_DIR, 'datasets/other_sketch_types/shape')
        else:
            NotImplementedError
        if debug:
            self.name_list = self.name_list[:100]


        print(
            "using "
            + str(len(self.name_list))
            + " shapes "
        )
    def __len__(self):
        return len(self.name_list)

    def get_pc(self, pc_path):
        # pc_path = os.path.join(dir, item)
        pc_array = np.loadtxt(pc_path, delimiter=",").astype('float32')
        np.random.seed(0)
        pc_array = farthest_point_sample(pc_array, 4096)
        pc_array = normalize_to_box(pc_array)[0]*0.4
        return pc_array
    
    def __getitem__(self, idx):
        # shape_index = self.all_pc_names.index(self.name_list[idx])
        if self.dataset in ['orignal', 'FVRS_M']:
            sketch_pc = self.sketch_pc[idx]
            shape_index = self.all_pc_names.index(self.name_list[idx])
            shape_pc = self.shape_pc[shape_index]
        else:
            item = self.name_list[idx]
            if self.dataset == 'human_sketch':
                sketch_path = os.path.join(self.sketch_dir, item + '.txt')
            elif self.dataset == 'network':
                sketch_path = os.path.join(self.sketch_dir, item + '_opt_quad_network_20_aggredated.txt')

            sketch_pc = self.get_pc(sketch_path)
            shape_pc = self.get_pc(os.path.join(self.shape_dir, item + '_opt.txt'))
        shape_pc = np.concatenate((np.expand_dims(sketch_pc, axis=0), np.expand_dims(shape_pc, axis=0)), axis=0)
        return shape_pc, idx 
    
def make_dataset(split='test'):
    from utils.pc_utils import farthest_point_sample, pc_normalize
    sketch_path = f'/vol/vssp/datasets/multiview/SDF_ShapeNet/BP-Net/data/all_vox256_img/sketch/sketch_with_shape_{split}.txt'
    sketch_name_list = [line.rstrip() for line in open(sketch_path)]
    # vox_list = []
    # for item in sketch_name_list:
    #     index = shape_name_list.index('03001627/' + item)
    #     vox_list.append(index)


    npy_dir = '/vol/vssp/datasets/multiview/3VS/datasets/SketchyVR/aligned_sketch'

    npy_array = []
    for name in sketch_name_list:
        npy_file = np.load(f'{npy_dir}/{name}.npy')
        npy_file = farthest_point_sample(npy_file, 4096)
        # new_pc = rotate_point_cloud(torch.tensor(npy_file), dim='y', angle=180)
        # pc, center, scale = normalize_to_box(new_pc)
        points = pc_normalize(npy_file)
        npy_array.append(np.array(points))

    # data_voxels = data_dict['voxels'][:] #(8762, 64, 64, 64, 1)
    # new_vox = data_voxels[vox_list] #(180, 64, 64, 64, 1)



    out_h5_file = f'/scratch/data/Neural-Template/datasets/sketch/sketch_4096_{split}.hdf5'
    with h5py.File(out_h5_file, 'w') as f:
        f.create_dataset('pcs', data=np.array(npy_array), compression="gzip")
        # f.create_dataset('shape_pcs', data=shape_pcs, compression="gzip")


if __name__ == "__main__":
    from utils.pc_utils import vis_pc
    import utils.provider as provider
    split = 'train'
    train_samples = Sketch_SDFSamples(split, 4096, sample_extra=1, load_ram=False, debug=False)
    pc_data, sdf, index = train_samples[200]
    sketch_pc = pc_data[0]
    shape_pc =  torch.tensor(pc_data[1]).unsqueeze(0)
    sdf_pc = sdf[:1, :, :3]

    extra_shape_pc = pc_data[2]
    pc_to_scale = torch.cat([shape_pc, sdf_pc], dim=1)
    new_pc, _ = provider.apply_random_scale_xyz(pc_to_scale, scale=[0.5, 2.])
    new_shape_pc, new_sdf_pc = torch.split(new_pc, [4096, 4096], dim = 1)

    colums = 4
    import matplotlib.pyplot as plt
    f = plt.figure()
    f.add_subplot(1,colums, 1)
    plt.imshow(vis_pc(shape_pc[0]))
    f.add_subplot(1,colums, 2)
    plt.imshow(vis_pc(new_shape_pc[0]))
    f.add_subplot(1,colums, 3)
    plt.imshow(vis_pc(sdf_pc[0]))
    f.add_subplot(1,colums, 4)
    plt.imshow(vis_pc(new_sdf_pc[0]))

    plt.title(index)
    plt.savefig(f'{split}_{index}.png')
    quit()