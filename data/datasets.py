import glob
import logging
from operator import index
import numpy as np
import os
import random
import torch
import torch.utils.data
import h5py
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DATA_DIR = os.getenv('DATA_DIR')

class SketchSamples(torch.utils.data.Dataset):
    def __init__(self,
                 datasets_dir: str,
                 split: str,
                 data_path: str,
                 auto_encoder=None,
                 max_batch=32,
                 sample_interval=1):
        super(SketchSamples).__init__()
        # Load sketch data
        sketch_data_path = os.path.join(datasets_dir, f'sketch/sketch_4096_{split}.hdf5')
        with h5py.File(sketch_data_path, 'r') as f:
            self.data_sketch = f['pc'][:]
        name_list = os.path.join(datasets_dir, f'sketch/sketch_index_{split}.txt')
        self.name_list = [line.rstrip() for line in open(name_list)]

        # Load Shape data
        shape_data_path = os.path.join(datasets_dir, data_path)
        if os.path.exists(shape_data_path):
            data_dict = h5py.File(shape_data_path, 'r')
        else:
            print("error: cannot load "+shape_data_path)
            exit(0)
        self.data_sketch = data_dict['sketch'][:]
        label_txt_path = data_path[:-5] + '.txt'
        self.obj_paths = [os.path.basename(line).rstrip('\n') for line in open(label_txt_path, mode='r').readlines()]

        # Load shape data
        self.data_voxels = data_dict['voxels'][:]
        self.data_voxels = np.reshape(self.data_voxels, [-1, 1, self.data_voxels.shape[1], self.data_voxels.shape[2],
                                                         self.data_voxels.shape[3]])

        # GT Latect vector
        if auto_encoder is not None:
            self.extract_latent_vector(data_voxels = self.data_voxels, auto_encoder = auto_encoder, max_batch = max_batch)

        ### interval
        self.sample_interval = sample_interval
    
    def __len__(self):
        return self.data_sketch.shape[0] // self.sample_interval

    def __getitem__(self, idx):
        idx = idx * self.sample_interval

        sketch = self.data_sketch[idx]

        if hasattr(self, 'latent_vectors'):
            latent_vector_gt = self.latent_vectors[idx]
        else:
            latent_vector_gt = None

        processed_inputs = sketch, latent_vector_gt

        return processed_inputs, idx
    def extract_latent_vector(self, data_voxels,  auto_encoder, max_batch):


        num_batch = int(np.ceil(data_voxels.shape[0] / max_batch))

        results = []
        print("start to extract GT!!!")
        with tqdm(range(num_batch), unit='batch') as tlist:
            for i in tlist:
                batched_voxels = data_voxels[i*max_batch:(i+1)*max_batch].astype(np.float32)
                batched_voxels = torch.from_numpy(batched_voxels).float().to(device)

                latent_vectors = auto_encoder.encoder(batched_voxels).detach().cpu().numpy()
                results.append(latent_vectors)

        if len(results) == 1:
            self.latent_vectors = results
        else:
            self.latent_vectors = np.concatenate(tuple(results), axis = 0)
        print("Done the extraction of GT!!!")

class MultiviewImgDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', 
                 data_type = 'sketch',
                 num_views=12, 
                 view=1):

        self.eval = self.split in ['test', 'val']
        self.shape_id = []
        name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}_sketch.txt')
        self.name_list = [line.rstrip() for line in open(name_list_path)]

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
                    os.path.join(DATA_DIR, 'view_based', data_type, '{}_{}.png'.format(line, view)) for i in range(3)]

                all_files.extend(sketch_paths)
        else:
            self.num_views = 1
            for line in self.name_list:
                sketch_paths = [(os.path.join(DATA_DIR, 'view_based', data_type, '{}_{}.png'.format(line, view)))]

                all_files.extend(sketch_paths)
        self.shape_id.append(line)

        ## Select subset for different number of views
        self.file_paths = all_files

        print('The size of %s data is %d' % (set, len(self.name_list)))
        if self.test_mode:
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
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.file_paths[idx * self.num_views + i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        # return (class_id, torch.stack(imgs), self.file_paths[idx * self.num_views:(idx + 1) * self.num_views])
        return (torch.stack(imgs), idx)
    
class ImNetImageSamples(torch.utils.data.Dataset):
    def __init__(self,
                 data_path: str,
                 auto_encoder=None,
                 max_batch=32,
                 sample_interval=1,
                 image_idx=None,
                 sample_voxel_size: int = 64,
                 ):
        super(ImNetImageSamples, self).__init__()
        data_dict = h5py.File(data_path, 'r')
        self.data_voxels = data_dict['voxels'][:]
        self.data_voxels = np.reshape(self.data_voxels, [-1, 1, self.data_voxels.shape[1], self.data_voxels.shape[2],
                                                         self.data_voxels.shape[3]])
        self.data_values = data_dict['values_' + str(sample_voxel_size)][:].astype(np.float32)
        self.data_points = (data_dict['points_' + str(sample_voxel_size)][:].astype(np.float32) + 0.5) / 256 - 0.5

        ### get file
        label_txt_path = data_path[:-5] + '.txt'
        self.obj_paths = [os.path.basename(line).rstrip('\n') for line in open(label_txt_path, mode='r').readlines()]

        ### extract the latent vector
        if auto_encoder is not None:
            self.extract_latent_vector(data_voxels = self.data_voxels, auto_encoder = auto_encoder, max_batch = max_batch)

        ### interval
        self.sample_interval = sample_interval

        ### views num
        self.view_num = 24
        self.view_size = 137
        self.crop_size = 128
        self.crop_edge = self.view_size - self.crop_size

        ### pixels
        self.crop_size = 128
        offset_x = int(self.crop_edge / 2)
        offset_y = int(self.crop_edge / 2)
        self.data_pixels = np.reshape(
            data_dict['pixels'][:, :, offset_y:offset_y + self.crop_size, offset_x:offset_x + self.crop_size],
            [-1, self.view_num, 1, self.crop_size, self.crop_size])

        self.image_idx = image_idx

    def __len__(self):
        return self.data_pixels.shape[0] // self.sample_interval

    def __getitem__(self, idx):

        idx = idx * self.sample_interval

        if self.image_idx is None:
            view_index = np.random.randint(0, self.view_num)
        else:
            view_index = self.image_idx

        image = self.data_pixels[idx, view_index].astype(np.float32) / 255.0


        if hasattr(self, 'latent_vectors'):
            latent_vector_gt = self.latent_vectors[idx]
        else:
            latent_vector_gt = None

        processed_inputs = image, latent_vector_gt
        
        return processed_inputs, idx

    def extract_latent_vector(self, data_voxels,  auto_encoder, max_batch):


        num_batch = int(np.ceil(data_voxels.shape[0] / max_batch))

        results = []
        print("start to extract GT!!!")
        with tqdm(range(num_batch), unit='batch') as tlist:
            for i in tlist:
                batched_voxels = data_voxels[i*max_batch:(i+1)*max_batch].astype(np.float32)
                batched_voxels = torch.from_numpy(batched_voxels).float().to(device)

                latent_vectors = auto_encoder.encoder(batched_voxels).detach().cpu().numpy()
                results.append(latent_vectors)

        if len(results) == 1:
            self.latent_vectors = results
        else:
            self.latent_vectors = np.concatenate(tuple(results), axis = 0)
        print("Done the extraction of GT!!!")


class ImNetSamples(torch.utils.data.Dataset):
    def __init__(self,
                 data_path: str,
                 sample_voxel_size: int,
                 interval=1):
        super(ImNetSamples, self).__init__()
        self.sample_voxel_size = sample_voxel_size
        data_dict = h5py.File(data_path, 'r')
        self.data_points = (data_dict['points_' + str(self.sample_voxel_size)][:].astype(np.float32) + 0.5) / 256 - 0.5
        self.data_values = data_dict['values_' + str(self.sample_voxel_size)][:].astype(np.float32)
        self.data_voxels = data_dict['voxels'][:]
        self.data_voxels = np.reshape(self.data_voxels, [-1, 1, self.data_voxels.shape[1], self.data_voxels.shape[2],
                                                         self.data_voxels.shape[3]])

        ### get file
        label_txt_path = data_path[:-5] + '.txt'
        self.obj_paths = [os.path.basename(line).rstrip('\n') for line in open(label_txt_path, mode='r').readlines()]

        ## interval
        self.interval = interval


    def __len__(self):
        return self.data_points.shape[0] // self.interval

    def __getitem__(self, idx):

        idx = idx * self.interval

        processed_inputs = self.data_voxels[idx].astype(np.float32), self.data_points[idx].astype(np.float32), \
                           self.data_values[idx]


        return processed_inputs, idx

class ImNetSketchSample(torch.utils.data.Dataset):
    def __init__(self,
                 datasets_dir: str,
                 phase: str,
                 sample_voxel_size: int,
                 load_latent_code=True,
                 interval=1,
                 inference=False, debug=False):
        super(ImNetSketchSample, self).__init__()
        self.sample_voxel_size = sample_voxel_size
        self.inference = inference
        self.debug = debug
        # Load sketch data
        sketch_data_path = os.path.join(datasets_dir, f'sketch/sketch_4096_{phase}.hdf5')
        with h5py.File(sketch_data_path, 'r') as f:
            self.sketch_pc = f['pcs'][:]
        name_list = os.path.join(datasets_dir, f'sketch/sketch_with_shape_{phase}.txt')
        self.name_list = [line.rstrip() for line in open(name_list)]

        # Load shape data
        self.shape_pc = np.load(os.path.join(datasets_dir, 'shape/pc_4096.npy'))
        self.sketch_to_shape_map = [int(line.rstrip()) for line in open(os.path.join(datasets_dir, f'shape/sketch_{phase}_index.txt'))]
        ## Load voxel
        shape_data_path = os.path.join(datasets_dir, f'sketch/shape_{phase}.hdf5')
        with h5py.File(shape_data_path, 'r') as f:
            self.data_points = f['points_' + str(self.sample_voxel_size)][:]
            self.data_values = f['values_' + str(self.sample_voxel_size)][:]


        # GT Latect vector
        if load_latent_code:
            self.latent_code = np.load(os.path.join(datasets_dir, f'sketch/latent_code_{phase}.npy'))

        


    def __len__(self):
        if self.debug:
            return min(100, len(self.name_list))

        return len(self.name_list) 

    def __getitem__(self, idx):
        sketch_pc = self.sketch_pc[idx]
        shape_index = self.sketch_to_shape_map[idx]
        shape_pc = self.shape_pc[shape_index]
        if self.inference:
            processed_inputs = sketch_pc, shape_pc, self.latent_code[idx]
            return processed_inputs, idx

        processed_inputs = sketch_pc, shape_pc, self.latent_code[idx], self.data_points[idx], self.data_values[idx]
        return processed_inputs, idx


class NTSamples(torch.utils.data.Dataset):
    def __init__(self,
                 split: str,
                 sample_voxel_size: int,
                 load_latent_code=True,
                 interval=1,
                 inference=False, debug=False):
        super(ImNetSketchSample, self).__init__()
        self.sample_voxel_size = sample_voxel_size
        self.inference = inference
        self.debug = debug

        name_list_path = os.path.join(DATA_DIR, f'split/sdf_{split}.txt')
        self.name_list = [line.rstrip() for line in open(name_list_path)]
        if debug:
            self.name_list = self.name_list[:100]

        # Load shape data
        self.shape_pc = np.load(os.path.join(DATA_DIR, 'shape/pc_4096.npy'))
        self.sketch_to_shape_map = [int(line.rstrip()) for line in open(os.path.join(datasets_dir, f'shape/sketch_{phase}_index.txt'))]
        ## Load voxel
        shape_data_path = os.path.join(datasets_dir, f'sketch/shape_{phase}.hdf5')
        with h5py.File(shape_data_path, 'r') as f:
            self.data_points = f['points_' + str(self.sample_voxel_size)][:]
            self.data_values = f['values_' + str(self.sample_voxel_size)][:]


        # GT Latect vector
        if load_latent_code:
            self.latent_code = np.load(os.path.join(datasets_dir, f'sketch/latent_code_{phase}.npy'))

        
    def __len__(self):
        if self.debug:
            return min(100, len(self.name_list))

        return len(self.name_list) 

    def __getitem__(self, idx):
        sketch_pc = self.sketch_pc[idx]
        shape_index = self.sketch_to_shape_map[idx]
        shape_pc = self.shape_pc[shape_index]
        if self.inference:
            processed_inputs = sketch_pc, shape_pc, self.latent_code[idx]
            return processed_inputs, idx

        processed_inputs = sketch_pc, shape_pc, self.latent_code[idx], self.data_points[idx], self.data_values[idx]
        return processed_inputs, idx



def vis(split='test', index = 1):
    sketch_path = f'/vol/vssp/datasets/multiview/SDF_ShapeNet/BP-Net/data/all_vox256_img/sketch/sketch_with_shape_{split}.txt'
    sketch_name_list = [line.rstrip() for line in open(sketch_path)]

    name_list_file = f'/vol/vssp/datasets/multiview/SDF_ShapeNet/BP-Net/data/all_vox256_img/all_vox256_img_{split}.txt'
    shape_name_list = [line.rstrip() for line in open(name_list_file)]

    #Shape_data
    data_path = f'/vol/vssp/datasets/multiview/SDF_ShapeNet/BP-Net/data/all_vox256_img/all_vox256_img_{split}.hdf5'
    data_dict = h5py.File(data_path, 'r')
    shape_images = data_dict['pixels'][:]
    #Shape pc data
    data_path = '/scratch/data/DeepMetaHandles/pc_4096.npy'
    pc_data = np.load(data_path)
    pc_index_list = [line.rstrip() for line in open(f'/scratch/data/DeepMetaHandles/sketch_{split}_index.txt')]

    name = sketch_name_list[index]
    # sketch_data
    data_path = f'/scratch/data/Neural-Template/datasets/sketch/sketch_4096_{split}.hdf5'
    data_dict = h5py.File(data_path, 'r')
    sketch_data = data_dict['pcs'][:]
    sketch_pc = sketch_data[index]
    shape_img = shape_images[shape_name_list.index('03001627/' + name)]
    shape_pc = pc_data[int(pc_index_list[index])]

    import matplotlib.pyplot as plt
    f = plt.figure()
    f.add_subplot(1,3, 1)
    plt.imshow(vis_pc(sketch_pc))
    f.add_subplot(1,3, 2)
    plt.imshow(shape_img[6], cmap='gray')
    f.add_subplot(1,3, 3)
    plt.imshow(vis_pc(shape_pc))
    plt.title(index)
    plt.savefig(f'{split}_{index}.png')
    quit()

def make_dataset(split='test'):
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

def load_model():
    import importlib
    from models.network import AutoEncoder
    from train.implicit_trainer import Trainer
    ## import config here
    resume_path = '/vol/vssp/datasets/multiview/SDF_ShapeNet/NT-Net/phase_2_model/model_epoch_2_300.pth'
    auto_encoder_config_path = '/vol/vssp/datasets/multiview/SDF_ShapeNet/NT-Net/phase_2_model/config.py'

    auto_spec = importlib.util.spec_from_file_location('*', auto_encoder_config_path)

    auto_config = importlib.util.module_from_spec(auto_spec)
    auto_spec.loader.exec_module(auto_config)

    ### Network
    auto_encoder = AutoEncoder(config=auto_config)

    network_state_dict = torch.load(resume_path)
    network_state_dict = Trainer.process_state_dict(network_state_dict, type = 1)
    auto_encoder.load_state_dict(network_state_dict)
    auto_encoder.eval()
    auto_encoder.to(device)
    return auto_encoder

def extract_latent_vector(data_voxels, save_dir, max_batch=32):
    data_voxels = np.reshape(data_voxels, [-1, 1, data_voxels.shape[1], data_voxels.shape[2],
                                                    data_voxels.shape[3]])

    auto_encoder = load_model()

    num_batch = int(np.ceil(data_voxels.shape[0] / max_batch))

    results = []
    print("start to extract GT!!!")
    with tqdm(range(num_batch), unit='batch') as tlist:
        for i in tlist:
            batched_voxels = data_voxels[i*max_batch:(i+1)*max_batch].astype(np.float32)
            batched_voxels = torch.from_numpy(batched_voxels).float().to(device)

            latent_vectors = auto_encoder.encoder(batched_voxels).detach().cpu().numpy()
            results.append(latent_vectors)

    if len(results) == 1:
        latent_vectors = results
    else:
        latent_vectors = np.concatenate(tuple(results), axis = 0)
    print("Done the extraction of GT!!!")
    save_path = os.path.join(save_dir, 'latent_code.npy')
    np.save(save_path, latent_vectors)
    print('Save latent code:', latent_vectors.shape)



def get_latent_code(split='train'):
    save_dir = '/scratch/data/Neural-Template/datasets/sketch'
    out_h5_file = f'/scratch/data/Neural-Template/datasets/sketch/shape_{split}.hdf5'
    if not os.path.exists(out_h5_file):
        data_path = f'/vol/vssp/datasets/multiview/SDF_ShapeNet/BP-Net/data/all_vox256_img/all_vox256_img_{split}.hdf5'
        data_dict = h5py.File(data_path, 'r')
        data_voxels = data_dict['voxels'][:]
        data_points = {}
        data_values = {}

        sketch_path = f'/vol/vssp/datasets/multiview/SDF_ShapeNet/BP-Net/data/all_vox256_img/sketch/sketch_with_shape_{split}.txt'
        sketch_name_list = [line.rstrip() for line in open(sketch_path)]

        name_list_file = f'/vol/vssp/datasets/multiview/SDF_ShapeNet/BP-Net/data/all_vox256_img/all_vox256_img_{split}.txt'
        shape_name_list = [line.rstrip() for line in open(name_list_file)]

        index_list = [shape_name_list.index(f'03001627/{name}') for name in sketch_name_list]

        voxels = data_voxels[index_list]
        with h5py.File(out_h5_file, 'w') as f:
            f.create_dataset('voxels', data=voxels, compression="gzip")
            for sample_voxel_size in [16, 32 ,64]:
                data_points = (data_dict['points_' + str(sample_voxel_size)][:].astype(np.float32) + 0.5) / 256 - 0.5
                f.create_dataset('points_' + str(sample_voxel_size), data=data_points[index_list], compression="gzip")
                data_values = data_dict['values_' + str(sample_voxel_size)][:].astype(np.float32)
                f.create_dataset('values_' + str(sample_voxel_size), data=data_values[index_list], compression="gzip")

    else:
        data_dict = h5py.File(out_h5_file, 'r')
        data_voxels = data_dict['voxels'][:]



    # extract_latent_vector(data_voxels, save_dir=save_dir)
if __name__ == "__main__":
    import sys
    sys.path.append('/user/HS229/ll00931/projects/Neural-Template/')

    # for mode in ['test']:
    #     make_dataset(split=mode)
    #     vis(split=mode)
    get_latent_code()
    # dataset = ImNetSketchSample(datasets_dir='/scratch/data/Neural-Template/datasets', phase='train', sample_voxel_size=32)
    # print(dataset.latent_code.shape)