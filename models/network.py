
import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from models.encoder.dgcnn import DGCNN
from models.encoder.cnn_3d import CNN3D, CNN3DDouble
from models.encoder.image import ImageEncoder
from models.encoder.sketch import SketchEncoder
from models.decoder.flow import FlowDecoder
from models.decoder.sdf import SDFDecoder
from models.decoder.bsp import BSPDecoder
from utils.ply_utils import triangulate_mesh_with_subdivide
from typing import Union
import mcubes
import math
from utils.other_utils import get_mesh_watertight, write_ply_polygon

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SketchAutoEncoder(nn.Module):
    ## init
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sketch_encoder = SketchEncoder(config = self.config)
        self.auto_encoder = None


    def set_autoencoder(self, network):
        self.auto_encoder = network

class ImageAutoEncoder(nn.Module):
    ## init
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config = self.config)
        self.auto_encoder = None


    def set_autoencoder(self, network):
        self.auto_encoder = network



class AutoEncoder(nn.Module):
    ## init
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.encoder_type == 'DGCNN':
            self.encoder = DGCNN(config=config)
        elif config.encoder_type == '3DCNN':
            if hasattr(self.config, 'use_double_encoder') and self.config.use_double_encoder:
                self.encoder = CNN3DDouble(config=config)
            else:
                self.encoder = CNN3D(config=config)

        elif config.encoder_type == 'Image':
            self.encoder = ImageEncoder(config=config)
        elif config.encoder_type == 'PN++':
            self.encoder = SketchEncoder(config=config)
        else:
            raise Exception("Encoder type not found!")

        if config.decoder_type == 'Flow':
            self.decoder = FlowDecoder(config=config)
        elif config.decoder_type == 'SDF':
            self.decoder = SDFDecoder(config=config)
        else:
            raise Exception("Decoder type not found!")

    def forward(self, inputs, coordinates_inputs):
        embedding = self.encoder(inputs)
        results = self.decoder(embedding, coordinates_inputs)
        return results


    def create_coordinates(self, resolution, space_range):
        dimensions_samples = np.linspace(space_range[0], space_range[1], resolution)
        x, y, z = np.meshgrid(dimensions_samples, dimensions_samples, dimensions_samples)
        x, y, z = x[:, :, :, np.newaxis], y[:, :, :, np.newaxis], z[:, :, :, np.newaxis]
        coordinates = np.concatenate((x, y, z), axis=3)
        coordinates = coordinates.reshape((-1, 3))
        coordinates = torch.from_numpy(coordinates).float().unsqueeze(0).cuda(device)
        return coordinates


    def save_bsp_deform(self, inputs: torch.Tensor, file_path: Union[None, str],
                        resolution: int = 16, max_batch=100000, space_range=(-1, 1), thershold_1=0.01, thershold_2=0.01, save_output=True, embedding=None):

        assert (self.config.decoder_type == 'Flow' or self.config.decoder_type == 'MVP') and self.config.flow_use_bsp_field

        ## build the coordinates
        coordinates = self.create_coordinates(resolution, space_range)

        ## convex weigth
        convex_layer_weights = self.decoder.bsp_field.convex_layer_weights.detach().cpu().numpy()

        ## get plane
        if embedding is None:
            inputs = inputs.unsqueeze(0)
            embedding = self.encoder(inputs)

        vertices, polygons, vertices_deformed, polygons_deformed, vertices_convex, bsp_convex_list = self.generate_deform_bsp(
            convex_layer_weights, coordinates, embedding, file_path, max_batch,
            resolution, thershold_1, thershold_2, save_output=save_output)

        return vertices, polygons, vertices_deformed, polygons_deformed, embedding, vertices_convex, bsp_convex_list

    def extract_prediction(self, embedding, coordinates, max_batch):

        coordinates = coordinates.unsqueeze(0)
        batch_num = int(np.ceil(coordinates.shape[1] / max_batch))

        results = []
        for i in range(batch_num):
            coordinates_inputs = coordinates[:, i * max_batch:(i + 1) * max_batch]
            result = self.decoder(embedding, coordinates_inputs)[1][0].detach().cpu().numpy()  ## for flow only
            results.append(result)

        if len(results) == 1:
            return results[0]
        else:
            return np.concatenate(tuple(results), axis=0)

    def generate_deform_bsp(self, convex_layer_weights, coordinates, embedding, file_path, max_batch,
                            resolution, thershold_1, thershold_2,
                            save_output=True):

        if hasattr(self.config, 'flow_use_split_dim') and self.config.flow_use_split_dim:
            embedding_1 = embedding[:, :self.config.decoder_input_embbeding_size]
            embedding_2 = embedding[:, self.config.decoder_input_embbeding_size:]
        else:
            embedding_1 = embedding
            embedding_2 = embedding

        bsp_convex_list = self.extract_bsp_convex(convex_layer_weights, coordinates, embedding, max_batch, resolution,
                                                  thershold_1, thershold_2)

        vertices, polygons, vertices_convex, polygons_convex = get_mesh_watertight(bsp_convex_list)

        vertices = np.array(vertices)
        vertices, polygons = triangulate_mesh_with_subdivide(vertices, polygons)

        vertices_result = self.deform_vertices(embedding_1, max_batch, vertices)

        if save_output:
            write_ply_polygon(file_path[:-4] + '_deformed.ply', vertices_result, polygons)
            write_ply_polygon(file_path[:-4] + '_original.ply', vertices, polygons)


        return vertices, polygons, vertices_result, polygons, vertices_convex, bsp_convex_list

    def extract_bsp_convex(self, convex_layer_weights, coordinates, embedding, max_batch, resolution, thershold_1,
                           thershold_2):

        if hasattr(self.config, 'flow_use_split_dim') and self.config.flow_use_split_dim:
            embedding_1 = embedding[:, :self.config.decoder_input_embbeding_size]
            embedding_2 = embedding[:, self.config.decoder_input_embbeding_size:]
        else:
            embedding_1 = embedding
            embedding_2 = embedding

        ## plane
        plane_parms = self.decoder.bsp_field.plane_encoder(embedding_2).cpu().detach().numpy()
        convex_predictions = []
        c_dim = self.decoder.bsp_field.c_dim
        for i in range(coordinates.size(1) // max_batch + 1):
            result = self.decoder(embedding, coordinates[:, i * max_batch:(i + 1) * max_batch])
            result = result[0]

            convex_prediction = result.squeeze(0).detach().cpu().numpy()
            convex_predictions.append(convex_prediction)
        if len(convex_predictions) > 1:
            convex_predictions = np.concatenate(tuple(convex_predictions), axis=0)
        else:
            convex_predictions = convex_predictions[0]
        convex_predictions = np.abs(convex_predictions.reshape((resolution, resolution, resolution, c_dim)))
        convex_predictions_float = convex_predictions < thershold_1
        convex_predictions_sum = np.sum(convex_predictions_float, axis=3)
        bsp_convex_list = []
        p_dim = self.decoder.bsp_field.p_dim
        cnt = 0
        for i in range(c_dim):
            slice_i = convex_predictions_float[:, :, :, i]
            if np.max(slice_i) > 0:  # if one voxel is inside a convex
                if np.min(
                        convex_predictions_sum - slice_i * 2) >= 0:  # if this convex is redundant, i.e. the convex is inside the shape
                    convex_predictions_sum = convex_predictions_sum - slice_i
                else:
                    box = []
                    for j in range(p_dim):
                        if convex_layer_weights[j, i] > thershold_2:
                            a = -plane_parms[0, 0, j]
                            b = -plane_parms[0, 1, j]
                            c = -plane_parms[0, 2, j]
                            d = -plane_parms[0, 3, j]
                            box.append([a, b, c, d])
                    if len(box) > 0:
                        bsp_convex_list.append(np.array(box, np.float32))

                cnt += 1
            print(f"{i} done! ")
        print(f'with {len(bsp_convex_list)} convex and enter to function {cnt}')
        return bsp_convex_list

    def deform_vertices(self, embedding, max_batch, vertices, terminate_time = None):
        ### deform the vertices
        vertices_torch = torch.from_numpy(np.array(vertices)).float().to(device).unsqueeze(0)
        vertices_result = []
        for i in range(int(np.ceil(vertices_torch.size(1) / max_batch))):
            result = self.decoder.reverse_flow(embedding, vertices_torch[:, i * max_batch:(i + 1) * max_batch], terminate_time = terminate_time)
            deformed_vertices = result.squeeze(0).detach().cpu().numpy()
            vertices_result.append(deformed_vertices)
        vertices_result = np.concatenate(vertices_result, axis=0)
        return vertices_result

    def undeform_vertices(self, embedding, max_batch, vertices, terminate_time = None):
        ### deform the vertices
        vertices_torch = torch.from_numpy(np.array(vertices)).float().to(device).unsqueeze(0)
        vertices_result = []
        for i in range(int(np.ceil(vertices_torch.size(1) / max_batch))):
            result = self.decoder.forward_flow(embedding, vertices_torch[:, i * max_batch:(i + 1) * max_batch], terminate_time = terminate_time)
            undeformed_vertices = result.squeeze(0).detach().cpu().numpy()
            vertices_result.append(undeformed_vertices)
        vertices_result = np.concatenate(vertices_result, axis=0)
        return vertices_result

class SubspaceLayer(nn.Module):
    def __init__(self, dim, n_basis):
        super(SubspaceLayer, self).__init__()

        self.U = nn.Parameter(torch.empty(n_basis, dim))    # (6,96)
        nn.init.orthogonal_(self.U)
        self.L = nn.Parameter(torch.FloatTensor([3 * i for i in range(n_basis, 0, -1)]))    # (6)
        self.mu = nn.Parameter(torch.zeros(dim))    # (96)

    def forward(self, z):
        return (self.L * z) @ self.U + self.mu

class EigenBlock_s(nn.Module):
    def __init__(
        self,
        in_dim,
        in_channels,
        n_basis
    ):
        super().__init__()

        self.convFeat1 = nn.Linear(in_dim, 128, 1)
        self.convFeat2 = nn.Linear(128, n_basis, 1)
        self.projection = SubspaceLayer(dim=in_channels, n_basis=n_basis)
        # self.subspace_conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        z = self.convFeat1(z)
        z = self.convFeat2(z)
        h = self.projection(z)#.view(h.shape)
        # h = h + z * self.gamma
        h = torch.sigmoid(h)

        return z, h

class EigenBlock_t(nn.Module):
    def __init__(
        self,
        in_dim,
        in_channels,
        n_basis
    ):
        super().__init__()

        self.convFeat1 = nn.Linear(in_dim, 128, 1)
        self.convFeat2 = nn.Linear(128, n_basis, 1)
        self.projection = SubspaceLayer(dim=in_channels, n_basis=n_basis)
        # self.subspace_conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat):
        z = self.convFeat1(feat)
        z = self.convFeat2(z)
        h = self.projection(z)#.view(h.shape)
        # h = h + z * self.gamma
        h = feat + torch.sigmoid(h)

        return z, h

class SDFAutoEncoder(AutoEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        from models.encoder.pointnet2_cls_msg import get_model
        self.encoder = get_model()
        self.decoder = SDFDecoder(config=config)

    def forward(self, pc_data, xyz):
        lat_vecs = self.encoder(pc_data)
        # lat_vecs_batch = lat_vecs.repeat(self.config.SamplesPerScene, 1)
        lat_vecs_batch = torch.repeat_interleave(lat_vecs, self.config.SamplesPerScene, dim = 0)
        input = torch.cat([lat_vecs_batch, xyz], dim=1)
        pred_sdf = self.decoder(input)
        return pred_sdf, lat_vecs
    
class ShapeSketchAutoEncoder(AutoEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = SketchEncoder(config=config)

        if config.decoder_type == 'Flow':
            self.decoder = FlowDecoder(config=config)
        elif config.decoder_type == 'SDF':
            self.decoder = SDFDecoder(config=config)
        else:
            raise Exception("Decoder type not found!")
        
        if hasattr(config, 'handle_network') :
            if config.handle_network == 'Eigen':
                # self.convFeat = nn.Linear(2 * self.config.decoder_input_embbeding_size, 128, 1)
                if self.config.eigen_s:
                    self.eigen_net_s = EigenBlock_s(in_dim=2 * self.config.decoder_input_embbeding_size, in_channels=self.config.decoder_input_embbeding_size, n_basis=config.basis_s)
                if self.config.eigen_t:
                    self.eigen_net_t = EigenBlock_t(in_dim=self.config.decoder_input_embbeding_size, in_channels=self.config.decoder_input_embbeding_size, n_basis=config.basis_t)
            elif config.handle_network == 'VAE':
                from models.encoder.vae import VAE
                self.vae_net = VAE(in_dim=self.config.decoder_input_embbeding_size, latent_dim=self.config.decoder_input_embbeding_size)
            elif config.handle_network == 'flow':
                from models.encoder.flow import get_flow
                self.flow = get_flow(style_dim=self.config.decoder_input_embbeding_size, width=512, depth=4, condition_size=self.config.decoder_input_embbeding_size)


        self.encode_func = self.config.encode_func if hasattr(self.config, 'encode_func') else 'encode'

    def encode(self, sketch_points, shape_points, is_training):
        #extract handle for z_s and z_t and use ahndle as embedding
        inputs = torch.cat([sketch_points, shape_points])
        embedding = self.encoder(inputs, is_training=is_training)
        if self.config.handle_network == 'Eigen':
            # z_s = embedding[:, :self.config.decoder_input_embbeding_size]
            # z_s_new = self.eigen_net(z_s.clone())
            # embedding[:, :self.config.decoder_input_embbeding_size] = z_s_new
            z_s = self.eigen_net_s(embedding)
            z_t = self.eigen_net_t(embedding)
            new_emb = torch.cat([z_s, z_t], dim=1) 
            return new_emb
        return embedding

    def encode_v2(self, sketch_points, shape_points, is_training):
        # only do eigenblock to z_s
        inputs = torch.cat([sketch_points, shape_points])
        embedding = self.encoder(inputs, is_training=is_training)
        if self.config.handle_network == 'Eigen':
            # z_s = embedding[:, :self.config.decoder_input_embbeding_size].clone()
            z_t = embedding[:, self.config.decoder_input_embbeding_size:].clone()
            # z_t = self.convFeat(embedding)
            # embedding[:, :self.config.decoder_input_embbeding_size] = z_s_new
            handle, z_s = self.eigen_net_s(embedding)
            new_emb = torch.cat([z_s, z_t], dim=1)
            return handle, new_emb
        return None, embedding

    def encode_v3(self, sketch_points, shape_points, is_training):
        # only do eigenblock to z_t
        inputs = torch.cat([sketch_points, shape_points])
        embedding = self.encoder(inputs, is_training=is_training)
        if self.config.handle_network == 'Eigen':
            z_s = embedding[:, :self.config.decoder_input_embbeding_size].clone()
            z_t = embedding[:, self.config.decoder_input_embbeding_size:].clone()
            # z_t = self.convFeat(embedding)
            # embedding[:, :self.config.decoder_input_embbeding_size] = z_s_new
            handle, z_t = self.eigen_net_t(z_t)
            new_emb = torch.cat([z_s, z_t], dim=1)
            return handle, new_emb
        return None, embedding

    def encode_v4(self, sketch_points, shape_points, is_training):
        # only do VAE to z_t
        inputs = torch.cat([sketch_points, shape_points])
        embedding = self.encoder(inputs, is_training=is_training)
        z_s = embedding[:, :self.config.decoder_input_embbeding_size].clone()
        z_t = embedding[:, self.config.decoder_input_embbeding_size:].clone()
        # z_t = self.convFeat(embedding)
        # embedding[:, :self.config.decoder_input_embbeding_size] = z_s_new
        z_t, mu, log_var = self.vae_net(z_t)
        new_emb = torch.cat([z_s, z_t], dim=1)
        return [mu, log_var], new_emb

    def forward(self, sketch_points, shape_points, coordinates_inputs, is_training):
        embedding = getattr(self, self.encode_func)(sketch_points, shape_points, is_training=is_training)
        coordinates_inputs = coordinates_inputs.repeat(2,1,1)
        results = self.decoder(embedding, coordinates_inputs)
        # del coordinates_inputs
        # prediction = getattr(self, self.config.forward_name)(sketch_points, shape_points, coordinates_inputs, is_training)
        return results

    def forward_v1(self, sketch_points, shape_points, coordinates_inputs, is_training):
        inputs = torch.cat([sketch_points, shape_points])
        embedding = self.encoder(inputs, is_training)
        if self.config.handle_network == 'Eigen':
            z_s = embedding[:, :self.config.decoder_input_embbeding_size].clone()
            z_s = self.eigen_net(z_s)
            embedding[:, :self.config.decoder_input_embbeding_size] = z_s
        coordinates_inputs = coordinates_inputs.repeat(2,1,1)
        results = self.decoder(embedding, coordinates_inputs)
        return results

    def forward_v2(self, sketch_points, shape_points, coordinates_inputs):
        # sketch_emb = [shape_z_s, sketch_z_t]
        inputs = torch.cat([sketch_points, shape_points])
        embedding = self.encoder(inputs)
        sketch_num, shape_num = sketch_points.shape[0], shape_points.shape[0]
        sketch_z, shape_z = torch.split(embedding, [sketch_num, shape_num])
        shape_z_s = shape_z[:, :self.config.decoder_input_embbeding_size].clone()
        if self.config.handle_network == 'Eigen':
            shape_z_s = self.eigen_net(shape_z_s)
            shape_z[:, :self.config.decoder_input_embbeding_size] = shape_z_s
        sketch_z[:, :self.config.decoder_input_embbeding_size] = shape_z_s[:sketch_num]
        embedding = torch.cat([sketch_z, shape_z])
        coordinates_inputs = coordinates_inputs.repeat(2,1,1)
        results = self.decoder(embedding, coordinates_inputs)
        return results



    def orthogonal_regularizer(self):
        reg = []
        for layer in self.modules():
            if isinstance(layer, SubspaceLayer):
                UUT = layer.U @ layer.U.t()
                reg.append(
                    ((UUT - torch.eye(UUT.shape[0], device=UUT.device)) ** 2).mean()
                )
        return sum(reg) / len(reg)

    def cov_loss(self, coef):
        B, N_basis = coef.shape
        coef = coef - coef.mean(dim=0)
        cov = torch.bmm(coef.view(B, N_basis, 1),
                        coef.view(B, 1, N_basis))
        cov = cov.sum(dim=0) / (B - 1)
        cov_loss = cov.norm(p=1, dim=(0, 1))
        return cov_loss

    def sp_loss(self, coef):
        B, N_basis = coef.shape

        reg = [coef.view(B, N_basis).norm(p=1, dim=-1).mean()]
        for layer in self.modules():
            if isinstance(layer, SubspaceLayer):
                reg.append(layer.U.view(N_basis, self.config.decoder_input_embbeding_size).norm(p=1, dim=-1).mean())
        return sum(reg) / len(reg)



if __name__ == '__main__':
    network = None