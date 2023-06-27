import torch.nn as nn
import pytorch3d.utils
from utils.pc_utils import normalize_to_box, sample_farthest_points
import torch
from einops import rearrange
from models.deformer.cage import deform_with_MVC

def weights_init(m):
    """
    initialize the weighs of the network for Convolutional layers and batchnorm layers

    From https://github.com/yifita/deep_cage
    """
    if isinstance(m, (torch.nn.modules.conv._ConvNd, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        torch.nn.init.constant_(m.bias, 0.0)
        torch.nn.init.constant_(m.weight, 1.0)

class Conv1d(nn.Module):
    """
    1dconvolution with custom normalization and activation
    
    From https://github.com/yifita/deep_cage
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 activation=None, normalization=None, momentum=0.01, conv_params={}):
        super(Conv1d, self).__init__()
        self.activation = activation
        self.normalization = normalization
        bias = not normalization and bias
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias, **conv_params)

        if normalization is not None:
            if self.normalization == 'batch':
                self.norm = nn.BatchNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            elif self.normalization == 'instance':
                self.norm = nn.InstanceNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            else:
                raise ValueError(
                    "only \"batch/instance\" normalization permitted.")

        # activation
        if activation is not None:
            if self.activation == 'relu':
                self.act = nn.ReLU()
            elif self.activation == 'elu':
                self.act = nn.ELU(alpha=1.0)
            elif self.activation == 'lrelu':
                self.act = nn.LeakyReLU(0.1)
            elif self.activation == "tanh":
                self.act = nn.Tanh()
            else:
                raise ValueError("only \"relu/elu/lrelu/tanh\" implemented")

    def forward(self, x, epoch=None):
        x = self.conv(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        return x

class Linear(nn.Module):
    """
    1dconvolution with custom normalization and activation
    
    From https://github.com/yifita/deep_cage
    """

    def __init__(self, in_channels, out_channels, bias=True,
                 activation=None, normalization=None, momentum=0.01):
        super(Linear, self).__init__()
        self.activation = activation
        self.normalization = normalization
        bias = not normalization and bias
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

        if normalization is not None:
            if self.normalization == 'batch':
                self.norm = nn.BatchNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            elif self.normalization == 'instance':
                self.norm = nn.InstanceNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            else:
                raise ValueError(
                    "only \"batch/instance\" normalization permitted.")

        # activation
        if activation is not None:
            if self.activation == 'relu':
                self.act = nn.ReLU()
            elif self.activation == 'elu':
                self.act = nn.ELU(alpha=1.0)
            elif self.activation == 'lrelu':
                self.act = nn.LeakyReLU(0.1)
            elif self.activation == "tanh":
                self.act = nn.Tanh()
            else:
                raise ValueError("only \"relu/elu/lrelu/tanh\" implemented")

    def forward(self, x, epoch=None):
        x = self.linear(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        return x

class PointNetfeat(nn.Module):
    """
    From https://github.com/yifita/deep_cage
    """
    def __init__(self, dim=3, num_points=2500, global_feat=True, trans=False, bottleneck_size=512, activation="relu", normalization=None):
        super().__init__()
        self.conv1 = Conv1d(dim, 64, 1, activation=activation, normalization=normalization)
        # self.stn_embedding = STN(num_points = num_points, K=64)
        self.conv2 = Conv1d(64, 128, 1, activation=activation, normalization=normalization)
        self.conv3 = Conv1d(128, bottleneck_size, 1, activation=None, normalization=normalization)
        #self.mp1 = torch.nn.MaxPool1d(num_points)

        self.trans = trans
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = self.conv1(x)
        pointfeat = x
        x = self.conv2(x)
        x = self.conv3(x)
        x,_ = torch.max(x, dim=2)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(batchsize, -1, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

class MLPDeformer2(nn.Module):
    """
    From https://github.com/yifita/deep_cage
    """
    def __init__(self, dim, bottleneck_size, npoint, residual=True, normalization=None):
        super().__init__()
        self.npoint = npoint
        self.dim = dim
        self.residual = residual
        self.layers = nn.Sequential(
                Linear(bottleneck_size, 512, activation="lrelu", normalization=normalization),
                Linear(512, 256, activation="lrelu", normalization=normalization),
                Linear(256, npoint*dim)
            )
    def forward(self, code):
        B, _ = code.shape
        x = self.layers(code)
        x = x.reshape(B, self.dim, self.npoint)
        return x


class KP_Deformer(nn.Module):
    def __init__(
        self,
        config
    ):
        super(KP_Deformer, self).__init__()
        self.config = config
        self.cage_size = self.config.DeformerSpecs['cage_size']
        self.ico_sphere_div = self.config.DeformerSpecs['ico_sphere_div']
        self.num_points = self.config.num_points
        self.num_structure_points = self.config.DeformerSpecs['num_structure_points']
        self.feat_dim = self.config.DeformerSpecs['feat_dim']
        template_vertices, template_faces = self.create_cage()
        self.init_template(template_vertices, template_faces)
        self.init_networks()
        self.apply(weights_init)
        # self.KP_dist_fun = 

    def create_cage(self):
        # cage (1, N, 3)
        mesh = pytorch3d.utils.ico_sphere(self.ico_sphere_div, device='cuda:0')
        init_cage_V = mesh.verts_padded()
        init_cage_F = mesh.faces_padded()
        init_cage_V = self.cage_size * normalize_to_box(init_cage_V)[0]
        init_cage_V = init_cage_V.transpose(1, 2)
        return init_cage_V, init_cage_F
    
    def init_template(self, template_vertices, template_faces):
        # save template as buffer
        self.register_buffer("template_faces", template_faces)
        self.register_buffer("template_vertices", template_vertices)
        
        # n_keypoints x number of vertices
        self.influence_param = nn.Parameter(torch.zeros(self.num_structure_points, self.template_vertices.shape[2]), requires_grad=True)
   
    def init_networks(self):
        dim = 3
        num_point = self.num_points
        n_keypoints = self.num_structure_points
        bottleneck_size = self.feat_dim #256
        d_residual = True
        normalization = None
        # keypoint predictor
        shape_encoder_kpt = nn.Sequential(
            PointNetfeat(dim=dim, num_points=num_point, bottleneck_size=bottleneck_size),
            Linear(bottleneck_size, bottleneck_size, activation="lrelu", normalization=normalization))
        nd_decoder_kpt = MLPDeformer2(dim=dim, bottleneck_size=bottleneck_size, npoint=n_keypoints,
                                residual=d_residual, normalization=normalization)
        self.keypoint_predictor = nn.Sequential(shape_encoder_kpt, nd_decoder_kpt)

        # influence predictor
        influence_size = n_keypoints * self.template_vertices.shape[2]
        shape_encoder_influence = nn.Sequential(
            PointNetfeat(dim=dim, num_points=num_point, bottleneck_size=influence_size),
            Linear(influence_size, influence_size, activation="lrelu", normalization=normalization))
        dencoder_influence = nn.Sequential(
                Linear(influence_size, influence_size, activation="lrelu", normalization=normalization),
                Linear(influence_size, influence_size, activation=None, normalization=None))
        self.influence_predictor = nn.Sequential(shape_encoder_influence, dencoder_influence)

    def forward(self, source_shape, target_shape):
        source_shape = source_shape.transpose(1, 2)
        target_shape = target_shape.transpose(1, 2)
        
        """
        source_shape (B,3,N)
        target_shape (B,3,M)
        """
        B, _, _ = source_shape.shape

        if target_shape is not None:
            shape = torch.cat([source_shape, target_shape], dim=0)
        else:
            shape = source_shape
        
        keypoints = self.keypoint_predictor(shape)
        keypoints = torch.clamp(keypoints, -1.0, 1.0)
        if target_shape is not None:
            source_keypoints, target_keypoints = torch.split(keypoints, B, dim=0)
        else:
            source_keypoints = keypoints

        self.keypoints = keypoints

        n_fps = 2 * self.hparams.num_structure_points
        self.init_keypoints = sample_farthest_points(shape, n_fps)

        if target_shape is not None:
            source_init_keypoints, target_init_keypoints = torch.split(self.init_keypoints, B, dim=0)
        else:
            source_init_keypoints = self.init_keypoints
            target_init_keypoints = None

        cage = self.template_vertices
        cage = self.optimize_cage(cage, source_shape)

        outputs = {
            "cage": cage.transpose(1, 2),
            "cage_face": self.template_faces,
            "source_keypoints": source_keypoints.transpose(1, 2),
            "target_keypoints": target_keypoints.transpose(1, 2),
            'source_init_keypoints': source_init_keypoints,
            'target_init_keypoints': target_init_keypoints
        }

        self.influence = self.influence_param[None]
        self.influence_offset = self.influence_predictor(source_shape)
        self.influence_offset = rearrange(
            self.influence_offset, 'b (k c) -> b k c', k=self.influence.shape[1], c=self.influence.shape[2])
        self.influence = self.influence + self.influence_offset

        distance = torch.sum((source_keypoints[..., None] - cage[:, :, None]) ** 2, dim=1)
        n_influence = int((distance.shape[2] / distance.shape[1]) * self.hparams.n_influence_ratio)
        n_influence = max(5, n_influence)
        threshold = torch.topk(distance, n_influence, largest=False)[0][:, :, -1]
        threshold = threshold[..., None]
        keep = distance <= threshold
        influence = self.influence * keep

        base_cage = cage
        keypoints_offset = target_keypoints - source_keypoints
        cage_offset = torch.sum(keypoints_offset[..., None] * influence[:, None], dim=2)
        new_cage = base_cage + cage_offset

        cage = cage.transpose(1, 2)
        new_cage = new_cage.transpose(1, 2)
        deformed_shapes, weights, _ = deform_with_MVC(cage, new_cage, self.template_faces.expand(B, -1, -1), source_shape.transpose(1, 2), verbose=True)

        deformed_keypoints = self.keypoint_predictor(deformed_shapes.transpose(1, 2))
        outputs.update({
            "cage": cage,
            "cage_face": self.template_faces,
            "new_cage": new_cage,
            "deformed": deformed_shapes,
            "deformed_keypoints": deformed_keypoints.transpose(1, 2),
            "weight": weights,
            "influence": influence})

        return outputs

if __name__ == "__main__":
    import importlib
    from data.SDF_datasets import SDFSamples
    from utils.model_utils import load_model
    from utils.pc_utils import farthest_point_sample
    from pytorch_lightning.utilities.cloud_io import load as pl_load
    resume_path = 'configs/sdf/autoencoder_codereg_8192_sketch_fg.py'
    spec = importlib.util.spec_from_file_location('*', resume_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    source_index = 0
    target_index = 1
    deformer = KP_Deformer(config).eval().cuda()
    resume_ckpt = '/mnt/disk1/ling/SketchGen/experiments/keypoint_deformer/last.ckpt'
    # deformer, _ = load_model(deformer, None, resume_ckpt, None)
    # ckpt = pl_load(resume_ckpt, map_location=lambda storage, loc: storage)  
    deformer.load_state_dict(ckpt['state_dict'])

    train_samples = SDFSamples('train', config.SamplesPerScene, load_ram=False, debug=True)
    source = farthest_point_sample(train_samples[source_index][0], 1024)
    target = farthest_point_sample(train_samples[target_index][0], 1024)
    source_shape = torch.tensor(source).cuda().unsqueeze(0)
    target_shape = torch.tensor(target).cuda().unsqueeze(0)

    deformed = deformer(source_shape, target_shape)['deformed'] 
    from utils.pc_utils import vis_pc
    import matplotlib.pyplot as plt
    f = plt.figure()
    f.add_subplot(1,3, 1)
    plt.imshow(vis_pc(source))
    f.add_subplot(1,3, 2)
    plt.imshow(vis_pc(target))
    f.add_subplot(1,3, 3)
    plt.imshow(vis_pc(deformed.unsqueeze(0).cpu().numpy()))
    plt.title(f'deform_{source_index}_{target_index}')
    plt.savefig(f'deform_{source_index}_{target_index}.png')
    quit()