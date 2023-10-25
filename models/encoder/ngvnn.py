import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import os

CACHE_DIR = os.getenv('CACHE_DIR')

'''
VGG (
  (features): Sequential (
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    (2): ReLU (inplace)
    (3): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    (6): ReLU (inplace)
    (7): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    (10): ReLU (inplace)
    (11): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    (13): ReLU (inplace)
    (14): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (15): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (16): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (17): ReLU (inplace)
    (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (19): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (20): ReLU (inplace)
    (21): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (22): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (23): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (24): ReLU (inplace)
    (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (27): ReLU (inplace)
    (28): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  )
  (classifier): Sequential (
    (0): Linear (25088 -> 4096)
    (1): ReLU (inplace)
    (2): Dropout (p = 0.5)
    (3): Linear (4096 -> 4096)
    (4): ReLU (inplace)
    (5): Dropout (p = 0.5)
    (6): Linear (4096 -> 1000)
  )
)
'''
nclasses = 10
original_model = models.vgg11_bn(pretrained=False)


# original_model.classifier._modules['6'] = nn.Linear(4096, nclasses)
def Net_Classifier(nfea=512, nclasses=10):
    return nn.Linear(nfea, nclasses)


class Net_Prev(nn.Module):
    def __init__(self, pretraining=False, num_views=12, ngram_filter_sizes=[3, 5, 7], num_filters=512):
        super(Net_Prev, self).__init__()
        self.num_views = num_views
        os.environ['TORCH_HOME'] = CACHE_DIR
        model = models.vgg11_bn(pretrained=pretraining)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(*list(model.classifier.children())[:2])

        embedding_dim = 4096
        # num_filters = 512
        self._ngram_filter_sizes = ngram_filter_sizes #[3, 5, 7]
        self._convolution_layers = [nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=ngram_size) for ngram_size in self._ngram_filter_sizes]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        self._activation = nn.ReLU()
        self._softmax = nn.Softmax(dim=1)
        self._layernorm = nn.LayerNorm(num_filters)

        self._fc1 = nn.Linear(num_filters * len(self._ngram_filter_sizes), num_filters)

    def forward(self, x):
        #feature extractor
        y = self.features(x)
        y = self.avgpool(y)
        y = self.classifier(torch.flatten(y, 1))
        y = y.view((int(x.shape[0]/self.num_views), -1, y.shape[-1])) #[n, V, D]

        # N-gram + attention
        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            token = self._activation(convolution_layer(y.transpose(1, 2)))# [n, D', num_gram]
            g_p = token.max(dim=2)[0]# [n, D']
            phi = torch.bmm(token.transpose(1, 2), g_p.unsqueeze(-1)) / (token.shape[1] ** .5)
            beta = self._softmax(phi)# [n, num_gram, 1]
            g_a = torch.bmm(beta.transpose(1, 2), token.transpose(1, 2))# [n, 1, D']
            g = g_a.squeeze(1) + g_p# [n, D']
            g = self._layernorm(g)
            filter_outputs.append(g)

        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]
        feature = self._fc1(maxpool_output)
        return feature
import os
class Net_Whole(nn.Module):
    def __init__(self, pretraining=False, nfea=4096, num_filters=512):
        super(Net_Whole, self).__init__()
        os.environ['TORCH_HOME'] = '/vol/research/sketching/projects'
        net = models.vgg11_bn(pretrained=pretraining)
        self.features = net.features
        classifier = net.classifier
        # classifier._modules['6'] = nn.Linear(4096, nclasses)
        self.modules_list = nn.ModuleList([module for module in classifier.children()])
        self._fc1 = nn.Linear(nfea, num_filters)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.modules_list[0](x)
        x = self.modules_list[1](x)
        x = self.modules_list[2](x)
        x = self.modules_list[3](x)
        x = self.modules_list[4](x)
        x = self.modules_list[5](x)
        out1 = self._fc1(x)
        # out2 = self.modules_list[6](out1)
        return out1  # [out1, out2]

class Net_Whole_resnet(nn.Module):
    def __init__(self, pretraining=False, num_filters=512):
        super(Net_Whole_resnet, self).__init__()
        net = models.resnet50(pretrained=pretraining)
        self.features = nn.ModuleList([module for module in net.children()][:-1])
        num_ftrs = net.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_filters)

    def forward(self, x):
        for i in range(len(self.features)):
            x = self.features[i](x)
        # print(.shape)
        x = self.fc(torch.squeeze(x))
        return x


if __name__ == '__main__':
    pool_idx = 13
    # avoid  pool at relu layer, because if relu is inplace, then
    # may cause misleading
    model_prev_pool = Net_Prev().cuda()
    # ipdb.set_trace()
    x = Variable(torch.rand(12*2, 3, 224, 224).cuda())
    bp = model_prev_pool(x)
    print(bp.shape)

    whole = Net_Whole().cuda()
    # ipdb.set_trace()
    x = Variable(torch.rand(2, 3, 224, 224).cuda())
    o2 = whole(x)
    print(o2.shape)

