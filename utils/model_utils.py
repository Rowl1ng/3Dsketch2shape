import os
import torch

import json

logs_filename = "Logs.pth"
specifications_filename = "specs.json"
model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"

def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))

def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir

def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )

def load_model_parameters(experiment_directory, checkpoint, encoder, decoder):

    filename = os.path.join(
        experiment_directory, model_params_subdir, "encoder_" + checkpoint + ".pth"
    )
    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)
    encoder.load_state_dict(data["model_state_dict"])

    filename = os.path.join(
        experiment_directory, model_params_subdir, "decoder_" + checkpoint + ".pth"
    )
    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)
    decoder.load_state_dict(data["model_state_dict"])

    return data["epoch"]

def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]

def loss_NCE(query, indices, latent_vecs, method='mean'):
    '''
        query: B x 512, latent vecs from Encoder
        gt_latvecs: B x 512, latent vecs from AutoDecoder
    '''
    B = query.shape[0]
    L = latent_vecs.shape[0]
    D = query.shape[1]

    all = torch.nn.functional.l1_loss(
        query.view(B, 1, D).repeat(1, L, 1),
        latent_vecs.view(1, L, D).repeat(B, 1, 1),
        reduction='none'
    ).sum(dim=2) # B x L

    loss = torch.nn.functional.cross_entropy(
        -all, indices, reduction=method
    )

    return loss

def loss_NCE_adapted(query, indices, gt, p=1):
    '''
        query: B x N x 512, latent vecs from Encoder
        gt_latvecs: B x 512, latent vecs from AutoDecoder
    '''
    B, N, _ = query.shape

    query_dist = torch.cdist(query, query, p=p)
    query_gt_dist = torch.cdist(query, gt.view(B, 1, -1), p=p)
    diag = torch.diag_embed(query_gt_dist.view(B, -1))
    final = diag + query_dist
    loss = torch.nn.functional.cross_entropy(
                -final.reshape(-1, N), indices.repeat(B), reduction='mean'
            )

    return loss


def get_spec_with_default(specs, key, default):
    if hasattr(specs, key):
        return getattr(specs, key)
    else:
        return default

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs.LearningRateSchedule

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules

def adjust_learning_rate(lr_schedules, optimizer, epoch):

    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

def get_latest_ckpt(exp_path, prefix):
    from glob import glob
    files = glob(os.path.join(exp_path, prefix + '*.pth'))
    if len(files) > 0:
        file = sorted(files, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split('_')[-1]))[-1]
    else:
        file = None
    optimizer_path = None
    if file is not None:
        # file = str(file)
        optimizer_path = file.replace('model_epoch', 'optimizer_epoch') 
    return file, optimizer_path

def process_state_dict(network_state_dict, type = 0):

    if torch.cuda.device_count() >= 2 and type == 0:
        for key, item in list(network_state_dict.items()):
            if key[:7] != 'module.':
                new_key = 'module.' + key
                network_state_dict[new_key] = item
                del network_state_dict[key]
    else:
        for key, item in list(network_state_dict.items()):
            if key[:7] == 'module.':
                new_key = key[7:]
                network_state_dict[new_key] = item
                del network_state_dict[key]

    return network_state_dict



def load_model(network, optimizer, resume_ckpt, optimizer_path):
    # if wandb.run.resumed and enable_wandb:
    #     resume_ckpt_wandb = wandb.restore(resume_ckpt)
    network_state_dict = torch.load(resume_ckpt)
    new_net_dict = network.state_dict()
    network_state_dict = process_state_dict(network_state_dict)
    pretrained_dict = {k: v for k, v in network_state_dict.items() if k in new_net_dict}
    new_net_dict.update(pretrained_dict)
    network.load_state_dict(new_net_dict)
    network.train()
    print(f"Reloaded the network from {resume_ckpt}")

    if optimizer_path is not None:
        optimizer_state_dict = torch.load(optimizer_path)
        optimizer.load_state_dict(optimizer_state_dict)
        print(f"Reloaded the optimizer from {optimizer_path}")
    return network, optimizer

def dict_from_module(module):
    context = {}
    for setting in vars(module).keys():
        # you can write your filter here
        if setting not in ['torch', 'nn']:
            context[setting] = str(vars(module)[setting])

    return context