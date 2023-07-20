#!/usr/bin/env python3

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import json
import importlib
import numpy as np
import tqdm
import wandb
from utils import model_utils as util

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EXP_DIR = os.getenv('EXP_DIR')
# os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_MODE"] = "offline"
wandb.init(project="SDFAutoEncoder", name="SDF_L1_NCE_finetune")

def save_logs(
    experiment_directory,
    loss_sdf_log,
    loss_sdf_test_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss_sdf": loss_sdf_log,
            "loss_sdf_test": loss_sdf_test_log,
        },
        os.path.join(experiment_directory, util.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, util.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss_sdf"],
        data["loss_sdf_test"],
        data["epoch"],
    )


def clip_logs(loss_sdf_log,loss_regl_log,loss_sdf_test_log,loss_regl_test_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_sdf_log) // len(lr_log)

    loss_sdf_log = loss_sdf_log[: (iters_per_epoch * epoch)]
    loss_regl_log = loss_regl_log[: (iters_per_epoch * epoch)]

    iters_per_epoch = len(loss_sdf_test_log) // len(lr_log)
    loss_sdf_test_log = loss_sdf_test_log[: (iters_per_epoch * epoch)]
    loss_regl_test_log = loss_regl_test_log[: (iters_per_epoch * epoch)]

    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_sdf_log,loss_regl_log,loss_sdf_test_log,loss_regl_test_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def loss_NCE(query, indices, latent_vecs):
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
        -all, indices, reduction='sum'
    )

    return loss


def main_function(experiment_directory, continue_from):

    # specs = util.load_experiment_specifications(experiment_directory)
    ## import config here
    spec = importlib.util.spec_from_file_location('*', args.experiment_directory)
    config = importlib.util.module_from_spec(spec)

    # print("Experiment description: \n" + ' '.join([str(elem) for elem in specs["Description"]]))

    data_source = config.DataSource
    train_split_file = config.TrainSplit
    test_split_file = config.TestSplit

    from models.encoder.pointnet2_cls_msg import PointNetEncoder
    from models.decoder.sdf import SDFDecoder

    encoder = PointNetEncoder(k=config.CodeLength).to(device)
    decoder = SDFDecoder(config=config).to(device)

    # arch_encoder = __import__("lib.models." + config.NetworkEncoder"], fromlist=["PointNetEncoder"])
    # arch_decoder = __import__("lib.models." + config["NetworkDecoder"], fromlist=["DeepSDF"])
    latent_size = config.CodeLength

    checkpoints = list(
        range(
            config.SnapshotFrequency,
            config.NumEpochs + 1,
            config.SnapshotFrequency,
        )
    )

    for checkpoint in config.AdditionalSnapshots:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = util.get_learning_rate_schedules(spec)

    grad_clip = util.get_spec_with_default(spec, "GradientClipNorm", None)

    def save_latest(epoch):
        util.save_model(experiment_directory, "encoder_latest.pth", encoder, epoch)
        util.save_model(experiment_directory, "decoder_latest.pth", decoder, epoch)
        util.save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)

    def save_checkpoints(epoch):
        util.save_model(experiment_directory, "encoder_" + str(epoch) + ".pth", encoder, epoch)
        util.save_model(experiment_directory, "decoder_" +str(epoch) + ".pth", decoder, epoch)
        util.save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)

    def signal_handler(sig, frame):
        print("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = util.get_spec_with_default(config, "SamplesPerScene", 8192)
    scene_per_batch = util.get_spec_with_default(config, "ScenesPerBatch", 16)
    clamp_dist = util.get_spec_with_default(config, "ClampingDistance", 0.1)
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = util.get_spec_with_default(config, "CodeRegularization", True)
    code_reg_lambda = util.get_spec_with_default(config, "CodeRegularizationLambda", 1e-4)

    # encoder = arch_encoder.PointNetEncoder(k=latent_size).cuda()
    # decoder = arch_decoder.DeepSDF(latent_size, **specs["NetworkSpecs"]).cuda()

    print("training with {} GPU(s)".format(torch.cuda.device_count()))

    encoder = torch.nn.DataParallel(encoder)
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = config.NumEpochs
    log_frequency = util.get_spec_with_default(config, "LogFrequency", 17)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    with open(test_split_file, "r") as f:
        test_split = json.load(f)


    num_data_loader_threads = util.get_spec_with_default(config, "DataLoaderThreads", 16)
    print("loading data with {} threads".format(num_data_loader_threads))

    sdf_dataset = lib.data.SketchSDF(
        data_source, train_split, num_samp_per_scene, is_train=True, num_views = specs["NumberOfViews"]
    )
    sdf_dataset_test = lib.data.SketchSDF(
        data_source, test_split, num_samp_per_scene, is_train=False, num_views = specs["NumberOfViews"]
    )

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )
    sdf_loader_test = data_utils.DataLoader(
        sdf_dataset_test,
        batch_size=scene_per_batch,
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    print("There are {} training samples".format(len(sdf_dataset)))
    print("There are {} test samples".format(len(sdf_dataset_test)))

    lat_vecs = torch.load(experiment_directory+'/LatentCodes/latest.pth')['latent_codes']['weight'].to(device)

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0)/10,
            },
            {
                "params": encoder.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_sdf_log = []
    loss_sdf_test_log = []
    start_epoch = 1

    if continue_from is not None:

        print('continuing from "{}"'.format(continue_from))

        try:
            model_epoch = util.load_model_parameters(experiment_directory, continue_from, encoder, decoder)

            optimizer_epoch = util.load_optimizer(experiment_directory, continue_from + ".pth", optimizer_all)

            loss_sdf_log,loss_sdf_test_log, log_epoch = load_logs(experiment_directory)
            if not log_epoch == model_epoch:
                loss_sdf_log,loss_sdf_test_log= clip_logs(loss_sdf_log,loss_sdf_test_log)

            if not (model_epoch == optimizer_epoch):
                raise RuntimeError(
                    "epoch mismatch: {} vs {} vs {}".format(
                        model_epoch, optimizer_epoch, log_epoch
                    ))
            start_epoch = model_epoch + 1

        except:
            print ('failed to load from continue. Starting from scratch.')

    print("starting from epoch {}".format(start_epoch))
    print("Number of encoder parameters: {}".format(sum(p.data.nelement() for p in encoder.parameters() if p.requires_grad)))
    print("Number of decoder parameters: {}".format(sum(p.data.nelement() for p in decoder.parameters() if p.requires_grad)))

    wandb.config = {
        "learning_rate": lr_schedules[1],
        "epochs": num_epochs,
        "batch_size": scene_per_batch}
    wandb.watch([encoder, decoder])

    for epoch in range(start_epoch, num_epochs + 1):

        print("epoch {}...".format(epoch))

        decoder.train()
        encoder.train()

        # adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        for sdf_data, pointset, name, indices in tqdm.tqdm(sdf_loader):

            optimizer_all.zero_grad()

            # Process the input data
            sdf_data.requires_grad = False
            xyz = sdf_data[:, :, 0:3].to(device)
            sdf_gt = sdf_data[:, :, 3].reshape(-1,1).to(device)
            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            vecs = encoder(pointset)

            # L1 based NCE Loss
            nce_reg_loss = loss_NCE(vecs, indices.to(device), lat_vecs)

            # DeepSDF branch
            batch_vecs = vecs.view(vecs.shape[0], 1, vecs.shape[1]).repeat(1, xyz.shape[1], 1).reshape(-1, latent_size)

            gt_batch_vecs = lat_vecs[indices.view(-1).to(device)]
            l1_reg_loss = loss_l1(vecs, gt_batch_vecs)

            pred_sdf = decoder(batch_vecs, xyz.reshape(-1, 3))

            if enforce_minmax:
                pred_sdf = torch.clamp(pred_sdf, minT, maxT)

            sdf_loss = loss_l1(pred_sdf, sdf_gt.to(device)) / pred_sdf.shape[0]

            if do_code_regularization:
                l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                reg_loss = (
                    code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                ) / pred_sdf.shape[0]
                batch_loss = sdf_loss + reg_loss.to(device)
            else:
                batch_loss = sdf_loss
            
            batch_loss = batch_loss + nce_reg_loss + l1_reg_loss
            batch_loss.backward()

            loss_sdf_log.append(sdf_loss.cpu())

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)

            optimizer_all.step()

        # wandb logging
        wandb.log({'NCE_loss': nce_reg_loss}, step=epoch)
        wandb.log({'L1_loss': l1_reg_loss}, step=epoch)
        wandb.log({'SDF_loss': sdf_loss}, step=epoch)
        wandb.log({"total train loss": batch_loss}, step=epoch)


        if epoch in checkpoints:
            print ('saving checkpoints')
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:

            print ('Training -- SDF_loss: {}, NCE_loss: {}, L1_loss: {}, total_loss: {}'.format(
                sdf_loss.item(), nce_reg_loss.item(), l1_reg_loss.item(), batch_loss.item()))

            encoder.eval()
            decoder.eval()

            for sdf_data, pointset, name, indices in tqdm.tqdm(sdf_loader_test):

                with torch.no_grad():
                    # Process the input data
                    sdf_data.requires_grad = False
                    xyz = sdf_data[:, :, 0:3].to(device)
                    sdf_gt = sdf_data[:, :, 3].reshape(-1,1).to(device)
                    if enforce_minmax:
                        sdf_gt = torch.clamp(sdf_gt, minT, maxT)

                    vecs = encoder(pointset)
                    # DeepSDF branch
                    batch_vecs = vecs.view(vecs.shape[0], 1, vecs.shape[1]).repeat(1, xyz.shape[1], 1).reshape(-1, latent_size)
                    pred_sdf = decoder(batch_vecs, xyz.reshape(-1, 3))

                    if enforce_minmax:
                        pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                    sdf_loss = loss_l1(pred_sdf, sdf_gt.to(device)) / pred_sdf.shape[0]

                    if do_code_regularization:
                        l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                        reg_loss = (
                            code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                        ) / pred_sdf.shape[0]

                        batch_loss = sdf_loss + reg_loss.to(device)
                    else:
                        batch_loss = sdf_loss

                    wandb.log({"val loss": batch_loss}, step=epoch)
                    loss_sdf_test_log.append(sdf_loss.cpu())

            print ('Val Loss: ', batch_loss.item())

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_sdf_log,
                loss_sdf_test_log,
                epoch,
            )
            loss_sdf_log = []
            loss_sdf_test_log = []


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )

    args = arg_parser.parse_args()
    main_function(args.experiment_directory, args.continue_from)