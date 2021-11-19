import argparse
import json
import numpy as np
import torch
from Losses.Losses import ProbHuberLoss
import WFLW.wflw_normalizer
import WFLW.WFLW
import Logger.Logger as LoggerLib

from ConfigParser.ConfigParser import TrainingSettings, TrainingType
import ConfigParser.ConfigParser as ConfigParser
import CocoKeypoints
import Utils.KeypointNormalizer
import tqdm
import Mpii_pose.normalize_data as MpiiNormalizeData
import Mpii_pose.Dataset as MpiiDataset
import DS_300W_LP.Dataset
import DS_300W_LP.normalize_data



def set_start_head_weights(head, start_diagonal):
    num_keypoints = head.bias.shape[0]//5
    with torch.no_grad():
        head.weight.fill_(0.0)
        head.bias.fill_(0.0)
        prec_offset = num_keypoints*2
        head.bias[0:prec_offset].fill_(0.0)
        for keypoint_idx in range(num_keypoints):
            head.bias[keypoint_idx*5+2] = start_diagonal
            head.bias[keypoint_idx*5+4] = start_diagonal


def extract_dataset_specific_stuff(training_type, split_type):
    if training_type == TrainingType.COCO_KP:
        num_points = 17
        string_name = 'COCO_keypoints'
        dataset = CocoKeypoints.PreprocessedCocoKeypoints(split=split_type)
        normalizer_mean = CocoKeypoints.Coco_mean
        normalizer_half_prec = CocoKeypoints.Coco_half_prec
        normalizer = Utils.KeypointNormalizer.TorchNormalizer(normalizer_mean, normalizer_half_prec)
        raise Exception("COCO training might be broken, not used in a while")
        epoch_logger = LoggerLib.coco_create_tf_epoch_logger_TODO
    elif training_type == TrainingType.WFLW:
        num_points = 98
        string_name = 'wflw_keypoints'
        dataset = WFLW.WFLW.PreprocessedWflwfKeypointsDataset(split=split_type)
        normalizer_mean = WFLW.wflw_normalizer.wflw_mean
        normalizer_half_prec = WFLW.wflw_normalizer.wflw_half_prec
        normalizer = Utils.KeypointNormalizer.TorchNormalizer(normalizer_mean, normalizer_half_prec)
        epoch_logger = LoggerLib.wflw_create_tf_epoch_logger
    elif training_type == TrainingType.DS_300W_LP:
        num_points = 68
        string_name = 'DS_300W_LP'
        dataset = DS_300W_LP.Dataset.Preprocessed300WLPKeypointsDataset(split=split_type)
        normalizer_mean = DS_300W_LP.normalize_data.mean
        normalizer_half_prec = DS_300W_LP.normalize_data.half_prec
        normalizer = Utils.KeypointNormalizer.TorchNormalizer(normalizer_mean, normalizer_half_prec)
        epoch_logger = LoggerLib.basic_logger
    elif training_type == TrainingType.MPII_KP:
        num_points = 16
        string_name = 'MPII_pose'
        dataset = MpiiDataset.PreprocessedMpiiDataset(split=split_type)
        normalizer_mean = MpiiNormalizeData.mean
        normalizer_half_prec = MpiiNormalizeData.half_prec
        normalizer = Utils.KeypointNormalizer.TorchNormalizer(normalizer_mean, normalizer_half_prec)
        epoch_logger = LoggerLib.basic_logger
    else:
        raise Exception("unknown training type: {}".format(training_type))
    return dataset, num_points, string_name, normalizer, epoch_logger

def create_dataloader(dataset, batch_size):
    train_dataset = dataset.get_train()
    eval_dataset = dataset.get_eval()
    dataloader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=True)
    dataloader_eval = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval

def set_seeds(run_index=0):
    print(run_index)
    np.random.seed(9001+run_index*7001)
    torch.manual_seed(9002+run_index*7001)

def init_final_conv(conv_module):
    conv_shape = conv_module.weight.shape
    assert(conv_shape[1] == 1)
    assert(conv_shape[2] == 7)
    assert(conv_shape[3] == 7)
    with torch.no_grad():
        conv_prototype = torch.zeros((conv_shape[0], 1, 7,7))
        conv_prototype[:, 0, 3,3] = 1
        conv_prototype[:, 0, 3,4] = 0.5
        conv_prototype[:, 0, 4,3] = 0.5
        conv_prototype[:, 0, 2,3] = 0.5
        conv_prototype[:, 0, 3,2] = 0.5
        conv_module.weight.copy_(conv_prototype)


def init_head_weights(network, loss_func):
    init_final_conv(network.final_conv)
    init_single_head_weights(network.head, loss_func)

def init_single_head_weights(head, loss_func):
    if isinstance(loss_func, ProbHuberLoss):
        min_reasonable_half_prec = loss_func.min_reasonable
        max_reasonable_half_prec = loss_func.max_reasonable
        start_val = 1.0
        set_start_head_weights(head, -(max_reasonable_half_prec-min_reasonable_half_prec)/2+start_val-min_reasonable_half_prec)
    else:
        with torch.no_grad():
            head.bias.fill_(0)
            head.weight.fill_(0)

def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--run_name', type=str, default='dummy')
    arg_parser.add_argument('--run_index', type=int, default=0)
    arg_parser.add_argument('config_file', type=str)
    arg_parser.add_argument('--resume_epoch', type=int, default=0)
    args = arg_parser.parse_args()
    run_name = args.run_name
    config_file = args.config_file
    resume_epoch = args.resume_epoch
    run_index = args.run_index
    with open(config_file, 'rb') as f:
        config_dict = json.load(f)
    train_settings = TrainingSettings.deserialize(config_dict)
    return train_settings, run_name, resume_epoch, run_index

def main():
    train_settings, run_name, resume_epoch, run_index = parse_arguments()
    set_seeds(run_index)
    device = torch.device('cuda')
    dtype = torch.float32
    batch_size = train_settings.batch_size
    loss_func = train_settings.loss_type
    network = train_settings.network_backbone
    if isinstance(train_settings.pretrain, ConfigParser.LoadLocalPretrainWeights):
        pretrain = train_settings.pretrain
        log_dir = 'logs/{}/{}'.format(pretrain.training_type, pretrain.training_name)
        pretrain_logger = LoggerLib.Logger(log_dir, None, config=None, load=True)
        pretrain_logger.load_network_weights(pretrain.epoch, network, device, transfer=True)
        del pretrain_logger


    dataset, num_points, training_type_name, kp_normalizer, epoch_logger = extract_dataset_specific_stuff(train_settings.training_type, train_settings.split_type)

    kp_normalizer=kp_normalizer.to(device)

    init_head_weights(network, loss_func)


    dataloader_train, dataloader_eval = create_dataloader(dataset, batch_size)

    logger_should_resume = resume_epoch != 0
    if run_index == 0:
        log_dir = 'logs/{}/{}'.format(training_type_name, run_name)
    else:
        log_dir = 'logs/{}/{}_run_{}'.format(training_type_name, run_name, run_index)
    loggers = LoggerLib.Logger(log_dir, epoch_logger, config=train_settings.serialize(), load=logger_should_resume)
    if logger_should_resume:
        logger.load_network_weights(resume_epoch, network, device)
        cur_epoch = resume_epoch + 1 # plus one because weights saved at end of epoch
    else:
        cur_epoch = 0
    network = network.to(device)
    num_epochs = train_settings.lr_schedule.num_epochs()
    non_pretrain_wd_weights = list(network.final_conv.parameters()) + [network.head.weight]
    non_pretrain_non_wd_weights = [network.head.bias]
    non_wd_weights = list(set(network.parameters())-set(non_pretrain_wd_weights))
    for epoch in range(cur_epoch, num_epochs):
        verbose = epoch % 30 == 9 or epoch == num_epochs-1
        cur_lr = train_settings.lr_schedule.get_learning_rate(epoch)
        if epoch < 3:
            opt_weights_wd = non_pretrain_wd_weights
            opt_weights_non_wd = non_pretrain_non_wd_weights
        else:
            opt_weights_wd = non_pretrain_wd_weights
            opt_weights_non_wd = non_wd_weights
        optimizer = train_settings.optimizer_type([{'params': opt_weights_non_wd}, {'params':opt_weights_wd, 'weight_decay':1e-1}], cur_lr)
        network.train()
        logger = loggers.get_train_logger(epoch, verbose)
        run_epoch(dataloader_train, network, loss_func, kp_normalizer, optimizer, logger, device, dtype)

        network.eval()
        logger = loggers.get_validation_logger(epoch, verbose)
        with torch.no_grad():
            run_epoch(dataloader_eval, network, loss_func, kp_normalizer, None, logger, device, dtype)

        if epoch == num_epochs-1:
            loggers.save_network(epoch, network)


def run_epoch(dataloader, network, loss_func, kp_normalizer, optimizer, logger, device, dtype):
    if len(dataloader) == 0:
        return
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
    for im, keypoints, weights, keypoint_type in tqdm.tqdm(dataloader, mininterval=2.0, maxinterval=10.0):
        B = keypoints.shape[0]
        N = keypoints.shape[1]
        im = im.to(device)
        keypoints = keypoints.to(device)
        weights = weights.to(device)
        im = (im - im_mean)/im_std
        keypoint_type = keypoint_type.numpy()
        out = network(im)
        # sometimes it is nicer to define the loss function in the image coordinate system, but I don't do that here
        keypoints_regress = kp_normalizer.normalize(keypoints)
        loss = 0.0
        out = out.view(B, N, 5)
        losses, half_prec, modes = loss_func(out, keypoints_regress, weights)
        loss += torch.mean(losses)

        keypoints_pred = kp_normalizer.denormalize(modes.detach())
        np_modes =  keypoints_pred.cpu().numpy()
        np_losses = losses.detach().cpu().numpy()
        np_kp = keypoints.cpu().numpy()
        logger.add_samples(np_losses, np_modes, np_kp, keypoint_type)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del loss
        del im, keypoints, weights, keypoint_type, out, keypoints_regress, losses, half_prec, modes
    logger.finish()


if __name__ == '__main__':
    main()
