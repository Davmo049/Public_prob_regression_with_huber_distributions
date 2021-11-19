import torch
import numpy as np
import os
import json
import argparse

import tqdm

import WFLW.WFLW as WFLW
import WFLW.wflw_normalizer
import Mpii_pose.normalize_data as MpiiNormalizeData
import Mpii_pose.Dataset as MpiiDataset

from ConfigParser.ConfigParser import TrainingSettings, TrainingType, SplitType

import Utils.KeypointNormalizer
import Logger.Logger as LoggerLib

import HuberFusion.HuberFusion as HuberFusion

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

def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--run_name', type=str, default='dummy')
    arg_parser.add_argument('--training_type', type=str, default='none')
    arg_parser.add_argument('--epoch', type=int, default=0)
    arg_parser.add_argument('--run_index', type=int, default=0)
    args = arg_parser.parse_args()
    run_name = args.run_name
    training_type = args.training_type
    if training_type == 'mpii':
        training_type = TrainingType.MPII_KP
    elif training_type == 'wflw':
        training_type = TrainingType.WFLW
    else:
        raise Exception('unknown training type {}'.format(training_type))
    run_index = args.run_index
    if run_index != 0:
        run_name = run_name + '_run_{}'.format(run_index)
    else:
        pass # Do nothing (run_name = run_name)
    return training_type, run_name, args.epoch, run_index

def main():
    dtype = torch.float32
    fusion_type = 'prob'
    device = torch.device('cuda')
    training_type, run_name, epoch, run_index = parse_arguments()

    if training_type == TrainingType.MPII_KP:
        dataset, num_points, training_type_string_name, kp_normalizer, epoch_logger = extract_dataset_specific_stuff(training_type, SplitType.DEPLOY)
        eval_dataset = dataset.get_test()
    else:
        dataset, num_points, training_type_string_name, kp_normalizer, epoch_logger = extract_dataset_specific_stuff(training_type, SplitType.EVAL)
        eval_dataset = dataset.get_eval()

    log_dir = os.path.join('logs', training_type_string_name, run_name)
    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'rb') as f:
        config_dict = json.load(f)
    train_settings = TrainingSettings.deserialize(config_dict)
    loss_func = train_settings.loss_type
    network = train_settings.network_backbone
    load_logger = LoggerLib.Logger(log_dir, None, config=None, load=True)
    network = network.to(device)
    load_logger.load_network_weights(epoch, network, device)
    kp_normalizer = kp_normalizer.to(device)

    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1).to(device)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1).to(device)
    if training_type == TrainingType.WFLW:
        errs = []
    elif training_type == TrainingType.MPII_KP:
        ret = {}
    network.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(eval_dataset))):
            if training_type == TrainingType.MPII_KP:
                im, keypoints, keypoint_types, full_im_shape, bounding_circle, raw_annotation = eval_dataset.create_eval_batch(i)
            else:
                im, keypoints, keypoint_types, full_im_shape, bounding_circle = eval_dataset.create_eval_batch(i)

            im = torch.tensor(im, dtype=torch.float32).to(device)

            keypoints = torch.tensor(keypoints, dtype=torch.float32).to(device)
            im_normalized = (im - im_mean)/im_std
            out = network(im_normalized).view(-1, num_points, 5)
            keypoints_regress = kp_normalizer.normalize(keypoints)
            losses, half_precs, modes = loss_func(out, keypoints_regress, 1.0)
            loss = torch.mean(losses)
            keypoints_pred = kp_normalizer.denormalize(modes.detach())
            precs = torch.bmm(half_precs.view(-1,2,2), half_precs.view(-1, 2, 2)).view(half_precs.shape)
            precs = kp_normalizer.denormalize_prec(precs)
            np_prec = precs.cpu().detach().numpy()
            np_kp = keypoints.cpu().numpy()
            np_modes =  keypoints_pred.cpu().numpy()

            fusion_modes_np, fusion_precs_np = eval_dataset.convert_back_eval_preds_to_unaug(full_im_shape, np_modes, np_prec, bounding_circle)

            if fusion_type == 'prob':
                fusion_modes_np, fusion_precs_np = HuberFusion.combine_hubers(fusion_modes_np, fusion_precs_np)
            else:
                fusion_modes_np = np.mean(fusion_modes_np, axis=0)
                fusion_precs_np = fusion_precs_np[0]
            np_losses = losses.detach().cpu().numpy()
            np_kp = np_kp[0]
            # fusion done, dataset specific stuff below
            if training_type == TrainingType.WFLW:
                ip = np.linalg.norm(np_kp[60,:]-np_kp[72,:])
                diff = np.linalg.norm((np_kp-fusion_modes_np), axis=1)
                normalized_diff = diff/ip*100
                errs.append(np.mean(normalized_diff))
            elif training_type == TrainingType.MPII_KP:
                mode_in_orig_im, prec_in_orig_im = eval_dataset.eval_preprocessors[0].inverse_map(224, bounding_circle, fusion_modes_np, fusion_precs_np)
                points = list(map(list, mode_in_orig_im+1))
                std_sum = []
                for p in prec_in_orig_im:
                    try:
                        vals, vecs = np.linalg.eig(p)
                        std_sum.append(np.sqrt(1/vals[0])+np.sqrt(1/vals[1]))
                    except np.linalg.LinalgError:
                        std_sum.append(112)
                        

                im_data = ret.get(raw_annotation['image_orig'], {})
                im_data[str(raw_annotation['center'][0])+'_'+str(raw_annotation['center'][1])] = (points, std_sum)
                ret[raw_annotation['image_orig']] = im_data

    if training_type == TrainingType.WFLW:
        print('mean: {}'.format(np.mean(errs)))
        print('FR: {}'.format(np.mean(np.array(errs) > 10.0)))
    elif training_type == TrainingType.MPII_KP:
        if run_index != 0:
            # bug if run_index > 10
            dirname = os.path.join('fusion_dumps', '{}_{}'.format(run_name[:-6], fusion_type))
        else:
            dirname = os.path.join('fusion_dumps', '{}_{}'.format(run_name, fusion_type))
        try:
            os.makedirs(dirname)
        except FileExistsError:
            pass
        outfile = os.path.join(dirname, '{}.json'.format(run_index))
        with open(outfile, 'w') as f:
            json.dump(ret, f, separators=(',\n', ': '))


if __name__ == '__main__':
    main()
