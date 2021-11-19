import numpy as np
import tensorboardX
import os
import torch
import pprint
import json

class Logger():
    def __init__(self, logger_path, create_tf_epoch_logger, config=None, load=False):
        self.create_tf_epoch_logger = create_tf_epoch_logger
        self.logger_path = logger_path # for example base_dir/logs/Coco/ExperimentName
        if not load and os.path.exists(logger_path):
            raise Exception("rerunning old training")
        if not load:
            os.makedirs(logger_path)
            os.makedirs(os.path.join(logger_path, 'saved_weights'))
        if config is not None:
            if not load:
                with open(os.path.join(logger_path, 'config.json'), 'w') as f:
                    json.dump(config, f)
            else:
                pass
                # TODO assert configs are the same
        self.tb_summary_writer = tensorboardX.SummaryWriter(os.path.join(self.logger_path, 'tf_logging'))

    def get_train_logger(self, epoch, verbose=False):
        return self.create_tf_epoch_logger(self.tb_summary_writer, True, epoch, verbose)

    def get_validation_logger(self, epoch, verbose=False):
        return self.create_tf_epoch_logger(self.tb_summary_writer, False, epoch, verbose)

    def save_network(self, epoch, model):
        path = os.path.join(self.logger_path, 'saved_weights', 'state_dict_{}.pkl'.format(epoch))
        torch.save(model.state_dict(), path)

    def load_network_weights(self, epoch, model, device, transfer=False):
        path = os.path.join(self.logger_path, 'saved_weights', 'state_dict_{}.pkl'.format(epoch))
        with open(path, 'rb') as f:
            state_dict = torch.load(f, map_location=device)
        if transfer:
            for v in ['head.weight', 'head.bias', 'end.2.weight', 'end.2.bias']:
                if v in state_dict:
                    del state_dict[v]
        strict=not(transfer)
        model.load_state_dict(state_dict, strict=strict)

def coco_create_tf_epoch_logger(*args):
    return CocoTfLogger(*args)

class CocoTfLogger():
    KEYPOINTS_NAME = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_should', 'r_should', 'l_elbow', 'r_elbox', 'l_hand', 'r_hand', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_foot', 'r_foot']
    def __init__(self, tb_writer, training, epoch, verbose):
        self.tb_writer = tb_writer
        self.training = training
        self.epoch = epoch
        self.verbose = verbose
        self.metrics = CocoTfMetrics()

    def add_samples(self, *args):
        self.metrics.add_samples(*args)

    def finish(self):
        loss, (mean_parts1, mean_parts2), (mean1, mean2) = self.metrics.get_stats()
        if self.training:
            split_str = 'train'
        else:
            split_str = 'eval'
        self.tb_writer.add_scalar('{}/loss'.format(split_str), loss, self.epoch)
        self.tb_writer.add_scalar('{}/mean1'.format(split_str), mean1, self.epoch)
        self.tb_writer.add_scalar('{}/mean2'.format(split_str), mean2, self.epoch)
        for part_mean, part_name in zip(mean_parts1, self.KEYPOINTS_NAME):
            self.tb_writer.add_scalar('{}/{}/mean1'.format(split_str, part_name), part_mean, self.epoch)
        for part_mean, part_name in zip(mean_parts2, self.KEYPOINTS_NAME):
            self.tb_writer.add_scalar('{}/{}/mean2'.format(split_str, part_name), part_mean, self.epoch)
        print(split_str)
        print(self.epoch)
        print(mean1)
        print(mean2)
        if self.verbose:
            print(mean_parts1)
            print(mean_parts2)

class CocoTfMetrics():
    N_BODY_PARTS = 17
    def __init__(self):
        self.sum_errors_1 = np.zeros(shape=(self.N_BODY_PARTS), dtype=np.float)
        self.N_errors_1 = np.zeros(shape=(self.N_BODY_PARTS), dtype=np.float)
        self.sum_errors_2 = np.zeros(shape=(self.N_BODY_PARTS), dtype=np.float)
        self.N_errors_2 = np.zeros(shape=(self.N_BODY_PARTS), dtype=np.float)
        self.losses_sum = 0.0
        self.losses_N = 0

    def add_samples(self, losses, prediction_mode, ground_truth, gt_type):
        # prediction mode BxNx2
        # gt BxNx2
        # gt_type is BxN, 0 for invalid, 1 and 2 for occluded/visible
        diff = ground_truth-prediction_mode
        error = np.linalg.norm(diff, axis=2)
        mask1 = gt_type == 1
        mask2 = gt_type == 2
        self.sum_errors_1 += np.sum(error*mask1, axis=0)
        self.N_errors_1 += np.sum(mask1, axis=0)
        self.sum_errors_2 += np.sum(error*mask2, axis=0)
        self.N_errors_2 += np.sum(mask2, axis=0)
        self.losses_sum += np.sum(losses)
        self.losses_N += len(losses)

    def get_stats(self):
        mean1 = self.sum_errors_1/self.N_errors_1
        mean2 = self.sum_errors_2/self.N_errors_2
        loss = self.losses_sum/self.losses_N
        return loss, (mean1, mean2), (np.mean(mean1), np.mean(mean2))

# WFLW logger

def wflw_create_tf_epoch_logger(*args):
    return WflwTfLogger(*args)

wflw_keypoint_to_region_map = {}
for i in range(0, 33):
    wflw_keypoint_to_region_map[i] = 'chin'
for i in range(33, 51):
    wflw_keypoint_to_region_map[i] = 'eyebrow'
for i in range(51, 60):
    wflw_keypoint_to_region_map[i] = 'nose'
for i in range(60, 76):
    wflw_keypoint_to_region_map[i] = 'eye'
wflw_keypoint_to_region_map[96] = 'eye'
wflw_keypoint_to_region_map[97] = 'eye'
for i in range(76, 96):
    wflw_keypoint_to_region_map[i] = 'mouth'

indices_per_type = {}
for index, name in wflw_keypoint_to_region_map.items():
    new_count = indices_per_type.get(name, 0) + 1
    indices_per_type[name] = new_count

class WflwTfLogger():
    def __init__(self, tb_writer, training, epoch, verbose):
        self.tb_writer = tb_writer
        self.training = training
        self.epoch = epoch
        self.verbose = verbose
        self.metrics = WflwTfMetrics()

    def add_samples(self, *args):
        self.metrics.add_samples(*args)

    def finish(self):
        loss, avg_pixel_dist, avg_nme, pixel_dists, nme_per_part = self.metrics.get_stats()
        if self.training:
            split_str = 'train'
        else:
            split_str = 'eval'
        self.tb_writer.add_scalar('{}/loss'.format(split_str), loss, self.epoch)
        self.tb_writer.add_scalar('{}/mean_err'.format(split_str), avg_pixel_dist, self.epoch)
        self.tb_writer.add_scalar('{}/mean_nme'.format(split_str), avg_nme, self.epoch)
        for part_name, part_mean in pixel_dists.items():
            self.tb_writer.add_scalar('{}/{}/mean'.format(split_str, part_name), part_mean, self.epoch)
        for part_name, part_nme in nme_per_part.items():
            self.tb_writer.add_scalar('{}/{}/nme'.format(split_str, part_name), part_nme, self.epoch)
        print('{} epoch: {}'.format(split_str, self.epoch))
        print('loss: {}'.format(loss))
        print('avg error {}'.format(avg_pixel_dist))
        print('avg nme {}'.format(avg_nme))
        if self.verbose:
            print('distance')
            pprint.pprint(pixel_dists)
            print('nme')
            pprint.pprint(nme_per_part)

class WflwTfMetrics():
    def __init__(self):
        self.sum_errors = np.zeros(shape=(98), dtype=np.float)
        self.losses_sum = 0.0
        self.sum_nme = np.zeros(shape=(98), dtype=np.float)
        self.N = 0

    def add_samples(self, losses, prediction_mode, ground_truth, keypoint_type):
        # prediction mode BxNx2
        # gt BxNx2
        # gt_type is BxN, 0 for invalid, 1 and 2 for occluded/visible
        # keypoint_type is unused, just for consistency with coco logger
        interocular = np.linalg.norm(ground_truth[:, 60, :]-ground_truth[:, 72, :], axis=1)
        diff = ground_truth-prediction_mode
        error = np.linalg.norm(diff, axis=2)
        error_nme = (error*100)/interocular.reshape(-1, 1)
        self.sum_errors += np.sum(error, axis=0)
        self.sum_nme += np.sum(error_nme, axis=0)
        self.losses_sum += np.sum(losses)
        self.N += len(losses)

    def get_stats(self):
        all_errors = self.sum_errors/self.N
        all_nme = self.sum_nme / self.N
        loss = self.losses_sum/self.N
        mean_error = np.mean(all_errors)
        mean_nme = np.mean(all_nme)
        sum_error_per_region = {}
        sum_nme_per_region = {}
        for i in range(98):
            region = wflw_keypoint_to_region_map[i]
            cur_err = sum_error_per_region.get(region, 0)
            cur_nme = sum_nme_per_region.get(region, 0)
            cur_err += all_errors[i]
            cur_nme += all_nme[i]
            sum_error_per_region[region] = cur_err
            sum_nme_per_region[region] = cur_nme
        for k in sum_error_per_region.keys():
            sum_error_per_region[k] /= indices_per_type[k]
            sum_nme_per_region[k] /= indices_per_type[k]
        return loss, mean_error, mean_nme, sum_error_per_region, sum_nme_per_region

def basic_logger(*args):
    return BasicTfLogger(*args)

class BasicTfLogger():
    def __init__(self, tb_writer, training, epoch, verbose):
        self.tb_writer = tb_writer
        self.training = training
        self.epoch = epoch
        self.verbose = verbose
        self.metrics = BasicTfMetrics()

    def add_samples(self, *args):
        self.metrics.add_samples(*args)

    def finish(self):
        loss, avg_pixel_dist, pixel_dists = self.metrics.get_stats()
        if self.training:
            split_str = 'train'
        else:
            split_str = 'eval'
        self.tb_writer.add_scalar('{}/loss'.format(split_str), loss, self.epoch)
        self.tb_writer.add_scalar('{}/mean_err'.format(split_str), avg_pixel_dist, self.epoch)
        for i in range(16):
            self.tb_writer.add_scalar('{}/err_{}'.format(split_str, i), pixel_dists[i], self.epoch)
        print('{} epoch: {}'.format(split_str, self.epoch))
        print('loss: {}'.format(loss))
        print('avg error {}'.format(avg_pixel_dist))
        if self.verbose:
            print('errors')
            print(pixel_dists)

class BasicTfMetrics():
    def __init__(self):
        self.sum_errors = 0.0
        self.N_per_kp = 0.0
        self.losses_sum = 0.0
        self.N = 0

    def add_samples(self, losses, prediction_mode, ground_truth, keypoint_type):
        # prediction mode BxNx2
        # gt BxNx2
        # gt_type is BxN, 0 for invalid, 1 and 2 for occluded/visible
        diff = ground_truth-prediction_mode
        mask = keypoint_type != 0
        error = np.linalg.norm(diff, axis=2)
        self.sum_errors += np.sum(error*mask, axis=0)
        self.N_per_kp += np.sum(mask, axis=0)
        self.losses_sum += np.sum(losses)
        self.N += len(losses)

    def get_stats(self):
        all_errors = self.sum_errors/self.N_per_kp
        mean_error = np.mean(all_errors)
        loss = self.losses_sum / self.N
        return loss, mean_error, all_errors
