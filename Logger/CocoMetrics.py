import numpy as np
import tensorboardX
import os
import torch

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
                    json.dump(config.json_serialize(), f)
        self.tb_summary_writer = tensorboardX.SummaryWriter(os.path.join(self.logger_path, 'tf_logging'))

    def get_train_logger(self, epoch, verbose=False):
        return self.create_tf_epoch_logger(self.tb_summary_writer, True, epoch, verbose)

    def get_validation_logger(self, epoch, verbose=False):
        return self.create_tf_epoch_logger(self.tb_summary_writer, False, epoch, verbose)

    def save_network(self, epoch, model):
        path = os.path.join(self.logger_path, 'saved_weights', 'state_dict_{}.pkl'.format(epoch))
        torch.save(model.state_dict(), path)

    def load_network_weights(self, epoch, model, device):
        path = os.path.join(self.logger_path, 'saved_weights', 'state_dict_{}.pkl'.format(epoch))
        with open(path, 'rb') as f:
            state_dict = torch.load(f, map_location=device)
        model.load_state_dict(state_dict)

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
