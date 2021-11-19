import json
import numpy as np
from enum import IntEnum
from Losses.Losses import LossType
import torch

import NetArch.NetArch

class TrainingType(IntEnum):
    COCO_KP=0
    WFLW=1
    DS_300W_LP=2
    MPII_KP=3

    @staticmethod
    def to_string(v):
        return v.name.lower()

    @staticmethod
    def from_string(v):
        s = training_type_str_to_enum.get(v.lower(), None)
        if s is None:
            raise Exception('could not resolve {}'.format(v))
        return s

training_type_str_to_enum={}
for v in TrainingType:
    training_type_str_to_enum[v.name.lower()] = v


class OptimizerType(IntEnum):
    SGD=0
    ADAM=1

    @staticmethod
    def to_string(v):
        return v.name.lower()

    @staticmethod
    def from_string(v):
        s = optimizer_type_str_to_enum.get(v.lower(), None)
        if s is None:
            raise Exception('could not resolve {}'.format(v))
        return s

optimizer_type_str_to_enum={}
for v in OptimizerType:
    optimizer_type_str_to_enum[v.name.lower()] = v

class SplitType(IntEnum):
    DEVELOP=0 # eval is part of full train
    EVAL=1 # keep train/eval as is
    MINI=2 # to test training works
    DEPLOY=3 # train on train+eval, no eval since test is not visible.

    @staticmethod
    def to_string(v):
        return v.name.lower()

    @staticmethod
    def from_string(v):
        s = split_type_str_to_enum.get(v.lower(), None)
        if s is None:
            raise Exception('could not resolve {}'.format(v))
        return s

split_type_str_to_enum={}
for v in SplitType:
    split_type_str_to_enum[v.name.lower()] = v



def serialize_optimizer(opt_type):
    if opt_type == torch.optim.SGD:
        return OptimizerType.SGD
    elif opt_type == torch.optim.Adam:
        return OptimizerType.ADAM

def deserialize_optimizer(opt_enum):
    if opt_enum == OptimizerType.SGD:
        return torch.optim.SGD
    elif opt_enum == OptimizerType.ADAM:
        return torch.optim.Adam
    else:
        raise Exception("unknown optimizer_enum: {}".format(opt_enum))


class TrainingSettings():
    def __init__(self, training_type, split_type, batch_size, loss_type, optimizer_type, lr_schedule, network_backbone, pretrain):
        self.training_type = training_type
        self.split_type = split_type
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.lr_schedule = lr_schedule
        self.network_backbone = network_backbone
        self.pretrain=pretrain

    def serialize(self):
        return {'training_type': TrainingType.to_string(self.training_type),
                'split_type':  SplitType.to_string(self.split_type),
                'loss_type':  self.loss_type.serialize(),
                'batch_size':  self.batch_size,
                'optimizer_type': OptimizerType.to_string(serialize_optimizer(self.optimizer_type)),
                'lr_schedule':self.lr_schedule.serialize(),
                'network_backbone_settings': NetArch.NetArch.serialize_network(self.network_backbone),
                'pretrain': self.pretrain.serialize()
                }

    @staticmethod
    def deserialize(dic):
        training_type = TrainingType.from_string(dic['training_type'])
        split_type = SplitType.from_string(dic['split_type'])
        batch_size = int(dic['batch_size'])
        loss_type = LossType.deserialize(dic['loss_type'])
        optimizer_type = deserialize_optimizer(OptimizerType.from_string(dic['optimizer_type']))
        lr_schedule = LearningRateSchedule.deserialize(dic['lr_schedule'])
        pretrain = deserialize_pretrain(dic['pretrain'])
        network_backbone = NetArch.NetArch.deserialize_network(dic['network_backbone_settings'], pretrain=isinstance(pretrain, ImageNetPretrain))
        return TrainingSettings(training_type, split_type, batch_size, loss_type, optimizer_type, lr_schedule, network_backbone, pretrain)

class LearningRateSchedule():
    def __init__(self, epoch_learning_rates):
        # epoch learning rates is a list of (epoch, learning_rate) sorted by epoch
        self.check_argument(epoch_learning_rates)
        self.epoch_learning_rates = epoch_learning_rates
        self.current_idx = 0

    def get_learning_rate(self, epoch):
        assert(epoch >= self.epoch_learning_rates[self.current_idx][0])
        if len(self.epoch_learning_rates) == self.current_idx+1:
            return self.epoch_learning_rates[-1][1]
        else:
            while self.epoch_learning_rates[self.current_idx+1][0] < epoch:
                self.current_idx += 1
                if len(self.epoch_learning_rates) == self.current_idx+1:
                    return self.epoch_learning_rates[-1][1]
            next_key_ep = self.epoch_learning_rates[self.current_idx+1][0]
            last_key_ep = self.epoch_learning_rates[self.current_idx][0]
            interp_factor = (epoch - last_key_ep) / (next_key_ep - last_key_ep)
            next_key_lr = self.epoch_learning_rates[self.current_idx+1][1]
            last_key_lr = self.epoch_learning_rates[self.current_idx][1]
            k = (next_key_lr-last_key_lr)
            return last_key_lr + k*interp_factor


    def num_epochs(self):
        return self.epoch_learning_rates[-1][0]

    @staticmethod
    def check_argument(arg):
        cur_epoch = -1
        assert(arg[0][0] == 0)
        for epoch, _ in arg:
            assert(epoch > cur_epoch)
            cur_epoch = epoch

    @staticmethod
    def deserialize(dic):
        return LearningRateSchedule(dic)

    def serialize(self):
        return self.epoch_learning_rates

class OutputStepSchedule():
    def __init__(self, epoch_and_active_steps):
        # input [(epoch, [step_min_out, step_max_out])]
        # assert epoch and active steps is sored w.r.t epoch
        self.epoch_and_active_steps = epoch_and_active_steps
        self.cur_step = 0

    def get_steps_for_epoch(self, epoch):
        while self.cur_step < len(self.epoch_and_active_steps)-1 and epoch >= self.epoch_and_active_steps[self.cur_step+1][0]:
            self.cur_step += 1
        do_not_modify_until, min_step, max_step = self.epoch_and_active_steps[self.cur_step][1]
        output_steps = list(range(min_step, max_step+1))
        return output_steps, do_not_modify_until

    def serialize(self):
        return self.epoch_and_active_steps

    @staticmethod
    def deserialize(dic):
        return OutputStepSchedule(dic)

def deserialize_pretrain(dic):
    if dic == False:
        return NoPretrain()
    elif dic == True:
        return ImageNetPretrain()
    elif dic['type'] == 'NoPretrain':
        return NoPretrain.deserialize(dic)
    elif dic['type'] == 'ImageNetPretrain':
        return ImageNetPretrain.deserialize(dic)
    elif dic['type'] == 'LoadLocalPretrainWeights':
        return LoadLocalPretrainWeights.deserialize(dic)
    else:
        raise Exception("Failed to parse pretrain settings")

class NoPretrain():
    def __init__(self):
        pass

    def serialize(self):
        return {'type': "NoPretrain"}

    @staticmethod
    def deserialize(dic):
        return NoPretrain()

class ImageNetPretrain():
    def __init__(self):
        pass

    def serialize(self):
        return {'type': "ImageNetPretrain"}

    @staticmethod
    def deserialize(dic):
        return ImageNetPretrain()

class LoadLocalPretrainWeights():
    def __init__(self, training_type, training_name, epoch):
        self.training_type = training_type
        self.training_name = training_name
        self.epoch = epoch

    def serialize(self):
        return {'type': 'LoadLocalPretrainWeights',
                'training_type': self.training_type,
                'training_name': self.training_name,
                'epoch': self.epoch}

    @staticmethod
    def deserialize(dic):
        return LoadLocalPretrainWeights(dic['training_type'], dic['training_name'], dic['epoch'])
