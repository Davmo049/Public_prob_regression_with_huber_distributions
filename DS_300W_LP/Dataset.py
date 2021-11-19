import os
import shutil
import json

import numpy as np
import cv2
import scipy.io
import tqdm

import ImageBuffer.ImageBuffer as ImageBuffer
from ImageTools import ImageTools as ImageTools

import general_utils.environment_variables
from ConfigParser.ConfigParser import SplitType

def mirror_idx_68():
    remap = {}
    # chin
    for i in range(17):
        remap[i] = 16-i
    # eyebrows
    remap[17] = 26
    remap[18] = 25
    remap[19] = 24
    remap[20] = 23
    remap[21] = 22
    # right -> left
    for i in range(17, 22):
        remap[remap[i]] = i
    #straight nose
    remap[27] = 27
    remap[28] = 28
    remap[29] = 29
    remap[30] = 30
    # bottom nose
    remap[31] = 35
    remap[32] = 34
    remap[33] = 33
    remap[34] = 32
    remap[35] = 31

    # eyes
    remap[36] = 45
    remap[37] = 44
    remap[38] = 43
    remap[39] = 42
    remap[40] = 47
    remap[41] = 46
    for i in range(36, 42):
        remap[remap[i]] = i

    # mouth
    # outer upper lip
    remap[48] = 54
    remap[49] = 53
    remap[50] = 52
    remap[51] = 51
    remap[52] = 50
    remap[53] = 49
    remap[54] = 48

    # outer lower lip
    remap[55] = 59
    remap[56] = 58
    remap[57] = 57
    remap[58] = 56
    remap[59] = 55

    # inner upper lip
    remap[60] = 64
    remap[61] = 63
    remap[62] = 62
    remap[63] = 61
    remap[64] = 60

    # inner lower lip
    remap[65] = 67
    remap[66] = 66
    remap[67] = 65

    ret = []
    for i in range(68):
        ret.append(remap[i])
    return ret

FLIP_ORDER = np.array(mirror_idx_68())

class Dataset_300W_LP_Raw():
    DATASET_SUBDIRS = ['AFW', 'HELEN', 'IBUG', 'LFPW']
    def __init__(self, dataset_dir=None):
        if dataset_dir is None:
            dataset_dir = general_utils.environment_variables.get_dataset_dir()
        file_dir = os.path.join(dataset_dir, 'facedatasets', '300W_LP')
        filepaths = self.get_files(file_dir, self.DATASET_SUBDIRS)
        self.file_dir = file_dir
        self.filepaths = filepaths

    def __getitem__(self, idx):
        filepath = os.path.join(self.file_dir, self.filepaths[idx])
        im_path = filepath + '.jpg'
        ann_path = filepath + '.mat'
        with open(im_path, 'rb') as f:
            im = cv2.imdecode(np.array(f.read(), dtype=np.uint8))
            im = im[:,:,::-1]
        ann_mat = scipy.io.loadmat(ann_path)
        points = ann_mat.points
        return im, points

    def preprocess_dataset(self, data_dir=None):
        if data_dir is None:
            dataset_dir = general_utils.environment_variables.get_dataset_dir()
        preproc_dir = os.path.join(dataset_dir, 'preproc_300W_LP')
        os.mkdir(preproc_dir)
        image_dir = os.path.join(preproc_dir, 'images')
        ann_dir = os.path.join(preproc_dir, 'annotations')
        splits_path = os.path.join(preproc_dir, 'splits.txt')
        os.mkdir(image_dir)
        os.mkdir(ann_dir)
        for i, filepath in tqdm.tqdm(enumerate(self.filepaths), total=len(self.filepaths)):
            im_out_path = os.path.join(image_dir, str(i) + '.jpg')
            im_in_path = os.path.join(self.file_dir, filepath + '.jpg')
            ann_out_path = os.path.join(ann_dir, str(i) + '.json')
            ann_in_path = os.path.join(self.file_dir, 'landmarks', filepath + '_pts.mat')
            shutil.copyfile(im_in_path, im_out_path)
            ann_mat = scipy.io.loadmat(ann_in_path)
            points = ann_mat['pts_3d']
            assert(len(points.shape) == 2)
            assert(points.shape[0] == 68)
            assert(points.shape[1] == 2)
            with open(ann_out_path, 'w') as f:
                dic = {'points': np_serialize(points.astype(np.float))}
                json.dump(dic, f)
        all_idx = np.arange(len(self.filepaths))
        eval_idx = all_idx[::4]
        train_idx = sorted(set(all_idx)-set(eval_idx))
        dump_splits(splits_path, (train_idx, eval_idx, all_idx))

    @staticmethod
    def get_files(base_dir, subdirs):
        files = []
        for subdir in subdirs:
            subdir_path = os.path.join(base_dir, subdir)
            for filename in os.listdir(subdir_path):
                if filename[-4:] =='.jpg':
                    files.append(os.path.join(subdir, filename[:-4]))
        return files

# TODO duplicated
def np_serialize(v):
    ret = []
    for i in range(len(v)):
        if isinstance(v[i], np.ndarray):
            ret.append(np_serialize(v[i]))
        else:
            ret.append(v[i])
    return ret

def np_deserialize(v):
    return np.array(v)

# TODO duplicated
def dump_splits(savefile, splits):
    with open(savefile, 'w') as f:
        for s in splits:
            f.write('{}\n'.format(' '.join(map(str,s))))

def load_splits(loadfile):
    ret = []
    with open(loadfile, 'r') as f:
        l = f.readline()
        while len(l) > 1:
            while l[-1] in {'\t', ' ', '\n', '\r'}:
                l = l[:-1]
            idxs = list(map(int, l.split(' ')))
            ret.append(idxs)
            l = f.readline()
    return ret


class PreprocessingInstance():
    def __init__(self, image_size, flip=False, rotation=0.0, scales=np.array([1.0, 1.0]), scale_angle=0.0,  translation_pre_perspective=np.array([0.0,0.0]), perspective_strength=0.0, perspective_angle=0.0, translation_post_perspective=np.array([0.0,0.0]), make_distances_consistent=False):
        self.image_size=image_size
        self.flip=flip
        self.rotation=rotation
        self.scales=scales
        self.scale_angle=scale_angle
        self.translation_pre_perspective=translation_pre_perspective
        self.perspective_strength = perspective_strength
        self.perspective_angle = perspective_angle
        self.translation_post_perspective = translation_post_perspective
        self.make_consistent = make_distances_consistent

    def get_coord_transform(self, image_in_shape):
        transforms = []
        circ_center, circ_radius = np.array((225.0, 225.0)), 175
        if self.flip:
             transforms.append(ImageTools.fliplr_as_affine(image_in_shape))
             circ_center[0] = image_in_shape[1]-circ_center[0]

        # move center of sphere to origin
        transforms.append(ImageTools.translation_as_affine(-circ_center))

        # random rotation
        transforms.append(ImageTools.rotate_as_affine(self.rotation))

        # scale_to_radius
        scale_post = np.array([1.0, 1.0])*(self.image_size/(2*circ_radius))
        transforms.append(ImageTools.scale_as_affine(0, scale_post))

        # random scaling
        transforms.append(ImageTools.scale_as_affine(self.scale_angle, self.scales))

        # random translation
        transforms.append(ImageTools.translation_as_affine(self.translation_pre_perspective))

        # random perspective
        rand_proj = ImageTools.perspective_as_affine(self.perspective_angle, self.perspective_strength)
        transforms.append(rand_proj)

        # random translation
        transforms.append(ImageTools.translation_as_affine(self.translation_post_perspective))

        # move center to middle of new image
        trans = np.array([self.image_size/2, self.image_size/2])
        transforms.append(ImageTools.translation_as_affine(trans))

        # finish
        transform = ImageTools.stack_affine_transforms(transforms)
        return transform

    def inverse_map(self, image_size, points, half_precs, target_map=None):
        transform = self.get_coord_transform(image_size)
        flip = self.flip
        remapping = np.linalg.inv(transform.A)
        if target_map is not None:
            transform_target = target_map.get_coord_transform(image_size)
            remapping = np.matmul(transform_target.A, remapping)
            flip = flip != target_map.flip # xor
        if flip:
            points = points[FLIP_ORDER]
            half_precs = half_precs[FLIP_ORDER]
        remapping = ImageTools.NpAffineTransforms(remapping)
        coords_new = remapping(points.transpose()).transpose()
        half_precs_new = []
        for coord, half_prec in zip(points, half_precs):
            jac = remapping.Jacobian(coord)
            var = np.linalg.inv(np.matmul(half_prec, half_prec))
            var_new = np.matmul(jac, var, jac.transpose())
            half_prec_new = scipy.linalg.sqrtm(np.linalg.inv(var_new)).real
            half_precs_new.append(half_prec_new)
        return coords_new, np.array(half_precs_new)

    def apply(self, image, points):
        image_shape = image.shape
        points = np.copy(points)
        transform = self.get_coord_transform(image_shape)
        if self.flip:
            points = points[FLIP_ORDER]

        net_in_image = ImageTools.np_warp_im(image, transform, (self.image_size, self.image_size))
        points_post_transform = transform(points.transpose()).transpose()

        keypoint_type = np.ones(68).astype(np.float32)
        return net_in_image, points_post_transform, keypoint_type

    @staticmethod
    def randomly_generate_augmentation(image_size=224, max_rot=30,max_first_rescale=1.2, max_relative_rescale=1.2):
        flip = np.random.randint(2) == 1
        max_translation_pre = 0.05
        max_translation_post = 0.05
        max_perspective_strength = 0.3
        use_less_augmentation = False
        if use_less_augmentation: # use less augmentations
            max_rot = max_rot/2
            max_first_rescale = 1.1
            max_relative_rescale = 1.1
            max_translation_pre /= 2
            max_translation_post /= 2
            max_perspective_strength /= 2

        rot_angle = (np.random.uniform(-1, 1))*(max_rot*(np.pi/180))

        f1 = np.log2(max_first_rescale)
        f2 = np.log2(max_relative_rescale)
        s1 = 2**np.random.uniform(-f1, f1)
        s2 = 2**np.random.uniform(-f2, f2)
        scale_angle = np.random.uniform(-1, 1)*np.pi
        scales = np.array([s1/s2, s1*s2])

        translation_pre_perspective = np.random.uniform(-1, 1, size=(2))*image_size*max_translation_pre

        perspective_angle = np.random.uniform(-1, 1)*np.pi
        perspective_strength = np.random.uniform(0,1)*max_perspective_strength/image_size

        # random translation
        translation_post_perspective = np.random.uniform(-1, 1, size=(2))*image_size*max_translation_post

        return PreprocessingInstance(image_size, flip=flip, rotation=rot_angle, scales=scales, scale_angle=scale_angle,  translation_pre_perspective=translation_pre_perspective, perspective_strength=perspective_strength, perspective_angle=perspective_angle, translation_post_perspective=translation_post_perspective)

    @staticmethod
    def get_non_augmented_preprocessor(image_size):
        return PreprocessingInstance(image_size)


class KeypointsDataset():
    def __init__(self, image_buffer, annotation_dir, indices, augment=False,image_size=224):
        self.indices = indices
        self.ann_dir = annotation_dir
        self.image_buffer = image_buffer
        self.image_size = image_size
        self.augment = augment
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.augment:
            preprocess = PreprocessingInstance.randomly_generate_augmentation(self.image_size)
        else:
            preprocess = PreprocessingInstance.get_non_augmented_preprocessor(self.image_size)
        return self.getitem_with_provided_preprocessor(idx, preprocess)

    def getitem_with_provided_preprocessor(self, idx, preprocess):
        index = self.indices[idx]

        ann_path = os.path.join(self.ann_dir, '{}.json'.format(index))

        full_im = self.image_buffer[index]
        with open(ann_path, 'r') as f:
            ann = json.load(f)
            points = np.array(ann['points'])-0.5
        full_im = full_im.astype(np.float)/255
        im, points, keypoint_type = preprocess.apply(full_im, points)
        return im.transpose(2,0,1).astype(np.float32), points.astype(np.float32), keypoint_type, np.ones(68).astype(np.float32)


class Preprocessed300WLPKeypointsDataset():
    def __init__(self, split=SplitType.DEVELOP, data_dir=None):
        if data_dir is None:
            data_dir = general_utils.environment_variables.get_dataset_dir()
        preproc_dir = os.path.join(data_dir, 'preproc_300W_LP')
        image_dir = os.path.join(preproc_dir, 'images')
        self.train_indices, self.eval_indices = self.load_splits(preproc_dir, split)
        all_indices = self.train_indices + self.eval_indices
        filenames = map(lambda x: os.path.join(image_dir, str(x) + '.jpg'), all_indices)
        self.image_buffer = ImageBuffer.ImageBuffer(filenames, all_indices)
        self.annotation_dir = os.path.join(preproc_dir, 'annotations')

    def get_train(self):
        return KeypointsDataset(self.image_buffer, self.annotation_dir, self.train_indices, augment=True)


    def get_eval(self):
        return KeypointsDataset(self.image_buffer, self.annotation_dir, self.eval_indices, augment=False)

    @staticmethod
    def load_splits(directory, split):
        # TODO duplicated code
        t_t, t_v, v = load_splits(os.path.join(directory, 'splits.txt'))
        if split == SplitType.DEVELOP:
            return t_t, t_v
        elif split == SplitType.EVAL:
            return t_t, t_v # No real evaluation set
        elif split == SplitType.DEPLOY:
            return sorted(t_t+t_v), []
        elif split == SplitType.MINI:
            train = t_t[::len(t_t)//(32)]
            val = t_v[::len(t_v)//(32)]
            return train, val
        else:
            raise Exception("UNKNOWN SPLIT {}".format(split))


