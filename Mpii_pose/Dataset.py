import shutil
import json
import os

import torch
import scipy.io
import numpy as np
import PIL.Image
from ConfigParser.ConfigParser import SplitType
from torch.utils.data import Dataset

import cv2

import ImageTools.ImageTools as ImageTools
import general_utils.environment_variables

# (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 10 - r wrist, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)

def MPII_mirror_idx():
    remap = {}
    remap[0] = 5
    remap[5] = 0
    remap[1] = 4
    remap[4] = 1
    remap[2] = 3
    remap[3] = 2
    remap[6] = 6
    remap[7] = 7
    remap[8] = 8
    remap[9]= 9
    remap[10] = 15
    remap[15] = 10
    remap[11] = 14
    remap[14] = 11
    remap[12] = 13
    remap[13] = 12
    ret = []
    for i in range(16):
        ret.append(remap[i])
    return ret

FLIP_ORDER = np.array(MPII_mirror_idx())
    

class ParseException(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class RawDataset():
    def __init__(self, dataset_dir=None):
        if dataset_dir is None:
            dataset_dir = general_utils.environment_variables.get_dataset_dir()
        mpii_pose_dir = os.path.join(dataset_dir, 'mpii_human_pose')
        self.image_dir = os.path.join(mpii_pose_dir, 'images')
        ann_file = os.path.join(mpii_pose_dir, 'ann', 'mpii_human_pose_v1_u12_1.mat')
        self.image_annos_train, self.image_annos_test = preprocess_ann(ann_file)

    def __getitem__(self, idx):
        im_ann = self.image_annos_train[idx]
        image_name = os.path.join(self.image_dir, im_ann['image_name'])
        impil = PIL.Image.open(image_name)
        im = np.array(impil.getdata()).reshape(impil.size[1], impil.size[0], 3)
        return im, im_ann['samples']

    def __len__(self):
        return len(self.image_annos_train)

    def preprocess_to_dir(dst_dir):
        pass


def preprocess_ann(filename):
    data = scipy.io.loadmat(filename)
    #.reshape(-1)[0]['annopoints'][0,0]['point'].reshape(-1)[0]['id'][0,0])
    data = data['RELEASE'][0,0]
    single_person = extract_single_person(data)
    image_train = extract_image_train(data)
    annolist_train, annolist_test = split_annolist_on_train(data['annolist'].reshape(-1),  image_train)
    sp_train, sp_test = split_single_person_on_train(single_person,  image_train)
    train_data = parse_annolist_train(annolist_train, sp_train)
    test_data = parse_annolist_test(annolist_test, sp_test)
    return train_data, test_data

def extract_single_person(data):
    cand = data['single_person'].reshape(-1)
    cand = list(map(lambda x: list(x.reshape(-1)-1), cand)) # each index has list of annotation indices, change 1 indexing to 0 indexing
    for image_data in cand:
        for l in image_data:
            if not isinstance(l, np.uint8):
                print(l)
                print(type(l))
                assert(False)
    return cand


def split_single_person_on_train(single_person,  image_train):
    ret_train = []
    ret_test = []
    assert(len(image_train) == len(single_person))
    for d, b in zip(single_person, image_train):
        if b == 1:
            ret_train.append(d)
        else:
            ret_test.append(d)
    return ret_train, ret_test


def extract_image_train(data):
    cand = data['img_train'].reshape(-1) # boolean vector of "is training"
    for x in cand:
        assert(isinstance(x, np.uint8))
    return cand

def split_annolist_on_train(data,  image_train):
    ret_train = []
    ret_test = []
    assert(len(image_train) == len(data))
    for d, b in zip(data, image_train):
        if b == 1:
            ret_train.append(d)
        else:
            print(d['annorect'].dtype.names)
            for x in d['annorect'].reshape(-1):
                if x is None:
                    print(d)
                    print(d.dtype.names)
                    print(d['annorect'])
                    print(d['image'])
            ret_test.append(d)
    return ret_train, ret_test

def parse_annolist_train(annolist, single_person_for_recs):
    ret = []
    for x in annolist:
        assert(x.shape == ())
    fails_train = 0
    for image_data, sp in zip(annolist, single_person_for_recs):
        image_name = image_data['image']['name'][0,0][0]
        assert(isinstance(image_name, str))
        samples_in_image, fails = parse_samples_train(image_data, sp)
        fails_train += fails
        ret.append({'samples': samples_in_image, 'image_name':image_name})
    print('failed to parse {} train samples'.format(fails_train))
    return ret


def parse_annolist_test(annolist, single_person_for_recs):
    ret = []
    for x in annolist:
        assert(x.shape == ())
    fails_test = 0
    for image_data, sp in zip(annolist, single_person_for_recs):
        image_name = image_data['image']['name'][0,0][0]
        assert(isinstance(image_name, str))
        samples_in_image, fails = parse_samples_test(image_data, sp)
        fails_test += fails
        ret.append({'samples': samples_in_image, 'image_name':image_name})
    print('failed to parse {} test samples'.format(fails_test))
    return ret

def parse_samples_train(data, sp):
    annorects = data['annorect'].reshape(-1)
    ret = []
    fails = 0
    for i, annorect in enumerate(annorects):
        try:
            scale = parse_scale(annorect)
            x,y = parse_objpos(annorect)
            joints = parse_joints(annorect)
            single_person = i in sp
            ret.append({'x_box': x, 'y_box': y, 'box_scale':scale, 'joints':joints, 'single_person': single_person})
        except ParseException:
            print(annorects)
            print(annorect)
            fails += 1
            continue
    return ret, fails

def parse_samples_test(image_data, sp):
    annorects = image_data['annorect'].reshape(-1)
    ret = []
    fails = 0
    for i, annorect in enumerate(annorects):
        if annorect is None:
            print('annorect is none')
            print(annorects)
            print(i)
            print(annorect)
            fails += 1
            continue
        try:
            scale = parse_scale(annorect)
            x,y = parse_objpos(annorect)
            single_person = i in sp
            ret.append({'x_box': x, 'y_box': y, 'box_scale':scale, 'single_person': single_person})
        except ParseException:
            print(sp,'x',len(annorects))
            print(annorects)
            print(annorect)
            fails += 1
            continue
    return ret, fails


def parse_scale(data):
    if 'scale' not in data.dtype.names:
        print('parse scale fail, no scale in data')
        print(data.shape)
        print(data.dtype.names)
        print('scale' in data.dtype.names)
        raise ParseException('failed parse')
    if len(data['scale'].reshape(-1)) == 0:
        print('parse scale fail')
        print(data['scale'].shape)
        raise ParseException('failed parse')
    scale = float(data['scale'][0,0])
    return float(scale)

def parse_objpos(data):
    objpos = data['objpos']
    x = objpos['x'][0,0][0,0]
    y = objpos['y'][0,0][0,0]
    assert(isinstance(x, np.uint16) or isinstance(x, np.uint8) or isinstance(x, np.int16) or isinstance(x, np.int8))
    assert(isinstance(y, np.uint16) or isinstance(y, np.uint8) or isinstance(y, np.int16) or isinstance(y, np.int8))
    x = int(x)
    y = int(y)
    return x, y

def parse_joints(data):
    joint_data = data['annopoints'].reshape(-1)
    assert(len(joint_data) == 1)
    joint_data = joint_data[0]

    x = joint_data['point']['x'].reshape(-1)
    y = joint_data['point']['y'].reshape(-1)
    idd = joint_data['point']['id'].reshape(-1)
    x = list(map(lambda x: float(x[0,0]), x))
    y = list(map(lambda x: float(x[0,0]), y))
    idd = list(map(lambda x: int(x[0,0]), idd))

    if 'is_visible' not in joint_data['point'].dtype.names:
        is_visible = [0 for _ in range(len(x))]
    else:
        is_visible = joint_data['point']['is_visible'].reshape(-1)
        is_visible_l = []
        for v in is_visible:
            if len(v) == 0:
                is_visible_l.append(0)
            else:
                is_visible_l.append(int(v))
        is_visible = is_visible_l
    joints = create_joint_vector(x, y, is_visible, idd)
    return joints

def create_joint_vector(x, y, is_visible, idd):
    N_at_idx = np.zeros((16))
    pos = np.zeros((16, 2))
    vis = np.zeros((16))
    for xi, yi, ivi, idi in zip(x,y, is_visible, idd):
        pos[idi] += np.array([xi, yi])
        vis[idi] += ivi
        N_at_idx[idi] += 1
    weights = (N_at_idx > 0).astype(np.float)
    mask = (N_at_idx > 0)
    pos[mask] = pos[mask] / N_at_idx[mask].reshape(-1, 1)
    return {'pos': pos, 'weights': weights, 'is_visible': vis}


def extractannopoint(anno):
    xs = list(map(lambda x: x['x'][0,0], anno['annopoints'][0,0]['point'].reshape(-1)))
    ys = list(map(lambda x: x['y'][0,0], anno['annopoints'][0,0]['point'].reshape(-1)))
    ids = list(map(lambda x: x['id'][0,0], anno['annopoints'][0,0]['point'].reshape(-1)))
    pos = np.zeros((16, 2))
    N_at_idx = np.zeros((16), dtype=np.int)
    for x, y, idd in zip(xs, ys, ids):
        pos[idd] += np.array([x,y])
        N_at_idx[idd] += 1
    mask = N_at_idx > 0
    weights = mask.astype(np.float32)
    pos[mask] = pos[mask] / N_at_idx[mask].reshape(-1, 1)
    return pos, weights

def preprocess_dataset(dataset_in, dataset_dir_out):
    train_samples = preprocess_train(dataset_in, dataset_dir_out)
    test_samples = preprocess_test(dataset_in, dataset_dir_out)
    count_file = os.path.join(dataset_dir_out, 'num_samples.txt')
    with open(count_file, 'w') as f:
        f.write(str(train_samples)+'\n')
        f.write(str(test_samples))

def preprocess_train(dataset_in, dataset_dir_out):
    i = 0
    os.makedirs(dataset_dir_out)
    image_dir_out = os.path.join(dataset_dir_out, 'images')
    os.makedirs(image_dir_out)
    annotation_dir_out = os.path.join(dataset_dir_out, 'annotations')
    os.makedirs(annotation_dir_out)
    for image_data in dataset_in.image_annos_train:
        image_name_in = os.path.join(dataset_in.image_dir, image_data['image_name'])
        samples_in_image = image_data['samples']
        for s in samples_in_image:
            image_name_out = os.path.join(image_dir_out, str(i)+'.jpg')
            shutil.copyfile(image_name_in, image_name_out)
            center = [s['x_box']-1, s['y_box']-1]
            assert(isinstance(center, list))
            assert(isinstance(center[0], int))
            assert(isinstance(center[1], int))
            radius = s['box_scale']*100
            assert(isinstance(radius, float))
            pos = list(map(lambda x: [x[0], x[1]], s['joints']['pos']))
            assert(isinstance(pos, list))
            for x in pos:
                assert(isinstance(x[0], float))
                assert(isinstance(x[1], float))
            weights = list(s['joints']['weights'])
            assert(isinstance(weights, list))
            for x in weights:
                assert(isinstance(x, float))
            sample = {'center': center, 'radius': radius, 'points': pos, 'weights': weights, 'single_person': s['single_person']}
            image_name_out = os.path.join(annotation_dir_out, str(i)+'.json')
            with open(image_name_out, 'w') as f:
                json.dump(sample, f)
            i += 1
    return i

def preprocess_test(dataset_in, dataset_dir_out):
    i = 0
    image_dir_out = os.path.join(dataset_dir_out, 'test_images')
    os.makedirs(image_dir_out)
    annotation_dir_out = os.path.join(dataset_dir_out, 'test_annotations')
    os.makedirs(annotation_dir_out)
    for image_data in dataset_in.image_annos_test:
        image_name_in = os.path.join(dataset_in.image_dir, image_data['image_name'])
        samples_in_image = image_data['samples']
        for s in samples_in_image:
            image_name_out = os.path.join(image_dir_out, str(i)+'jpg')
            shutil.copyfile(image_name_in, image_name_out)
            center = [s['x_box']-1, s['y_box']-1]
            assert(isinstance(center, list))
            assert(isinstance(center[0], int))
            assert(isinstance(center[1], int))
            radius = s['box_scale']*100
            assert(isinstance(radius, float))
            sample = {'center': center, 'radius': radius, 'image_orig': image_data['image_name']}
            with open(image_name_out, 'w') as f:
                json.dump(sample, f)
            i += 1
    return i


class PreprocessedMpiiDataset():
    def __init__(self, split=SplitType.DEVELOP, data_dir=None):
        if data_dir is None:
            data_dir = general_utils.environment_variables.get_dataset_dir()
        preproc_dir = os.path.join(data_dir, 'mpii_pose_preprocessed')
        self.train_image_dir = os.path.join(preproc_dir, 'images')
        self.train_annotation_dir = os.path.join(preproc_dir, 'annotations')
        self.train_idx, self.eval_idx, self.test_idx, use_test_dir = self.load_splits(split)
        if use_test_dir:
            self.test_image_dir = os.path.join(preproc_dir, 'test_images')
            self.test_annotation_dir = os.path.join(preproc_dir, 'test_annotations')
        else:
            self.test_image_dir = self.train_image_dir
            self.test_annotation_dir = self.train_annotation_dir

    def get_train(self):
        return KeypointsDataset(self.train_image_dir, self.train_annotation_dir, self.train_idx, augment=True, load_joints=True)

    def get_eval(self):
        return KeypointsDataset(self.test_image_dir, self.test_annotation_dir, self.eval_idx, augment=False, load_joints=True)

    def get_test(self):
        return KeypointsDataset(self.test_image_dir,  self.test_annotation_dir, self.test_idx, augment=False, load_joints=False)

    @staticmethod
    def load_splits(split):
        n_train_samples = 28883
        n_test_samples = 11731
        if split == SplitType.DEVELOP or split == SplitType.EVAL:
            all_samples = set(range(n_train_samples))
            train_idxs = list(map(int, np.arange(n_train_samples*0.7)/0.7))
            eval_idxs = sorted(list(all_samples - set(train_idxs)))
            return train_idxs, eval_idxs, eval_idxs, False
        elif split == SplitType.DEPLOY:
            return list(range(n_train_samples)), [], list(range(n_test_samples)), True
        elif split == SplitType.MINI:
            train = list(range(128))
            val = list(range(128, 256))
            return train, val, val, False
        else:
            raise Exception("UNKNOWN SPLIT {}".format(split))


class KeypointsDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, indices, augment=False, load_joints=False):
        self.image_size = 224
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.indices = indices
        self.augment = augment
        self.load_joints = load_joints
        self.eval_preprocessors = self.create_eval_preprocessors(self.image_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        im_path = os.path.join(self.image_dir, str(index)+'.jpg')
        ann_path = os.path.join(self.annotation_dir, str(index)+'.json')

        with open(im_path, 'rb') as f:
            data = f.read()
        buf = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        im = im[:,:,::-1]
        im = im.astype(np.float)/255

        with open(ann_path, 'r') as f:
            ann = json.load(f)
        center = np.array(ann['center']).astype(np.float) -1.0 # matlab indexing
        radius = ann['radius']
        if self.load_joints:
            points = np.array(ann['points']) - 1.0 # matlab indexing
            weights = np.array(ann['weights'])
            single_person = ann['single_person']
        else:
            points = np.zeros((16, 2))
            weights = np.zeros((16))
            single_person = False

        if self.augment:
            preprocess = PreprocessingInstance.randomly_generate_augmentation(self.image_size, not single_person)
        else:
            preprocess = PreprocessingInstance.get_non_augmented_preprocessor(self.image_size)

        im_in, points, weights = preprocess.apply(im, points, weights, center, radius)
        return im_in.transpose(2,0,1).astype(np.float32), points.astype(np.float32), weights.astype(np.int), weights

    def create_eval_batch(self, idx):
        index = self.indices[idx]
        im_path = os.path.join(self.image_dir, str(index)+'.jpg')
        ann_path = os.path.join(self.annotation_dir, str(index)+'.json')

        with open(im_path, 'rb') as f:
            data = f.read()
        buf = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        im = im[:,:,::-1]
        full_im = im.astype(np.float)/255

        with open(ann_path, 'r') as f:
            ann = json.load(f)
        center = np.array(ann['center']).astype(np.float) -1.0 # matlab indexing
        radius = ann['radius']

        if self.load_joints:
            points = np.array(ann['points']) - 1.0 # matlab indexing
            weight_ann = np.array(ann['weights'])
        else:
            points = np.zeros((16, 2))
            weight_ann = np.zeros((16))

        ims = np.empty((len(self.eval_preprocessors), self.image_size, self.image_size, 3))
        keypoints = np.zeros((len(self.eval_preprocessors), 16, 2))
        weights = np.zeros((len(self.eval_preprocessors), 16))

        for preproc_idx, preprocess in enumerate(self.eval_preprocessors):
            im, keypoints_post_transform, weight = preprocess.apply(full_im, points, weight_ann, center, radius)
            ims[preproc_idx] = im
            keypoints[preproc_idx] = keypoints_post_transform
            weights[preproc_idx] = weight

        bounding_circle = (center, radius)
        return ims.transpose(0, 3,1,2).astype(np.float32), keypoints.astype(np.float32), weights.astype(np.int), full_im.shape, bounding_circle, ann

    @staticmethod
    def create_eval_preprocessors(image_size):
        augs = []
        augs.append(PreprocessingInstance(image_size))
        augs.append(PreprocessingInstance(image_size, flip=True))
        return augs

    def convert_back_eval_preds_to_unaug(self, image_size, modes, precs, bounding_circle):
        T0 = self.eval_preprocessors[0]
        ret_modes = []
        ret_precs = []
        for mode, prec, preproc in zip(modes, precs, self.eval_preprocessors):
            m, prec_ret = preproc.inverse_map(image_size, bounding_circle, mode, prec, target_map=T0)
            ret_modes.append(m)
            ret_precs.append(prec_ret)
        return np.array(ret_modes), np.array(ret_precs)

class PreprocessingInstance():
    def __init__(self, image_size, flip=False, rotation=0.0, scales=np.array([1.0, 1.0]), scale_angle=0.0,  translation_pre_perspective=np.array([0.0,0.0]), perspective_strength=0.0, perspective_angle=0.0, translation_post_perspective=np.array([0.0,0.0])):
        self.image_size=image_size
        self.flip=flip
        self.rotation=rotation
        self.scales=scales
        self.scale_angle=scale_angle
        self.translation_pre_perspective=translation_pre_perspective
        self.perspective_strength = perspective_strength
        self.perspective_angle = perspective_angle
        self.translation_post_perspective = translation_post_perspective

    def get_coord_transform(self, image_in_shape, bounding_circle):
        transforms = []
        circ_center, circ_radius = bounding_circle
        circ_center = np.copy(circ_center)
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

    def inverse_map(self, image_size, bounding_circle, points, precs, target_map=None):
        transform = self.get_coord_transform(image_size, bounding_circle)
        flip = self.flip
        remapping = np.linalg.inv(transform.A)
        if target_map is not None:
            transform_target = target_map.get_coord_transform(image_size, bounding_circle)
            remapping = np.matmul(transform_target.A, remapping)
            flip = flip != target_map.flip # xor
        if flip:
            points = points[FLIP_ORDER]
            precs = precs[FLIP_ORDER]
        remapping = ImageTools.NpAffineTransforms(remapping)
        coords_new = remapping(points.transpose()).transpose()
        precs_new = []
        for coord, prec in zip(points, precs):
            jac = remapping.Jacobian(coord)
            var = np.linalg.inv(prec)
            var_new = np.matmul(jac, np.matmul(var, jac.transpose()))
            prec_new = np.linalg.inv(var_new)
            precs_new.append(prec_new)
        return coords_new, np.array(precs_new)

    def apply(self, image, points, weights, center, radius):
        image_shape = image.shape
        points = np.copy(points)
        bounding_circle = (center, radius)
        transform = self.get_coord_transform(image_shape, bounding_circle)
        if self.flip:
            points = points[FLIP_ORDER]
            weights = weights[FLIP_ORDER]

        net_in_image = ImageTools.np_warp_im(image, transform, (self.image_size, self.image_size))
        points_post_transform = transform(points.transpose()).transpose()

        return net_in_image, points_post_transform, weights

    @staticmethod
    def randomly_generate_augmentation(image_size=224, max_rot=30,max_first_rescale=1.2, max_relative_rescale=1.2, keep_origin_fix=True):
        flip = np.random.randint(2) == 1
        max_translation_pre = 0.05
        max_translation_post = 0.05
        max_perspective_strength = 0.3
        if keep_origin_fix:
            max_translation_pre = 0.0
            max_translation_post = 0.0


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
