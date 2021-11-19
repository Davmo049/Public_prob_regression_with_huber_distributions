from general_utils.environment_variables import get_dataset_dir
import os
import json
import copy
import torch
import numpy as np
from enum import IntEnum
from PIL import Image

import mathlib
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import Dataset
from ImageTools import ImageTools as ImageTools
import subprocess

from mathlib.covers import minimum_covering_sphere
import matplotlib
import scipy
import scipy.linalg
import ImageBuffer.ImageBuffer as ImageBuffer
from ConfigParser.ConfigParser import SplitType

def WFLW_mirror_idx():
    remap = {}
    # chin
    for i in range(33):
        remap[i] = 32-i
    # eyebrows
    remap[33] = 46
    remap[34] = 45
    remap[35] = 44
    remap[36] = 43
    remap[37] = 42
    remap[38] = 50
    remap[39] = 49
    remap[40] = 48
    remap[41] = 47
    # right -> left
    for i in range(33, 42):
        remap[remap[i]] = i
    # eyes
    remap[60] = 72
    remap[61] = 71
    remap[62] = 70
    remap[63] = 69
    remap[64] = 68
    remap[65] = 75
    remap[66] = 74
    remap[67] = 73
    # right -> left
    for i in range(60, 68):
        remap[remap[i]] = i
    remap[96] = 97
    remap[97] = 96
    # nose
    for i in range(51, 55):
        remap[i] = i
    for i in range(5):
        remap[55+i] = 59-i
    # mouth
    for i in range(82-76+1):
        remap[76+i] = 82-i
    for i in range(87-83+1):
        remap[83+i] = 87-i
    for i in range(92-88+1):
        remap[88+i] = 92-i
    remap[93] = 95
    remap[94] = 94
    remap[95] = 93
    ret = []
    for i in range(98):
        ret.append(remap[i])
    return ret

FLIP_ORDER = np.array(WFLW_mirror_idx())

class WflwAnnotation():
    def __init__(self, points, pose, expression, illumination, makeup, occlusion, blur):
        self.points = points
        self.pose = pose
        self.expression = expression
        self.illumination = illumination
        self.makeup = makeup
        self.occlusion = occlusion
        self.blur = blur

    def serialize(self):
        return {'points': np_serialize(self.points),
                'pose': pose,
                'expression': self.expression,
                'illumination': self.illumination,
                'makeup': self.makeup,
                'occlusion': self.occlusion,
                'blur': self.blur}

    @staticmethod
    def deserialize(dic):
        return WflwAnnotation(np_deserialize(dic['points']), dic['pose'], dic['expression'], dic['illumination'], dic['makeup'], dic['occlusion'], dic['blur'])

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

class RawWflwAnnotation(WflwAnnotation):
    def __init__(self, bounding_box, points, pose, expression, illumination, makeup, occlusion, blur):
        super().__init__(points, pose, expression, illumination, makeup, occlusion, blur)
        self.bounding_box = bounding_box


class PreprocessesedWflwAnnotation(WflwAnnotation):
    def __init__(self, circle, points, pose, expression, illumination, makeup, occlusion, blur):
        super().__init__(points, pose, expression, illumination, makeup, occlusion, blur)
        self.circle_center = circle[0]
        self.circle_radius = circle[1]

    def serialize(self):
        return {'circle': {
                    'center': np_serialize(self.circle_center),
                    'radius': self.circle_radius},
                'points': np_serialize(self.points),
                'pose': self.pose,
                'expression': self.expression,
                'illumination': self.illumination,
                'makeup': self.makeup,
                'occlusion': self.occlusion,
                'blur': self.blur}

    @staticmethod
    def deserialize(dic):
        circle = (np_deserialize(dic['circle']['center']), dic['circle']['radius'])
        return PreprocessesedWflwAnnotation(circle, np_deserialize(dic['points']), dic['pose'], dic['expression'], dic['illumination'], dic['makeup'], dic['occlusion'], dic['blur'])


class RawWflw():
    def __init__(self, train=True, dataset_path=None):
        if dataset_path is None:
            dataset_path = get_dataset_dir()
        wflw_path = os.path.join(dataset_path, 'facedatasets', '300W')
        if train:
            annotation_file = 'list_98pt_rect_attr_train.txt'
        else:
            annotation_file = 'list_98pt_rect_attr_test.txt'
        annotation_filepath = os.path.join(wflw_path, 'WFLW_annotations', 'list_98pt_rect_attr_train_test', annotation_file)
        self.image_dir = os.path.join(wflw_path, 'WFLW_images')
        self.annotations = self.load_annotations(annotation_filepath)

    def __getitem__(self, idx):
        image_name, annotation = self.annotations[idx]
        im_path = os.path.join(self.image_dir, image_name)
        img_PIL = Image.open(im_path)
        data = img_PIL.getdata()
        full_im = np.array(data).reshape(img_PIL.size[1], img_PIL.size[0], 3)
        return full_im, annotation

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def load_annotations(filepath):
        ret = []
        with open(filepath, 'r') as f:
            for l in f.readlines():
                while l[-1] in {'\r', '\n'}:
                    l = l[:-1]
                parts = l.split(' ')
                parts_data = parts[:2*98]
                metadata = parts[2*98:]
                points = np.empty(shape=(98,2))
                for part_idx in range(98):
                    xi = float(parts_data[part_idx*2])
                    yi = float(parts_data[part_idx*2+1])
                    points[part_idx, 0] = xi
                    points[part_idx, 1] = yi
                bbx = np.array([int(metadata[0]), int(metadata[1]), int(metadata[2]), int(metadata[3])]) # xmin, ymin, xmax, ymax
                pose = bool(metadata[4])
                expression = bool(metadata[5])
                illumination = bool(metadata[6])
                makeup = bool(metadata[7])
                occlusion = bool(metadata[8])
                blur = bool(metadata[9])
                image_name = metadata[10]
                annotation = RawWflwAnnotation(bbx, points, pose, expression, illumination, makeup, occlusion, blur)
                ret.append((image_name, annotation))
        return ret

def crop_jpeg(path_src, path_dst, w,h,x,y):
    call = ['jpegtran', '-perfect', '-crop', '{}x{}+{}+{}'.format(w, h,x,y), '-outfile', path_dst, path_src]
    proc = subprocess.Popen(' '.join(call), shell=True)
    proc.wait()

def export_wflw(dataset_dir=None, valtrain_keep=0.3):
    train_split = []
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    wflw_train = RawWflw(train=True, dataset_path=dataset_dir)
    wflw_val = RawWflw(train=False, dataset_path=dataset_dir)
    wflw_preprocessed = os.path.join(dataset_dir, 'WFLW_keypoint_preproc_jpeg')
    os.mkdir(wflw_preprocessed)
    wflw_preprocessed_images = os.path.join(wflw_preprocessed, 'images')
    annotations_preprocessed_dir = os.path.join(wflw_preprocessed, 'annotations')
    os.mkdir(annotations_preprocessed_dir)
    os.mkdir(wflw_preprocessed_images)
    train_samples = preprocess_dataset_jpeg(wflw_train, wflw_preprocessed_images, annotations_preprocessed_dir, offset=0)
    val_samples = preprocess_dataset_jpeg(wflw_val, wflw_preprocessed_images, annotations_preprocessed_dir, offset=train_samples)

    # save splits
    split_train_val = np.floor(np.arange(train_samples*0.3)/0.3).astype(np.int)
    all_train_indices = range(train_samples)
    split_train_train = sorted(list(set(all_train_indices)-set(split_train_val)))
    split_val = np.arange(train_samples, val_samples)
    split_file = os.path.join(wflw_preprocessed, 'splits.txt')
    dump_splits(split_file, (split_train_train, split_train_val, split_val))

def preprocess_dataset_png(dataset, image_save_dir, annotations_save_dir, offset):
    if offset != 0:
        file_should_exist = os.path.join(image_save_dir, '{}.png'.format(offset-1))
        file_should_not_exist = os.path.join(image_save_dir, '{}.png'.format(offset))
        if not os.path.exists(file_should_exist):
            print(' file "{}" does not exist'.format(file_should_exist))
            assert(False)
        if os.path.exists(file_should_not_exist):
            print(' file "{}" does exist, but shouldnt'.format(file_should_not_exist))
            assert(False)
    index = offset

    num_skipped = 0
    for im, ann in tqdm.tqdm(dataset):
        ret = preprocess_raw_sample(im, ann)
        if ret is None:
            num_skipped+=1
            continue
        im, ann = ret
        # visualize_sample(im, ann)
        save_sample(im, ann, image_save_dir, annotations_save_dir, index)
        index += 1
    return index

def preprocess_dataset_jpeg(dataset, image_save_dir, annotations_save_dir, offset):
    if offset != 0:
        file_should_exist = os.path.join(image_save_dir, '{}.jpg'.format(offset-1))
        file_should_not_exist = os.path.join(image_save_dir, '{}.jpg'.format(offset))
        if not os.path.exists(file_should_exist):
            print(' file "{}" does not exist'.format(file_should_exist))
            assert(False)
        if os.path.exists(file_should_not_exist):
            print(' file "{}" does exist, but shouldnt'.format(file_should_not_exist))
            assert(False)
    index = offset

    num_skipped = 0
    for idx in tqdm.tqdm(range(len(dataset))):
        image_name, annotation = dataset.annotations[idx]
        im = Image.open(os.path.join(dataset.image_dir, image_name))
        annotation, (left, right, top, bot) = get_jpeg_crop_coords(im, annotation)
        src_path = os.path.join(dataset.image_dir, image_name)
        dst_path = os.path.join(image_save_dir, str(index)+'.jpg')
        crop_jpeg(src_path, dst_path, right-left,bot-top,left,top)

        im = np.array(im.getdata(), dtype=np.uint8).reshape(im.size[1], im.size[0], 3)
        im_crop = im[top:bot, left:right]
        im_loaded = Image.open(dst_path)
        im_loaded = np.array(im_loaded.getdata(), dtype=np.uint8).reshape(im_loaded.size[1], im_loaded.size[0], 3).astype(np.int)
        assert(np.all(im_loaded==im_crop))

        ann_save_path = os.path.join(annotations_save_dir, str(index)+'.json')
        with open(ann_save_path, 'w') as f:
            json.dump(annotation.serialize(), f)
        index += 1
    return index

def save_sample(im, annotation, im_save_dir, ann_save_dir, index):
    im_path = os.path.join(im_save_dir, str(index)+'.png')
    ann_save_path = os.path.join(ann_save_dir, str(index)+'.json')
    PIL_im = Image.fromarray(im).convert('RGB')
    PIL_im.save(im_path)
    with open(ann_save_path, 'w') as f:
        json.dump(annotation.serialize(), f)


def get_jpeg_crop_coords(im, annotation):
    # aug order: flip, shift, affine, scale, rotation
    oversize_crop = 1.0 # keep 5% of width/height extra compared to segmentation
    rescale_min = 0.8 # "zoom out" at most to 80% of original size
    rescale_max = 1.3 # # zoom in at most to 130% of original size
    shift = 0.1 # shift at most 10% of width/height

    max_affine_zoom_out = 0.4 # zoom in/out at most 40% due to affine transform, center of affine is the welzl median
    max_affine_zoom_in = 0.4 # zoom in/out at most 40% due to affine transform, center of affine is the welzl median
    c = min(1-1/np.sqrt(1+max_affine_zoom_out), 1/np.sqrt(1-max_affine_zoom_in)-1)
    affine_radius_out = 1/(1-c)
    oversize_aug = shift + (1/rescale_min)*affine_radius_out # keep 5% of width/height extra compared to segmentation
    oversize_final = shift + (1/rescale_min)*affine_radius_out*oversize_crop
    middle_x = (annotation.bounding_box[0]+annotation.bounding_box[2])/2
    middle_y = (annotation.bounding_box[1]+annotation.bounding_box[3])/2
    radius = np.sqrt((annotation.bounding_box[0]-middle_x)**2+(annotation.bounding_box[1]-middle_y)**2)
    center = np.array([middle_x, middle_y])
    left = max(0, np.floor(center[0]+0.5-radius*oversize_final).astype(np.int))
    x_quant_size = 8*im.layer[0][2] # id, v(ertical?)samp, h(orizontal?)samp, quantization_table
    y_quant_size = 8*im.layer[0][1]
    left = int((left//x_quant_size)*x_quant_size)
    right = np.ceil(center[0]+0.5+radius*oversize_final).astype(np.int)
    right = min(im.size[0], int(np.ceil(right/x_quant_size)*x_quant_size))
    top = max(0, np.floor(center[1]+0.5-radius*oversize_final).astype(np.int))
    top = int((top//y_quant_size)*y_quant_size)
    bot = np.ceil(center[1]+0.5+radius*oversize_final).astype(np.int)
    bot = min(im.size[1], int(np.ceil(bot/y_quant_size)*y_quant_size))
    center_new = center + 0.5 - np.array((left, top))

    keypoints_translated = annotation.points - np.array((left, top)).reshape(1, 2)
    keypoints_translated += 0.5

    circle_out = (center_new, radius)
    annotation = PreprocessesedWflwAnnotation(circle_out, keypoints_translated, annotation.pose, annotation.expression, annotation.illumination, annotation.makeup, annotation.occlusion, annotation.blur)
    return annotation, (left, right, top, bot)

def preprocess_raw_sample(im, annotation):
    # aug order: flip, shift, affine, scale, rotation
    oversize_crop = 1.0 # keep 5% of width/height extra compared to segmentation
    rescale_min = 0.8 # "zoom out" at most to 80% of original size
    rescale_max = 1.3 # # zoom in at most to 130% of original size
    shift = 0.1 # shift at most 10% of width/height

    max_affine_zoom_out = 0.4 # zoom in/out at most 40% due to affine transform, center of affine is the welzl median
    max_affine_zoom_in = 0.4 # zoom in/out at most 40% due to affine transform, center of affine is the welzl median
    c = min(1-1/np.sqrt(1+max_affine_zoom_out), 1/np.sqrt(1-max_affine_zoom_in)-1)
    affine_radius_out = 1/(1-c)
    oversize_aug = shift + (1/rescale_min)*affine_radius_out # keep 5% of width/height extra compared to segmentation
    oversize_final = shift + (1/rescale_min)*affine_radius_out*oversize_crop
    middle_x = (annotation.bounding_box[0]+annotation.bounding_box[2])/2
    middle_y = (annotation.bounding_box[1]+annotation.bounding_box[3])/2
    radius = np.sqrt((annotation.bounding_box[0]-middle_x)**2+(annotation.bounding_box[1]-middle_y)**2)
    center = np.array([middle_x, middle_y])
    # translate crop
    left = max(0, np.floor(center[0]+0.5-radius*oversize_final).astype(np.int))
    right = min(im.shape[1]-1, np.ceil(center[0]+0.5+radius*oversize_final).astype(np.int))
    top = max(0, np.floor(center[1]+0.5-radius*oversize_final).astype(np.int))
    bot = min(im.shape[0]-1, np.ceil(center[1]+0.5+radius*oversize_final).astype(np.int))
    center_new = center+0.5 - np.array((left, top))
    im_translated = im[top:bot, left:right]

    keypoints_translated = annotation.points - np.array((left, top)).reshape(1, 2)
    keypoints_translated += 0.5

    # scale crop
    downscale_factor = max(1, np.ceil(radius/224).astype(np.int))
    im_scaled = np.zeros((im_translated.shape[0]//downscale_factor, im_translated.shape[1]//downscale_factor, 3), dtype=np.float64)

    prescale_h = downscale_factor*(im_translated.shape[0]//downscale_factor)
    prescale_w = downscale_factor*(im_translated.shape[1]//downscale_factor)
    im_prescaled = im_translated[:prescale_h, :prescale_w, :]
    for r in range(downscale_factor):
        for c in range(downscale_factor):
            im_scaled += im_prescaled[r::downscale_factor, c::downscale_factor,:]
    im_scaled /= downscale_factor**2
    im_scaled = np.round(im_scaled).astype(np.uint8)
    keypoints_scaled = np.copy(keypoints_translated)
    keypoints_scaled /= downscale_factor
    keypoints_scaled -= 0.5

    scaled_center = center_new/downscale_factor-0.5
    scaled_radius = radius/downscale_factor
    circle_out = (scaled_center, scaled_radius)
    annotation = PreprocessesedWflwAnnotation(circle_out, keypoints_scaled, annotation.pose, annotation.expression, annotation.illumination, annotation.makeup, annotation.occlusion, annotation.blur)
    return im_scaled, annotation

def visualize_sample(image, ann):
    print(image[0,0])
    plt.imshow(image)
    x = list(map(lambda x: x[0], ann.points))
    y = list(map(lambda x: x[1], ann.points))
    plt.plot(x,y,'rx')
    import pprint
    pprint.pprint(ann.serialize())
    plt.show()

def dump_splits(savefile, splits):
    with open(savefile, 'w') as f:
        for s in splits:
            f.write('{}\n'.format(' '.join(map(str,s))))

class PreprocessedWflwfKeypointsDataset():
    def __init__(self, split=SplitType.DEVELOP, data_dir=None):
        if data_dir is None:
            data_dir = get_dataset_dir()
        wflw_preproc_dir = os.path.join(data_dir, 'WFLW_keypoint_preproc_jpeg')
        image_dir = os.path.join(wflw_preproc_dir, 'images')
        self.train_indices, self.eval_indices = self.load_splits(wflw_preproc_dir, split)
        all_indices = self.train_indices + self.eval_indices
        filenames = map(lambda x: os.path.join(image_dir, str(x) + '.jpg'), all_indices)
        self.image_buffer = ImageBuffer.ImageBuffer(filenames, all_indices)
        self.annotation_dir = os.path.join(wflw_preproc_dir, 'annotations')

    def get_train(self):
        return WflwKeypointsDataset(self.image_buffer, self.annotation_dir, self.train_indices, augment=True)


    def get_eval(self):
        return WflwKeypointsDataset(self.image_buffer,  self.annotation_dir, self.eval_indices, augment=False)

    @staticmethod
    def load_splits(directory, split):
        # TODO duplicated code
        t_t, t_v, v = load_splits(os.path.join(directory, 'splits.txt'))
        if split == SplitType.DEVELOP:
            return t_t, t_v
        elif split == SplitType.EVAL:
            return sorted(t_t+t_v), v
        elif split == SplitType.DEPLOY:
            return sorted(t_t+t_v+v), []
        elif split == SplitType.MINI:
            train = t_t[::len(t_t)//(32)]
            val = t_v[::len(t_v)//(32)]
            return train, val
        else:
            raise Exception("UNKNOWN SPLIT {}".format(split))

# TODO duplicated
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


class WflwPreprocessingInstance():
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

    def apply(self, image, ann):
        image_shape = image.shape
        points = np.copy(ann.points)
        points -= 0.5
        bounding_circle = (ann.circle_center, ann.circle_radius)
        transform = self.get_coord_transform(image_shape, bounding_circle)
        if self.flip:
            points = points[FLIP_ORDER]

        net_in_image = ImageTools.np_warp_im(image, transform, (self.image_size, self.image_size))
        points_post_transform = transform(points.transpose()).transpose()
        annotation_post_transform = copy.copy(ann)
        if self.make_consistent:
            points_consistent = self.make_annotation_consistent(points_post_transform)
        else:
            points_consistent = points_post_transform

        annotation_post_transform.points = points_consistent
        keypoint_type = np.ones(98).astype(np.float32)
        return net_in_image, annotation_post_transform, keypoint_type

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

        return WflwPreprocessingInstance(image_size, flip=flip, rotation=rot_angle, scales=scales, scale_angle=scale_angle,  translation_pre_perspective=translation_pre_perspective, perspective_strength=perspective_strength, perspective_angle=perspective_angle, translation_post_perspective=translation_post_perspective, make_distances_consistent=True)

    @staticmethod
    def get_non_augmented_preprocessor(image_size):
        return WflwPreprocessingInstance(image_size)

    @staticmethod
    def get_eval_augmentations(image_size):
        augs = []
        augs.append(WflwPreprocessingInstance(image_size))
        augs.append(WflwPreprocessingInstance(image_size, flip=True))
        return augs

    @staticmethod
    def make_annotation_consistent(points):
        # points are 98x2
        ret = np.copy(points)
        ret[1:16] = WflwPreprocessingInstance.helper_consistent(points, range(0,17))

        ret[17:32] = WflwPreprocessingInstance.helper_consistent(points, range(16,33))
        # eyebrows TODO
        # I skip doing this for non-chins. It seems like some annotations are adjusted after getting automatically generated
        # ret[52:54] = WflwPreprocessingInstance.helper_consistent(points, range(51,55))
        # ret[56] = WflwPreprocessingInstance.helper_consistent(points, range(55,58))
        # ret[58] = WflwPreprocessingInstance.helper_consistent(points, range(57,60))
        # ret[61] = WflwPreprocessingInstance.helper_consistent(points, range(60,63))
        # ret[63] = WflwPreprocessingInstance.helper_consistent(points, range(62,65))
        # ret[65] = WflwPreprocessingInstance.helper_consistent(points, range(64,67))
        # ret[67] = WflwPreprocessingInstance.helper_consistent(points, [66, 67, 60])
        # ret[69] = WflwPreprocessingInstance.helper_consistent(points, range(68,71))
        # ret[71] = WflwPreprocessingInstance.helper_consistent(points, range(70,73))
        # ret[73] = WflwPreprocessingInstance.helper_consistent(points, range(72,75))
        # ret[75] = WflwPreprocessingInstance.helper_consistent(points, [74,75, 68])
        # ret[81] = WflwPreprocessingInstance.helper_consistent(points, range(80,83))
        # ret[83:85] = WflwPreprocessingInstance.helper_consistent(points, range(82,86))
        # ret[86:88] = WflwPreprocessingInstance.helper_consistent(points, [85,86,87, 76])
        # ret[89] = WflwPreprocessingInstance.helper_consistent(points, range(88, 91))
        # ret[91] = WflwPreprocessingInstance.helper_consistent(points, range(90, 93))
        # ret[93] = WflwPreprocessingInstance.helper_consistent(points, range(92, 95))
        # ret[93] = WflwPreprocessingInstance.helper_consistent(points, range(92, 95))
        # ret[95] = WflwPreprocessingInstance.helper_consistent(points, [94, 95, 88])
        return ret

    @staticmethod
    def helper_consistent(points, idxs):
        pidx = points[idxs]
        d = np.linalg.norm(pidx[1:]-pidx[:-1], axis=1)
        cd = np.zeros(len(idxs))
        cd[1:] = np.cumsum(d)
        cd = cd/cd[-1]
        samplepos = np.arange(1, len(idxs)-1)/(len(idxs)-1)
        xsample = np.interp(samplepos, cd, pidx[:,0])
        ysample = np.interp(samplepos, cd, pidx[:,1])
        return np.stack((xsample, ysample), axis=1)



class WflwKeypointsDataset():
    def __init__(self, image_buffer, annotation_dir, indices, augment=False,image_size=224):
        self.indices = indices
        self.ann_dir = annotation_dir
        self.image_buffer = image_buffer
        self.image_size = image_size
        self.augment = augment
        self.eval_preprocessors = WflwPreprocessingInstance.get_eval_augmentations(image_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.augment:
            preprocess = WflwPreprocessingInstance.randomly_generate_augmentation(self.image_size)
        else:
            preprocess = WflwPreprocessingInstance.get_non_augmented_preprocessor(self.image_size)
        return self.getitem_with_provided_preprocessor(idx, preprocess)

    def getitem_with_provided_preprocessor(self, idx, preprocess):
        index = self.indices[idx]

        ann_path = os.path.join(self.ann_dir, '{}.json'.format(index))

        full_im = self.image_buffer[index]
        with open(ann_path, 'r') as f:
            ann = json.load(f)
            ann = PreprocessesedWflwAnnotation.deserialize(ann)
        full_im = full_im.astype(np.float)/255
        im, annotation_post_transform, keypoint_type = preprocess.apply(full_im, ann)
        return im.transpose(2,0,1).astype(np.float32), annotation_post_transform.points.astype(np.float32), keypoint_type, np.ones(98).astype(np.float32)

    def create_eval_batch(self, idx):
        index = self.indices[idx]

        ann_path = os.path.join(self.ann_dir, '{}.json'.format(index))

        full_im = self.image_buffer[index]

        with open(ann_path, 'r') as f:
            ann = json.load(f)
            ann = PreprocessesedWflwAnnotation.deserialize(ann)

        full_im = full_im.astype(np.float)/255
        ims = np.empty((len(self.eval_preprocessors), self.image_size, self.image_size, 3))
        keypoints = np.empty((len(self.eval_preprocessors), 98, 2))
        keypoint_types = np.empty((len(self.eval_preprocessors), 98))

        bounding_circle = (ann.circle_center, ann.circle_radius)
        for preproc_idx, preprocess in enumerate(self.eval_preprocessors):
            im, keypoints_post_transform, keypoint_type = preprocess.apply(full_im, ann)
            ims[preproc_idx] = im
            keypoints[preproc_idx] = keypoints_post_transform.points
            keypoint_types[preproc_idx] = keypoint_type
        im_tensors = torch.tensor(ims.transpose(0,3,1,2).astype(np.float32))
        return im_tensors, keypoints, keypoint_types, full_im.shape, bounding_circle

    def convert_back_eval_preds_to_unaug(self, image_size, modes, precs, bounding_circle):
        T0 = self.eval_preprocessors[0]
        ret_modes = []
        ret_precs = []
        for mode, prec, preproc in zip(modes, precs, self.eval_preprocessors):
            m, prec_ret = preproc.inverse_map(image_size, bounding_circle, mode, prec, target_map=T0)
            ret_modes.append(m)
            ret_precs.append(prec_ret)
        return np.array(ret_modes), np.array(ret_precs)
