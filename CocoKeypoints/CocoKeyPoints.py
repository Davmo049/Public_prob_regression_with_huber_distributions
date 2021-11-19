from general_utils.environment_variables import get_dataset_dir
import os
import torch
import numpy as np
from enum import IntEnum
import json
from PIL import Image
import mathlib
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import Dataset
from ImageTools import ImageTools as ImageTools

from mathlib.covers import minimum_covering_sphere
from ConfigParser.ConfigParser import SplitType

import matplotlib
import scipy
import scipy.linalg

# posprocessed stats, assuming netinput is 224x224
Coco_mean = np.array([
 [112.11429992,  49.86835563],
 [114.67978472,  43.66795015],
 [109.52308929,  43.70595513],
 [118.43303935,  44.60971139],
 [105.66148043,  44.71130425],
 [123.96409906,  71.97750981],
 [100.15112657,  71.94578632],
 [129.54589202, 103.37780944],
 [ 94.66143426, 103.265769  ],
 [123.14474463, 109.31973259],
 [101.02758434, 109.52484962],
 [118.9460166 , 133.28276459],
 [105.19053389, 133.29365153],
 [117.74087727, 153.64111874],
 [106.22360524, 153.80412156],
 [117.04024744, 190.34717109],
 [107.32986176, 190.60791506]])

Coco_half_prec = np.array([
  [[ 2.56211106e-02,  6.08902798e-05],
   [ 6.08902798e-05,  3.58556393e-02]],
  [[ 2.49840259e-02, -6.25063596e-04],
   [-6.25063596e-04,  3.82517951e-02]],
  [[ 2.49452757e-02,  8.54828179e-04],
   [ 8.54828179e-04,  3.81318726e-02]],
  [[ 2.50266902e-02, -1.64356829e-03],
   [-1.64356829e-03,  3.71121347e-02]],
  [[ 2.49358927e-02,  1.92925248e-03],
   [ 1.92925248e-03,  3.70924947e-02]],
  [[ 2.62125404e-02, -2.66909111e-03],
   [-2.66909111e-03,  2.91943140e-02]],
  [[ 2.62250755e-02,  2.86227033e-03],
   [ 2.86227033e-03,  2.91927878e-02]],
  [[ 2.45901662e-02, -1.91472639e-03],
   [-1.91472639e-03,  2.32919558e-02]],
  [[ 2.46182611e-02,  1.97803376e-03],
   [ 1.97803376e-03,  2.33024642e-02]],
  [[ 2.37310186e-02, -4.10082599e-04],
   [-4.10082599e-04,  2.20709897e-02]],
  [[ 2.36657006e-02,  4.74012218e-04],
   [ 4.74012218e-04,  2.20708361e-02]],
  [[ 3.05850810e-02, -1.09656943e-03],
   [-1.09656943e-03,  2.32569006e-02]],
  [[ 3.06156014e-02,  9.17328582e-04],
   [ 9.17328582e-04,  2.32539831e-02]],
  [[ 2.90821654e-02,  5.04401554e-04],
   [ 5.04401554e-04,  2.94763083e-02]],
  [[ 2.93081201e-02, -4.98816816e-04],
   [-4.98816816e-04,  2.96619593e-02]],
  [[ 2.17568864e-02,  4.01964013e-04],
   [ 4.01964013e-04,  3.14025656e-02]],
  [[ 2.18908208e-02, -8.27671983e-04],
   [-8.27671983e-04,  3.20129888e-02]]])

class RawCoco():
    def __init__(self, train=True, dataset_path=None):
        if dataset_path is None:
            dataset_path = get_dataset_dir()
        coco_path = os.path.join(dataset_path, 'COCO')
        if train:
            json_file = 'person_keypoints_train2017.json'
        else:
            json_file = 'person_keypoints_val2017.json'
        annotation_path = os.path.join(coco_path, 'annotations', json_file)
        with open(annotation_path, 'r') as f:
            self.all_data = json.load(f)
        self.annotations = self.all_data['annotations']
        if train:
            self.imdir = os.path.join(coco_path, 'train2017')
        else:
            self.imdir = os.path.join(coco_path, 'val2017')

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        rel_impath = '{0:012}.jpg'.format(ann['image_id'])
        impath = os.path.join(self.imdir, rel_impath)
        with open(impath, 'rb') as f:
            img_PIL = Image.open(f)
            img_PIL.convert('RGB')
            data = img_PIL.getdata()
            if isinstance(data[0], np.int):
                img = np.array(data).reshape(img_PIL.size[1], img_PIL.size[0]).reshape(img_PIL.size[1], img_PIL.size[0],1).repeat(3,2)
            else:
                img = np.array(data).reshape(img_PIL.size[1], img_PIL.size[0], 3)
        return img, ann

    def __len__(self):
        return len(self.annotations)

def search_for_suspect_samples(dataset_dir=None):
    def search_for_suspect_in_dataset(ds):
        for i in range(len(ds.annotations)):
            ann = ds.annotations[i]
            if not np.any(ann['keypoints']):
                continue
            segments = ann['segmentation']
            points = []
            for j in range(len(segments)):
                for i in range(len(segments[j])//2):
                    x = segments[j][i*2]
                    y = segments[j][i*2+1]
                    points.append((x,y))
            points = np.array(points)
            ret = mathlib.covers.min_sphere(points)
            if ret is None:
                print(i)
                print('wtf')
            center, radius = ret
            # import matplotlib.patches as patches
            # circle = patches.Circle((center[0], center[1]),radius,fill=False, edgecolor='blue',linestyle='dotted',linewidth='2.2')
            # plt.gca().add_patch(circle)
            # plt.scatter(points[:,0], points[:,1])
            # plt.show()

    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    coco_train = RawCoco(train=True, dataset_path=dataset_dir)
    coco_val = RawCoco(train=False, dataset_path=dataset_dir)
    print('train')
    search_for_suspect_in_dataset(coco_train)
    print('val')
    search_for_suspect_in_dataset(coco_val)

def export_coco(dataset_dir=None, valtrain_keep=0.3):
    train_split = []
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    coco_train = RawCoco(train=True, dataset_path=dataset_dir)
    coco_val = RawCoco(train=False, dataset_path=dataset_dir)
    coco_preprocessed = os.path.join(dataset_dir, 'COCO_keypoint_preproc')
    os.mkdir(coco_preprocessed)
    coco_preprocessed_images = os.path.join(coco_preprocessed, 'images')
    annotations_preprocessed_dir = os.path.join(coco_preprocessed, 'annotations')
    os.mkdir(annotations_preprocessed_dir)
    os.mkdir(coco_preprocessed_images)
    train_samples = preprocess_dataset(coco_train, coco_preprocessed_images, annotations_preprocessed_dir, offset=0)
    val_samples = preprocess_dataset(coco_val, coco_preprocessed_images, annotations_preprocessed_dir, offset=train_samples)

    # save splits
    split_train_val = np.floor(np.arange(train_samples*0.3)/0.3).astype(np.int)
    all_train_indices = range(train_samples)
    split_train_train = sorted(list(set(all_train_indices)-set(split_train_val)))
    split_val = np.arange(train_samples, val_samples)
    split_file = os.path.join(coco_preprocessed, 'splits.txt')
    dump_splits(split_file, (split_train_train, split_train_val, split_val))

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

def preprocess_dataset(dataset, image_save_dir, annotations_save_dir, offset):
    if offset != 0:
        file_should_exist = os.path.join(image_save_dir, '{}.png'.format(offset-1))
        file_should_not_exist = os.path.join(image_save_dir, '{}.png'.format(offset))
        if not os.path.exists(file_should_exist):
            print(' file "{}" does not exist"'.format(file_should_exist))
            assert(False)
        if os.path.exists(file_should_not_exist):
            print(' file "{}" does exist, but shouldnt"'.format(file_should_not_exist))
            assert(False)
    returned_annotations = []
    index = offset

    num_skipped = 0
    for im, ann in tqdm.tqdm(dataset):
        ret = preprocess_raw_sample(im, ann)
        if ret is None:
            num_skipped+=1
            continue
        im, keypoints, segments, bounding_circle = ret
        # visualize_sample(im, keypoints, segments, bounding_circle)
        save_sample(im, keypoints, bounding_circle, image_save_dir, annotations_save_dir, index)
        index += 1
    return index

def visualize_sample(im, kp, segments, bounding_circle):
    plt.imshow(im/255)
    mask = kp[:, 2] > 0.5
    plt.scatter(kp[mask, 0], kp[mask, 1])
    for s in segments:
        x = list(s[:,0])
        x += [x[0]]
        y = list(s[:,1])
        y += [y[0]]
        plt.plot(x,y)
    plt.show()

def preprocess_raw_sample(im, ann):
    # aug order: flip, shift, affine, scale, rotation
    oversize_crop = 1.1 # keep 5% of width/height extra compared to segmentation
    rescale_min = 0.8 # "zoom out" at most to 80% of original size
    rescale_max = 1.3 # # zoom in at most to 130% of original size
    shift = 0.1 # shift at most 10% of width/height

    max_affine_zoom_out = 0.4 # zoom in/out at most 40% due to affine transform, center of affine is the welzl median
    max_affine_zoom_in = 0.4 # zoom in/out at most 40% due to affine transform, center of affine is the welzl median
    c = min(1-1/np.sqrt(1+max_affine_zoom_out), 1/np.sqrt(1-max_affine_zoom_in)-1)
    affine_radius_out = 1/(1-c)
    oversize_aug = shift + (1/rescale_min)*affine_radius_out # keep 5% of width/height extra compared to segmentation
    oversize_final = shift + (1/rescale_min)*affine_radius_out*oversize_crop
    if np.all(np.array(ann['keypoints'][2::3]) == 0):
        print(ann['keypoints'])
        return None
    segments = ann['segmentation']
    points = []
    for j in range(len(segments)):
        for i in range(len(segments[j])//2):
            x = segments[j][i*2]
            y = segments[j][i*2+1]
            points.append((x,y))
    points = np.array(points)
    ret = mathlib.covers.min_sphere(points)
    if ret is None:
        num_skipped += 1
        print('Could not find bounding sphere {}'.format(points))
        return None
    else:
        center, radius = ret
    # translate crop
    left = max(0, np.floor(center[0]+0.5-radius*oversize_final).astype(np.int))
    right = min(im.shape[1]-1, np.ceil(center[0]+0.5+radius*oversize_final).astype(np.int))
    top = max(0, np.floor(center[1]+0.5-radius*oversize_final).astype(np.int))
    bot = min(im.shape[0]-1, np.ceil(center[1]+0.5+radius*oversize_final).astype(np.int))
    center_new = center+0.5 - np.array((left, top))
    im_translated = im[top:bot, left:right]
    segment_translated = []
    for j in range(len(ann['segmentation'])):
        segm = ann['segmentation'][j]
        points = np.empty((len(segm)//2, 2), dtype=np.float64)
        points[:,0] = np.array(segm[::2])-left+0.5
        points[:,1] = np.array(segm[1::2])-top+0.5
        segment_translated.append(points)
    center_translated = center-np.array((left, top))
    keypoints = ann['keypoints']
    keypoints_translated = np.empty((len(keypoints)//3, 3), dtype=np.float64)
    for i in range(3):
        keypoints_translated[:,i] = keypoints[i::3]
    keypoints_translated[:, :2] -= np.array((left, top)).reshape(1, 2)
    keypoints_translated[:, :2] += 0.5

    # scale crop
    downscale_factor = max(1, np.ceil(radius/112).astype(np.int))
    im_scaled = np.zeros((im_translated.shape[0]//downscale_factor, im_translated.shape[1]//downscale_factor, 3), dtype=np.float64)

    prescale_h = downscale_factor*(im_translated.shape[0]//downscale_factor)
    prescale_w = downscale_factor*(im_translated.shape[1]//downscale_factor)
    im_prescaled = im_translated[:prescale_h, :prescale_w, :]
    for r in range(downscale_factor):
        for c in range(downscale_factor):
            im_scaled += im_prescaled[r::downscale_factor, c::downscale_factor,:]
    im_scaled /= downscale_factor**2
    segments_scaled = []
    for segments in segment_translated:
        segments_scaled.append(segments/downscale_factor-0.5)
    keypoints_scaled = np.copy(keypoints_translated)
    keypoints_scaled[:,:2] /= downscale_factor
    keypoints_scaled[:,:2] -= 0.5
    keypoints_scaled[keypoints_scaled[:, 2]<0.5, :2] = 0


    scaled_center = center_new/downscale_factor-0.5
    scaled_radius = radius/downscale_factor
    return im_scaled, keypoints_scaled, segments_scaled, (scaled_center, scaled_radius)

def save_sample(im, keypoints, circle, image_save_dir, annotations_save_dir, index):
    im_file = os.path.join(image_save_dir, '{}.png'.format(index))
    im = np.round(np.clip(im, 0, 255)).astype(np.uint8)
    PIL_im = Image.fromarray(im).convert('RGB')
    PIL_im.save(im_file)
    segments_as_list = []
    ann = CocoPreprocAnnFormat(circle, keypoints)
    annotation_file_path = os.path.join(annotations_save_dir, '{}.json'.format(index))
    with open(annotation_file_path, 'w') as f:
        json.dump(ann.to_dict(), f)

class CocoPreprocAnnFormat():
    KEYPOINTS_NAME = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_should', 'r_should', 'l_elbow', 'r_elbow', 'l_hand', 'r_hand', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_foot', 'r_foot']
    NAME_TO_IDX = {k: i for i,k in enumerate(KEYPOINTS_NAME)}
    FLIP_ORDER = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
    def __init__(self, circle, keypoints):
        self.circle_center = circle[0]
        self.circle_radius = circle[1]
        self.keypoints = keypoints

    def to_dict(self):
        return {
            'circle': {
                'center': numpy_to_list_of_lists(self.circle_center),
                'radius': self.circle_radius
            },
            'keypoints': numpy_to_list_of_lists(self.keypoints)
        }

    @staticmethod
    def from_dict(dic):
        circle_dic = dic['circle']
        circle = (np.array(circle_dic['center']), circle_dic['radius'])
        keypoints = np.array(dic['keypoints'])
        return CocoPreprocAnnFormat(circle, keypoints)


def numpy_to_list_of_lists(arr):
    if len(arr.shape) == 0:
        return arr
    if len(arr.shape) == 1:
        return list(arr)
    return list(map(numpy_to_list_of_lists, arr))

class PreprocessedCocoKeypoints():
    def __init__(self, split=SplitType.DEVELOP, data_dir=None):
        if data_dir is None:
            data_dir = get_dataset_dir()
        coco_preproc_dir = os.path.join(data_dir, 'COCO_keypoint_preproc')
        self.image_dir = os.path.join(coco_preproc_dir, 'images')
        self.annotation_dir = os.path.join(coco_preproc_dir, 'annotations')
        self.train_indices, self.eval_indices = self.load_splits(coco_preproc_dir, split)

    def get_train(self):
        return CocoKeypointsDataset(self.image_dir, self.annotation_dir, self.train_indices, augment=True)


    def get_eval(self):
        return CocoKeypointsDataset(self.image_dir, self.annotation_dir, self.eval_indices, augment=False)

    @staticmethod
    def load_splits(directory, split):
        t_t, t_v, v = load_splits(os.path.join(directory, 'splits.txt'))
        if split == SplitType.DEVELOP:
            return t_t, t_v
        elif split == SplitType.EVAL:
            return sorted(t_t+t_v), v
        elif split == SplitType.DEPLOY:
            return sorted(t_t+t_v+v), []
        elif split == SplitType.MINI:
            train = t_t[::len(t_t)//320]
            val = t_t[::len(t_v)//320]
            return train, val
        else:
            raise Exception("UNKNOWN SPLIT {}".format(split))


def std_to_ellipsoid(std):
    eigs, vecs = np.linalg.eig(std)
    r1 = max(eigs[0], eigs[1])
    r2 = min(eigs[0], eigs[1])
    if eigs[0] > eigs[1]:
        angle = np.arctan2(vecs[1,0], vecs[0,0])
    else:
        angle = np.arctan2(vecs[1,1], vecs[0,1])
    return r1, r2, angle*180/np.pi

def plot_coco_preproc(im, keypoints, kp_type, pred_kps, half_prec):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    kpx_start = []
    kpx_end = []
    kpy_start = []
    kpy_end = []
    ellipses = []
    string_edges = [['l_should', 'r_should'], ['l_should', 'l_elbow'], ['l_should', 'l_hip'], ['l_elbow', 'l_hand'], ['l_hip', 'r_hip'], ['l_hip', 'l_knee'], ['l_knee', 'l_foot'], ['r_should', 'r_elbow'], ['r_should', 'r_hip'], ['r_elbow', 'r_hand'], ['r_hip', 'r_knee'], ['r_knee', 'r_foot']]
    edges = list(map(lambda x: list(map(lambda x: CocoPreprocAnnFormat.NAME_TO_IDX[x], x)), string_edges))
    skel_start_x = []
    skel_start_y = []
    skel_end_x = []
    skel_end_y = []
    for edge in edges:
        s_idx = edge[0]
        e_idx = edge[1]
        skel_start_x.append(pred_kps[s_idx, 0])
        skel_start_y.append(pred_kps[s_idx, 1])
        skel_end_x.append(pred_kps[e_idx, 0])
        skel_end_y.append(pred_kps[e_idx, 1])
    for i in range(keypoints.shape[0]):
        x_true,y_true = keypoints[i, :]
        x_pred,y_pred = pred_kps[i, :]
        std = np.linalg.inv(half_prec[i])
        r1 , r2, angle = std_to_ellipsoid(std)
        ell_color = 'r'
        if i % 2 == 1:
            ell_color = 'g'
        ell = matplotlib.patches.Ellipse((x_pred, y_pred), r1, r2, angle=angle, fill=False, edgecolor=ell_color, linewidth=2.0)
        ellipses.append(ell)
        v = kp_type[i]
        if v != 0:
            kpx_start.append(x_true)
            kpy_start.append(y_true)
            kpx_end.append(x_pred)
            kpy_end.append(y_pred)
        plt.text(x_pred-10, y_pred-10, CocoPreprocAnnFormat.KEYPOINTS_NAME[i], fontsize=15, c='r')

    plt.imshow(im)
    plt.plot([kpx_start, kpx_end], [kpy_start, kpy_end], c='r')
    plt.plot([skel_start_x, skel_end_x], [skel_start_y, skel_end_y], c='b')
    for ell in ellipses:
        ax.add_artist(ell)
    plt.show()

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

    def inverse_map(self, image_size, bounding_circle, coords, half_precs, target_map=None):
        transform = self.get_coord_transform(image_size, bounding_circle)
        flip = self.flip
        remapping = np.linalg.inv(transform.A)
        if target_map is not None:
            transform_target = target_map.get_coord_transform(image_size, bounding_circle)
            remapping = np.matmul(transform_target.A, remapping)
            flip = flip != target_map.flip # xor
        if flip:
            coords = coords[CocoPreprocAnnFormat.FLIP_ORDER]
            half_precs = half_precs[CocoPreprocAnnFormat.FLIP_ORDER]
        remapping = ImageTools.NpAffineTransforms(remapping)
        coords_new = remapping(coords.transpose()).transpose()
        half_precs_new = []
        for coord, half_prec in zip(coords, half_precs):
            jac = remapping.Jacobian(coord)
            var = np.linalg.inv(np.matmul(half_prec, half_prec))
            var_new = np.matmul(jac, var, jac.transpose())
            half_prec_new = scipy.linalg.sqrtm(np.linalg.inv(var_new)).real
            half_precs_new.append(half_prec_new)
        return coords_new, np.array(half_precs_new)

    def apply(self, image, ann):
        image_shape = image.shape
        keypoints = ann.keypoints
        bounding_circle = (ann.circle_center, ann.circle_radius)
        transform = self.get_coord_transform(image_shape, bounding_circle)
        if self.flip:
            keypoints = keypoints[CocoPreprocAnnFormat.FLIP_ORDER]

        net_in_image = ImageTools.np_warp_im(image, transform, (self.image_size, self.image_size))
        keypoints_post_transform = transform(keypoints[:, :2].transpose()).transpose()
        keypoint_type = keypoints[:, 2].astype(np.int)
        return net_in_image, keypoints_post_transform, keypoint_type


    @staticmethod
    def randomly_generate_augmentation(image_size=224, max_rot=30,max_first_rescale=1.2, max_relative_rescale=1.2):
        flip = np.random.randint(2) == 1
        max_translation_pre = 0.05
        max_translation_post = 0.05
        max_perspective_strength = 0.3
        if False: # use less augmentations
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

    @staticmethod
    def get_eval_augmentations(image_size):
        old_state = np.random.get_state()
        np.random.seed(1234567)
        N = 2
        augs = []
        augs.append(PreprocessingInstance.get_non_augmented_preprocessor(image_size))
        if N > 1:
            augs.append(PreprocessingInstance(image_size, flip=True))
        for i in range(N-2):
            augs.append(PreprocessingInstance.randomly_generate_augmentation(image_size))
        np.random.set_state(old_state)
        return augs

class CocoKeypointsDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, indices, augment=False,image_size=224):
        self.indices = indices
        self.ann_dir = annotation_dir
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        self.eval_preprocessors = PreprocessingInstance.get_eval_augmentations(image_size)

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

        im_path = os.path.join(self.image_dir, '{}.png'.format(index))
        ann_path = os.path.join(self.ann_dir, '{}.json'.format(index))
        img_PIL = Image.open(im_path)

        data = img_PIL.getdata()
        full_im = np.array(data).reshape(img_PIL.size[1], img_PIL.size[0], 3)
        with open(ann_path, 'r') as f:
            ann = json.load(f)
            ann = CocoPreprocAnnFormat.from_dict(ann)


        full_im = full_im.astype(np.float)/255
        # v=0 -> not labeled
        # v=1 -> labeled but not visible
        # v=2 -> labeled and visible
        # weight = 0 if not labled,
        # weight = 1 otherwise
        im, keypoints_post_transform, keypoint_type = preprocess.apply(full_im, ann)
        weights = keypoint_type > 0
        return im.transpose(2,0,1).astype(np.float32), keypoints_post_transform.astype(np.float32), weights.astype(np.float32), keypoint_type

    def create_eval_batch(self, idx):
        index = self.indices[idx]

        im_path = os.path.join(self.image_dir, '{}.png'.format(index))
        ann_path = os.path.join(self.ann_dir, '{}.json'.format(index))
        img_PIL = Image.open(im_path)

        data = img_PIL.getdata()
        full_im = np.array(data).reshape(img_PIL.size[1], img_PIL.size[0], 3)

        with open(ann_path, 'r') as f:
            ann = json.load(f)
            ann = CocoPreprocAnnFormat.from_dict(ann)


        full_im = full_im.astype(np.float)/255
        ims = np.empty((len(self.eval_preprocessors), self.image_size, self.image_size, 3))
        keypoints = np.empty((len(self.eval_preprocessors), 17, 2))
        keypoint_types = np.empty((len(self.eval_preprocessors), 17))

        bounding_circle = (ann.circle_center, ann.circle_radius)
        for preproc_idx, preprocess in enumerate(self.eval_preprocessors):
            im, keypoints_post_transform, keypoint_type = preprocess.apply(full_im, ann)
            ims[preproc_idx] = im
            keypoints[preproc_idx] = keypoints_post_transform
            keypoint_types[preproc_idx] = keypoint_type
        im_tensors = torch.tensor(ims.transpose(0,3,1,2).astype(np.float32))
        return im_tensors, keypoints, keypoint_types, full_im.shape, bounding_circle

    def convert_back_eval_preds_to_unaug(self, image_size, modes, half_precs, bounding_circle):
        T0 = self.eval_preprocessors[0]
        ret_modes = []
        ret_half_precs = []
        for mode, half_prec, preproc in zip(modes, half_precs, self.eval_preprocessors):
            m, hp = preproc.inverse_map(image_size, bounding_circle, mode, half_prec, target_map=T0)
            ret_modes.append(m)
            ret_half_precs.append(hp)
        return np.array(ret_modes), np.array(ret_half_precs)
