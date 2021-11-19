

import numpy as np
import torch

from ImageTools import ImageTools as ImageTools
from Losses.Losses import ProbHuberLoss
import CocoKeypoints
from ImageTools.ImageTools import NpAffineTransforms
import matplotlib.pyplot as plt

def get_skeletons_from_bounding_spheres(image, bounding_circles, net, net_size=224, device='cpu', dtype=torch.float32):
    """
       image is np array of HWC format range 0-1.
       bbxs is list of tuples (center_x, center_y, radius) floats
       net creates 17x2 + 17x2x2 mean and half_precision estimates
       net_size is expected (square) input size to network.
    """
    net.eval()
    loss_func = ProbHuberLoss(30.0/224, 50.0)
    kp_normalizer = CocoKeypoints.TorchNormalizer(device)

    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
    return_modes = []
    for circle in bounding_circles:
        x = circle[0]
        y = circle[1]
        radius = circle[2]
        scale_change = net_size / (2*radius)
        transform_translate_centering = ImageTools.translation_as_affine((-x,-y)) # origin at center of circle
        transform_scale = ImageTools.scale_as_affine(0,(scale_change, scale_change))
        transform_translate_crop = ImageTools.translation_as_affine((net_size/2, net_size/2))
        im_to_crop = ImageTools.stack_affine_transforms([transform_translate_centering, transform_scale, transform_translate_crop])
        crop_to_im = NpAffineTransforms(np.linalg.inv(im_to_crop.A))

        crop = ImageTools.np_warp_im(image, im_to_crop, (net_size, net_size))

        # get network prediction

        im_torch = torch.tensor(crop.astype(np.float).transpose(2,0,1), device=device, dtype=dtype).unsqueeze(0)
        im_torch_normalized = (im_torch - im_mean) / im_std
        out = net(im_torch_normalized).view(1, 17, 5)
        keypoints_regress = torch.zeros((1,17,2), device=device, dtype=dtype)
        weights = torch.ones((1,17), device=device, dtype=dtype)
        losses, half_prec, modes = loss_func(out, keypoints_regress, weights)
        keypoints_pred = kp_normalizer.denormalize(modes.detach())
        half_prec = kp_normalizer.denormalize_half_prec(half_prec)
        keypoints_pred = keypoints_pred.cpu().detach().numpy().reshape(17, 2).transpose(1,0)

        # convert prediction to original image
        mode = crop_to_im(keypoints_pred)
        return_modes.append(mode)
    return return_modes




