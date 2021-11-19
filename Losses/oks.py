import numpy as np
import torch

# OKS is a COCO-keypoint specific loss


oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
oks_vars = (oks_sigmas*2)**2 # in reference code. cocoeval.py:206

def get_oks(predictions, gts, bounding_boxes, kp_types):
    # preds, are Bx17x2
    # gt is has size Bx17x2
    # bounding_bpx has size Bx4 (minx, miny, w,h)
    # kp_type is Bx17 == 0 if point not marked
    B = len(predictions)
    ret = []
    for bidx in range(B):
        markings = kp_types[bidx]
        bbx = bounding_boxes[bidx]
        gt = gts[bidx]
        prediction = predictions[bidx]

        markings_mask = markings>0
        has_markings = np.any(markings_mask)
        area = bbx[2]*bbx[3]

        if has_markings:
            err_norm = np.linalg.norm(prediction-gt, axis=0)
        else: # The really strange case
            left_edge = np.ones(17)*(bbx[0]-bbx[2])
            right_edge = np.ones(17)*(bbx[0]+2*bbx[2])
            top_edge = np.ones(17)*(bbx[1]-bbx[3])
            bottom_edge = np.ones(17)*(bbx[1]+2*bbx[3])
            left_err = left_edge - prediction[0]
            right_err = prediction[0]-right_edge
            top_err = top_edge - prediction[1]
            bottom_err = prediction[1] - bottom_edge
            dx = np.maximum(np.maximum(np.zeros(17), left_err), right_err)
            dy = np.maximum(np.maximum(np.zeros(17), top_err), bottom_err)
            err_norm = np.sqrt(dx**2 + dy**2)
        normalized_2norm = err_norm**2 / (oks_vars * 2 * area + np.spacing(1))
        if has_markings:
            err_norm = err_norm[markings_mask]
        oks_per_kp = np.exp(-normalized_2norm)
        oks = np.mean(oks_per_kp)
        # if has_markings:
        #    ret.append(oks)
        ret.append(oks)
    return ret
