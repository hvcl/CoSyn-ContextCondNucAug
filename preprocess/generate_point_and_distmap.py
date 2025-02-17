import warnings

# Essentials
import os
import glob
import cv2 as cv
import numpy as np
import scipy.io as sio
import numpy as np
from tqdm import tqdm
import random
import tifffile

# Image functions
from scipy.ndimage import measurements
from PIL import Image

## Set your data paths
PATH_TARGET_CLASSES = '/data_path/classes/'
PATH_TARGET_INSTANCES = '/data_path/instances/'
PATH_TARGET_POINTS = '/data_path/points/'
PATH_TARGET_DISTS = '/data_path/dist_maps/'

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def gen_instance_hv_map(ann): #, crop_shape):
    """Input annotation must be of original shape.
    
    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    orig_ann = ann.copy()  # instance ID map
    orig_ann = np.pad(orig_ann, (2, 2))
    # fixed_ann = fix_mirror_padding(orig_ann)
    # re-cropping with fixed instance id map
    # crop_ann = cropping_center(fixed_ann, crop_shape)
    # TODO: deal with 1 label warning
    # crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(orig_ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(orig_ann == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    x_map = x_map[2:-2, 2:-2]
    y_map = y_map[2:-2, 2:-2]

    hv_map = (np.dstack([x_map, y_map]) + 1) / 2
    
    # return hv_map
    return hv_map


# Create splits
splits = ["train", "test"] # or ["train", "val", "test"]
for splt in splits:
    save_dist_dir = os.path.join(PATH_TARGET_DISTS, splt)
    save_point_dir = os.path.join(PATH_TARGET_POINTS, splt)
    os.makedirs(save_dist_dir, exist_ok=True)
    os.makedirs(save_point_dir, exist_ok=True)

    lbl_pths = sorted(glob.glob(os.path.join(PATH_TARGET_CLASSES, splt, "*.png")))
    inst_pths = sorted(glob.glob(os.path.join(PATH_TARGET_INSTANCES, splt, "*.png")))
    for idx, (lbl_pth, inst_pth) in tqdm(enumerate(zip(lbl_pths, inst_pths)), desc='# processing'):
        fn = os.path.basename(lbl_pth)
        fn = fn.split('.png')[0]

        lbl = np.array(Image.open(lbl_pth), dtype=np.uint8)
        inst = np.array(Image.open(inst_pth), dtype=np.int16)

        num_inst = len(np.unique(inst)[1:])
        for cur_i, cur_val in enumerate(np.unique(inst)[1:]):
            inst[inst==cur_val] = cur_i + 1
        
        inst = inst.astype(np.uint8)

        hv_map = gen_instance_hv_map(inst)
        save_dst_pth = os.path.join(save_dist_dir, fn + '.npy')
        np.save(save_dst_pth, hv_map)

        cnt = np.zeros_like(inst)
        cntr_lst = []
        for i in list(np.unique(inst)[1:]):
            inst_i = (inst==i).astype(np.uint8)
            contours, hierarchy = cv.findContours(inst_i, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL, RETR_TREE
            cntr_lst.extend(contours)

        num_cntr = len(cntr_lst)

        for idx, e in enumerate(cntr_lst):
            M = cv.moments(e)
            cX, cY = 0, 0
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv.boundingRect(e)
                cX = x + w // 2
                cY = y + h // 2
            ctd = [cX, cY]
            cnt[ctd[1], ctd[0]] = 1
            
        cnt[cnt>0] = lbl[cnt>0]
        point_map = cnt
        save_pth = os.path.join(save_point_dir, fn)
        Image.fromarray(point_map).save(save_pth)