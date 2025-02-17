import json
import os
import os.path as osp
import random
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, InterpolationMode, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToTensor

LizardClass = namedtuple('LizardClass', ['name', 'id', 'has_instances', 'color'])
classes = [
    LizardClass('Background',         0, True,  (  0,   0,   0)), # black
    LizardClass('Neutrophil',         1, True,  (204,   0,   0)), # red
    LizardClass('Epithelial',         2, True,  ( 76, 153,   0)), # green
    LizardClass('Lymphocyte',         3, True,  (255, 153,  51)), # orange
    LizardClass('Plasma',             4, True,  (153,  76, 255)), # purple
    LizardClass('Eosinophil',         5, True,  (255,  51, 153)), # pink
    LizardClass('Connective_tissue',  6, True,  (  51, 51, 255)), # blue
]
num_classes = 7
mapping_id = torch.tensor([x.id for x in classes])
colors = torch.tensor([cls.color for cls in classes])

LIZARD_CLASSES = ['', 'neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil', 'connective tissue']

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    return (img + 1) * 0.5


def unnormalize_and_clamp_to_zero_to_one(img):
    return torch.clamp(unnormalize_to_zero_to_one(img.cpu()), 0, 1)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class ToTensorNoNorm():
    def __call__(self, X_i):
        X_i = np.array(X_i)

        if len(X_i.shape) == 2:
            # Add channel dim.
            X_i = X_i[:, :, None]

        return torch.from_numpy(np.array(X_i, copy=False)).permute(2, 0, 1)


def interpolate_3d(x, *args, **kwargs):
    return F.interpolate(x.unsqueeze(0), *args, **kwargs).squeeze(0)


class RandomResize(nn.Module):
    def __init__(self, scale=(0.5, 2.0), mode='nearest'):
        super().__init__()
        self.scale = scale
        self.mode = mode

    def get_random_scale(self):
        return random.uniform(*self.scale)

    def forward(self, x):
        random_scale = self.get_random_scale()
        x = interpolate_3d(x, scale_factor=random_scale, mode=self.mode)
        return x


def read_jsonl(jsonl_path):
    import jsonlines
    lines = []
    with jsonlines.open(jsonl_path, 'r') as f:
        for line in f.iter():
            lines.append(line)
    return lines


class LizardDataset(Dataset):
    def __init__(
        self,
        root="",
        split='train',
        side_x=128,
        side_y=128,
        shuffle=False,
        augmentation_type='flip',
    ):
        super().__init__()
        self.root = Path(root)
        self.image_dir = os.path.join(self.root, 'images', split)
        self.label_dir = os.path.join(self.root, 'classes', split)
        self.point_dir = os.path.join(self.root, 'points', split)
        self.dist_dir = os.path.join(self.root, 'distances', split)

        self.split = split
        self.shuffle = shuffle
        self.side_x = side_x
        self.side_y = side_y

        if augmentation_type == 'none':
            self.augmentation = Compose([
                Resize((side_x, side_y), interpolation=InterpolationMode.NEAREST),
                # ToTensor(),
            ])
        elif augmentation_type == 'flip':
            self.augmentation = Compose([
                Resize((side_x, side_y), interpolation=InterpolationMode.NEAREST),
                RandomHorizontalFlip(p=0.5),
                # ToTensor(),
            ])
        elif 'resizedCrop' in augmentation_type:
            scale = [float(s) for s in augmentation_type.split('_')[1:]]
            assert len(scale) == 2, scale
            self.augmentation = Compose([
                RandomResize(scale=scale, mode='nearest'),
                RandomCrop((1024, 2048)),
                Resize((side_x, side_y), interpolation=InterpolationMode.NEAREST),
                RandomHorizontalFlip(p=0.5),
                # ToTensor(),
            ])
        elif 'lizard' in augmentation_type:
            self.augmentation = Compose([
                # Resize((500, 500), interpolation=InterpolationMode.NEAREST),
                # RandomCrop((side_x, side_y)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                # ToTensor(),
            ])
        else:
            raise NotImplementedError(augmentation_type)

        # verification
        self.images = sorted([osp.join(self.image_dir, file) for file in os.listdir(self.image_dir)
                              if "." in file and file.split(".")[-1].lower() in ["jpg", "jpeg", "png", "gif"]])
        self.labels = sorted([osp.join(self.label_dir, file) for file in os.listdir(self.label_dir)
                              if "." in file and file.split(".")[-1].lower() in ["jpg", "jpeg", "png", "gif"]])
        self.points = sorted([osp.join(self.point_dir, file) for file in os.listdir(self.point_dir)
                              if "." in file and file.split(".")[-1].lower() in ["jpg", "jpeg", "png", "gif"]])
        self.dists = sorted([osp.join(self.dist_dir, file) for file in os.listdir(self.dist_dir)
                              if "." in file and file.split(".")[-1].lower() in ["jpg", "jpeg", "png", "gif"]])

        assert len(self.images) == len(self.labels) == len(self.points) == len(self.dists), f'{len(self.images)} != {len(self.labels)} != {len(self.points)} != {len(self.dists)}'
        for img, lbl, pnt, dist in zip(self.images, self.labels, self.points, self.dists):
            assert osp.splitext(osp.basename(img))[0] == osp.splitext(osp.basename(lbl))[0]
            assert osp.splitext(osp.basename(img))[0] == osp.splitext(osp.basename(pnt))[0]
            assert osp.splitext(osp.basename(img))[0] == osp.splitext(osp.basename(dist))[0]

    def __len__(self):
        return len(self.images)

    def random_sample(self):
        return self.__getitem__(random.randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def set_prompt(self, cls):
        prompt = "high quality histopathology colon tissue image"
        cls_now = np.unique(cls)
        # shuffle classes and add to prompt
        if len(cls_now)==1:
            prompt += " with no nucleus"
        else:
            prompt += " including nuclei types of "
            included = []
            for i in cls_now[1:]:
                included.append(LIZARD_CLASSES[int(i)])
            random.shuffle(included)
            if len(included) > 1:
                included[-1] = "and " + included[-1]
                if len(included)==2:
                    included = ' '.join(included)
                else:
                    included = ', '.join(included)
            else:
                included = included[0]
            prompt += included
        return prompt

    def __getitem__(self, idx):
        # load image
        try:
            original_pil_image  = Image.open(self.images[idx]).convert("RGB")
            original_pil_target = Image.open(self.labels[idx])
            original_pil_point  = Image.open(self.points[idx])
            original_pil_dist     = Image.open(self.dists[idx])
        except (OSError, ValueError) as e:
            print(f"An exception occurred trying to load file {self.images[idx]}.")
            print(f"Skipping index {idx}")
            return self.skip_sample(idx)

        # Transforms
        image = ToTensor()(original_pil_image)
        label = ToTensorNoNorm()(original_pil_target).float()
        point = ToTensorNoNorm()(original_pil_point).float()
        dist = ToTensor()(original_pil_dist).float()
        img_dist_lbl_pnt = self.augmentation(torch.cat([image, dist, label, point])) # 0,1,2: image, 3: dist, 4: label, 5:point
        
        aug_img_dist = img_dist_lbl_pnt[:4]
        aug_lbl = img_dist_lbl_pnt[4].unsqueeze(0)
        aug_pnt = img_dist_lbl_pnt[5].unsqueeze(0)

        caption = self.set_prompt(label)
        
        return aug_img_dist, aug_lbl, caption, aug_pnt

def transform_lbl(lbl: torch.Tensor, *args, **kwargs):
    lbl = lbl.long()
    if lbl.size(1) == 1:
        # Remove single channel axis.
        lbl = lbl[:, 0]
    rgbs = colors[lbl]
    rgbs = rgbs.permute(0, 3, 1, 2)
    return rgbs / 255.


def get_text(cls):
    prompt = "high quality histopathology colon tissue image"
    cls_now = np.unique(cls)
    # shuffle classes and add to prompt
    if len(cls_now)==1:
        prompt += " with no nucleus"
    else:
        prompt += " including nuclei types of "
        included = []
        for i in cls_now[1:]:
            included.append(LIZARD_CLASSES[int(i)])
        random.shuffle(included)
        if len(included) > 1:
            included[-1] = "and " + included[-1]
            if len(included)==2:
                included = ' '.join(included)
            else:
                included = ', '.join(included)
        else:
            included = included[0]
        prompt += included
    return prompt