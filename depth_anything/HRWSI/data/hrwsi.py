# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    def __init__(self, resize_shape):
        self.normalize = transforms.Normalize(
            mean=[0.4268, 0.4177, 0.3832], std=[0.2402, 0.2401, 0.2459])
        # self.normalize = lambda x : x
        self.resize = transforms.Resize(resize_shape)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        image = self.resize(image)
        depth = self.resize(depth)

        return {'image': image, 'depth': depth, 'dataset': "hrwsi"}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class HRWSI(Dataset):
    def __init__(self, data_dir_root, resize_shape):
        import csv

        # image paths are of the form <data_dir_root>/{outleft, depthmap}/*.png
        self.image_files, self.depth_files = [], []

        with open(f"{data_dir_root}/train.csv", 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                self.image_files.append(os.path.join(data_dir_root,row[0]))
                self.depth_files.append(os.path.join(data_dir_root,row[1]))

        self.transform = ToTensor(resize_shape)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(depth_path), dtype=np.float32) /255.0

        # depth[depth > 8] = -1
        depth = depth[..., None]

        sample = dict(image=image, depth=depth)
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.image_files)


def get_hrwsi_loader(data_dir_root, resize_shape, batch_size=1, ddp=False, ddp_rank=0, ddp_world_size=1, **kwargs):
    # dataset = HRWSI(data_dir_root, resize_shape)
    # return DataLoader(dataset, batch_size, **kwargs)
    dataset = HRWSI(data_dir_root, resize_shape)
    
    # Create sampler for DDP
    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=False,
            seed=42  # You can make this configurable
        )
        # When using a distributed sampler, don't shuffle in the DataLoader
        kwargs['shuffle'] = False
        kwargs['sampler'] = sampler
    
    return DataLoader(dataset, batch_size, **kwargs)