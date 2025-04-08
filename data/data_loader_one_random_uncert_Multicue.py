from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
import scipy.io
import random


# def pad_to_multiple_of_eight(tensor):
#     _, h, w = tensor.shape
#     h_pad = (32 - h % 32) % 32
#     w_pad = (32 - w % 32) % 32
#     padded_tensor = torch.nn.functional.pad(tensor, (0.9_mask_a1+a2.9_mask_a1, w_pad, 0.9_mask_a1+a2.9_mask_a1, h_pad), mode='reflect')
#     return padded_tensor


def pad_to_multiple_of_eight(tensor):
    _, h, w = tensor.shape
    h_pad = (32 - h % 32) % 32
    w_pad = (32 - w % 32) % 32
    # 在 PyTorch 1.7 中，可能需要分两步进行填充
    # 首先填充宽度维度
    tensor = torch.nn.functional.pad(tensor, (0, w_pad), mode='reflect')
    # 然后填充高度维度
    if h_pad > 0:
        # 需要一个额外的维度来处理高度填充
        new_shape = list(tensor.shape)
        new_shape[-2] += h_pad  # 增加高度维度
        new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
        new_tensor[:, :tensor.shape[1], :] = tensor
        tensor = new_tensor
    return tensor


class BSDS_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='/data_file', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, 'train_val_all.lst')

        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)


    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file = img_lb_file[0]
            label_list = []
            for i_label in range(1, len(img_lb_file)):
                lb = scipy.io.loadmat(join(self.root, img_lb_file[i_label]))
                lb = np.asarray(lb['edge_gt'])
                label = torch.from_numpy(lb)
                label = label.float()
                label_list.append(label.unsqueeze(0))
            labels = torch.cat(label_list, 0)
            lb_mean = pad_to_multiple_of_eight(labels.mean(dim=0).unsqueeze(0))
            lb_std = pad_to_multiple_of_eight(labels.std(dim=0).unsqueeze(0))
            lb_index = random.randint(2, len(img_lb_file)) - 1
            lb_file = img_lb_file[lb_index]
        else:
            img_file = self.filelist[index].rstrip()

        img = imageio.imread(join(self.root, img_file))
        img = transforms.ToTensor()(img)
        img = pad_to_multiple_of_eight(img.float())

        if self.split == "train":
            lb = scipy.io.loadmat(join(self.root, lb_file))
            lb = np.asarray(lb['edge_gt'])
            label = torch.from_numpy(lb)
            label = label.float().unsqueeze(0)
            label = pad_to_multiple_of_eight(label)

            return img, label, lb_mean, lb_std
        else:
            return img


