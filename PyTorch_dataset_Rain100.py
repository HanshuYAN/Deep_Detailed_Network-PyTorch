#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a re-implementation of training code of our paper:
# X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. “Removing Rain from Single Images via a Deep Detail Network”, CVPR, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)

import os
import random

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data



def is_img(x):
    if x.endswith('.png') and not(x.startswith('._')):
        return True
    else:
        return False

def _np2Tensor(img):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    return tensor

class Rain100(data.Dataset):
    def __init__(self, args, isTrain=True):
        self.args=args
        self.isTrain = isTrain
        if isTrain:
            self.patch_size = args.patch_size

        self._set_filesystem(args.input_path, args.gt_path)

        if args.ext == 'img':
            self.images_n, self.images_c = self._scan()
        elif args.ext.find('sep') >= 0:
            self.images_n, self.images_c = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_n:
                    hr = misc.imread(v)
                    if np.max(hr) > 1:
                        hr = hr /255.0
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for v in self.images_c:
                    lr = misc.imread(v)
                    if np.max(lr) > 1:
                        lr = lr /255.0
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, lr)

            self.images_n = [
                v.replace(self.ext, '.npy') for v in self.images_n
            ]
            self.images_c = [
                v.replace(self.ext, '.npy') for v in self.images_c
            ]
        else:
            print('Please define data type')

        # self.repeat = 210000 * 20 // len(self.images_n) #{ len(dataloader) * batch_size / len(img_list)}
        self.repeat = 2

    def _set_filesystem(self, dir_n, dir_c):
        self.dir_n = dir_n
        self.dir_c = dir_c
        self.ext = '.png'
        print('********* {}: dir_n and dir_c ******'.format(self.isTrain))
        print(self.dir_n)
        print(self.dir_c)

    def _scan(self):
        
        list_c = sorted(
            [os.path.join(self.dir_c, x) for x in os.listdir(self.dir_c) if is_img(x)])
        list_n = [os.path.splitext(x)[0]+'.png' for x in list_c]
        # list_n = [os.path.splitext(x)[0]+'x2.png' for x in list_c]
        list_n = [os.path.join(self.dir_n, os.path.split(x)[-1]) for x in list_n]
        # list_n = sorted(
        #     [os.path.join(self.dir_n, x) for x in os.listdir(self.dir_n) if is_img(x)])
        if self.isTrain:
            return list_n[0:1700], list_c[0:1700]
        else:
            return list_n[1700:1800], list_c[1700:1800]

    def __getitem__(self, idx):
        img_n, img_c, _, _ = self._load_file(idx)
        assert img_n.shape==img_c.shape

        if self.isTrain:
            x = random.randint(0,img_n.shape[0] - self.patch_size)
            y = random.randint(0,img_n.shape[1] - self.patch_size)
            img_n = img_n[x : x+self.patch_size, y : y+self.patch_size, :]
            img_c = img_c[x : x+self.patch_size, y : y+self.patch_size, :]

        img_n = _np2Tensor(img_n)
        img_c = _np2Tensor(img_c)
        return img_n, img_c

    def __len__(self):
        if self.isTrain:
            return len(self.images_n) * self.repeat
        else:
            return len(self.images_n)

    def _get_index(self, idx):
        if self.isTrain:
            return idx % len(self.images_n)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        file_n = self.images_n[idx]
        file_c = self.images_c[idx]
        if self.args.ext == 'img':
            img_n = misc.imread(file_n)
            if np.max(img_n)>1: img_n = img_n/255.0
            img_c = misc.imread(file_c)
            if np.max(img_c)>1: img_c = img_c/255.0
        elif self.args.ext.find('sep') >= 0:
            img_n = np.load(file_n)
            img_c = np.load(file_c)
        else:
            assert False

        filename_n = os.path.splitext(os.path.split(file_n)[-1])[0]
        filename_c = os.path.splitext(os.path.split(file_c)[-1])[0]

        return img_n, img_c, filename_n, filename_c

class Rain100_Test(data.Dataset):
    def __init__(self, args):
        self.args=args

        self._set_filesystem(args.input_path_test, args.gt_path_test) 
        self.images_n, self.images_c = self._scan()

    def _set_filesystem(self, dir_n, dir_c):
        self.dir_n = dir_n
        self.dir_c = dir_c
        self.ext = '.png'
        print('********* dir_n and dir_c ******')
        print(self.dir_n)
        print(self.dir_c)

    def _scan(self):
        
        list_c = sorted(
            [os.path.join(self.dir_c, x) for x in os.listdir(self.dir_c) if is_img(x)])
        list_n = [os.path.splitext(x)[0]+'x2.png' for x in list_c]
        list_n = [os.path.join(self.dir_n, os.path.split(x)[-1]) for x in list_n]
        # list_n = sorted(
        #     [os.path.join(self.dir_n, x) for x in os.listdir(self.dir_n) if is_img(x)])

        return list_n, list_c


    def __getitem__(self, idx):
        img_n, img_c, filename_n, filename_c = self._load_file(idx)
        assert img_n.shape==img_c.shape

        img_n = _np2Tensor(img_n)
        img_c = _np2Tensor(img_c)
        return img_n, img_c, filename_n, filename_c

    def __len__(self):
        return len(self.images_n)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        file_n = self.images_n[idx]
        file_c = self.images_c[idx]
        img_n = misc.imread(file_n)
        if np.max(img_n)>1: img_n = img_n/255.0
        img_c = misc.imread(file_c)
        if np.max(img_c)>1: img_c = img_c/255.0

        filename_n = os.path.splitext(os.path.split(file_n)[-1])[0]
        filename_c = os.path.splitext(os.path.split(file_c)[-1])[0]

        return img_n, img_c, filename_n, filename_c


if __name__=='__main__':

    # Prepare Data
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--patch_size', default=64)
    parser.add_argument('--input_path', default="./Data/Rain100_Train/rain/")
    parser.add_argument('--gt_path', default="./Data/Rain100_Train/norain/")
    parser.add_argument('--ext', default='img')
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    dataset = Rain100(args, isTrain=False)

    # img_n,img_c,file_n, file_c = dataset._load_file(1)
    img_n,img_c = dataset.__getitem__(10)
    import pdb; pdb.set_trace()
    
    plt.figure(1)
    plt.imshow(np.transpose(img_n.numpy(),(1,2,0)))
    plt.figure(2)
    plt.imshow(np.transpose(img_c.numpy(),(1,2,0)))
    plt.show()