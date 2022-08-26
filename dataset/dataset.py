#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import lmdb
import six
import sys
import bisect
import warnings
from PIL import Image
import numpy as np
import cv2
import os

sys.path.append('../')
from utils import str_filt
from utils.labelmaps import get_vocabulary, labels2strs
from IPython import embed

from utils import utils_deblur
from utils import utils_sisr as sr
from utils import utils_image as util
import imgaug.augmenters as iaa

from scipy import io as sio
scale = 0.90
kernel = utils_deblur.fspecial('gaussian', 15, 1.)
noise_level_img = 0.


def rand_crop(im):
    w, h = im.size
    p1 = (random.uniform(0, w*(1-scale)), random.uniform(0, h*(1-scale)))
    p2 = (p1[0] + scale*w, p1[1] + scale*h)
    return im.crop(p1 + p2)


def central_crop(im):
    w, h = im.size
    p1 = (((1-scale)*w/2), (1-scale)*h/2)
    p2 = ((1+scale)*w/2, (1+scale)*h/2)
    return im.crop(p1 + p2)


def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im


class lmdbDataset(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, word2idx=None):
        super(lmdbDataset, self).__init__()
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot creat lmdb from %s' % root)
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
            print("root:", root)
            print("nSamples:", nSamples)

            # for batching
            self.hr_keys = [key for key, _ in txn.cursor() if b'hr' in key]
            self.lr_keys = [key for key, _ in txn.cursor() if b'lr' in key]
            self.label_keys = [key for key, _ in txn.cursor() if b'label' in key]

        self.voc_type = voc_type
        self.max_len = max_len
        self.word2idx = word2idx

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # index += 1
        txn = self.env.begin(write=False)
        label_key = self.label_keys[index]
        HR_key = self.hr_keys[index]    # 128*32
        LR_key = self.lr_keys[index]    # 64*16
        word = ""
        try:
            img_HR = buf2PIL(txn, HR_key, 'RGB')
            img_LR = buf2PIL(txn, LR_key, 'RGB')
            word = txn.get(label_key)
            if word is None:
                print("None word:", label_key)
                word = " "
            else:
                word = str(word.decode())
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)

        return img_HR, img_LR, label_str


class lmdbDataset_kor(Dataset):
    def __init__(self, root=None, voc_type='korean', max_len=100, word2idx=None):
        super(lmdbDataset_kor, self).__init__()
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot creat lmdb from %s' % root)
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
            print("root:", root)
            print("nSamples:", nSamples)

            # for batching
            self.hr_keys = [key for key, _ in txn.cursor() if b'hr' in key]
            self.lr_keys = [key for key, _ in txn.cursor() if b'lr' in key]
            self.label_keys = [key for key, _ in txn.cursor() if b'label' in key]

        self.voc_type = voc_type
        self.max_len = max_len
        self.word2idx = word2idx

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # index += 1
        txn = self.env.begin(write=False)
        label_key = self.label_keys[index]
        HR_key = self.hr_keys[index]    # 128*32
        LR_key = self.lr_keys[index]    # 64*16
        word = ""
        try:
            img_HR = buf2PIL(txn, HR_key, 'RGB')
            img_LR = buf2PIL(txn, LR_key, 'RGB')
            word = txn.get(label_key)
            if word is None:
                print("None word:", label_key)
                word = " "
            else:
                word = str(word.decode())
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)

        target = [self.word2idx.get(ch, 1) for ch in label_str]
        target.insert(0, 2)
        target.append(3)
        target = np.array(target)

        return img_HR, img_LR, label_str, target


class lmdbDataset_eng(Dataset):
    def __init__(self, root=None, voc_type='korean', max_len=100, word2idx=None):
        super(lmdbDataset_eng, self).__init__()
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot creat lmdb from %s' % root)
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
            print("root:", root)
            print("nSamples:", nSamples)

            # for batching
            self.hr_keys = [key for key, _ in txn.cursor() if b'hr' in key]
            self.lr_keys = [key for key, _ in txn.cursor() if b'lr' in key]
            self.label_keys = [key for key, _ in txn.cursor() if b'label' in key]

        self.voc_type = voc_type
        self.max_len = max_len
        self.word2idx = word2idx

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # index += 1
        txn = self.env.begin(write=False)
        label_key = self.label_keys[index]
        HR_key = self.hr_keys[index]    # 128*32
        LR_key = self.lr_keys[index]    # 64*16
        word = ""
        try:
            img_HR = buf2PIL(txn, HR_key, 'RGB')
            img_LR = buf2PIL(txn, LR_key, 'RGB')
            word = txn.get(label_key)
            if word is None:
                print("None word:", label_key)
                word = " "
            else:
                word = str(word.decode())
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)

        target = [self.word2idx.get(ch, 1) for ch in label_str]
        target.insert(0, 2)
        target.append(3)
        target = np.array(target)

        return img_HR, img_LR, label_str, target


class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC, aug=None, blur=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask
        self.aug = aug

        self.blur = blur

    def __call__(self, img, ratio_keep=False):

        size = self.size

        if ratio_keep:
            ori_width, ori_height = img.size
            ratio = float(ori_width) / ori_height

            if ratio < 3:
                width = 100# if self.size[0] == 32 else 50
            else:
                width = int(ratio * self.size[1])

            size = (width, self.size[1])

        # print("size:", size)
        img = img.resize(size, self.interpolation)

        if self.blur:
            # img_np = np.array(img)
            # img_np = cv2.GaussianBlur(img_np, (5, 5), 1)
            #print("in degrade:", np.unique(img_np))
            # img_np = noisy("gauss", img_np).astype(np.uint8)
            # img_np = apply_brightness_contrast(img_np, 40, 40).astype(np.uint8)
            # img_np = JPEG_compress(img_np)

            # img = Image.fromarray(img_np)
            pass

        if not self.aug is None:
            img_np = np.array(img)
            # print("imgaug_np:", imgaug_np.shape)
            imgaug_np = self.aug(images=img_np[None, ...])
            img = Image.fromarray(imgaug_np[0, ...])

        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor


class NormalizeOnly(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC, aug=None, blur=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask
        self.aug = aug

        self.blur = blur

    def __call__(self, img, ratio_keep=False):

        size = self.size

        if ratio_keep:
            ori_width, ori_height = img.size
            ratio = float(ori_width) / ori_height

            if ratio < 3:
                width = 100# if self.size[0] == 32 else 50
            else:
                width = int(ratio * self.size[1])

            size = (width, self.size[1])

        # print("size:", size)
        # img = img.resize(size, self.interpolation)

        if self.blur:
            img_np = np.array(img)
            # img_np = cv2.GaussianBlur(img_np, (5, 5), 1)
            #print("in degrade:", np.unique(img_np))
            # img_np = noisy("gauss", img_np).astype(np.uint8)
            # img_np = apply_brightness_contrast(img_np, 40, 40).astype(np.uint8)
            # img_np = JPEG_compress(img_np)

            img = Image.fromarray(img_np)

        if not self.aug is None:
            img_np = np.array(img)
            # print("imgaug_np:", imgaug_np.shape)
            imgaug_np = self.aug(images=img_np[None, ...])
            img = Image.fromarray(imgaug_np[0, ...])

        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor



class resizeNormalizeRandomCrop(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img, interval=None):

        w, h = img.size

        if w < 32 or not interval is None:
            img = img.resize(self.size, self.interpolation)
            img_tensor = self.toTensor(img)
        else:
            np_img = np.array(img)
            h, w = np_img.shape[:2]
            np_img_crop = np_img[:, int(w * interval[0]):int(w * interval[1])]
            # print("size:", self.size, np_img_crop.shape, np_img.shape, interval)
            img = Image.fromarray(np_img_crop)
            img = img.resize(self.size, self.interpolation)
            img_tensor = self.toTensor(img)

        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor


class resizeNormalizeKeepRatio(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img, label_str):
        o_w, o_h = img.size

        ratio = o_w / float(o_h)
        re_h = self.size[1]
        re_w = int(re_h * ratio)
        if re_w > self.size[0]:
            img = img.resize(self.size, self.interpolation)
            img_tensor = self.toTensor(img).float()
        else:
            img = img.resize((re_w, re_h), self.interpolation)
            img_np = np.array(img)
            # if len(label_str) > 4:
            #     print("img_np:", img_np.shape)

            shift_w = int((self.size[0] - img_np.shape[1]) / 2)
            re_img = np.zeros((self.size[1], self.size[0], img_np.shape[-1]))
            re_img[:, shift_w:img_np.shape[1]+shift_w] = img_np

            re_img = Image.fromarray(re_img.astype(np.uint8))

            img_tensor = self.toTensor(re_img).float()

            if o_h / o_w < 0.5 and len(label_str) > 4:
                # cv2.imwrite("mask_h_" + label_str + ".jpg", re_mask.astype(np.uint8))
                # cv2.imwrite("img_h_" + label_str + ".jpg", np.array(re_img))
                # print("img_np_h:", o_h, o_w, img_np.shape, label_str)
                pass

        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            if re_w > self.size[0]:
                # img = img.resize(self.size, self.interpolation)

                re_mask_cpy = np.ones((mask.size[1], mask.size[0]))

                mask = self.toTensor(mask)
                img_tensor = torch.cat((img_tensor, mask), 0).float()
            else:
                mask = np.array(mask)
                mask = cv2.resize(mask, (re_w, re_h), cv2.INTER_NEAREST)
                shift_w = int((self.size[0] - mask.shape[1]) / 2)

                # print("resize mask:", mask.shape)

                re_mask = np.zeros((self.size[1], self.size[0]))

                re_mask_cpy = re_mask.copy()
                re_mask_cpy[:, shift_w:mask.shape[1] + shift_w] = np.ones(mask.shape)

                re_mask[:, shift_w:mask.shape[1] + shift_w] = mask
                '''
                if o_h / o_w > 2 and len(label_str) > 4:
                    cv2.imwrite("mask_" + label_str + ".jpg", re_mask.astype(np.uint8))
                    cv2.imwrite("img_" + label_str + ".jpg", re_img.astype(np.uint8))
                    print("img_np:", o_h, o_w, img_np.shape, label_str)

                if o_h / o_w < 0.5 and len(label_str) > 4:
                    cv2.imwrite("mask_h_" + label_str + ".jpg", re_mask.astype(np.uint8))
                    cv2.imwrite("img_h_" + label_str + ".jpg", re_img.astype(np.uint8))
                    print("img_np_h:", o_h, o_w, img_np.shape, label_str)
                '''
                re_mask = self.toTensor(re_mask).float()
                img_tensor = torch.cat((img_tensor, re_mask), 0)

        return img_tensor, torch.tensor(cv2.resize(re_mask_cpy, (self.size[0] * 2, self.size[1] * 2), cv2.INTER_NEAREST)).float()


class alignCollate_syn(object):
    def __init__(self, imgH=64,
                 imgW=256,
                 down_sample_scale=4,
                 keep_ratio=False,
                 min_ratio=1,
                 mask=False,
                 alphabet=53,
                 train=True,
                 y_domain=False
                 ):

        sometimes = lambda aug: iaa.Sometimes(0.2, aug)

        aug = [
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.AverageBlur(k=(1, 5)),
            iaa.MedianBlur(k=(3, 7)),
            iaa.BilateralBlur(
                d=(3, 9), sigma_color=(10, 250), sigma_space=(10, 250)),
            iaa.MotionBlur(k=3),
            iaa.MeanShiftBlur(),
            iaa.Superpixels(p_replace=(0.1, 0.5), n_segments=(1, 7))
        ]

        self.aug = iaa.Sequential([sometimes(a) for a in aug], random_order=True)

        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask

        imgH = self.imgH
        imgW = self.imgW

        self.transform = resizeNormalize((imgW, imgH), self.mask)
        self.transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask, blur=True)
        self.transform_pseudoLR = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask, aug=self.aug)

        self.train = train

    def degradation(self, img_L):
        # degradation process, blur + bicubic downsampling + Gaussian noise
        # if need_degradation:
        # img_L = util.modcrop(img_L, sf)
        img_L = np.array(img_L)
        # print("img_L_before:", img_L.shape, np.unique(img_L))
        img_L = sr.srmd_degradation(img_L, kernel)

        noise_level_img = 0.
        # print("unique:", np.unique(img_L))
        img_L = img_L + np.random.normal(0, noise_level_img, img_L.shape)

        # print("img_L_after:", img_L_beore.shape, img_L.shape, np.unique(img_L))

        return Image.fromarray(img_L.astype(np.uint8))

    def __call__(self, batch):
        images, images_lr, _, _, label_strs = zip(*batch)

        # [self.degradation(image) for image in images]
        # images_hr = images
        '''
        images_lr = [image.resize(
            (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),
            Image.BICUBIC) for image in images]

        if self.train:
            if random.random() > 1.5:
                images_hr = [image.resize(
                (image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale),
                Image.BICUBIC) for image in images]
            else:
                images_hr = images
        else:
            images_hr = images
            #[image.resize(
            #    (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),
            #    Image.BICUBIC) for image in images]
        '''
        # images_hr = [self.degradation(image) for image in images]
        images_hr = images
        #images_lr = [image.resize(
        #     (image.size[0] // 4, image.size[1] // 4),
        #     Image.BICUBIC) for image in images_lr]
        # images_lr = images

        #images_lr_new = []
        #for image in images_lr:
        #    image_np = np.array(image)
        #    image_aug = self.aug(images=image_np[None, ])[0]
        #    images_lr_new.append(Image.fromarray(image_aug))
        #images_lr = images_lr_new

        images_hr = [self.transform(image) for image in images_hr]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        if self.train:
            images_lr = [image.resize(
            (image.size[0] // 2, image.size[1] // 2), # self.down_sample_scale
            Image.BICUBIC) for image in images_lr]
        else:
            pass
        #    # for image in images_lr:
        #    #     print("images_lr:", image.size)
        #    images_lr = [image.resize(
        #         (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),  # self.down_sample_scale
        #        Image.BICUBIC) for image in images_lr]
        #    pass
        # images_lr = [self.degradation(image) for image in images]
        images_lr = [self.transform2(image) for image in images_lr]

        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        max_len = 26

        label_batches = []
        weighted_tics = []
        weighted_masks = []

        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                word = [ch for ch in word]
                word[2] = "e"
                word = "".join(word)

            if len(word) <= 1:
                pass
            elif len(word) < 26 and len(word) > 1:
                #inter_com = 26 - len(word)
                #padding = int(inter_com / (len(word) - 1))
                #new_word = word[0]
                #for i in range(len(word) - 1):
                #    new_word += "-" * padding + word[i + 1]

                #word = new_word
                pass
            else:
                word = word[:26]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            if len(label_list) <= 0:
                # blank label
                weighted_masks.append(0)
            else:
                weighted_masks.extend(label_list)

            labels = torch.tensor(label_list)[:, None].long()
            label_vecs = torch.zeros((labels.shape[0], self.alsize))
            # print("labels:", labels)
            #if labels.shape[0] > 0:
            #    label_batches.append(label_vecs.scatter_(-1, labels, 1))
            #else:
            #    label_batches.append(label_vecs)

            if labels.shape[0] > 0:
                label_vecs = torch.zeros((labels.shape[0], self.alsize))
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
                weighted_tics.append(1)
            else:
                label_vecs = torch.zeros((1, self.alsize))
                label_vecs[0, 0] = 1.
                label_batches.append(label_vecs)
                weighted_tics.append(0)

        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)

        # print(images_lr.shape, images_hr.shape)

        return images_hr, images_lr, images_hr, images_lr, label_strs, label_rebatches, torch.tensor(weighted_masks).long(), torch.tensor(weighted_tics)


class alignCollate(alignCollate_syn):
    """
    For using CRNN as TPG.
    """
    def __call__(self, batch):
        images_HR, images_lr, label_strs = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW

        images_HR = [self.transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [self.transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        return images_HR, images_lr, label_strs


class alignCollate_kor(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, label_strs, targets = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW

        images_HR = [self.transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [self.transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        # max_len = max(len(target) for target in targets)
        max_len = 27

        targets_ = []
        for target in targets:
            d = max_len - target.shape[0]
            target = np.pad(target, (0, d), 'constant')
            targets_.append(target)
        target_batch = torch.LongTensor(np.array(targets_))

        return images_HR, images_lr, label_strs, target_batch


class alignCollate_eng(alignCollate_syn):
    """
    For using CDistNet_eng as TPG.
    """
    def __call__(self, batch):
        images_HR, images_lr, label_strs, targets = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW

        images_HR = [self.transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [self.transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        # max_len = max(len(target) for target in targets)
        max_len = 27

        targets_ = []
        for target in targets:
            d = max_len - target.shape[0]
            if d < 0:
                target = target[:max_len]
                target[-1] = 3
            else:
                target = np.pad(target, (0, d), 'constant')
            targets_.append(target)
        target_batch = torch.LongTensor(np.array(targets_))

        return images_HR, images_lr, label_strs, target_batch


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
