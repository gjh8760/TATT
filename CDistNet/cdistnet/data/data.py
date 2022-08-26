import os
import re
import glob
import time
import copy
import codecs
import pickle
import numpy as np
import six
from PIL import Image
from PIL import ImageFile
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from torchvision.transforms import functional as F
from torchvision import transforms
import argparse
from mmcv import Config
from tqdm import tqdm
import lmdb
from typing import Sized, Optional, Iterator

from cdistnet.data.transform import CVGeometry, CVColorJitter, CVDeterioration
from cdistnet.data.jamoconverter import JamoConverter

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_vocab(vocab=None, vocab_size=None):
    """
    Load vocab from disk. The first four items in the vocab should be <PAD>, <UNK>, <S>, </S>
    """
    # print('Load set vocabularies as %s.' % vocab)
    vocab = [' ' if len(line.split()) == 0 else line.split()[0] for line in codecs.open(vocab, 'r', 'utf-8')]
    vocab = vocab[:vocab_size]
    assert len(vocab) == vocab_size
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


class PILDataset(Dataset):
    """Read images as PIL"""

    def __init__(
            self,
            image_dir,
            word2idx,
            idx2word,
            size=(100, 32),
            rotate=False,
            max_width=256,
            rgb2gray=True,
            keep_aspect_ratio=False,
            is_lower=False,
            data_aug=True,
            train_val_test=0,
            no_label=False,
            dst_vocab=None,
            is_master=True,
    ):
        self.image_dir = image_dir
        self.gt = dict()
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.rgb2gray = rgb2gray
        self.width = size[0]
        self.height = size[1]
        self.rotate = rotate
        self.keep_aspect_ratio = keep_aspect_ratio
        self.max_width = max_width
        self.is_lower = is_lower
        self.data_aug = data_aug
        self.train_val_test = train_val_test
        self.is_master = is_master
        self.dst_vocab = dst_vocab
        self.jamoconverter = JamoConverter()
        if self.is_master:
            print("preparing data ...")
            print("path:{}".format(image_dir + '/gt.txt'))
        if not no_label:
            with open(image_dir + '/gt.txt', 'r', encoding='UTF-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(' ')
                    image_name, label = line[0], ' '.join(line[1:])
                    label = label.lower() if is_lower else label
                    label = self.filter_label(label)
                    label = self.convert_label(label)
                    self.gt[image_name] = label
        else:
            exts = ('.png', '.PNG', '.JPG', '.jpg')
            image_paths = []
            for ext in exts:
                image_paths.extend(glob.glob(os.path.join(image_dir, '*' + ext)))
            image_paths = sorted(image_paths)
            for image_path in image_paths:
                image_basename = os.path.basename(image_path)
                image_name = os.path.splitext(image_basename)[0]
                self.gt[image_name] = '' # empty string
            
        self.data = list(self.gt.items())

        # No transformation for val, test
        if self.train_val_test == 0 and self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])

    def __len__(self):
        return len(self.data)

    def filter_label(self, label):
        """
        Filter out unwanted characters in labels.
        self.dst_vocab path should contain the following keywords for correct filtering :
        - num : numbers
        - sym : symbols
        - eng : lowercase alphabet
        - Eng : uppercase alphabet
        - kor : Korean
        - jamo : Korean in jamo
        """
        re_pattern = '[^'
        if 'num' in self.dst_vocab: re_pattern += '0-9'
        if 'sym' in self.dst_vocab: re_pattern += '~`!@#$%^&*()-_+={[}]|\:;"<,>.?/'
        if 'eng' in self.dst_vocab: re_pattern += 'a-z'
        if 'Eng' in self.dst_vocab: re_pattern += 'A-Z'
        if 'kor' in self.dst_vocab or 'jamo' in self.dst_vocab: re_pattern += '가-힣'
        re_pattern += ']+'
        label = re.sub(re_pattern, '', label)
        return label

    def convert_label(self, label):
        """
        Jamo format needs label conversion
        """
        if 'jamo' in self.dst_vocab:
            label = self.jamoconverter.text2label(label)
        else:
            pass
        return label

    def __getitem__(self, idx):
        image_name = self.data[idx][0]
        exts = ('.png', '.PNG', '.JPG', '.jpg')
        for ext in exts:
            try:
                image_path = os.path.join(self.image_dir, image_name + ext)
                image = Image.open(image_path)
            except:
                pass
        if self.rgb2gray:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        if self.rotate:
            h, w = image.height, image.width
            if h > w:
                image = image.rotate(90, expand=True)
        im = image
        if self.train_val_test == 0 and self.data_aug:
            image = self.augment_tfs(image)
        if self.keep_aspect_ratio:
            h, w = image.height, image.width
            ratio = w / h
            image = image.resize((min(max(int(self.height * ratio), self.height), self.max_width), self.height), Image.ANTIALIAS)
            # height is self.height
            # width keeps aspect ratio as long as
            # self.height < width < self.max_width
        else:
            image = image.resize((self.width, self.height), Image.ANTIALIAS)
        image = np.array(image)
        if self.rgb2gray:
            image = np.expand_dims(image, -1)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 128. - 1.
        # image is normalized to [-1, 1]

        label = self.data[idx][1]
        target = [self.word2idx.get(ch, 1) for ch in label]
        target.insert(0, 2) # <sos>
        target.append(3) # <eos>
        target = np.array(target)

        if self.train_val_test == 2:
            # test
            return image, label, image_name, im
        return image, target


class LMDBDataset(Dataset):
    """Load from lmdb file."""

    def __init__(
            self,
            image_dir,
            word2idx,
            idx2word,
            size=(100, 32),
            rotate=True,
            max_width=256,
            rgb2gray=True,
            keep_aspect_ratio=False,
            is_lower=False,
            data_aug=True,
            train_val_test=0,
            no_label=False,
            dst_vocab=None,
            is_master=True
    ):
        self.image_dir = image_dir
        self.gt = dict()
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.rgb2gray = rgb2gray
        self.width = size[0]
        self.height = size[1]
        self.rotate = rotate
        self.keep_aspect_ratio = keep_aspect_ratio
        self.max_width = max_width
        self.is_lower = is_lower
        self.data_aug = data_aug
        self.train_val_test = train_val_test
        self.is_master = is_master
        self.dst_vocab = dst_vocab
        self.jamoconverter = JamoConverter()

        if self.is_master:
            print("preparing data ...")
            print("path:{}".format(image_dir))
        self.env = lmdb.open(str(image_dir), readonly=True, lock=False, readahead=False, meminit=False)
        assert self.env, f'Cannot open LMDB dataset from {image_dir}.'
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('num-samples'.encode()))
        if self.is_master:
            print("samples = {}".format(self.length))

        # No transformation for val, test
        if self.train_val_test == 0 and self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])

    def __len__(self):
        return self.length

    def filter_label(self, label):
        """
        Filter out unwanted characters in labels.
        self.dst_vocab path should contain the following keywords for correct filtering :
        - num : numbers
        - sym : symbols
        - eng : lowercase alphabet
        - Eng : uppercase alphabet
        - kor : Korean
        - jamo : Korean in jamo
        """
        re_pattern = '[^'
        if 'num' in self.dst_vocab: re_pattern += '0-9'
        if 'sym' in self.dst_vocab: re_pattern += '~`!@#$%^&*()-_+={[}]|\:;"<,>.?/'
        if 'eng' in self.dst_vocab: re_pattern += 'a-z'
        if 'Eng' in self.dst_vocab: re_pattern += 'A-Z'
        if 'kor' in self.dst_vocab or 'jamo' in self.dst_vocab: re_pattern += '가-힣'
        re_pattern += ']+'
        label = re.sub(re_pattern, '', label)
        return label

    def convert_label(self, label):
        """
        Jamo format needs label conversion
        """
        if 'jamo' in self.dst_vocab:
            label = self.jamoconverter.text2label(label)
        else:
            pass
        return label

    def get(self, idx):
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{idx+1:09d}', f'label-{idx+1:09d}'
            label = str(txn.get(label_key.encode()), 'utf-8')  # label
            label = label.lower() if self.is_lower else label
            label = self.filter_label(label)
            label = self.convert_label(label)
            imgbuf = txn.get(image_key.encode())  # image
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            if self.rgb2gray:
                image = Image.open(buf).convert('L')
            else:
                image = Image.open(buf).convert('RGB')
            return image, label, image_key

    def __getitem__(self, idx):
        image, label, image_key = self.get(idx)
        if self.rotate:
            h, w = image.height, image.width
            if h > w:
                image = image.rotate(90, expand=True)
        im = image
        if self.train_val_test == 0 and self.data_aug:
            image = self.augment_tfs(image)
        if self.keep_aspect_ratio:
            h, w = image.height, image.width
            ratio = w / h
            image = image.resize(
                (min(max(int(self.height * ratio), self.height), self.max_width), self.height),
                Image.ANTIALIAS
            )
        else:
            image = image.resize((self.width, self.height), Image.ANTIALIAS)
        image = np.array(image)
        if self.rgb2gray:
            image = np.expand_dims(image, -1)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 128. - 1.

        target = [self.word2idx.get(ch, 1) for ch in label]
        target.insert(0, 2)
        target.append(3)
        target = np.array(target)
        if self.train_val_test == 2:
            # test
            return image, label, image_key, im
        return image, target


class MyConcatDataset(ConcatDataset):
    # if 'attribute' is not an attribute of ConcatDataset, get attribute from Dataset
    def __getattr__(self, attribute):
        return getattr(self.datasets[0], attribute)


def collate_fn(insts):
    # padding for normal size
    try:
        src_insts, tgt_insts = list(zip(*insts))
        src_insts = src_pad(src_insts)
        tgt_insts = tgt_pad(tgt_insts)
    except:
        return None
    return src_insts, tgt_insts


def collate_fn_test(insts):
    try:
        src_insts, gt_insts, name_insts, src_orig_insts = list(zip(*insts))
        src_insts = src_pad(src_insts)
    except:
        return None
    return src_insts, gt_insts, name_insts, src_orig_insts


def src_pad(insts):
    """
    Pad zeros to unify image widths in batch.
    """
    max_w = max(inst.shape[-1] for inst in insts)
    insts_ = []
    for inst in insts:
        d = max_w - inst.shape[-1]
        inst = np.pad(inst, ((0, 0), (0, 0), (0, d)), 'constant')
        insts_.append(inst)
    insts = torch.tensor(np.array(insts_)).to(torch.float32)
    return insts


def tgt_pad(insts):
    """
    Pad <PAD>(==0) to unify target sequence lengths in batch.
    """
    max_len = max(len(inst) for inst in insts)
    insts_ = []
    for inst in insts:
        d = max_len - inst.shape[0]
        inst = np.pad(inst, (0, d), 'constant')
        insts_.append(inst)
    batch_seq = torch.LongTensor(np.array(insts_))
    return batch_seq


def dataset_bag(ds_type, cfg, paths, word2idx, idx2word, train_val_test):
    datasets = [ds_type(
            image_dir=path,
            word2idx=word2idx,
            idx2word=idx2word,
            size=(cfg.width, cfg.height),
            rotate=cfg.rotate,
            max_width=cfg.max_width,
            rgb2gray=cfg.rgb2gray,
            keep_aspect_ratio=cfg.keep_aspect_ratio,
            is_lower=cfg.is_lower,
            data_aug=cfg.data_aug,
            train_val_test=train_val_test, # 0 train, 1 val, 2 test
            no_label=cfg.test.no_label,
            dst_vocab=cfg.dst_vocab,
            is_master=cfg.dist_train == False or cfg.local_rank == 0) for path in paths]
    if len(datasets) > 1: return MyConcatDataset(datasets)
    else: return datasets[0]


def make_dataloader(cfg, is_train=True, val_image_dir=None):
    vocab = cfg.dst_vocab
    vocab_size = cfg.dst_vocab_size
    word2idx, idx2word = load_vocab(vocab, vocab_size)

    if cfg.train.is_train_images or cfg.val.is_val_images:
        ds_type = PILDataset
    else:
        ds_type = LMDBDataset
    
    train_val_test = 0 if is_train == True else 1
    if train_val_test == 0:
        dataset = dataset_bag(ds_type, cfg,
                    cfg.train.image_dir,
                    word2idx, idx2word, train_val_test=train_val_test)
    else:
        dataset = dataset_bag(ds_type, cfg,
                    [val_image_dir],
                    word2idx, idx2word, train_val_test=train_val_test)
                    
    if cfg.dist_train:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg.train.batch_size if is_train else cfg.val.batch_size,
            num_workers=cfg.train.num_workers if is_train else cfg.val.num_workers,
            pin_memory=False if cfg.keep_aspect_ratio else True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg.train.batch_size if is_train else cfg.val.batch_size,
            shuffle=False, # data is shuffled in Dataset
            num_workers=cfg.train.num_workers if is_train else cfg.val.num_workers,
            pin_memory=False if cfg.keep_aspect_ratio else True,
            collate_fn=collate_fn,
        )
    return dataloader


def make_dataloader_test(cfg, image_dir):
    vocab = cfg.dst_vocab
    vocab_size = cfg.dst_vocab_size
    word2idx, idx2word = load_vocab(vocab, vocab_size)
    if cfg.train.is_train_images or cfg.test.is_test_images:
        ds_type = PILDataset
    else:
        ds_type = LMDBDataset
    dataset = dataset_bag(ds_type, cfg,
                          paths = [image_dir],
                          word2idx=word2idx, idx2word=idx2word, train_val_test=2)
    if cfg.dist_train:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_workers,
        pin_memory=False if cfg.keep_aspect_ratio else True,
        collate_fn=collate_fn_test,
        sampler=train_sampler if cfg.dist_train else None,
    )
    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CDistNet')
    parser.add_argument('--config', type=str, help='train config file path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    data_loader = make_dataloader(cfg, is_train=True)
    for idx, batch in enumerate(data_loader):
        print(batch[0].shape)
