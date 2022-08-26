"""
Convert between images dataset and lmdb dataset.

"""
import os
import six
import lmdb
import argparse
import re
from tqdm import tqdm
from mmcv import Config
from PIL import Image

def parse_args():
    def str2bool(x):
        return x.lower() in ['true']
    parser = argparse.ArgumentParser()
    parser.add_argument('--to_lmdb', type=str2bool, default=False, help='condition of whether images dataset to lmdb (True) or lmdb to images dataset (False).')
    parser.add_argument('--config', type=str, default='configs/CDistNet_config_convert.py', help='path to CDistNet configuration file.' )
    parser.add_argument('--convert_only_labels', type=str2bool, default=False, help='save only labels.' )
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    return cfg, args


def convert_lmdb_to_images(cfg):
    lmdb_dirs = list()
    lmdb_dirs.extend(cfg.val.image_dir)
    lmdb_dirs.extend(cfg.train.image_dir)
    for lmdb_dir in lmdb_dirs:
        print(f'Processing {lmdb_dir}...')
        image_dir = lmdb_dir.replace('lmdb', 'images')
        if not os.path.exists(image_dir): os.makedirs(image_dir)
        labels = list()
        env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            length = int(txn.get('num-samples'.encode()))
            print("samples = {}".format(length))
            for idx in tqdm(range(length)):
                image_key, label_key = f'image-{idx+1:09d}', f'label-{idx+1:09d}'
                label = str(txn.get(label_key.encode()), 'utf-8')  # label
                if not args.convert_only_labels:
                    imgbuf = txn.get(image_key.encode())  # image
                    buf = six.BytesIO()
                    buf.write(imgbuf)
                    buf.seek(0)
                    image = Image.open(buf).convert('RGB')
                    image.save(os.path.join(image_dir, image_key + '.png'))
                label_ = label
                label = re.sub('[^0-9a-zA-Z]+', '', label)
                if label != label_:
                    print(f'filtered alphanumeric : {label_}\t->\t{label}')
                label_ = label
                label = label[:30]
                if label != label_:
                    print(f'cut length 30 : {label_}\t->\t{label}')
                labels.append(' '.join([image_key, label]))
        with open(os.path.join(image_dir, 'gt.txt'), 'w') as f:
            f.writelines([f'{label}\n' for label in labels])


def convert_images_to_lmdb(cfg):
    image_dirs = list()
    image_dirs.extend(cfg.val.image_dir)
    image_dirs.extend(cfg.train.image_dir)
    for image_dir in image_dirs:
        print(f'Processing {image_dir}...')
        lmdb_dir = image_dir.replace('images', 'lmdb')
        if not os.path.exists(lmdb_dir): os.makedirs(lmdb_dir)
        with open(os.path.join(image_dir, 'gt.txt'), encoding='utf-8') as f:
            labels = f.read().splitlines()
        labels = [label.split(' ') for label in labels]
        map_size = 20*1024**3 if 'train' in image_dir else 4*1024**3
        env = lmdb.open(lmdb_dir, map_size=map_size)
        with env.begin(write=True) as txn:
            txn.put('num-samples'.encode(), str(len(labels)).encode())
            for image_key, label in tqdm(labels):
                label_key = image_key.replace('image', 'label')
                with open(os.path.join(image_dir, image_key + '.jpg'), 'rb') as f:
                    imgbuf = f.read()
                txn.put(image_key.encode(), imgbuf)
                txn.put(label_key.encode(), label.encode())
        print(f'Created dataset with {len(labels)} samples')


def main(cfg, args):
    if args.to_lmdb:
        convert_images_to_lmdb(cfg)
    else:
        convert_lmdb_to_images(cfg)


if __name__ == "__main__":
    cfg, args = parse_args()
    main(cfg, args)