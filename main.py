import yaml
import sys
import numpy as np
import torch
import argparse
import os
from easydict import EasyDict


# Fix random seed
import random
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config, args):
    if args.kor:
        from interfaces.super_resolution_kor import TextSR
    elif args.tpg == 'cdistnet_eng':
        from interfaces.super_resolution_eng_cdistnet import TextSR
    else:
        from interfaces.super_resolution_eng import TextSR
    Mission = TextSR(config, args)

    if args.test:
        Mission.test()
    else:
        Mission.train()


if __name__ == '__main__':

    def str2bool(x):
        return x.lower() in 'true'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='/data/gjh8760/Dataset/STISR/TextZoom/test/easy', help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    parser.add_argument('--vis_dir', type=str, default='TATT/', help='')
    parser.add_argument('--STN', type=str2bool, default='True', help='')
    parser.add_argument('--mask', action='store_true', default=True, help='')
    parser.add_argument('--gradient', action='store_true', default=True, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--stu_iter', type=int, default=1, help='Default is set to 1, must be used with --arch=tsrn_tl_cascade')
    parser.add_argument('--test_model', type=str, default='ASTER', help='aster, crnn, moran')  # CDistNet for kor
    parser.add_argument('--tpg', type=str, default="CRNN", help='model to generate TP')        # CDistNet for kor
    parser.add_argument('--config', type=str, default='super_resolution.yaml')
    parser.add_argument('--kor', type=str2bool, default='False', help='True for korean model, False for English model')
    # parser.add_argument('--config_cdistnet', type=str, default='CDistNet/ckpts/yeti_2022-07-11/CDistNet_config_kor.py', help='train config file path for CDistNet')
    parser.add_argument('--tssim_rotation_degree', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--tpg_lr', type=float, default=0.0001, help='')
    parser.add_argument('--training_stablize', action='store_true', default=False)

    args = parser.parse_args()

    if args.kor:
        args.config = 'super_resolution_kor.yaml'
        args.tpg = 'cdistnet_kor'
        args.test_model = 'cdistnet_kor'
    else:
        args.config = 'super_resolution.yaml'

    config_path = os.path.join('config', args.config)
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)

    config.TRAIN.lr = args.learning_rate

    main(config, args)
