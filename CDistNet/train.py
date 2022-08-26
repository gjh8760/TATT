import os
import shutil
import argparse
import logging
from mmcv import Config
from thop import profile
import torch
import torch.nn as nn
import torch.distributed as dist

from cdistnet.engine.trainer import Trainer
from cdistnet.model.model import build_CDistNet
from cdistnet.data.data import make_dataloader
from cdistnet.utils.tensorboardx import TensorboardLogger

def parse_args():
    def str2bool(x):
        return x.lower() == 'true'
    parser = argparse.ArgumentParser(description='Train CDistNet')
    parser.add_argument('--config', type=str, default='configs/CDistNet_config.py',help='train config file path')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index for training')
    parser.add_argument('--initial_test', default=False, type=str2bool, help='test model save & validation at start of training')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.dist_train:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        cfg.local_rank = args.local_rank
    if cfg.dist_train == False or args.local_rank == 0:
        if not os.path.exists(cfg.train.saved_model_dir):
            os.makedirs(cfg.train.saved_model_dir)
        # shutil.copy(args.config, cfg.train.saved_model_dir)   # TODO: remove comment
    return cfg,args

def getlogger(model_dir):
    logger = logging.getLogger('CDistNet')
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(os.path.join(model_dir, 'log.txt' ))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger
    
def train(cfg, args):
    if cfg.dist_train:
        dist.init_process_group(backend='nccl')
        cfg.train.batch_size //= cfg.num_procs
        cfg.train.num_workers //= cfg.num_procs
        cfg.val.batch_size //= cfg.num_procs
        cfg.val.num_workers //= cfg.num_procs
    # model
    model = build_CDistNet(cfg)
    # dataloaders
    val_dataloaders = []
    for val_image_dir in cfg.val.image_dir:
        val_dataloaders.append(make_dataloader(cfg, is_train=False, val_image_dir=val_image_dir))
    train_dataloader = make_dataloader(cfg, is_train=True)
    
    # logger
    is_master = cfg.dist_train == False or cfg.local_rank == 0
    logger = getlogger(cfg.train.saved_model_dir) if is_master else None
    tb_logger = TensorboardLogger(cfg.train.saved_model_dir) if is_master else None
    if is_master:
        logger.info("================================================\nSTART TRAINING\n")
        logger.info(f"config file: {args.config}")
        logger.info(f"save dir: {cfg.train.saved_model_dir}")
    trainer = Trainer(
        model=model,
        saved_model = cfg.train.saved_model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloaders,
        num_epochs=cfg.train.num_epochs,
        logger=logger,
        tb_logger=tb_logger,
        display_iter=cfg.train.display_iter,
        tfboard_iter=cfg.train.tfboard_iter,
        val_iter=cfg.train.val_iter,
        model_dir=cfg.train.saved_model_dir,
        label_smoothing=cfg.train.label_smoothing, # TODO : figure out what label_smoothing does
        grads_clip=cfg.train.grads_clip,
        cfg=cfg,
        args=args,
    )
    trainer.fit()

def main():
    cfg,args = parse_args()
    train(cfg,args)

if __name__ == '__main__':
    main()
