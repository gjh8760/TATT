import os
import shutil
import argparse
import codecs
import subprocess
import csv
import glob
from tqdm import tqdm
from mmcv import Config
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import random
from PIL import Image
from statistics import mean


from cdistnet.data.data import make_dataloader_test
from cdistnet.model.translator import Translator
from cdistnet.model.model import build_CDistNet
from cdistnet.data.jamoconverter import JamoConverter

import warnings
warnings.filterwarnings(action='ignore')


def parse_args():
    def str2bool(x):
        return x.lower() in ['true']
    parser = argparse.ArgumentParser(description='Train CDistNet')
    parser.add_argument('--config', default='ckpts/yeti_2022-07-11/CDistNet_config_kor.py', help='train config file path')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index for training')
    args = parser.parse_args()
    return args


def init_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)


def load_vocab(vocab=None, vocab_size=None):
    """
    Load vocab from disk. The fisrt four items in the vocab should be <PAD>, <UNK>, <S>, </S>
    """
    # print('Load set vocabularies as %s.' % vocab)
    vocab = [' ' if len(line.split()) == 0 else line.split()[0] for line in codecs.open(vocab, 'r', 'utf-8')]
    vocab = vocab[:vocab_size]
    assert len(vocab) == vocab_size
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def average_models(model, models):
    """Average models into model"""
    with torch.no_grad():
        for key in model.state_dict().keys():
            v = []
            for m in models:
                v.append(m.state_dict()[key])
            v = torch.sum(torch.stack(v), dim=0) / len(v)
            model.state_dict()[key].copy_(v)


def get_predictions(translator, idx2word, b_image, b_gt, b_name, rgb2gray, jamoconverter):
    """Return predicted words, groundtruth words, and image-keys

    Returns:
        predictions (list, tuple) : Element is tuple of predicted word, gt word, image-key
            e.g. ('privat', 'private', 'image-000000122')
    """
    predictions = []

    if rgb2gray == False:
        b_image = b_image[:, :3, :, :]
    else:
        b_image = b_image[:, 0:1, :, :]

    batch_hyp, _ = translator.translate_batch(images=b_image)

    for idx, seq in enumerate(batch_hyp):
        # convert char_idx to char
        seq = [x for x in seq if x != 3]
        pred = [idx2word[x] for x in seq]
        pred = ''.join(pred)
        gt = b_gt[idx]
        gt = ''.join(gt)
        image_key = b_name[idx]
        if jamoconverter is not None:
            gt = jamoconverter.str2text(gt)
            pred = jamoconverter.str2text(pred)
        predictions.append((pred, gt, image_key))
    
    return predictions


def get_accuracy(predictions):
    total_word = len(predictions)
    correct_word = 0
    for pred, gt, name in predictions:
        if pred.lower() == gt.lower():
            correct_word += 1
        else:
            continue
    return correct_word / total_word * 100.0


def save_tensor_as_image(image_tensor, name='temp.png'):
    """
    Function for debugging purposes.
    Save tensor image in debug folder.
    """
    image_tensor = image_tensor.clone().detach().cpu().numpy()
    image_tensor = image_tensor.transpose(1, 2, 0)
    image_tensor += 1.0
    image_tensor *= 128
    im = Image.fromarray(image_tensor.astype(np.uint8))
    os.makedirs('./debug', exist_ok=True)
    im.save(f'./debug/{name}')


def eval(cfg, args, saved_model_path):
    # load model
    model = build_CDistNet(cfg)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(f'cuda:{args.gpu}')
    device = torch.device('cuda')
    ckpt = torch.load(saved_model_path, map_location=device)
    try:
        model.load_state_dict(ckpt['model'])
    except:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # ineffective for producing same results across GPUs
    ##########################
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    ##########################


    # prediction utils
    translator = Translator(cfg, model=model)
    _, idx2word = load_vocab(cfg.dst_vocab, cfg.dst_vocab_size)
    jamoconverter = JamoConverter() if 'jamo' in cfg.dst_vocab else None

    accs = []
    # set path to save evaluation results
    results_dir = os.path.join(cfg.test.saved_model_dir, 'results')
    init_dir(results_dir)
    for eval_dataset_dir in cfg.test.image_dir:
        dataset_name = os.path.basename(eval_dataset_dir)
        print(f"dataset name: {dataset_name}")
        # set path to save error images
        os.makedirs(os.path.join(cfg.test.saved_model_dir, 'err'), exist_ok=True)
        if cfg.test.save_err:
            err_image_dir = os.path.join(cfg.test.saved_model_dir, 'err', 'images', dataset_name)
            init_dir(err_image_dir)
        # set path to save correct images
        os.makedirs(os.path.join(cfg.test.saved_model_dir, 'corr'), exist_ok=True)
        if cfg.test.save_corr:
            corr_image_dir = os.path.join(cfg.test.saved_model_dir, 'corr', 'images', dataset_name)
            init_dir(corr_image_dir)

        test_dataloader = make_dataloader_test(cfg, eval_dataset_dir)
        
        predictions = []
        err_predictions = []
        corr_predictions = []
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            b_image, b_gt, b_name, b_orig_image = batch[0], batch[1], batch[2], batch[3]
            # for i in range(len(b_image)):
            #     save_tensor_as_image(b_image[i], f'{i}.png')
            b_predictions = get_predictions(translator, idx2word, b_image, b_gt, b_name, cfg.rgb2gray, jamoconverter)
            predictions.extend(b_predictions)
            
            # save error images
            for image, prediction in zip(b_orig_image, b_predictions):
                pred, gt, image_key = prediction
                if gt.lower() == pred.lower():
                    if cfg.test.save_corr:
                        image.save(os.path.join(corr_image_dir, f'{image_key}_{pred}_{gt}.png'))
                        corr_predictions.append(prediction)
                    else:
                        corr_predictions.append(prediction)
                else:
                    if cfg.test.save_err:
                        image.save(os.path.join(err_image_dir, f'{image_key}_{pred}_{gt}.png'))
                        err_predictions.append(prediction)
                    else:
                        err_predictions.append(prediction)
        
        if not len(dataset_name):
            dataset_name = 'dataset'
            
        # write error
        err_path = os.path.join(cfg.test.saved_model_dir, 'err', dataset_name + '.csv')
        with open(err_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Pred', 'Gt', 'Image-Key'])
            for err_prediction in err_predictions:
                writer.writerow(err_prediction)

        # write correct
        corr_path = os.path.join(cfg.test.saved_model_dir, 'corr', dataset_name + '.csv')
        with open(corr_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Pred', 'Gt', 'Image-Key'])
            for corr_prediction in corr_predictions:
                writer.writerow(corr_prediction)
                
        # write results
        results_path = os.path.join(results_dir, dataset_name + '.csv')
        with open(results_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Pred', 'Gt', 'Image-Key'])
            for prediction in predictions:
                writer.writerow(prediction)

        acc = get_accuracy(predictions)
        print(f"accuracy : {acc:.1f} %\n")
        accs.append(acc)


    avg_acc = mean(accs)
    accs.append(avg_acc)
    result_line = [f"{acc:.1f}" for acc in accs]
    result_line.insert(0, saved_model_path.split('/')[-1])
    print(os.path.join(cfg.test.saved_model_dir, 'result.csv'))
    with open(os.path.join(cfg.test.saved_model_dir, 'result.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(result_line)
    
    print(f"\n\naverage accuracy : {avg_acc:.1f} %\n")


def main():
    """Evaluate the saved ckpts on benchmark.
    Evaluate single ckpt / all ckpts in directory.
    """
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.dist_train = False
    # cfg.local_rank = 0

    # prepare results.csv
    headers = [os.path.basename(dir) for dir in cfg.test.image_dir]
    headers.insert(0, 'Ckpt')
    headers.append('Average')
    headers.append('Notes')
    result_path = os.path.join(cfg.test.saved_model_dir, 'result.csv')
    if not os.path.exists(result_path):
        with open(result_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    # start evalulation
    saved_model_paths = sorted(glob.glob(cfg.test.saved_model_dir + '/*.pth'))
    if not cfg.test.eval_all:
        saved_model_paths = [saved_model_paths[-1]]
    for saved_model_path in saved_model_paths:
        print(f"model: {saved_model_path}")
        eval(cfg, args, saved_model_path)


if __name__ == '__main__':
    main()
