import argparse
import os
import re
from mmcv import Config
from binascii import unhexlify
import sys
sys.path.append(os.getcwd())
from collections import Counter
from cdistnet.data.jamoconverter import JamoConverter

def parse_args():
    def str2bool(x):
        return x.lower() in ['true']
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/CDistNet_config_convert.py', help='path to CDistNet configuration file.' )
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    return cfg, args

def add_sequence_symbols(vocab, is_jamo=False):
    """
    Add sequence symbols to vocab.
    <pad> : pad
    <unk> : unkown
    <sos> : start of sequence
    <eos> : end of sequence
    <del> : delimiter for jamo sequence
    """
    if is_jamo:
        vocab = ['<pad>', '<unk>', '<sos>', '<eos>', '<del>'] + vocab
    else:
        vocab = ['<pad>', '<unk>', '<sos>', '<eos>'] + vocab
    return vocab

def main(cfg, args):
    vocab_dir = './cdistnet/utils'

    special_symbols = '~`!@#$%^&*()_-+={[}]|\:;<,>.?/'
    special_symbols = [sym for sym in special_symbols]
    special_symbols.extend(["'", '"'])
    special_symbols = sorted(special_symbols)
    
    # Jamo
    jamo = JamoConverter().vocab
    jamo = add_sequence_symbols(jamo, is_jamo=True)
    vocab_path = os.path.join(vocab_dir, f'dict_{len(jamo)}_jamo.txt')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for item in jamo:
            f.write(f'{item}\n')
        print(f'Saved vocab : jamo (len : {len(jamo)})')
    
    # From euc-kr encoding
    # len(euc_kr) = 2350
    euc_kr = []
    for code in range(int('b0a1', 16), int('c8ff', 16)):
        try:
            code = hex(code)
            code = code[2:]
            code = unhexlify(code)
            kor_syl = code.decode('euc-kr')
            euc_kr.append(kor_syl)
        except:
            pass
    euc_kr = add_sequence_symbols(euc_kr, is_jamo=False)
    vocab_path = os.path.join(vocab_dir, f'dict_{len(euc_kr)}_kor.txt')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for item in euc_kr:
            f.write(f'{item}\n')
        print(f'Saved vocab : kor (len : {len(euc_kr)})')

    # From train/val set of AI Hub
    # len(chars) = 1849
    # len(set(chars) - set(euc_kr)) = 139
    label_dirs = list()
    label_dirs.extend(cfg.val.image_dir)
    label_dirs.extend(cfg.train.image_dir)
    label_paths = [os.path.join(label_dir, 'gt.txt') for label_dir in label_dirs]
    texts = []
    for label_path in label_paths:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            lines = [line.split(' ')[-1] for line in lines]
            texts.extend(lines)
    chars = []
    for text in texts:
        text_split = [c for c in text if re.sub('[^가-힣]+', '', c) == c]
        chars.extend(text_split)
    keys = list(Counter(chars).keys())
    values = list(Counter(chars).values())
    num_chars = sum(values)
    chars = sorted(list(set(chars)))
    chars_minus_euckr = list(set(chars) - set(euc_kr))
    num_chars_minus_euckr = sum([values[keys.index(c)] for c in chars_minus_euckr])
    print(f'Number of chars - euc_kr : {100 * num_chars_minus_euckr / num_chars:.2f} %')

if __name__ == '__main__':
    cfg, args = parse_args()
    main(cfg, args)