import os
import shutil
import random

def init_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)

src_dir =  '../data/korean/images/evaluation'
dst_dir = '../data/korean/images/evaluation_subset'
init_dir(dst_dir)

label_path = os.path.join(src_dir, 'gt.txt')
with open(label_path, encoding='utf-8') as f:
    label = f.read().splitlines()

# subset_indices = list(range(32772, 32843))
subset_indices = list(range(len(label)))
random.shuffle(subset_indices)
subset_indices = sorted(subset_indices[:10000])

dst_label_path = os.path.join(dst_dir, 'gt.txt')
with open(dst_label_path, 'w', encoding='utf-8') as f:
    for i, subset_idx in enumerate(subset_indices):
        src_image_name = f'image-{subset_idx:09d}'
        dst_image_name = f'image-{i + 1:09d}'
        dst_text = label[subset_idx - 1].split(' ')[1]
        f.write(f'{dst_image_name} {dst_text}\n')
        shutil.copyfile(os.path.join(src_dir, src_image_name + '.jpg'), os.path.join(dst_dir, dst_image_name + '.jpg'))

