# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import tempfile
import zipfile
from PIL import Image
import mmcv
import numpy as np
import json
from mmcv import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert LoveDA dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='LoveDA folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args

def convert_json_to_label(ann_file):
    pil_label = Image.open(ann_file)
    label = np.asarray(pil_label)
    min_val = np.min(label)
    max_val = np.max(label)
    print(min_val, max_val)
    sample_class_stats = {}
    for c in range(1, 8):
        n = int(np.sum(label == c))
        if n > 0:
            sample_class_stats[int(c)] = n
    sample_class_stats['file'] = ann_file
    return sample_class_stats

def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)

def main():
    args = parse_args()
    # dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('/home/Hung_Data/HungData/mmseg_data/Datasets/LoveDA/loveDA', 'loveDA_rural_train')
    else:
        out_dir = args.out_dir
    nproc = 3
    print('Making directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'test'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    tmp_dir = "/home/Hung_Data/HungData/mmseg_data/Datasets/LoveDA/loveDA"
    lst_file_ann = []
    for dataset in ['Train']:
        # zip_file = zipfile.ZipFile(
        #     os.path.join(dataset_path, dataset + '.zip'))
        # zip_file.extractall(tmp_dir)
        data_type = dataset.lower()
        for location in ['Rural']:
            for image_type in ['images_png', 'masks_png']:
                if image_type == 'images_png':
                    dst = osp.join(out_dir, 'img_dir', data_type)
                else:
                    dst = osp.join(out_dir, 'ann_dir', data_type)
                src_dir = osp.join(tmp_dir, dataset, location,
                                   image_type)
                print(src_dir,tmp_dir)
                src_lst = os.listdir(src_dir)
                for file in src_lst:
                    shutil.copy(osp.join(src_dir, file), dst)
                    if "ann_dir/" in osp.join(dst, file) and "train/" in osp.join(dst, file):
                        lst_file_ann.append(osp.join(src_dir, file))
                        # sample_class_stats = convert_json_to_label(osp.join(dst, file))
                        # print(sample_class_stats)
    if nproc > 1:
        sample_class_stats = mmcv.track_parallel_progress(convert_json_to_label, lst_file_ann, nproc)
    else:
        sample_class_stats = mmcv.track_progress(convert_json_to_label, lst_file_ann)
    save_class_stats(out_dir, sample_class_stats)
        # print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()