import numpy as np
import bob.learn.em
import bob.io.base
import matio
import argparse
import os
import sys


def get_img_list(path):
    with open(path) as f:
        lines = f.readlines()
        label_img_dict = dict()

        current_label = ''
        pre_label = ''
        img_list = []
        i = 0
        for line in lines:
            i = i + 1
            label = line.strip().split('/')[0]
            if label != current_label:
                if len(img_list) > 0:
                    label_img_dict[pre_label] = img_list
                    img_list = []
                current_label = label
                img_list.append(line.strip())
            else:
                img_list.append(line.strip())
                pre_label = current_label
                if i == len(lines):
                    label_img_dict[pre_label] = img_list
        return label_img_dict


def get_order_train_image(feature_path, nums):
    label_feature_dict = get_img_list(feature_path)
    result = []

    for key in label_feature_dict:
        each_class = []
        cur_fea_list = label_feature_dict[key]
        for i in range(nums):
            each_class.append(cur_fea_list[i])
        result.append(each_class)
    return result

if __name__ == '__main__':
    feature_path = './feature_list.txt'
    nums = 20
    save_list = './vggface2_per%s.lst'%nums
    result = get_order_train_image(feature_path, nums)
    with open(save_list, 'w') as f:
        for i in range(len(result)):
            for j in range(nums):
                f.write(result[i][j] + '\n')
            
