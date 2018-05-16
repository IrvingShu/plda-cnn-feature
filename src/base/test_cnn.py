import numpy as np
import bob.learn.em
import bob.io.base
import matio
import argparse
import os
import sys
import os.path as osp

fea_type = '_feat.bin'

def AugDistance(fea1, fea2):
    
    pass

if __name__ == '__main__':
    #root_folder = '/workspace/data/vgg-features_0410/vggface-r100-spa-m2.0-ep96/vggface2_train_aligned_112x112/aligned_imgs'
    root_folder ='/workspace/data/face-idcard-1M/features/insightface-r100-spa-m2.0-ep96/'
    with open('../../data/pair.txt') as f, open('./result.txt','w')as f2:
        lines = f.readlines()
        for line in lines:
            fea1_name = line.strip().split(' ')[0] + fea_type
            fea2_name = line.strip().split(' ')[1] + fea_type
            
            fea1  = matio.load_mat(osp.join(root_folder, fea1_name)).flatten()
            fea2  = matio.load_mat(osp.join(root_folder, fea2_name)).flatten()

	    dist = np.dot(fea1, fea2) / (np.linalg.norm(fea1, ord=2) * np.linalg.norm(fea2, ord=2))                       
 
            f2.write('%s  %s: %f\n'%(fea1_name, fea2_name, dist))
