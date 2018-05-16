import numpy as np
import bob.learn.em
import bob.io.base
import matio
import argparse
import os
import sys
import os.path as osp


plda_model_file = '../model/plda.hdf5'
fea_type = '_feat.bin'
def load_plda_base(plda_model_file):
    #load model
    plda_hdf5file = bob.io.base.HDF5File(plda_model_file)
    plda_base = bob.learn.em.PLDABase(plda_hdf5file)
    return plda_base

def plda_score(model, samples):
    return model.compute_log_likelihood(samples)


if __name__ == '__main__':
    #root_folder = '/workspace/data/vgg-features_0410/vggface-r100-spa-m2.0-ep96/vggface2_train_aligned_112x112/aligned_imgs'
    root_folder ='/workspace/data/face-idcard-1M/features/insightface-r100-spa-m2.0-ep96/'
    plda_model_file = '../model/plda.hdf5'
    #plda_base = load_plda_base(plda_model_file)
    #plda_machine = bob.learn.em.PLDAMachine(plda_base)
    with open('../data/pair.txt') as f, open('./result.txt','w')as f2:
        lines = f.readlines()
        for line in lines:
            fea1_name = line.strip().split(' ')[0] + fea_type
            fea2_name = line.strip().split(' ')[1] + fea_type
            
            fea1  = matio.load_mat(osp.join(root_folder, fea1_name)).flatten()
            fea2  = matio.load_mat(osp.join(root_folder, fea2_name)).flatten()
            plda_base = load_plda_base(plda_model_file)
            trainer = bob.learn.em.PLDATrainer()
            plda1 = bob.learn.em.PLDAMachine(plda_base)
            sample = []
            sample.append(fea1) 
            tmp1 = np.array(sample, dtype=np.float64)
            
            trainer.enroll(plda1, tmp1) 
            #sample.append(fea1)
            #sample.append(fea2) 
            #score = plda_machine.log_likelihood_ratio(np.array(sample))
            sample2 = []
            sample2.append(fea2)
            tmp2 = np.array(sample2, dtype=np.float64)
            f2.write('%s  %s: %f\n'%(fea1_name, fea2_name, plda1(tmp2)))
