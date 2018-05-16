import numpy as np
import bob.learn.em
import bob.io.base
import matio
import argparse
import os
import sys


SUBSPACE_DIMENSION_OF_F = 384
SUBSPACE_DIMENSION_OF_G = 384

plda_model_file = '../model/plda_256.hdf5'


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


def read_feature(root_folder, feature_path):
    label_feature_dict = get_img_list(feature_path)
    feature_matrix = []

    for key in label_feature_dict:
        each_class_feature = []
        cur_fea_list = label_feature_dict[key]
        for i in range(len(cur_fea_list)):
            full_path = os.path.join(root_folder, cur_fea_list[i])
            x_vec = matio.load_mat(full_path).flatten()
            each_class_feature.append(x_vec)
        feature_matrix.append(np.array(each_class_feature, dtype=np.float64))
    return feature_matrix


def train_plda_base(root_folder, train_files):
    #load feature
    training_features = read_feature(root_folder, train_files)
    input_dimension = training_features[0].shape[1]
    print('Trainning PLDA base machine')
    plda_base = bob.learn.em.PLDABase(input_dimension, SUBSPACE_DIMENSION_OF_F, SUBSPACE_DIMENSION_OF_G)

    #create trainer
    t = bob.learn.em.PLDATrainer()
    # train machine
    bob.learn.em.train(t, plda_base, training_features)

    #write machines to file
    plda_hdf5file = bob.io.base.HDF5File(plda_model_file, 'w')
    plda_base.save(plda_hdf5file)


def load_plda_base(plda_model_file):
    #load model
    plda_hdf5file = bob.io.base.HDF5File(plda_model_file)
    plda_base = bob.learn.em.PLDABase(plda_hdf5file)
    return plda_base


def plda_score(model, samples):
    return model.compute_log_likelihood(samples)


def test():
    pass

if __name__ == '__main__':
    root_folder = '/workspace/data/vgg-features_0410/vggface-r100-spa-m2.0-ep96/vggface2_train_aligned_112x112/aligned_imgs'
    feature_list = './vggface2_per20.lst'
    #feature_mat = read_feature(root_folder,feature_list)
    #print(np.array(feature_mat).shape)
    #print(type(feature_mat[0]))
    train_plda_base(root_folder, feature_list)
    print('Finished Done')
    #test

