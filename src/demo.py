import numpy as np
import bob.learn.em
import bob.io.base
import matio
import argparse
import os
import sys


SUBSPACE_DIMENSION_OF_F = 1
SUBSPACE_DIMENSION_OF_G = 2

plda_model_file = '../model/test.hdf5'
'''
data1 = np.array([[3,-3,100], [4,-4,50],[40,-40,150]], dtype=np.float64)
data2 = np.array([[3,6,-50], [4,8,-100],[40,79,-800]], dtype=np.float64)

data = [data1, data2]

pldabase = bob.learn.em.PLDABase(3,1,2)
trainer = bob.learn.em.PLDATrainer()
bob.learn.em.train(trainer, pldabase, data, max_iterations=10)

plda = bob.learn.em.PLDAMachine(pldabase)
samples = np.array([[3.5,-3.4,102],[4.5,-4.3,56]], dtype=np.float64)
loglike = plda.compute_log_likelihood(samples)
print('loglike:%f'%loglike)
'''
samples = np.array([[3.5,-3.4,102],[4.5,-4.3,56]], dtype=np.float64)
#plda_hdf5file = bob.io.base.HDF5File(plda_model_file, 'w')
#pldabase.save(plda_hdf5file)

plda_hdf5file2 = bob.io.base.HDF5File(plda_model_file)
pldabase2 = bob.learn.em.PLDABase(plda_hdf5file2)
plda2 = bob.learn.em.PLDAMachine(pldabase2)
loglike = plda2.compute_log_likelihood(samples)
print('loglike2:%f'%loglike)
