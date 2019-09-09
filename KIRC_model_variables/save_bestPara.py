#!/usr/bin/env python
"""
save the weights using cPickle

Tongjun Gu
tgu@ufl.edu

"""

import os
import sys
import numpy as np
from six.moves import cPickle
import glob

result_dir = "./results_bestPara/para/"
parafiles = glob.glob("./results_bestPara/*_para_*")

for one in parafiles:
	data = np.loadtxt(one,delimiter='\t',dtype='float32')
	print(data.shape)
	base = os.path.basename(one)
	ofl = os.path.splitext(base)[0]
	f = open(result_dir + ofl + '.pkl', 'wb')
	cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()
 
	# example for loading the weights
	fo = open(result_dir + ofl + '.pkl', 'rb')
	loaded_obj = cPickle.load(fo)
	f.close()
	print(loaded_obj.shape)

