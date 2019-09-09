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

result_dir = "results_bestPara/para/"
parafiles = glob.glob("results_bestPara/[gm]*_para_w_h1.txt")

for one in parafiles:
	data = np.loadtxt(one,delimiter='\t',dtype='float32')
	print(data.shape)
	half = round(data.shape[0]/2)
	data1 = data[0:half, :]
	data2 = data[half:data.shape[0], :]

	base = os.path.basename(one)
	ofl = os.path.splitext(base)[0]
	f = open(result_dir + ofl + '_firstHalf.pkl', 'wb')
	cPickle.dump(data1, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()
	f = open(result_dir + ofl + '_secHalf.pkl', 'wb')
	cPickle.dump(data2, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()
 
