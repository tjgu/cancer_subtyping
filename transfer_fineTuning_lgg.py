#!/usr/bin/env python
"""
Subtype LGG patients based on the models learned from KIRC patients

The hidden layer of KIRC model is used as a pretrain model and then do fine tuning the variables using the LGG datasets with a lower learning rate at 0.0005

Run on the full LGG datasets

Tongjun Gu
tgu@ufl.edu

"""

import os
import sys
import numpy as np
from numpy import loadtxt
import dA_finetuning
import logistic_sgd
import classification as cla
from gc import collect as gc_collect
import cors_save_finetuning
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


np.warnings.filterwarnings('ignore') # Theano causes some warnings  

data_dir2="../lgg.sample1/"
datasets = ["mirna.expr", "protein.expr", "gn.expr", "methy", "cna.nocnv.nodup"]

n_epochs=5000
batch_size=50
corruption_level=0.2
lr2 = 0.0005
rng=np.random.RandomState(6000)
prt_nm = 189
methy_nm = 25062
cna_nm = 9176

result_prefix="./lgg/results_fineTuning9.2/"
inputdir="./results_bestPara/"
com_hidden = np.array([])
for one in datasets:
	filename2=data_dir2 + one + ".2017-06-19.txt";
	data2=np.loadtxt(filename2,delimiter='\t',dtype='float32')
	data2=np.transpose(data2)
	print ("LUAD data2.shape")
	print (data2.shape)

	data2_norm,data2_min,data2_max=cla.normalize_col_scale01(data2,tol=1e-10)
	test_set_x_org = data2_norm

	if one == "mirna.expr":
		mirna_expr_org_test = test_set_x_org
		n_hidden=50
		paranm = inputdir + one + "_para_w.txt"
		lines = loadtxt(paranm, comments="#")
		W = theano.shared(value=lines, name='W', borrow=True)
		paranm = inputdir + one + "_para_bv.txt"
		lines = loadtxt(paranm, comments="#")
		bvis = theano.shared(value=lines, borrow=True)
		paranm = inputdir + one + "_para_bh.txt"
		lines = loadtxt(paranm, comments="#")
		bhid = theano.shared(value=lines, name='b', borrow=True)
		model_trained_mirna_2, train_set_x_extr_mirna_2, training_time_mirna_2 = dA_finetuning.train_model(train_set_x_org=test_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)

		com_hidden = np.hstack([com_hidden, train_set_x_extr_mirna_2]) if com_hidden.size else train_set_x_extr_mirna_2
	if one == "protein.expr":
		data_permute_id = rng.permutation(prt_nm)
		test_set_x_org = test_set_x_org[:, data_permute_id]
		prt_expr_org_test = test_set_x_org
		filename=result_prefix + "prt_SameAsKIRC.txt"
		file_handle=open(filename,'w')
		np.savetxt(file_handle, test_set_x_org, delimiter="\t", header=" after normalization")
		file_handle.close()

		n_hidden=50
		paranm = inputdir + one + "_para_w.txt"
		lines = loadtxt(paranm, comments="#")
		W = theano.shared(value=lines, name='W', borrow=True)
		paranm = inputdir + one + "_para_bv.txt"
		lines = loadtxt(paranm, comments="#")
		bvis = theano.shared(value=lines, borrow=True)
		paranm = inputdir + one + "_para_bh.txt"
		lines = loadtxt(paranm, comments="#")
		bhid = theano.shared(value=lines, name='b', borrow=True)
		
		model_trained_prt_2, train_set_x_extr_prt_2, training_time_prt_2 = dA_finetuning.train_model(train_set_x_org=test_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)
		com_hidden = np.hstack([com_hidden, train_set_x_extr_prt_2]) if com_hidden.size else train_set_x_extr_prt_2
	if one == "gn.expr":
		gn_expr_org_test = test_set_x_org
		n_hidden=500
		paranm = inputdir + one + "_para_w_h1.txt"
		lines = loadtxt(paranm, comments="#")
		W = theano.shared(value=lines, name='W', borrow=True)
		paranm = inputdir + one + "_para_bv_h1.txt"
		lines = loadtxt(paranm, comments="#")
		bvis = theano.shared(value=lines, borrow=True)
		paranm = inputdir + one + "_para_bh_h1.txt"
		lines = loadtxt(paranm, comments="#")
		bhid = theano.shared(value=lines, name='b', borrow=True)
		model_trained_1_gn_2, train_set_x_extr_1_gn_2, training_time_1_gn_2 = dA_finetuning.train_model(train_set_x_org=test_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)

		n_hidden=50
		paranm = inputdir + one + "_para_w_h2.txt"
		lines = loadtxt(paranm, comments="#")
		W = theano.shared(value=lines, name='W', borrow=True)
		paranm = inputdir + one + "_para_bv_h2.txt"
		lines = loadtxt(paranm, comments="#")
		bvis = theano.shared(value=lines, borrow=True)
		paranm = inputdir + one + "_para_bh_h2.txt"
		lines = loadtxt(paranm, comments="#")
		bhid = theano.shared(value=lines, name='b', borrow=True)
		model_trained_2_gn, train_set_x_extr_2_gn, training_time_2_gn = dA_finetuning.train_model(train_set_x_org=train_set_x_extr_1_gn_2, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)
		print ("finish initial training for gn")

		com_hidden = np.hstack([com_hidden, train_set_x_extr_2_gn]) if com_hidden.size else train_set_x_extr_2_gn
	if one == "methy":
		data_permute_id = rng.permutation(methy_nm)
		test_set_x_org = test_set_x_org[:, data_permute_id]
		methy_expr_org_test = test_set_x_org
		filename=result_prefix + "methy_SameAsKIRC.txt"
		file_handle=open(filename,'w')
		np.savetxt(file_handle, test_set_x_org, delimiter="\t", header=" after normalization")
		file_handle.close()

		n_hidden=500
		paranm = inputdir + one + "_para_w_h1.txt"
		lines = loadtxt(paranm, comments="#")
		W = theano.shared(value=lines, name='W', borrow=True)
		paranm = inputdir + one + "_para_bv_h1.txt"
		lines = loadtxt(paranm, comments="#")
		bvis = theano.shared(value=lines, borrow=True)
		paranm = inputdir + one + "_para_bh_h1.txt"
		lines = loadtxt(paranm, comments="#")
		bhid = theano.shared(value=lines, name='b', borrow=True)

		model_trained_1_methy_2, train_set_x_extr_1_methy_2, training_time_1_methy_2 = dA_finetuning.train_model(train_set_x_org=test_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)

		n_hidden=50
		paranm = inputdir + one + "_para_w_h2.txt"
		lines = loadtxt(paranm, comments="#")
		W = theano.shared(value=lines, name='W', borrow=True)
		paranm = inputdir + one + "_para_bv_h2.txt"
		lines = loadtxt(paranm, comments="#")
		bvis = theano.shared(value=lines, borrow=True)
		paranm = inputdir + one + "_para_bh_h2.txt"
		lines = loadtxt(paranm, comments="#")
		bhid = theano.shared(value=lines, name='b', borrow=True)
		model_trained_2_methy, train_set_x_extr_2_methy, training_time_2_methy = dA_finetuning.train_model(train_set_x_org=train_set_x_extr_1_methy_2, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)
		print ("finish training for methy")

		com_hidden = np.hstack([com_hidden, train_set_x_extr_2_methy]) if com_hidden.size else train_set_x_extr_2_methy
	if one == "cna.nocnv.nodup":
		data_permute_id = rng.permutation(cna_nm)
		test_set_x_org = test_set_x_org[:, data_permute_id]
		cna_expr_org_test = test_set_x_org
		filename=result_prefix + "cna_SameAsKIRC.txt"
		file_handle=open(filename,'w')
		np.savetxt(file_handle, test_set_x_org, delimiter="\t", header=" after normalization")
		file_handle.close()

		n_hidden=500
		paranm = inputdir + one + "_para_w_h1.txt"
		lines = loadtxt(paranm, comments="#")
		W = theano.shared(value=lines, name='W', borrow=True)
		paranm = inputdir + one + "_para_bv_h1.txt"
		lines = loadtxt(paranm, comments="#")
		bvis = theano.shared(value=lines, borrow=True)
		paranm = inputdir + one + "_para_bh_h1.txt"
		lines = loadtxt(paranm, comments="#")
		bhid = theano.shared(value=lines, name='b', borrow=True)
		model_trained_1_cna_2, train_set_x_extr_1_cna_2, training_time_1_cna_2 = dA_finetuning.train_model(train_set_x_org=test_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)

		n_hidden=50
		paranm = inputdir + one + "_para_w_h2.txt"
		lines = loadtxt(paranm, comments="#")
		W = theano.shared(value=lines, name='W', borrow=True)
		paranm = inputdir + one + "_para_bv_h2.txt"
		lines = loadtxt(paranm, comments="#")
		bvis = theano.shared(value=lines, borrow=True)
		paranm = inputdir + one + "_para_bh_h2.txt"
		lines = loadtxt(paranm, comments="#")
		bhid = theano.shared(value=lines, name='b', borrow=True)
		model_trained_2_cna, train_set_x_extr_2_cna, training_time_2_cna = dA_finetuning.train_model(train_set_x_org=train_set_x_extr_1_cna_2, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)
		print ("finish training for cna")

		com_hidden = np.hstack([com_hidden, train_set_x_extr_2_cna]) if com_hidden.size else train_set_x_extr_2_cna

n_hidden=50
paranm = inputdir + "com_h0_para_w.txt"
lines = loadtxt(paranm, comments="#")
W = theano.shared(value=lines, name='W', borrow=True)
paranm = inputdir + "com_h0_para_bv.txt"
lines = loadtxt(paranm, comments="#")
bvis = theano.shared(value=lines, borrow=True)
paranm = inputdir + "com_h0_para_bh.txt"
lines = loadtxt(paranm, comments="#")
bhid = theano.shared(value=lines, name='b', borrow=True)
model_trained_1, com_hidden_x_extr_1, training_time_1 = dA_finetuning.train_model(train_set_x_org=com_hidden, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)
n_hidden=10
paranm = inputdir + "com_h1_para_w.txt"
lines = loadtxt(paranm, comments="#")
W = theano.shared(value=lines, name='W', borrow=True)
paranm = inputdir + "com_h1_para_bv.txt"
lines = loadtxt(paranm, comments="#")
bvis = theano.shared(value=lines, borrow=True)
paranm = inputdir + "com_h1_para_bh.txt"
lines = loadtxt(paranm, comments="#")
bhid = theano.shared(value=lines, name='b', borrow=True)
model_trained_21, com_hidden_x_extr_21, training_time_21 = dA_finetuning.train_model(train_set_x_org=com_hidden_x_extr_1, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)
n_hidden=1
paranm = inputdir + "com_h2_para_w.txt"
lines = loadtxt(paranm, comments="#")
W = theano.shared(value=lines, name='W', borrow=True)
paranm = inputdir + "com_h2_para_bv.txt"
lines = loadtxt(paranm, comments="#")
bvis = theano.shared(value=lines, borrow=True)
paranm = inputdir + "com_h2_para_bh.txt"
lines = loadtxt(paranm, comments="#")
bhid = theano.shared(value=lines, name='b', borrow=True)
model_trained_2, com_hidden_x_extr_2, training_time_2 = dA_finetuning.train_model(train_set_x_org=com_hidden_x_extr_21, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)

cors_save_finetuning.save_final_para(result_prefix, com_hidden_x_extr_2, model_trained_2)

[mirna_ave1, mirna_ave2] = cors_save_finetuning.mirna_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_mirna_2, com_hidden_x_extr_2, train_set_x_extr_mirna_2, mirna_expr_org_test, result_prefix)
[prt_ave1, prt_ave2] = cors_save_finetuning.prt_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_prt_2, com_hidden_x_extr_2, train_set_x_extr_prt_2, prt_expr_org_test, result_prefix)
[gn_ave1, gn_ave2] = cors_save_finetuning.gn_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_gn_2, model_trained_2_gn, com_hidden_x_extr_2, train_set_x_extr_2_gn, gn_expr_org_test, result_prefix)
[methy_ave1, methy_ave2] = cors_save_finetuning.methy_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_methy_2, model_trained_2_methy, com_hidden_x_extr_2, train_set_x_extr_2_methy, methy_expr_org_test, result_prefix)
[cna_ave1, cna_ave2] = cors_save_finetuning.cna_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_cna_2, model_trained_2_cna, com_hidden_x_extr_2, train_set_x_extr_2_cna, cna_expr_org_test, result_prefix)

cor1 = mirna_ave1 + prt_ave1 + gn_ave1 + methy_ave1 + cna_ave1
cor2 = mirna_ave2 + prt_ave2 + gn_ave2 + methy_ave2 + cna_ave2

cors = [mirna_ave1, prt_ave1, gn_ave1, methy_ave1, cna_ave1, 1000000, mirna_ave2, prt_ave2, gn_ave2, methy_ave2, cna_ave2]
filename=result_prefix + "cors_values.txt"
file_handle=open(filename,'w')
np.savetxt(file_handle, cors ,delimiter="\t")
file_handle.close()

para = ["lr:" + str(lr2), "cl:" + str(corruption_level), "bs:" + str(batch_size), "ep:" + str(n_epochs)]
cors1org = ["orgCor:", mirna_ave1, prt_ave1, gn_ave1, methy_ave1, cna_ave1]
cors2org = ["comCor:", mirna_ave2, prt_ave2, gn_ave2, methy_ave2, cna_ave2]
tot = ["addOrg:", cor1, cor2]

filename2=result_prefix + "parameters_cors.txt"
file_handle2=open(filename2,'a')
np.savetxt(file_handle2, para, delimiter="\t", fmt="%s")
np.savetxt(file_handle2, cors1org, delimiter="\t", fmt="%s")
np.savetxt(file_handle2, cors2org, delimiter="\t", fmt="%s")
np.savetxt(file_handle2, tot, delimiter="\t", fmt="%s")
np.savetxt(file_handle2, [''], delimiter="\t", fmt="%s")
file_handle2.close()

