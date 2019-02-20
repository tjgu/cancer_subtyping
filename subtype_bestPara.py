#!/usr/bin/env python
"""
Subtype KIRC patients by integrating Multi-platform datasets using stacked denosing autoencoder

Run on the full datasets

Tongjun Gu
tgu@ufl.edu

"""

import os
import sys
import numpy as np

import dA
import logistic_sgd
import classification as cl
from gc import collect as gc_collect
import cors_save

np.warnings.filterwarnings('ignore') # Theano causes some warnings  

data_dir="./data_kirc/kirc.sample/"
result_dir="./results_bestPara/"

datasets = ["mirna.expr", "protein.expr", "gn.expr", "methy", "cna.nocnv.nodup"]

learning_rate=0.01
corruption_level=0.2
n_epochs=5000
batch_size=50
rng=np.random.RandomState(6000)

com_hidden = np.array([])
for one in datasets:
	filename=data_dir + one + ".2017-04-10.txt";
	data=np.loadtxt(filename,delimiter='\t',dtype='float32')
	data=np.transpose(data)
	print data.shape
	train_set_x_org,data_min,data_max=cl.normalize_col_scale01(data,tol=1e-10)
	if one == "mirna.expr":
		mirna_expr_org = train_set_x_org
		n_hidden=50
		model_trained_mirna, train_set_x_extr_mirna, training_time_mirna = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		com_hidden = np.hstack([com_hidden, train_set_x_extr_mirna]) if com_hidden.size else train_set_x_extr_mirna
		cors_save.save_para_oneLayer(result_dir, one, model_trained_mirna)
	if one == "protein.expr":
		prt_expr_org = train_set_x_org
		n_hidden=50
		model_trained_prt, train_set_x_extr_prt, training_time_prt = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		com_hidden = np.hstack([com_hidden, train_set_x_extr_prt]) if com_hidden.size else train_set_x_extr_prt
		cors_save.save_para_oneLayer(result_dir, one, model_trained_prt)
	if one == "gn.expr":
		gn_expr_org = train_set_x_org
		n_hidden=500
		model_trained_1_gn, train_set_x_extr_1_gn, training_time_1_gn = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		n_hidden=50
		model_trained_2_gn, train_set_x_extr_2_gn, training_time_2_gn = dA.train_model(train_set_x_org=train_set_x_extr_1_gn, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		com_hidden = np.hstack([com_hidden, train_set_x_extr_2_gn]) if com_hidden.size else train_set_x_extr_2_gn
		cors_save.save_para_twoLayers(result_dir, one, model_trained_1_gn, model_trained_2_gn)
	if one == "methy":
		methy_expr_org = train_set_x_org
		n_hidden=500
		model_trained_1_methy, train_set_x_extr_1_methy, training_time_1_methy = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		n_hidden=50
		model_trained_2_methy, train_set_x_extr_2_methy, training_time_2_methy = dA.train_model(train_set_x_org=train_set_x_extr_1_methy, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		com_hidden = np.hstack([com_hidden, train_set_x_extr_2_methy]) if com_hidden.size else train_set_x_extr_2_methy
		cors_save.save_para_twoLayers(result_dir, one, model_trained_1_methy, model_trained_2_methy)
	if one == "cna.nocnv.nodup":
		cna_expr_org = train_set_x_org
		n_hidden=500
		model_trained_1_cna, train_set_x_extr_1_cna, training_time_1_cna = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		n_hidden=50
		model_trained_2_cna, train_set_x_extr_2_cna, training_time_2_cna = dA.train_model(train_set_x_org=train_set_x_extr_1_cna, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		com_hidden = np.hstack([com_hidden, train_set_x_extr_2_cna]) if com_hidden.size else train_set_x_extr_2_cna
		cors_save.save_para_twoLayers(result_dir, one, model_trained_1_cna, model_trained_2_cna)

n_hidden=50
model_trained_1, com_hidden_x_extr_1, training_time_1 = dA.train_model(train_set_x_org=com_hidden, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
n_hidden=10
model_trained_21, com_hidden_x_extr_21, training_time_21 = dA.train_model(train_set_x_org=com_hidden_x_extr_1, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
n_hidden=1
model_trained_2, com_hidden_x_extr_2, training_time_2 = dA.train_model(train_set_x_org=com_hidden_x_extr_21, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)

cors_save.mirna_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_mirna, com_hidden_x_extr_2, train_set_x_extr_mirna, mirna_expr_org, result_dir)
cors_save.prt_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_prt, com_hidden_x_extr_2, train_set_x_extr_prt, prt_expr_org, result_dir)
cors_save.gn_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_gn, model_trained_2_gn, com_hidden_x_extr_2, train_set_x_extr_2_gn, gn_expr_org, result_dir)
cors_save.methy_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_methy, model_trained_2_methy, com_hidden_x_extr_2, train_set_x_extr_2_methy, methy_expr_org, result_dir)
cors_save.cna_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_cna, model_trained_2_cna, com_hidden_x_extr_2, train_set_x_extr_2_cna, cna_expr_org, result_dir)

cors_save.save_final_para(result_dir, com_hidden_x_extr_2, model_trained_2)

