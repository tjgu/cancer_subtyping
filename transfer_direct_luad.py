#!/usr/bin/env python
"""
Subtype LUAD patients using the models learned directly from KIRC patients
The current version of the script doesnot load the model variables directly but re-run the SdA with KIRC dataset to obtain the KIRC model variables

Run on the full datasets of LUAD

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
import cors_save_test

np.warnings.filterwarnings('ignore') # Theano causes some warnings  

data_dir="./data_kirc/kirc.sample/"
data_dir_luad="./data_luad/luad.sample/"
result_dir_luad="./luad/results_direct_transfer/"

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
	data_norm_kirc,data_min,data_max=cl.normalize_col_scale01(data,tol=1e-10)

	filename=data_dir_luad + one + ".2017-04-10.txt";
	data_luad=np.loadtxt(filename,delimiter='\t',dtype='float32')
	data_luad=np.transpose(data_luad)
	print data_luad.shape
	data_norm_luad,data_min,data_max=cl.normalize_col_scale01(data_luad,tol=1e-10)

	test_set_x_org = data_norm_luad
        train_set_x_org = data_norm_kirc

	if one == "mirna.expr":
		mirna_expr_org = train_set_x_org
		mirna_expr_test = test_set_x_org
		n_hidden=50
		model_trained_mirna, train_set_x_extr_mirna, training_time_mirna = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		com_hidden = np.hstack([com_hidden, train_set_x_extr_mirna]) if com_hidden.size else train_set_x_extr_mirna
	if one == "protein.expr":
		prt_expr_org = train_set_x_org
		data_permute_id = rng.permutation(train_set_x_org.shape[1])
                test_set_x_org = test_set_x_org[:, data_permute_id]
		prt_expr_test = test_set_x_org
		n_hidden=50
		model_trained_prt, train_set_x_extr_prt, training_time_prt = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		com_hidden = np.hstack([com_hidden, train_set_x_extr_prt]) if com_hidden.size else train_set_x_extr_prt
	if one == "gn.expr":
		gn_expr_org = train_set_x_org
		gn_expr_test = test_set_x_org
		n_hidden=500
		model_trained_1_gn, train_set_x_extr_1_gn, training_time_1_gn = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		n_hidden=50
		model_trained_2_gn, train_set_x_extr_2_gn, training_time_2_gn = dA.train_model(train_set_x_org=train_set_x_extr_1_gn, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		com_hidden = np.hstack([com_hidden, train_set_x_extr_2_gn]) if com_hidden.size else train_set_x_extr_2_gn
	if one == "methy":
		methy_expr_org = train_set_x_org
		data_permute_id = rng.permutation(train_set_x_org.shape[1])
                test_set_x_org = test_set_x_org[:, data_permute_id]
		methy_expr_test = test_set_x_org
		n_hidden=500
		model_trained_1_methy, train_set_x_extr_1_methy, training_time_1_methy = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		n_hidden=50
		model_trained_2_methy, train_set_x_extr_2_methy, training_time_2_methy = dA.train_model(train_set_x_org=train_set_x_extr_1_methy, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		com_hidden = np.hstack([com_hidden, train_set_x_extr_2_methy]) if com_hidden.size else train_set_x_extr_2_methy
	if one == "cna.nocnv.nodup":
		cna_expr_org = train_set_x_org
		data_permute_id = rng.permutation(train_set_x_org.shape[1])
                test_set_x_org = test_set_x_org[:, data_permute_id]
		cna_expr_test = test_set_x_org
		n_hidden=500
		model_trained_1_cna, train_set_x_extr_1_cna, training_time_1_cna = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		n_hidden=50
		model_trained_2_cna, train_set_x_extr_2_cna, training_time_2_cna = dA.train_model(train_set_x_org=train_set_x_extr_1_cna, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
		com_hidden = np.hstack([com_hidden, train_set_x_extr_2_cna]) if com_hidden.size else train_set_x_extr_2_cna

n_hidden=50
model_trained_1, com_hidden_x_extr_1, training_time_1 = dA.train_model(train_set_x_org=com_hidden, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
n_hidden=10
model_trained_21, com_hidden_x_extr_21, training_time_21 = dA.train_model(train_set_x_org=com_hidden_x_extr_1, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
n_hidden=1
model_trained_2, com_hidden_x_extr_2, training_time_2 = dA.train_model(train_set_x_org=com_hidden_x_extr_21, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)

[rcor1_t, rcor2_t, mirna_ave1_t, mirna_ave2_t, prt_ave1_t, prt_ave2_t, gn_ave1_t, gn_ave2_t, methy_ave1_t, methy_ave2_t, cna_ave1_t, cna_ave2_t] = cors_save_test.test_cor(model_trained_2, model_trained_21, model_trained_1, model_trained_mirna, model_trained_prt, model_trained_1_gn, model_trained_2_gn, model_trained_1_methy, model_trained_2_methy, model_trained_1_cna, model_trained_2_cna, mirna_expr_test, prt_expr_test, gn_expr_test, methy_expr_test, cna_expr_test, result_dir_luad)

cor1_t = mirna_ave1_t + prt_ave1_t + gn_ave1_t + methy_ave1_t + cna_ave1_t
cor2_t = mirna_ave2_t + prt_ave2_t + gn_ave2_t + methy_ave2_t + cna_ave2_t

para = ["direct transfer\n" + "lr:" + str(learning_rate), "cl:" + str(corruption_level), "bs:" + str(batch_size), "ep:" + str(n_epochs)]
cors1org_t = ["orgCor_test:", mirna_ave1_t, prt_ave1_t, gn_ave1_t, methy_ave1_t, cna_ave1_t]
cors2org_t = ["comCor_test:", mirna_ave2_t, prt_ave2_t, gn_ave2_t, methy_ave2_t, cna_ave2_t]
tot = ["addTest:", cor1_t, cor2_t]

filename2=result_dir_luad + "parameters_cors.txt"
file_handle2=file(filename2,'a')
np.savetxt(file_handle2, para, delimiter="\t", fmt="%s")
np.savetxt(file_handle2, cors1org_t, delimiter="\t", fmt="%s")
np.savetxt(file_handle2, cors2org_t, delimiter="\t", fmt="%s")
np.savetxt(file_handle2, tot, delimiter="\t", fmt="%s")
np.savetxt(file_handle2, [''], delimiter="\t", fmt="%s")
file_handle2.close()

