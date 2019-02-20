#!/usr/bin/env python
"""
Subtype KIRC patients by integrating Multi-platform datasets using stacked denosing autoencoder

Test different parameters

Tongjun Gu
tgu@ufl.edu

"""

import os
import sys
import numpy as np

import dA
import logistic_sgd
import classification as cla
from gc import collect as gc_collect
import cors_save
import cors_save_test

np.warnings.filterwarnings('ignore') # Theano causes some warnings  

data_dir="./data_kirc/kirc.sample/"
datasets = ["mirna.expr", "protein.expr", "gn.expr", "methy", "cna.nocnv.nodup"]
lr = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
cl = [0.01, 0.05, 0.1, 0.2, 0.5]
bs = [10, 20, 50, 100]
ne = [1000, 2000, 5000]

rng=np.random.RandomState(6000)
num_samples=441
data_permute_id=rng.permutation(num_samples)

for onelr in lr:
	learning_rate = onelr
	for onecl in cl:
		corruption_level = onecl
		for onebs in bs:
			batch_size = onebs
			for onene in ne:
				n_epochs = onene
				result_prefix="./results_para_test/lr" + str(learning_rate) + "cl" + str(corruption_level) + "bs" + str(batch_size) + "ep" + str(n_epochs) + "_"
				result_prefix2="./results_para_test/"

				k=2
				num_split_each=num_samples // k
				res=num_samples % k
				num_samples_splits=num_split_each*np.ones([k],dtype=int)

				indices_folds=np.zeros([num_samples],dtype=int)

				if res>0:
					for r in range(res):
						num_samples_splits[r]=num_samples_splits[r]+1

				start=0
				end=0
				for i in range(k):
					start = end
					end = end+num_samples_splits[i]
					indices_folds[start:end]=i

				com_hidden = np.array([])
				for one in datasets:
					filename=data_dir + one + ".2017-04-10.txt";
					data=np.loadtxt(filename,delimiter='\t',dtype='float32')
					data=np.transpose(data)
					data=data[data_permute_id, :]
					print data.shape

					data_norm,data_min,data_max=cla.normalize_col_scale01(data,tol=1e-10)
					test_set_x_org = data_norm[indices_folds==1, :]
					train_set_x_org = data_norm[indices_folds!=1, :]

					if one == "mirna.expr":
						mirna_expr_org = train_set_x_org
						mirna_expr_org_test = test_set_x_org
						n_hidden=50
						model_trained_mirna, train_set_x_extr_mirna, training_time_mirna = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						com_hidden = np.hstack([com_hidden, train_set_x_extr_mirna]) if com_hidden.size else train_set_x_extr_mirna
					if one == "protein.expr":
						prt_expr_org = train_set_x_org
						prt_expr_org_test = test_set_x_org
						n_hidden=50
						model_trained_prt, train_set_x_extr_prt, training_time_prt = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						com_hidden = np.hstack([com_hidden, train_set_x_extr_prt]) if com_hidden.size else train_set_x_extr_prt
					if one == "gn.expr":
						gn_expr_org = train_set_x_org
						gn_expr_org_test = test_set_x_org
						n_hidden=500
						model_trained_1_gn, train_set_x_extr_1_gn, training_time_1_gn = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						n_hidden=50
						model_trained_2_gn, train_set_x_extr_2_gn, training_time_2_gn = dA.train_model(train_set_x_org=train_set_x_extr_1_gn, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						com_hidden = np.hstack([com_hidden, train_set_x_extr_2_gn]) if com_hidden.size else train_set_x_extr_2_gn
					if one == "methy":
						methy_expr_org = train_set_x_org
						methy_expr_org_test = test_set_x_org
						n_hidden=500
						model_trained_1_methy, train_set_x_extr_1_methy, training_time_1_methy = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						n_hidden=50
						model_trained_2_methy, train_set_x_extr_2_methy, training_time_2_methy = dA.train_model(train_set_x_org=train_set_x_extr_1_methy, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						com_hidden = np.hstack([com_hidden, train_set_x_extr_2_methy]) if com_hidden.size else train_set_x_extr_2_methy
					if one == "cna.nocnv.nodup":
						cna_expr_org = train_set_x_org
						cna_expr_org_test = test_set_x_org
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

				[mirna_ave1, mirna_ave2] = cors_save.mirna_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_mirna, com_hidden_x_extr_2, train_set_x_extr_mirna, mirna_expr_org, result_prefix)
				[prt_ave1, prt_ave2] = cors_save.prt_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_prt, com_hidden_x_extr_2, train_set_x_extr_prt, prt_expr_org, result_prefix)
				[gn_ave1, gn_ave2] = cors_save.gn_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_gn, model_trained_2_gn, com_hidden_x_extr_2, train_set_x_extr_2_gn, gn_expr_org, result_prefix)
				[methy_ave1, methy_ave2] = cors_save.methy_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_methy, model_trained_2_methy, com_hidden_x_extr_2, train_set_x_extr_2_methy, methy_expr_org, result_prefix)
				[cna_ave1, cna_ave2] = cors_save.cna_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_cna, model_trained_2_cna, com_hidden_x_extr_2, train_set_x_extr_2_cna, cna_expr_org, result_prefix)
				cor1 = mirna_ave1 + prt_ave1 + gn_ave1 + methy_ave1 + cna_ave1
				cor2 = mirna_ave2 + prt_ave2 + gn_ave2 + methy_ave2 + cna_ave2

				cors_save.save_final_para(result_prefix, com_hidden_x_extr_2, model_trained_2)

				[rcor1_t, rcor2_t, mirna_ave1_t, mirna_ave2_t, prt_ave1_t, prt_ave2_t, gn_ave1_t, gn_ave2_t, methy_ave1_t, methy_ave2_t, cna_ave1_t, cna_ave2_t] = cors_save_test.test_cor(model_trained_2, model_trained_21, model_trained_1, model_trained_mirna, model_trained_prt, model_trained_1_gn, model_trained_2_gn, model_trained_1_methy, model_trained_2_methy, model_trained_1_cna, model_trained_2_cna, mirna_expr_org_test, prt_expr_org_test, gn_expr_org_test, methy_expr_org_test, cna_expr_org_test, result_prefix)

				cor1_t = mirna_ave1_t + prt_ave1_t + gn_ave1_t + methy_ave1_t + cna_ave1_t
				cor2_t = mirna_ave2_t + prt_ave2_t + gn_ave2_t + methy_ave2_t + cna_ave2_t

				cors = [mirna_ave1, prt_ave1, gn_ave1, methy_ave1, cna_ave1, 1000000, mirna_ave2, prt_ave2, gn_ave2, methy_ave2, cna_ave2, 2000000, mirna_ave1_t, prt_ave1_t, gn_ave1_t, methy_ave1_t, cna_ave1_t, 3000000, mirna_ave2_t, prt_ave2_t, gn_ave2_t, methy_ave2_t, cna_ave2_t]
				filename=result_prefix + "cross_validation_" + str(n) + "_cors_values.txt"
				file_handle=file(filename,'w')
				np.savetxt(file_handle, cors ,delimiter="\t")
				file_handle.close()

				para = ["lr:" + str(learning_rate), "cl:" + str(corruption_level), "bs:" + str(batch_size), "ep:" + str(n_epochs)]
				cors1org = ["orgCor:", mirna_ave1, prt_ave1, gn_ave1, methy_ave1, cna_ave1]
				cors2org = ["comCor:", mirna_ave2, prt_ave2, gn_ave2, methy_ave2, cna_ave2]
				cors1org_t = ["orgCor_test:", mirna_ave1_t, prt_ave1_t, gn_ave1_t, methy_ave1_t, cna_ave1_t]
				cors2org_t = ["comCor_test:", mirna_ave2_t, prt_ave2_t, gn_ave2_t, methy_ave2_t, cna_ave2_t]
				tot = ["addOrg:", cor1, cor2, "addTest:", cor1_t, cor2_t]

				filename2=result_prefix2 + "parameters_cors.txt"
				file_handle2=file(filename2,'a')
			    	np.savetxt(file_handle2, para, delimiter="\t", fmt="%s")
			    	np.savetxt(file_handle2, cors1org, delimiter="\t", fmt="%s")
			    	np.savetxt(file_handle2, cors2org, delimiter="\t", fmt="%s")
			    	np.savetxt(file_handle2, cors1org_t, delimiter="\t", fmt="%s")
			    	np.savetxt(file_handle2, cors2org_t, delimiter="\t", fmt="%s")
			    	np.savetxt(file_handle2, tot, delimiter="\t", fmt="%s")
			    	np.savetxt(file_handle2, [''], delimiter="\t", fmt="%s")
				file_handle2.close()
				
