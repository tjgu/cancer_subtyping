#!/usr/bin/env python
"""
Multi-modal classification using denosing autoencoder

Tongjun Gu
tgu@ufl.edu

"""

import os
import sys
import numpy as np

import dA_finetuning
import dA
import logistic_sgd
import classification as cla
from gc import collect as gc_collect
import cors_save
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


np.warnings.filterwarnings('ignore') # Theano causes some warnings  

data_dir="./data_kirc/kirc.sample/"
data_dir2="./data_luad/luad.sample/"
datasets = ["mirna.expr", "protein.expr", "gn.expr", "methy", "cna.nocnv.nodup"]
lr = [0.01]
cl = [0.2]
bs = [50]
ne = [5000]

lr2 = [0.001]

rng=np.random.RandomState(6000)

for onelr in lr:
	learning_rate = onelr
	for onecl in cl:
		corruption_level = onecl
		for onebs in bs:
			batch_size = onebs
			for onene in ne:
				n_epochs = onene
				result_prefix="./luad/results_fineTuning/"
				com_hidden = np.array([])
				for one in datasets:
					filename=data_dir + one + ".2017-04-10.txt";
					data=np.loadtxt(filename,delimiter='\t',dtype='float32')
					data=np.transpose(data)
					print "KIRC data.shape"
					print data.shape

					filename2=data_dir2 + one + ".2017-04-10.txt";
                                        data2=np.loadtxt(filename2,delimiter='\t',dtype='float32')
                                        data2=np.transpose(data2)
                                        print "LUAD data2.shape"
                                        print data2.shape

					data_norm,data_min,data_max=cla.normalize_col_scale01(data,tol=1e-10)
					data2_norm,data2_min,data2_max=cla.normalize_col_scale01(data2,tol=1e-10)
					train_set_x_org = data_norm
					test_set_x_org = data2_norm

					if one == "mirna.expr":
						mirna_expr_org = train_set_x_org
						mirna_expr_org_test = test_set_x_org
						n_hidden=50
						model_trained_mirna, train_set_x_extr_mirna, training_time_mirna = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						print "finish initial training for miRNA"
						
						W = theano.shared(value=model_trained_mirna.W.get_value(), name='W', borrow=True)
						bvis = theano.shared(value=model_trained_mirna.b_prime.get_value(), borrow=True)
						bhid = theano.shared(value=model_trained_mirna.b.get_value(), name='b', borrow=True)
						model_trained_mirna_2, train_set_x_extr_mirna_2, training_time_mirna_2 = dA_finetuning.train_model(train_set_x_org=test_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2[0], corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)

						com_hidden = np.hstack([com_hidden, train_set_x_extr_mirna_2]) if com_hidden.size else train_set_x_extr_mirna_2
					if one == "protein.expr":
						prt_expr_org = train_set_x_org
						data_permute_id = rng.permutation(train_set_x_org.shape[1])
						test_set_x_org = test_set_x_org[:, data_permute_id]
						prt_expr_org_test = test_set_x_org
						n_hidden=50
						model_trained_prt, train_set_x_extr_prt, training_time_prt = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						print "finish initial training for protein"
						
						W = theano.shared(value=model_trained_prt.W.get_value(), name='W', borrow=True)
						bvis = theano.shared(value=model_trained_prt.b_prime.get_value(), borrow=True)
						bhid = theano.shared(value=model_trained_prt.b.get_value(), name='b', borrow=True)
						model_trained_prt_2, train_set_x_extr_prt_2, training_time_prt_2 = dA_finetuning.train_model(train_set_x_org=test_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2[0], corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)
						com_hidden = np.hstack([com_hidden, train_set_x_extr_prt_2]) if com_hidden.size else train_set_x_extr_prt_2
					if one == "gn.expr":
						gn_expr_org = train_set_x_org
						gn_expr_org_test = test_set_x_org
						n_hidden=500
						model_trained_1_gn, train_set_x_extr_1_gn, training_time_1_gn = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						W = theano.shared(value=model_trained_1_gn.W.get_value(), name='W', borrow=True)
						bvis = theano.shared(value=model_trained_1_gn.b_prime.get_value(), borrow=True)
						bhid = theano.shared(value=model_trained_1_gn.b.get_value(), name='b', borrow=True)
						model_trained_1_gn_2, train_set_x_extr_1_gn_2, training_time_1_gn_2 = dA_finetuning.train_model(train_set_x_org=test_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2[0], corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)
						n_hidden=50
						model_trained_2_gn, train_set_x_extr_2_gn, training_time_2_gn = dA.train_model(train_set_x_org=train_set_x_extr_1_gn_2, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						print "finish initial training for gn"

						com_hidden = np.hstack([com_hidden, train_set_x_extr_2_gn]) if com_hidden.size else train_set_x_extr_2_gn
					if one == "methy":
						methy_expr_org = train_set_x_org
						data_permute_id = rng.permutation(train_set_x_org.shape[1])
						test_set_x_org = test_set_x_org[:, data_permute_id]
						methy_expr_org_test = test_set_x_org
						n_hidden=500
						model_trained_1_methy, train_set_x_extr_1_methy, training_time_1_methy = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						W = theano.shared(value=model_trained_1_methy.W.get_value(), name='W', borrow=True)
						bvis = theano.shared(value=model_trained_1_methy.b_prime.get_value(), borrow=True)
						bhid = theano.shared(value=model_trained_1_methy.b.get_value(), name='b', borrow=True)
						model_trained_1_methy_2, train_set_x_extr_1_methy_2, training_time_1_methy_2 = dA_finetuning.train_model(train_set_x_org=test_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2[0], corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)
						n_hidden=50
						model_trained_2_methy, train_set_x_extr_2_methy, training_time_2_methy = dA.train_model(train_set_x_org=train_set_x_extr_1_methy_2, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						print "finish initial training for methy"

						com_hidden = np.hstack([com_hidden, train_set_x_extr_2_methy]) if com_hidden.size else train_set_x_extr_2_methy
					if one == "cna.nocnv.nodup":
						cna_expr_org = train_set_x_org
						data_permute_id = rng.permutation(train_set_x_org.shape[1])
						test_set_x_org = test_set_x_org[:, data_permute_id]
						cna_expr_org_test = test_set_x_org
						n_hidden=500
						model_trained_1_cna, train_set_x_extr_1_cna, training_time_1_cna = dA.train_model(train_set_x_org=train_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						W = theano.shared(value=model_trained_1_cna.W.get_value(), name='W', borrow=True)
						bvis = theano.shared(value=model_trained_1_cna.b_prime.get_value(), borrow=True)
						bhid = theano.shared(value=model_trained_1_cna.b.get_value(), name='b', borrow=True)
						model_trained_1_cna_2, train_set_x_extr_1_cna_2, training_time_1_cna_2 = dA_finetuning.train_model(train_set_x_org=test_set_x_org, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2[0], corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng, W=W, bhid=bhid, bvis=bvis)
						n_hidden=50
						model_trained_2_cna, train_set_x_extr_2_cna, training_time_2_cna = dA.train_model(train_set_x_org=train_set_x_extr_1_cna_2, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=learning_rate, corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
						print "finish initial training for cna"

						com_hidden = np.hstack([com_hidden, train_set_x_extr_2_cna]) if com_hidden.size else train_set_x_extr_2_cna
				
				n_hidden=50
				model_trained_1, com_hidden_x_extr_1, training_time_1 = dA.train_model(train_set_x_org=com_hidden, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2[0], corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
				n_hidden=10
				model_trained_21, com_hidden_x_extr_21, training_time_21 = dA.train_model(train_set_x_org=com_hidden_x_extr_1, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2[0], corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)
				n_hidden=1
				model_trained_2, com_hidden_x_extr_2, training_time_2 = dA.train_model(train_set_x_org=com_hidden_x_extr_21, training_epochs=n_epochs, batch_size=batch_size, n_hidden=n_hidden, learning_rate=lr2[0], corruption_level=corruption_level, cost_measure="cross_entropy", rng=rng)


				[mirna_ave1, mirna_ave2] = cors_save.mirna_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_mirna_2, com_hidden_x_extr_2, train_set_x_extr_mirna_2, mirna_expr_org_test, result_prefix)
				[prt_ave1, prt_ave2] = cors_save.prt_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_prt_2, com_hidden_x_extr_2, train_set_x_extr_prt_2, prt_expr_org_test, result_prefix)
				[gn_ave1, gn_ave2] = cors_save.gn_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_gn, model_trained_2_gn, com_hidden_x_extr_2, train_set_x_extr_2_gn, gn_expr_org_test, result_prefix)
				[methy_ave1, methy_ave2] = cors_save.methy_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_methy, model_trained_2_methy, com_hidden_x_extr_2, train_set_x_extr_2_methy, methy_expr_org_test, result_prefix)
				[cna_ave1, cna_ave2] = cors_save.cna_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_cna, model_trained_2_cna, com_hidden_x_extr_2, train_set_x_extr_2_cna, cna_expr_org_test, result_prefix)

				cor1 = mirna_ave1 + prt_ave1 + gn_ave1 + methy_ave1 + cna_ave1
				cor2 = mirna_ave2 + prt_ave2 + gn_ave2 + methy_ave2 + cna_ave2

				cors_save.save_final_para(result_prefix, com_hidden_x_extr_2, model_trained_2)

				cors = [mirna_ave1, prt_ave1, gn_ave1, methy_ave1, cna_ave1, 1000000, mirna_ave2, prt_ave2, gn_ave2, methy_ave2, cna_ave2]
				filename=result_prefix + "cors_values.txt"
				file_handle=file(filename,'w')
				np.savetxt(file_handle, cors ,delimiter="\t")
				file_handle.close()

			para = ["lr:" + str(learning_rate), "cl:" + str(corruption_level), "bs:" + str(batch_size), "ep:" + str(n_epochs)]
			cors1org = ["orgCor:", mirna_ave1, prt_ave1, gn_ave1, methy_ave1, cna_ave1]
			cors2org = ["comCor:", mirna_ave2, prt_ave2, gn_ave2, methy_ave2, cna_ave2]
			tot = ["addOrg:", cor1, cor2]

			filename2=result_prefix + "parameters_cors.txt"
		        file_handle2=file(filename2,'a')
		        np.savetxt(file_handle2, para, delimiter="\t", fmt="%s")
		        np.savetxt(file_handle2, cors1org, delimiter="\t", fmt="%s")
		        np.savetxt(file_handle2, cors2org, delimiter="\t", fmt="%s")
		        np.savetxt(file_handle2, tot, delimiter="\t", fmt="%s")
		        np.savetxt(file_handle2, [''], delimiter="\t", fmt="%s")
			file_handle2.close()
			
