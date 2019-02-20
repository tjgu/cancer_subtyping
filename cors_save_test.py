from __future__ import division
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
import numpy as np
import dA
import cors_save

def cor_pearson(df1, df2):
	cors = np.array([])
	for i in xrange(df1.shape[1]):
		cor = pearsonr(df1[:, i], df2[:, i])
		cors = np.vstack([cors, cor]) if len(cors) else cor
	cors_m = np.nanmean(cors[:, 0])
	return(cors, cors_m)


def cor_spearmanr(df1, df2):
	cors = []
	for i in xrange(df1.shape[1]):
		cor = spearmanr(df1[:, i], df2[:, i])
		cors = np.vstack([cors, cor]) if len(cors) else cor
	cors_m = np.nanmean(cors[:, 0])
	return(cors, cors_m)


def test_cor(model_trained_2, model_trained_21, model_trained_1, model_trained_mirna, model_trained_prt, model_trained_1_gn, model_trained_2_gn, model_trained_1_methy, model_trained_2_methy, model_trained_1_cna, model_trained_2_cna, mirna_expr_org_test, prt_expr_org_test, gn_expr_org_test, methy_expr_org_test, cna_expr_org_test, result_dir):
	mirna1_h = dA.test_model(model_trained_mirna, mirna_expr_org_test)
	prt1_h = dA.test_model(model_trained_prt, prt_expr_org_test)
	gn1_h = dA.test_model(model_trained_1_gn, gn_expr_org_test)
	gn2_h = dA.test_model(model_trained_2_gn, gn1_h)
	methy1_h = dA.test_model(model_trained_1_methy, methy_expr_org_test)
	methy2_h = dA.test_model(model_trained_2_methy, methy1_h)
	cna1_h = dA.test_model(model_trained_1_cna, cna_expr_org_test)
	cna2_h = dA.test_model(model_trained_2_cna, cna1_h)
	
	com_hidden = np.hstack([mirna1_h, prt1_h, gn2_h, methy2_h, cna2_h])
	com1_h = dA.test_model(model_trained_1, com_hidden)
	com21_h = dA.test_model(model_trained_21, com1_h)
	com2_h = dA.test_model(model_trained_2, com21_h)

	filename=result_dir + "com_h2_values_test.txt"
        file_handle=file(filename,'w')
        np.savetxt(file_handle, com2_h, delimiter="\t", header=" the values in the second combined hidden layer")
	
	[mirna_ave1_t, mirna_ave2_t] = cors_save2.mirna_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_mirna, com2_h, mirna1_h, mirna_expr_org_test, result_dir)
	[prt_ave1_t, prt_ave2_t] = cors_save2.prt_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_prt, com2_h, prt1_h, prt_expr_org_test, result_dir)
        [gn_ave1_t, gn_ave2_t] = cors_save2.gn_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_gn, model_trained_2_gn, com2_h, gn2_h, gn_expr_org_test, result_dir)
        [methy_ave1_t, methy_ave2_t] = cors_save2.methy_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_methy, model_trained_2_methy, com2_h, methy2_h, methy_expr_org_test, result_dir)
        [cna_ave1_t, cna_ave2_t] = cors_save2.cna_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_cna, model_trained_2_cna, com2_h, cna2_h, cna_expr_org_test, result_dir)

	rcor2_t = 5 - mirna_ave2_t - prt_ave2_t - gn_ave2_t - methy_ave2_t - cna_ave2_t
	rcor1_t = 5 - mirna_ave1_t - prt_ave1_t - gn_ave1_t - methy_ave1_t - cna_ave1_t
	return(rcor1_t, rcor2_t, mirna_ave1_t, mirna_ave2_t, prt_ave1_t, prt_ave2_t, gn_ave1_t, gn_ave2_t, methy_ave1_t, methy_ave2_t, cna_ave1_t, cna_ave2_t)


