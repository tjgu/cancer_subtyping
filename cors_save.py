from __future__ import division
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
import numpy as np

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

def mirna_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_mirna, com_hidden_x_extr_2, train_set_x_extr_mirna, mirna_expr_org, result_dir):
        mirna1 = model_trained_2.get_reconstructed_input(com_hidden_x_extr_2)
        mirna2 = model_trained_21.get_reconstructed_input(mirna1)
        mirna3 = model_trained_1.get_reconstructed_input(mirna2.eval())
        mirna4 = model_trained_mirna.get_reconstructed_input(mirna3.eval()[:, 0:50])
        mirna0 = model_trained_mirna.get_reconstructed_input(train_set_x_extr_mirna)
        [mirna1_cors, mirna1_ave_cor] = cor_pearson(mirna_expr_org, mirna0.eval())
        [mirna2_cors, mirna2_ave_cor] = cor_pearson(mirna_expr_org, mirna4.eval())

        filename=result_dir + "cors_mirna_final.txt"
        file_handle=file(filename,'w')
        np.savetxt(file_handle, [mirna2_ave_cor], delimiter="\t")
        np.savetxt(file_handle, mirna2_cors, delimiter="\t")
        file_handle.close()
        filename=result_dir + "cors_mirna_hidden.txt"
        file_handle=file(filename,'w')
        np.savetxt(file_handle, [mirna1_ave_cor], delimiter="\t")
        np.savetxt(file_handle, mirna1_cors, delimiter="\t")
        file_handle.close()
	print mirna1_ave_cor, mirna2_ave_cor
	return(mirna1_ave_cor, mirna2_ave_cor)

def prt_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_prt, com_hidden_x_extr_2, train_set_x_extr_prt, prt_expr_org, result_dir):
        prt1 = model_trained_2.get_reconstructed_input(com_hidden_x_extr_2)
        prt2 = model_trained_21.get_reconstructed_input(prt1)
        prt3 = model_trained_1.get_reconstructed_input(prt2)
        prt4 = model_trained_prt.get_reconstructed_input(prt3.eval()[:, 50:100])
        prt0 = model_trained_prt.get_reconstructed_input(train_set_x_extr_prt)
        [prt1_cors, prt1_ave_cor] = cor_pearson(prt_expr_org, prt0.eval())
        [prt2_cors, prt2_ave_cor] = cor_pearson(prt_expr_org, prt4.eval())

        filename=result_dir + "cors_prt_final.txt"
        file_handle=file(filename,'w')
        np.savetxt(file_handle, [prt2_ave_cor], delimiter="\t")
        np.savetxt(file_handle, prt2_cors, delimiter="\t")
        file_handle.close()
        filename=result_dir + "cors_prt_hidden.txt"
        file_handle=file(filename,'w')
        np.savetxt(file_handle, [prt1_ave_cor], delimiter="\t")
        np.savetxt(file_handle, prt1_cors, delimiter="\t")
        file_handle.close()
	print prt1_ave_cor, prt2_ave_cor
	return(prt1_ave_cor, prt2_ave_cor)

def gn_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_gn, model_trained_2_gn, com_hidden_x_extr_2, train_set_x_extr_2_gn, gn_expr_org, result_dir):
        gn1 = model_trained_2.get_reconstructed_input(com_hidden_x_extr_2)
        gn2 = model_trained_21.get_reconstructed_input(gn1)
        gn3 = model_trained_1.get_reconstructed_input(gn2)
        gn4 = model_trained_2_gn.get_reconstructed_input(gn3.eval()[:, 100:150])
        gn5 = model_trained_1_gn.get_reconstructed_input(gn4.eval())
        gn00 = model_trained_2_gn.get_reconstructed_input(train_set_x_extr_2_gn)
        gn01 = model_trained_1_gn.get_reconstructed_input(gn00.eval())
        [gn1_cors, gn1_ave_cor] = cor_pearson(gn_expr_org, gn01.eval())
        [gn2_cors, gn2_ave_cor] = cor_pearson(gn_expr_org, gn5.eval())

        filename=result_dir + "cors_gn_final.txt"
        file_handle=file(filename,'w')
        np.savetxt(file_handle, [gn2_ave_cor], delimiter="\t")
        np.savetxt(file_handle, gn2_cors, delimiter="\t")
        file_handle.close()
        filename=result_dir + "cors_gn_hidden.txt"
        file_handle=file(filename,'w')
        np.savetxt(file_handle, [gn1_ave_cor], delimiter="\t")
        np.savetxt(file_handle, gn1_cors, delimiter="\t")
        file_handle.close()
	print gn1_ave_cor, gn2_ave_cor
	return(gn1_ave_cor, gn2_ave_cor)

def methy_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_methy, model_trained_2_methy, com_hidden_x_extr_2, train_set_x_extr_2_methy, methy_expr_org, result_dir):
        methy1 = model_trained_2.get_reconstructed_input(com_hidden_x_extr_2)
        methy2 = model_trained_21.get_reconstructed_input(methy1)
        methy3 = model_trained_1.get_reconstructed_input(methy2)
        methy4 = model_trained_2_methy.get_reconstructed_input(methy3.eval()[:, 150:200])
        methy5 = model_trained_1_methy.get_reconstructed_input(methy4.eval())
        methy00 = model_trained_2_methy.get_reconstructed_input(train_set_x_extr_2_methy)
        methy01 = model_trained_1_methy.get_reconstructed_input(methy00.eval())
        [methy1_cors, methy1_ave_cor] = cor_pearson(methy_expr_org, methy01.eval())
        [methy2_cors, methy2_ave_cor] = cor_pearson(methy_expr_org, methy5.eval())

        filename=result_dir + "cors_methy_final.txt"
        file_handle=file(filename,'w')
        np.savetxt(file_handle, [methy2_ave_cor], delimiter="\t")
        np.savetxt(file_handle, methy2_cors, delimiter="\t")
        file_handle.close()
        filename=result_dir + "cors_methy_hidden.txt"
        file_handle=file(filename,'w')
        np.savetxt(file_handle, [methy1_ave_cor], delimiter="\t")
        np.savetxt(file_handle, methy1_cors, delimiter="\t")
        file_handle.close()
	print methy1_ave_cor, methy2_ave_cor
	return(methy1_ave_cor, methy2_ave_cor)

def cna_cor_save(model_trained_2, model_trained_21, model_trained_1, model_trained_1_cna, model_trained_2_cna, com_hidden_x_extr_2, train_set_x_extr_2_cna, cna_expr_org, result_dir):
        cna1 = model_trained_2.get_reconstructed_input(com_hidden_x_extr_2)
        cna2 = model_trained_21.get_reconstructed_input(cna1)
        cna3 = model_trained_1.get_reconstructed_input(cna2)
        cna4 = model_trained_2_cna.get_reconstructed_input(cna3.eval()[:, 200:250])
        cna5 = model_trained_1_cna.get_reconstructed_input(cna4.eval())
        cna00 = model_trained_2_cna.get_reconstructed_input(train_set_x_extr_2_cna)
        cna01 = model_trained_1_cna.get_reconstructed_input(cna00.eval())
        [cna1_cors, cna1_ave_cor] = cor_pearson(cna_expr_org, cna01.eval())
        [cna2_cors, cna2_ave_cor] = cor_pearson(cna_expr_org, cna5.eval())

        filename=result_dir + "cors_cna_final.txt"
        file_handle=file(filename,'w')
        np.savetxt(file_handle, [cna2_ave_cor], delimiter="\t")
        np.savetxt(file_handle, cna2_cors, delimiter="\t")
        file_handle.close()
        filename=result_dir + "cors_cna_hidden.txt"
        file_handle=file(filename,'w')
        np.savetxt(file_handle, [cna1_ave_cor], delimiter="\t")
        np.savetxt(file_handle, cna1_cors, delimiter="\t")
        file_handle.close()
	print cna1_ave_cor, cna2_ave_cor
	return(cna1_ave_cor, cna2_ave_cor)

def save_final_para(result_dir, com_hidden_x_extr_2, model_trained_2):
	filename=result_dir + "com_h2_values.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, com_hidden_x_extr_2, delimiter="\t", header=" the values in the second combined hidden layer")
	file_handle.close()
	filename=result_dir + "com_h2_parameters_w.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, model_trained_2.params[0].get_value(), delimiter="\t", header="the parameter of w in the second hidden layer")
	file_handle.close()
	filename=result_dir + "com_h2_parameters_bh.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, model_trained_2.params[1].get_value(), delimiter="\t", header="the parameter of b hidden in the second hidden layer")
	file_handle.close()
	filename=result_dir + "com_h2_parameters_bv.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, model_trained_2.params[2].get_value(), delimiter="\t", header="the parameter of b visible in the second hidden layer")
	file_handle.close()

def save_para_oneLayer(result_dir, paraOne, model_trained):
	filename=result_dir + paraOne + "_para_w.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, model_trained.params[0].get_value(), delimiter="\t", header="the parameter of w in the hidden layer")
	file_handle.close()
	filename=result_dir + paraOne + "_para_bh.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, model_trained.params[1].get_value(), delimiter="\t", header="the parameter of b hidden in the hidden layer")
	file_handle.close()
	filename=result_dir + paraOne + "_para_bv.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, model_trained.params[2].get_value(), delimiter="\t", header="the parameter of b visible in the hidden layer")
	file_handle.close()
	
def save_para_twoLayers(result_dir, paraOne, model_trained_1, model_trained_2):
	filename=result_dir + paraOne + "_para_w_h1.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, model_trained_1.params[0].get_value(), delimiter="\t", header="the parameter of w in the first hidden layer")
	file_handle.close()
	filename=result_dir + paraOne + "_para_bh_h1.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, model_trained_1.params[1].get_value(), delimiter="\t", header="the parameter of b hidden in the first hidden layer")
	file_handle.close()
	filename=result_dir + paraOne + "_para_bv_h1.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, model_trained_1.params[2].get_value(), delimiter="\t", header="the parameter of b visible in the first hidden layer")
	file_handle.close()
	
	filename=result_dir + paraOne + "_para_w_h2.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, model_trained_2.params[0].get_value(), delimiter="\t", header="the parameter of w in the second hidden layer")
	file_handle.close()
	filename=result_dir + paraOne + "_para_bh_h2.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, model_trained_2.params[1].get_value(), delimiter="\t", header="the parameter of b hidden in the second hidden layer")
	file_handle.close()
	filename=result_dir + paraOne + "_para_bv_h2.txt"
	file_handle=file(filename,'w')
	np.savetxt(file_handle, model_trained_2.params[2].get_value(), delimiter="\t", header="the parameter of b visible in the second hidden layer")
	file_handle.close()
