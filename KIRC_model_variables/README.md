
# KIRC_model_variables

This directory contains the model variables learned from the KIRC datasets: mirna.expr (miRNA expression), protein.expr (protein expression), gn.expr (gene expression), methy (methylation), cna.nocnv.nodup (CNA).

For miRNA and protein expression datasets, there are three variables: w, bh, and bv. such as for miRNA expression, the three variables were saved in the files of mirna.expr_para_w.pkl, mirna.expr_para_bh.pkl and mirna.expr_para_bv.pkl

For the gene expression, methylation and CNA, there are six variables with three from the first individual hidden layer: w_h1, bh_h1 and bv_h1; and three from the second individual hidden layer: w_h2, bh_h2 and bv_h2.

There are three joint hidden layers with name of com_h0, com_h1 and com_h2. There are three variables for each layer: w, hv and hb. 

The files with names starting with "cors_" are the files containing the correlations calculated with the reconstructed input from the LJHL and LIHL.

Some of the parameters are too larget to upload but they are availabe upon request.
