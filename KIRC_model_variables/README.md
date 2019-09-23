KIRC_model_variables
This directory contains the model variables learned from the KIRC datasets: mirna.expr (miRNA expression), protein.expr (protein expression), gn.expr (gene expression), methy (methylation), cna.nocnv.nodup (CNA). Initially the variables were saved in .txt files but they were too large to upload, thus, they were now saved in .h5 using h5py.

For miRNA and protein expression datasets, there are three variables: w, bh, and bv. such as for miRNA expression, the three variables were saved in the files of mirna.expr_para_w.h5, mirna.expr_para_bh.h5 and mirna.expr_para_bv.h5

For the gene expression, methylation and CNA, there are six variables with three from the first individual hidden layer: w_h1, bh_h1 and bv_h1; and three from the second individual hidden layer: w_h2, bh_h2 and bv_h2.

There are three joint hidden layers with name of com_h0, com_h1 and com_h2. There are three variables for each layer: w, hv and hb.

Some of the parameters are too larget to upload, thus they were split into two connected parts, such as methy_para_w_h1.h5 and gn.expr_para_w_h1.h5.

The hidden values that were used for subtyping are in the file of com_h2_values.txt.
