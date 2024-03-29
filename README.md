# cancer_subtyping
An update of the programs using tensorflow 2.0 is at https://github.com/zhaoxiwu2003/subtype_cancer_using_denoising_autoencoder_with_tied_weights.

The purpose of the project is to stratify the cancer patients into clinical associated subgroups to assist the personalized diagnosis and treatment. 
The algorithm used in this study is stacked denoising autoencoders (SdA).

To obtain the optimal parameters for the architecture of the SdA, we tested a serial of parameters in a wide range: the learning rate at 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3 and 0.5; the corruption level at 0.01, 0.05, 0.1, 0.2 and 0.5; the batch size at 10, 20, 50 and 100; and the epoch size at 1000, 2000 and 5000. The program used for this testing is subtype_parameters_test.py. Depending on the computational resources, it may take weeks or months to run subtype_parameters_test.py.

After selecting the best parameters, we ran subtype_bestPara.py on the whole datasets and saved the model variables for using in similar tasks in the directory of KIRC_model_variables. The details about each variable is in the readme.txt in the directory.

transfer_fineTuning_luad.py and transfer_fineTuning_lgg.py were run to subgroup the LUAD and LGG patients using the KIRC model as a pre-trained model.

All other scripts were called by the above programs. Some of the scripts were adapted from https://github.com/yifeng-li/DECRES: classification.py, dA.py, dA_finetuning.py and logistic_sgd.py

To run the programs, please change the directory for the input datasets if you save the input datasets in a different directory. And also make the directory for saving the output and add it to the program.The datsets are too large to upload but they are available upon request.
