#!/usr/bin/python

import os, sys
import argparse
import random


from tensorflow_legacy.outputfiles.save import *
from tensorflow_legacy.outputfiles.evaluation import *
from tensorflow_legacy.sits.readingsits import *
from keras.utils.np_utils import to_categorical

#-----------------------------------------------------------------------		
def main(sits_path_train, sits_path_test,regression, res_path, feature, archi, noarchi, norun, n_epochs, batch_size,class_label, n_channels, val_rate, out_layer):

	if archi == "complexity":
		from tensorflow_legacy.deeplearning.architecture_complexity import runArchi  # --- to be changed to test other configurations
	if archi == "rnn":
		from tensorflow_legacy.deeplearning.architecture_rnn import runArchi  # --- to be changed to test other configurations

	#-- Creating output path if does not exist
	if not os.path.exists(res_path):
		os.makedirs(res_path)

	#---- Evaluated metrics
	eval_label = ['OA', 'train_loss', 'train_time', 'test_time']	
	
	#---- String variables
	#train_str = 'train_class'
	#test_str = 'test_class'
	#train_str = 'train_dataset'
	#test_str = 'test_dataset'
	train_str = sits_path_train
	test_str = sits_path_test

	#---- Get filenames
	train_file = train_str + '.csv'
	test_file = test_str + '.csv'
	print("train_file: ", train_file)
	print("test_file: ", test_file)

	#---- output files			
	res_path = res_path + '/Archi' + str(noarchi) + '/'
	if not os.path.exists(res_path):
		os.makedirs(res_path)
	print("noarchi: ", noarchi)
	str_result = feature + '-' + os.path.basename(train_str) + '-noarchi' + str(noarchi) + '-norun' + str(norun)
	res_file = res_path + '/resultOA-' + str_result + '.csv'
	res_mat = np.zeros((len(eval_label),1))	
	traintest_loss_file = res_path + '/trainingHistory-' + str_result + '.csv'
	conf_file = res_path + '/confMatrix-' + str_result + '.csv'
	out_model_file = res_path + '/bestmodel-' + str_result + '.h5'


	#---- Downloading
	X_train, polygon_ids_train, y_train = readSITSData(train_file, regression)
	X_test,  polygon_ids_test, y_test = readSITSData(test_file, regression)

	if not regression:
		n_classes_test = len(np.unique(y_test))
		n_classes_train = len(np.unique(y_train))
		if (n_classes_test != n_classes_train):
			print("WARNING: different number of classes in train and test")
		n_classes = max(n_classes_train, n_classes_test)
		y_train_one_hot = to_categorical(y_train, n_classes)
		y_test_one_hot = to_categorical(y_test, n_classes)
	else:
		y_train_one_hot = y_train
		y_test_one_hot = y_test
	#print(X_train.shape)
	#print(type(X_train))
	#print(X_train)
	#---- Adding the features and reshaping the data if necessary
	X_train = addingfeat_reshape_data(X_train, feature, n_channels)
	X_test = addingfeat_reshape_data(X_test, feature, n_channels)
	#print(X_train.shape)


	#---- Normalizing the data per band
	minMaxVal_file = '.'.join(out_model_file.split('.')[0:-1])
	minMaxVal_file = minMaxVal_file + '_minMax.txt'
	if not os.path.exists(minMaxVal_file): 
		min_per, max_per = computingMinMax(X_train)
		save_minMaxVal(minMaxVal_file, min_per, max_per)
	else:
		min_per, max_per = read_minMaxVal(minMaxVal_file)
	X_train =  normalizingData(X_train, min_per, max_per)
	X_test =  normalizingData(X_test, min_per, max_per)
	
	#---- Extracting a validation set (if necesary)
	if val_rate > 0:
		X_train, y_train, X_val, y_val = extractValSet(X_train, polygon_ids_train, y_train, val_rate)
		if not regression:
			#--- Computing the one-hot encoding (recomputing it for train)
			y_train_one_hot = to_categorical(y_train, n_classes)
			y_val_one_hot = to_categorical(y_val, n_classes)
		else:
			y_train_one_hot = y_train
			y_val_one_hot = y_val

	
	
	
	if not os.path.isfile(res_file):
		if val_rate==0:
			res_mat[0,norun], res_mat[1,norun], model, model_hist, res_mat[2,norun], res_mat[3,norun] = \
				runArchi(noarchi, out_layer, regression, n_epochs, batch_size, X_train, y_train_one_hot, X_test, y_test_one_hot, out_model_file)
		else:
			res_mat[0,norun], res_mat[1,norun], model, model_hist, res_mat[2,norun], res_mat[3,norun] = \
				runArchi(noarchi, out_layer, regression, n_epochs, batch_size, X_train, y_train_one_hot, X_val, y_val_one_hot, X_test, y_test_one_hot, out_model_file)

		saveLossAcc(model_hist, traintest_loss_file)		
		p_test = model.predict(x=X_test)
		if not regression:
			#---- computing confusion matrices
			C = computingConfMatrix(y_test, p_test,n_classes)
			#---- saving the confusion matrix
			save_confusion_matrix(C, class_label, conf_file)
				
		print('Test Overall accuracy (OA): ', res_mat[0,norun])
		print('Val loss: ', res_mat[1,norun])
		print('Training time (s): ', res_mat[2,norun])
		print('Test time (s): ', res_mat[3,norun])
		
		#---- saving res_file
		saveMatrix(np.transpose(res_mat), res_file, eval_label)

#-----------------------------------------------------------------------		
if __name__ == "__main__":
	try:
		if len(sys.argv) == 1:
			prog = os.path.basename(sys.argv[0])
			print('      '+sys.argv[0]+' [options]')
			print("     Help: ", prog, " --help")
			print("       or: ", prog, " -h")
			print("example 1: python %s --sits_path path/to/sits_datasets --res_path path/to/results " %sys.argv[0])
			sys.exit(-1)
		else:
			parser = argparse.ArgumentParser(description='Running deep learning architectures on SITS datasets')
			parser.add_argument('--sits_path', dest='sits_path',
								help='path for train and test sits datasets',
								default=None)
			parser.add_argument('--res_path', dest='result_path',
								help='path where to store the datasets',
								default=None)
			parser.add_argument('--feat', dest='feature',
								help='used feature vector',
								default="SB")
			parser.add_argument('--noarchi', dest='noarchi', type=int,
								help='archi to run', default=2)
			parser.add_argument('--norun', dest='norun', type=int,
								help='run number', default=0)
			args = parser.parse_args()
			print(args.sits_path)
			main(args.sits_path, args.result_path, args.feature, args.noarchi, args.norun)
			print("0")
	except(RuntimeError):
		print >> sys.stderr
		sys.exit(1)

#EOF
