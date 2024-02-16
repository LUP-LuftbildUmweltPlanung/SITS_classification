#!/usr/bin/python

""" 
	Defining keras architecture.
	4.4. How big and deep model for our data?
	4.4.1. Width influence or the bias-variance trade-off
"""

import sys, os

from tensorflow_legacy.deeplearning.architecture_features import *
import keras
from keras import layers
from keras import backend as K
from keras.optimizer_v2.adam import Adam
from keras.layers import Conv1D, Conv2D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, GRU, Bidirectional, Flatten

#-----------------------------------------------------------------------
#---------------------- ARCHITECTURES
#------------------------------------------------------------------------	

#-----------------------------------------------------------------------		
def Archi_3CONV16_1FC256(X, nbclasses, regression, out_layer):

	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 16 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)

	if not regression:
		#-- SOFTMAX layer
		out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
	else:
		# -- output layers for different regression tasks
		if out_layer == "linear":
			out = Dense(units=1, activation='linear', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer == "sigmoid":
			out = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer =="relu":
			out = Dense(units=1, activation='relu', kernel_regularizer=l2(l2_rate))(X)
		else:
			assert False, "defined out_layer not existing!"
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV16_1FC256')	
	
	
#-----------------------------------------------------------------------		
def Archi_3CONV32_1FC256(X, nbclasses, regression, out_layer):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	print(l2_rate)
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 32 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	if not regression:
		#-- SOFTMAX layer
		out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
	else:
		# -- output layers for different regression tasks
		if out_layer == "linear":
			out = Dense(units=1, activation='linear', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer == "sigmoid":
			out = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer =="relu":
			out = Dense(units=1, activation='relu', kernel_regularizer=l2(l2_rate))(X)
		else:
			assert False, "defined out_layer not existing!"
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV32_1FC256')	


#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256(X, nbclasses, regression, out_layer):
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 64 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	if not regression:
		#-- SOFTMAX layer
		out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
	else:
		# -- output layers for different regression tasks
		if out_layer == "linear":
			out = Dense(units=1, activation='linear', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer == "sigmoid":
			out = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer =="relu":
			out = Dense(units=1, activation='relu', kernel_regularizer=l2(l2_rate))(X)
		else:
			assert False, "defined out_layer not existing!"
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256')	


#-----------------------------------------------------------------------		
def Archi_3CONV128_1FC256(X, nbclasses, regression, out_layer):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 128 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	if not regression:
		#-- SOFTMAX layer
		out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
	else:
		# -- output layers for different regression tasks
		if out_layer == "linear":
			out = Dense(units=1, activation='linear', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer == "sigmoid":
			out = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer =="relu":
			out = Dense(units=1, activation='relu', kernel_regularizer=l2(l2_rate))(X)
		else:
			assert False, "defined out_layer not existing!"
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV128_1FC256')	


#-----------------------------------------------------------------------		
def Archi_3CONV256_1FC256(X, nbclasses, regression, out_layer):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 256 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	if not regression:
		#-- SOFTMAX layer
		out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
	else:
		# -- output layers for different regression tasks
		if out_layer == "linear":
			out = Dense(units=1, activation='linear', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer == "sigmoid":
			out = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer =="relu":
			out = Dense(units=1, activation='relu', kernel_regularizer=l2(l2_rate))(X)
		else:
			assert False, "defined out_layer not existing!"

	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV256_1FC256')	


#-----------------------------------------------------------------------		
def Archi_3CONV512_1FC256(X, nbclasses, regression, out_layer):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 512 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	if not regression:
		#-- SOFTMAX layer
		out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
	else:
		# -- output layers for different regression tasks
		if out_layer == "linear":
			out = Dense(units=1, activation='linear', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer == "sigmoid":
			out = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer =="relu":
			out = Dense(units=1, activation='relu', kernel_regularizer=l2(l2_rate))(X)
		else:
			assert False, "defined out_layer not existing!"

	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV512_1FC256')	


#-----------------------------------------------------------------------		
def Archi_3CONV1024_1FC256(X, nbclasses, regression, out_layer):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 1024 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	if not regression:
		#-- SOFTMAX layer
		out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
	else:
		# -- output layers for different regression tasks
		if out_layer == "linear":
			out = Dense(units=1, activation='linear', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer == "sigmoid":
			out = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(l2_rate))(X)
		elif out_layer =="relu":
			out = Dense(units=1, activation='relu', kernel_regularizer=l2(l2_rate))(X)
		else:
			assert False, "defined out_layer not existing!"

	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV1024_1FC256')	


#--------------------- Switcher for running the architectures
def runArchi(noarchi, out_layer, regression, n_epochs, batch_size, *args):

	switcher = {		
		0: Archi_3CONV16_1FC256,
		1: Archi_3CONV32_1FC256,
		2: Archi_3CONV64_1FC256,
		3: Archi_3CONV128_1FC256,
		4: Archi_3CONV256_1FC256,
		5: Archi_3CONV512_1FC256,
		6: Archi_3CONV1024_1FC256,
	}
	func = switcher.get(noarchi, lambda: 0)

	if not regression:
		model = func(args[0], args[1].shape[1], regression, out_layer)
	else:
		model = func(args[0], 1, regression, out_layer)

	if len(args)==5:
		return trainTestModel_EarlyAbandon(regression, model, *args, n_epochs=n_epochs, batch_size=batch_size)
	elif len(args)==7:
		return trainValTestModel_EarlyAbandon(regression, model, *args, n_epochs=n_epochs, batch_size=batch_size)

#EOF
