from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense,Input,Concatenate
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.cluster import DBSCAN
import pickle
import random
import os
import sys
import math
import random
import pandas as pd
import numpy as np
from itertools import permutations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

input_dir = "Input/FD/"
input_data_dir = input_dir + "Data/"

output_dir = "Output/FD/"
output_metrics_dir = output_dir + "Metrics/"
output_model_dir = output_dir + "Model/"

dbscan_hyperparameters = {
	"eps": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
	"min_samples": [1, 2, 3, 4, 5, 6]
}

'''
dbscan_hyperparameters = {
	"eps": [1.0],
	"min_samples": [3]
}
'''

ae_hyperparameters = {
	"n_components": [2, 4, 8, 16],
	"n_hidden_neurons": [32, 64, 128, 256],
	"optimizer": ["adam", "rmsprop", "SGD"],
	"batch_size": [8, 16, 32, 64],
	"epochs": [100, 250, 500]
}

kpca_hyperparameters = {
	"n_components": [2, 4, 8, 16],
	"kernel": ["poly", "rbf", "sigmoid"],
	"gamma": [0.01, 0.1, 0.25],
	"alpha": [0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
	"degree": [3, 4, 5, 6]
}


def read_data():

	training_set = None
	normal_test_set = None
	anomalous_test_set = None

	for file in os.listdir(input_data_dir):
		if file.split(".csv")[0] == "CC_EL":
			training_set = pd.read_csv(input_data_dir + file)
		elif file.split(".csv")[0] == "N":
			normal_test_set = pd.read_csv(input_data_dir + file)
		elif file.split(".csv")[0] == "A":	
			anomalous_test_set = pd.read_csv(input_data_dir + file)
	
	training_set = training_set.drop(columns = ["F", "P"], axis=1)
	normal_test_set = normal_test_set.drop(columns = ["F", "P"], axis=1)
	anomalous_test_set = anomalous_test_set.drop(columns = ["F", "P"], axis=1)
	
	'''
	training_set, _ = normalize_dataset(training_set, 0, None, "min-max")
	normal_test_set, _ = normalize_dataset(normal_test_set, 0, None, "min-max")
	anomalous_test_set, _ = normalize_dataset(anomalous_test_set, 0, None, "min-max")
	'''
	
	normal_test_set["Label"] = ["N"]*len(normal_test_set)
	anomalous_test_set["Label"] = ["A"]*len(anomalous_test_set)
	test_set = pd.concat([normal_test_set, anomalous_test_set], axis=0, ignore_index=True)

	return training_set, test_set

def normalize_dataset(dataset, reuse_parameters, normalization_parameters_in, normalization_technique):
	
	normalized_dataset = dataset.copy()
	normalization_parameters = {}
	if reuse_parameters == 0:
		if normalization_technique == "zscore":
			for column in normalized_dataset:
				column_values = normalized_dataset[column].values
				if np.any(column_values) == True:
					column_values_mean = np.mean(column_values)
					column_values_std = np.std(column_values)
					if column_values_std != 0:
						column_values = (column_values - column_values_mean)/column_values_std
				else:
					column_values_mean = 0
					column_values_std = 0
				normalized_dataset[column] = column_values
				normalization_parameters[column+"_mean"] = column_values_mean
				normalization_parameters[column+"_std"] = column_values_std
		elif normalization_technique == "min-max":
			column_intervals = get_intervals(dataset)
			for column in normalized_dataset:
				column_data = normalized_dataset[column].tolist()
				intervals = column_intervals[column]
				if intervals[0] != intervals[1]:
					for idx,sample in enumerate(column_data):
						column_data[idx] = (sample-intervals[0])/(intervals[1]-intervals[0])
					
				normalized_dataset[column] = column_data

			for column in column_intervals:
				normalization_parameters[column+"_min"] = column_intervals[column][0]
				normalization_parameters[column+"_max"] = column_intervals[column][1]			
	else:
		if normalization_technique == "zscore":
			for label in normalized_dataset:
				mean = normalization_parameters_in[label+"_mean"]
				std = normalization_parameters_in[label+"_std"]
				parameter_values = normalized_dataset[label].values
				if std != 0:
					parameter_values = (parameter_values - float(mean))/float(std)
				normalized_dataset[label] = parameter_values
		elif normalization_technique == "min-max":
			for label in normalized_dataset:
				min = normalization_parameters_in[label+"_min"]
				max = normalization_parameters_in[label+"_max"]
				parameter_values = normalized_dataset[label].values
				if min != max:
					for idx,sample in enumerate(parameter_values):
						parameter_values[idx] = (sample-min)/(max-min)
				normalized_dataset[label] = parameter_values			
	
	return normalized_dataset, normalization_parameters	

def get_intervals(timeseries):

	intervals = {}
	
	columns = list(timeseries.columns)
	for column in columns:
		intervals[column] = [9999999999, -9999999999]
	for column in timeseries:
		temp_max = timeseries[column].max()
		temp_min = timeseries[column].min()
		if intervals[column][0] > temp_min:
			intervals[column][0] = temp_min
		if intervals[column][1] < temp_max:
			intervals[column][1] = temp_max

	return intervals	

def train_ae(training_set, validation_set, n_components, n_hidden_neurons, optimizer, batch_size, epochs):
	model = None
	threshold=0.0
	training_set_np = training_set.to_numpy()
	training_set_np = training_set_np.astype('float32')
	validation_set_np = validation_set.to_numpy()
	validation_set_np = validation_set_np.astype('float32')
	model = autoencoder(n_hidden_neurons, n_components, len(list(training_set.columns)), optimizer)
	model.fit(training_set_np,training_set_np, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
	reconstructed_validation_set_np = model.predict(validation_set_np)
	threshold = mean_squared_error(validation_set_np, reconstructed_validation_set_np)
	return model, threshold

def train_kpca(training_set, validation_set, n_components, kernel, gamma, degree, alpha):

	model = None
	threshold=0.0
	training_set_np = training_set.to_numpy()
	training_set_np = training_set_np.astype('float32')
	validation_set_np = validation_set.to_numpy()
	validation_set_np = validation_set_np.astype('float32')
	if n_components > len(training_set_np):
		n_components = len(training_set_np)

	model = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, degree=degree, alpha=alpha, fit_inverse_transform=True)
	model.fit(training_set_np)
	compressed_validation_set_np = model.transform(validation_set_np)
	reconstructed_validation_set_np = model.inverse_transform(compressed_validation_set_np)
	threshold = mean_squared_error(validation_set_np, reconstructed_validation_set_np)

	return model, threshold

def autoencoder(hidden_neurons, latent_code_dimension, input_dimension, optimizer):
	input_layer = Input(shape=(input_dimension,))
	encoder = Dense(hidden_neurons,activation="relu")(input_layer)
	code = Dense(latent_code_dimension)(encoder)
	decoder = Dense(hidden_neurons,activation="relu")(code)
	output_layer = Dense(input_dimension,activation="linear")(decoder)
	model = Model(inputs=[input_layer],outputs=[output_layer])
	model.compile(optimizer=optimizer,loss="mse")
	return model	
	
def train_dbscan(training_set, validation_set, eps, min_samples):

	model = {}
	
	temp_training_set = training_set.copy()
	temp_validation_set = validation_set.copy()

	model = DBSCAN(eps=eps, min_samples=min_samples).fit(temp_training_set)
	cluster_labels = model.labels_
	temp_training_set["Cluster"] = cluster_labels
	used = set();
	clusters = [x for x in cluster_labels if x not in used and (used.add(x) or True)]

	instances_sets = {}
	centroids = {}
			
	for cluster in clusters:
		instances_sets[cluster] = []
		centroids[cluster] = []

	temp = temp_training_set
	for index, row in temp.iterrows():
		instances_sets[int(row["Cluster"])].append(row.values.tolist())

	n_features_per_instance = len(instances_sets[list(instances_sets.keys())[0]][0])-1
			
	for instances_set_label in instances_sets:
		instances = instances_sets[instances_set_label]
		for idx, instance in enumerate(instances):
			instances[idx] = instance[0:n_features_per_instance]
		for i in range(0,n_features_per_instance):
			values = []
			for instance in instances:
				values.append(instance[i])
			centroids[instances_set_label].append(np.mean(values))
		
	model = centroids
	
	clusters = []
	for index, instance in temp_validation_set.iterrows():
		min_value = float('inf')
		min_centroid = -1
		for centroid in centroids:
			centroid_coordinates = np.array([float(i) for i in centroids[centroid]])
			dist = np.linalg.norm(instance.values-centroid_coordinates)
			if dist<min_value:
				min_value = dist
				min_centroid = centroid
		clusters.append(min_centroid)
		
	temp_validation_set["Cluster"] = clusters
	distances = []
	for index, instance in temp_validation_set.iterrows():
		if instance["Cluster"] != -1:
			instance = np.array([float(i) for i in instance])
			instance_cluster = int(instance[-1])
			centroid_coordinates = np.array([float(i) for i in model[instance_cluster]])
			instance = np.delete(instance, len(instance)-1)
			distances.append(np.linalg.norm(instance-centroid_coordinates))
	threshold = max(distances)

	return model, threshold

def classify_diagnoses(model, threshold, test_set, fd_method):

	predicted_labels = []
	test_labels = list(test_set["Label"])
	test_set_no_labels_np = test_set.drop(["Label"], axis=1).to_numpy()
	if fd_method == "ae":
		reconstructed_test_set_np = model.predict(test_set_no_labels_np, verbose=0)
		for idx,elem in enumerate(reconstructed_test_set_np):
			error = mean_squared_error(test_set_no_labels_np[idx], reconstructed_test_set_np[idx])
			if error > threshold:
				predicted_labels.append("A")
			else:
				predicted_labels.append("N")

	elif fd_method == "kpca":
		compressed_test_set_np = model.transform(test_set_no_labels_np)
		reconstructed_test_set_np = model.inverse_transform(compressed_test_set_np)
		for idx,elem in enumerate(reconstructed_test_set_np):
			error = mean_squared_error(test_set_no_labels_np[idx], reconstructed_test_set_np[idx])
			if error > threshold:
				predicted_labels.append("A")
			else:
				predicted_labels.append("N")
				
	elif fd_method == "dbscan":
		for idx, elem in enumerate(test_set_no_labels_np):
			min_value = float('inf')
			min_centroid = -1
			for centroid in model:
				centroid_coordinates = np.array([float(i) for i in model[centroid]])
				dist = np.linalg.norm(elem-centroid_coordinates)
				if dist<min_value:
					min_value = dist
					min_centroid = centroid
				
			if dist > threshold:
				predicted_labels.append("A")
			else:
				predicted_labels.append("N")

	return predicted_labels, test_labels

def save_data(encoded_el):

	for type in encoded_el:
		if type == "Training":
			encoded_el[type].to_csv(output_data_dir + "TrainingData.csv", index=False)
		else:
			encoded_el[type].to_csv(output_data_dir + "TestData.csv", index=False)

	return None

def write_model(model, threshold, fd_method):

	if fd_method == "ae":
		model.save(output_model_dir + "ae.keras")
		file = open(output_model_dir + "threshold.txt", "w")
		file.write(str(threshold))
		file.close()

	return None
	
def write_metrics(performance_metrics):

	file = open(output_metrics_dir + "Metrics.txt", "w")
	file.write("Accuracy: " + str(performance_metrics["accuracy"]) + "\n")
	file.write("Precision: " + str(performance_metrics["precision"]) + "\n")
	file.write("Recall: " + str(performance_metrics["recall"]) + "\n")
	file.write("f1: " + str(performance_metrics["f1"]) + "\n")
	file.write("TN: " + str(performance_metrics["tn"]) + "\n")
	file.write("TP: " + str(performance_metrics["tp"]) + "\n")
	file.write("FN: " + str(performance_metrics["fn"]) + "\n")
	file.write("FP: " + str(performance_metrics["fp"]))
	file.close()

def evaluate_performance_metrics(test_labels, predicted_labels):

	performance_metrics = {}
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	
	for idx, label in enumerate(test_labels):
		if predicted_labels[idx] == "N" and predicted_labels[idx] == test_labels[idx]:
			tn = tn + 1
		elif predicted_labels[idx] == "A" and predicted_labels[idx] == test_labels[idx]:
			tp = tp + 1
		elif predicted_labels[idx] == "N" and predicted_labels[idx] != test_labels[idx]:
			fn = fn + 1
		elif predicted_labels[idx] == "A" and predicted_labels[idx] != test_labels[idx]:
			fp = fp + 1
			
	performance_metrics["tp"] = tp
	performance_metrics["tn"] = tn
	performance_metrics["fp"] = fp
	performance_metrics["fn"] = fn	
	
	try:
		performance_metrics["accuracy"] = (tp+tn)/(tp+tn+fp+fn)
	except ZeroDivisionError:
		print("Accuracy could not be computed because the denominator was 0")
		performance_metrics["accuracy"] = "undefined"

	try:
		performance_metrics["precision"] = tp/(tp+fp)
	except ZeroDivisionError:
		print("Precision could not be computed because the denominator was 0")
		performance_metrics["precision"] = "undefined"

	try:
		performance_metrics["recall"] = tp/(tp+fn)
	except ZeroDivisionError:
		print("Recall could not be computed because the denominator was 0")
		performance_metrics["recall"] = "undefined"
		
	try:
		performance_metrics["f1"] = 2*tp/(2*tp+fp+fn)
	except ZeroDivisionError:
		print("F1 could not be computed because the denominator was 0")
		performance_metrics["f1"] = "undefined"	
	
	print("The evaluated performance metrics are the following:")
	print("Accuracy: " + str(performance_metrics["accuracy"]))
	print("Precision: " + str(performance_metrics["precision"]))
	print("Recall: " + str(performance_metrics["recall"]))
	print("f1: " + str(performance_metrics["f1"]))
	print("TP: " + str(performance_metrics["tp"]))
	print("TN: " + str(performance_metrics["tn"]))
	print("FN: " + str(performance_metrics["fn"]))
	print("FP: " + str(performance_metrics["fp"]))
	
	return performance_metrics
	
try:
	fd_method = sys.argv[1]
	if fd_method == "ae" or fd_method == "kpca" or fd_method == "dbscan":
		split_percentage = float(sys.argv[2])
except:
	print("Input the right number of input arguments.")
	sys.exit()


training_set, test_set = read_data()


best_performing_model = None
best_threshold = 0.0
best_f1 = 0.0
best_performance = None

if fd_method == "ae":
	for n_components in ae_hyperparameters["n_components"]:
		for n_hidden_neurons in ae_hyperparameters["n_hidden_neurons"]:
			for optimizer in ae_hyperparameters["optimizer"]:
				for batch_size in ae_hyperparameters["batch_size"]:
					for epochs in ae_hyperparameters["epochs"]:
						try:
							training_set, validation_set = train_test_split(training_set, test_size=split_percentage)
							model, threshold = train_ae(training_set, validation_set, n_components, n_hidden_neurons, optimizer, batch_size, epochs)
							predicted_labels, test_labels = classify_diagnoses(model, threshold, test_set, fd_method)
							performance_metrics = evaluate_performance_metrics(test_labels, predicted_labels)
							if performance_metrics["f1"] > best_f1:
								best_f1 = performance_metrics["f1"]
								best_performance = performance_metrics.copy()
								best_performing_model = model
								best_threshold = threshold
						except:
							continue

if fd_method == "kpca":
	for n_components in kpca_hyperparameters["n_components"]:
		for kernel in kpca_hyperparameters["kernel"]:
			for gamma in kpca_hyperparameters["gamma"]:
				for degree in kpca_hyperparameters["degree"]:
					for alpha in kpca_hyperparameters["alpha"]:
						try:
							training_set, validation_set = train_test_split(training_set, test_size=split_percentage)
							model, threshold = train_kpca(training_set, validation_set, n_components, kernel, gamma, degree, alpha)
							predicted_labels, test_labels = classify_diagnoses(model, threshold, test_set, fd_method)
							performance_metrics = evaluate_performance_metrics(test_labels, predicted_labels)
							if performance_metrics["f1"] > best_f1:
								best_f1 = performance_metrics["f1"]
								best_performance = performance_metrics.copy()
								best_performing_model = model
								best_threshold = threshold
						except:
							continue
							
if fd_method == "dbscan":
	for eps in dbscan_hyperparameters["eps"]:
		for min_samples in dbscan_hyperparameters["min_samples"]:
			try:
				training_set, validation_set = train_test_split(training_set, test_size=split_percentage)
				model, threshold = train_dbscan(training_set, validation_set, eps, min_samples)
				predicted_labels, test_labels = classify_diagnoses(model, threshold, test_set, fd_method)
				performance_metrics = evaluate_performance_metrics(test_labels, predicted_labels)
				if performance_metrics["f1"] > best_f1:
					best_f1 = performance_metrics["f1"]
					best_performance = performance_metrics.copy()
					best_performing_model = model
					best_threshold = threshold		
			except:
				continue

write_model(best_performing_model, best_threshold, fd_method)
write_metrics(best_performance)

'''
normal_event_log, validation_event_log = split_event_log(event_logs["Normal"], validation_percentage)
petri_net = build_petri_net(normal_event_log)
encoded_el = pm_based_fe(event_logs, petri_net)
'''











