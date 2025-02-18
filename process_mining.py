from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pm4py
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
import time
import func_timeout


input_dir = "Input/PM/"
input_eventlogs_dir = input_dir + "EventLogs/"

output_dir = "Output/PM/"
output_data_dir = output_dir + "Data/"
output_petrinet_dir = output_dir + "PetriNet/"
output_metrics_dir = output_dir + "Metrics/"

variant = ""

def read_event_logs():

	event_logs = {}

	event_logs["TR"] = xes_importer.apply(input_eventlogs_dir + "N_tr.xes")
	event_logs["N"] = xes_importer.apply(input_eventlogs_dir + "N_tst.xes")
	event_logs["A"] = xes_importer.apply(input_eventlogs_dir + "A.xes")

	return event_logs
	
def split_event_log(event_log, split_parameter):

	pd_event_log = []
	cc_event_log = []

	traces = []
	for trace in event_log:
		traces.append(trace)

	cc_event_log, pd_event_log = train_test_split(traces, test_size=split_parameter)
	cc_event_log = (pm4py.objects.log.obj.EventLog)(cc_event_log)
	pd_event_log = (pm4py.objects.log.obj.EventLog)(pd_event_log)
	
	return pd_event_log, cc_event_log

def process_discovery(event_log, variant):

	petri_net = {}

	if variant == "im":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_inductive(event_log)

	elif variant == "ilp":
		#petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=0)
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=1)

	return petri_net

def export_petri_net(petri_net):
	pnml_exporter.apply(petri_net["network"], petri_net["initial_marking"], output_petrinet_dir + "PN.pnml", final_marking = petri_net["final_marking"])

def pm_based_fe(event_logs, petri_net):

	encoded_event_logs = {}
	cc_timing = 0

	# Get the activities from both the petri net and event logs

	activities = []

	petri_net_activities = get_petri_net_activities(petri_net)
	activities = activities + petri_net_activities

	for event_log_type in event_logs:
		event_log_activities = get_event_log_activities(event_logs[event_log_type])
		activities = list(set(activities + event_log_activities))

	activities.sort()

	timings = []

	for event_log_type in event_logs:
		
		trace_wise_diagnoses_fitness_precision = []
		
		for trace in event_logs[event_log_type]:
			
			log = (pm4py.objects.log.obj.EventLog)([trace])
			timing = time.time()
			fitness, precision, aligned_traces = compute_fitness_precision(petri_net, log, "ALIGNMENT_BASED")
			misaligned_activities = compute_misaligned_activities(log, aligned_traces)
			timings.append(time.time() - timing)
			for activity in activities:
				if activity not in list(misaligned_activities.keys()):
					misaligned_activities[activity] = 0
			temp_list = []
			for sorted_key in sorted(misaligned_activities):
				temp_list.append(misaligned_activities[sorted_key])
			trace_wise_diagnoses_fitness_precision.append(temp_list + [fitness, precision])

		encoded_event_logs[event_log_type] = pd.DataFrame(columns = activities + ["F", "P"], data = trace_wise_diagnoses_fitness_precision)

	cc_timing = sum(timings)/len(timings)

	return encoded_event_logs, cc_timing

def get_event_log_activities(event_log):
	
	activities = []
	for trace in event_log:
		for event in trace:
			if event["concept:name"] not in activities:
				activities.append(event["concept:name"])	
					
	activites = list(set(activities))

	return activities

def get_petri_net_activities(petri_net):
	activities = []
	transitions = list(petri_net["network"]._PetriNet__get_transitions())

	for transition in transitions:
		transition = transition._Transition__get_label()
		if transition != None:
			activities.append(transition)

	return activities

def compute_fitness_precision(petri_net, event_log, cc_variant):

	log_fitness = 0.0
	aligned_traces = None
	parameters = {}
	parameters[log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY] = 'CaseID'
	
	if cc_variant == "ALIGNMENT_BASED":
		aligned_traces = alignments.apply_log(event_log, petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
		log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
		log_precision = pm4py.precision_alignments(event_log, petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"])
	elif cc_variant == "TOKEN_BASED":
		replay_results = tokenreplay.algorithm.apply(log = event_log, net = petri_net["network"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = tokenreplay.algorithm.Variants.TOKEN_REPLAY)
		log_fitness = replay_fitness.evaluate(results = replay_results, variant = replay_fitness.Variants.TOKEN_BASED)["log_fitness"]
		log_precision = pm4py.conformance.precision_token_based_replay()

	return log_fitness, log_precision, aligned_traces
	
def compute_misaligned_activities(event_log, aligned_traces):
	
	misaligned_activities = {}
	events = {}
	temp = []
	for aligned_trace in aligned_traces:
		temp.append(list(aligned_trace.values())[0])
	aligned_traces = temp
	for aligned_trace in aligned_traces:
		for move in aligned_trace:
			log_behavior = move[0]
			model_behavior = move[1]
			if log_behavior != model_behavior:
				if log_behavior != None and log_behavior != ">>":
					try:
						events[log_behavior] = events[log_behavior]+1
					except:
						events[log_behavior] = 0
						events[log_behavior] = events[log_behavior]+1
				elif model_behavior != None and model_behavior != ">>":
					try:
						events[model_behavior] = events[model_behavior] + 1
					except:
						events[model_behavior] = 0
						events[model_behavior] = events[model_behavior]+1
	while bool(events):
		popped_event = events.popitem()
		if popped_event[1] > 0:
			misaligned_activities[popped_event[0]] = popped_event[1]

	return misaligned_activities

def save_timing(cc_timing):

	file = open(output_metrics_dir + "cc_time.txt", "w")
	file.write(str(cc_timing))
	file.close()

	return None

def save_encoded_event_logs(encoded_event_logs):

	for event_log_type in encoded_event_logs:
		encoded_event_logs[event_log_type].to_csv(output_data_dir + event_log_type + ".csv", index=False)

	return None
	
try:
	split_parameter = float(sys.argv[1])
	variant = sys.argv[2]
	exp_type = sys.argv[3]
except:
	print("Enter the right number of input arguments.")
	sys.exit()
	
sound_petri_net_found = False
n_tries = 0

while sound_petri_net_found is False:	
	event_logs = read_event_logs()
	pd_event_log, cc_event_log = split_event_log(event_logs["TR"], split_parameter)
	del event_logs["TR"]
	
	if exp_type == "Modeling":
		del event_logs["A"]
		event_logs["N"] = random.sample(event_logs["N"], 100)
	elif exp_type == "AnomalyDetection":
		event_logs["CC_EL"] = cc_event_log
	
	petri_net = process_discovery(pd_event_log, variant)
	export_petri_net(petri_net)
	try:
		encoded_event_logs, cc_timing = func_timeout.func_timeout(timeout=10000, func=pm_based_fe, args=[event_logs, petri_net])
		sound_petri_net_found = True
	except func_timeout.exceptions.FunctionTimedOut:
		cc_timing = -1
		print("Couldn't compute alignments within the time limit.")
		break
	except:
		if n_tries < 2:
			n_tries += 1
			continue
		else:
			cc_timing = -2
			break
			
		
save_timing(cc_timing)
save_encoded_event_logs(encoded_event_logs)

