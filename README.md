# Requirements to run the methodology

This project has been executed on a Windows 10 machine with Python 3.11.5. A few libraries have been used within Python modules. Among these, there are:

- pm4py 2.7.11.11
- scipy 1.14.0
- scikit-learn 1.3.0
- tensorflow 2.18.0

Please note that the list above is not comprehensive and there could be other requirements for running the project.

# Execution instructions and project description

The methodology runs by executing the DOS experimentation.bat script. This script includes experimental parameters to set: 

- The process discovery variant (pd_variant)
- The dataset (dataset)
- The number of dataset replicas (n_dataset_replicas)
- The split parameter (split_parameter)
- The validation percentage (val_split)
- The anomaly detection methods (fd_methods)
- The type of the experiment (exp_type)

The exp_type variable controls which of the two experiments to perform. If exp_type is set to Modeling, the process_mining.py script will exclude anomalous behavior and only evaluate the process model quality by comparing normal test event logs with the reference Petri net. In particular, for each dataset replica:

- A split parameter value is chosen
- A process discovery variant is chosen
- The training event log of the dataset is split according to the split parameter value, and the reference Petri net is generated (process_mining.py)
- Anomalous behavior is left out (process_mining.py)
- The normal (test) event log is checked against the reference Petri net to evaluate its quality (process_mining.py)

Otherwise, if exp_type is set to AnomalyDetection, the process_mining.py script will include anomalous behavior when computing the conformance checking diagnoses against the reference Petri net. In particular, for each dataset replica:

- A split parameter value is chosen
- A process discovery variant is chosen
- The training event log of the dataset is split according to the split parameter value, and the reference Petri net is generated (process_mining.py)
- Validation, normal (test) and anomalous (test) event logs are checked against the reference Petri net to obtain the conformance checking diagnoses (process_mining.py)
- An anomaly detection method is chosen
- The anomaly detection method is trained against the validation event log (fault_detection.py)
- The anomaly detection effectiveness is calculated by classifying the normal (test) and anomalous (test) event logs (fault_detection.py)

For each quadruple of dataset replica, split variant, process discovery variant and anomaly detection method, the results and models are saved under the Results folder.
