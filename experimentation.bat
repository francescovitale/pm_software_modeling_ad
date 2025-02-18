:: Useful commands:
:: cd <directory>
:: copy path\to\file destination\path
:: xcopy path\to\dirs destination\path /E
:: rmdir "<dir_name>" /s /q
:: ren path\to\file <name>
:: del /F /Q path\to\file 

:: Options:
:: dataset=[RBC_HANDOVER_0.975, RBC_HANDOVER_0.95, RBC_HANDOVER_0.925, START_OF_MISSION_0.975, START_OF_MISSION_0.95, START_OF_MISSION_0.925]
:: n_dataset_replicas=<integer>
:: split_parameter=<float>
:: pd_variant=[im, ilp]
:: val_split=<float>
:: experimentation_type=[Modeling, AnomalyDetection]
:: fd_method=[kpca, dbscan, ae]

set pd_variant=ilp
set dataset=RBC_HANDOVER_0.975 
set n_dataset_replicas=9
set split_parameter=0.01
set val_split=0.1
set fd_methods=kpca ae dbscan
set exp_type=AnomalyDetection

for /D %%p IN ("Results\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for %%k in (%dataset%) do (

	mkdir Results\%%k

	for /l %%a in (0, 1, %n_dataset_replicas%) do (

		mkdir Results\%%k\%%a
		
		del /F /Q Input\PM\EventLogs\*

		copy Data\%%k\%%a\N_tr.xes Input\PM\EventLogs
		copy Data\%%k\%%a\N_tst.xes Input\PM\EventLogs
		copy Data\%%k\%%a\A.xes Input\PM\EventLogs

		for %%x in (%split_parameter%) do (

			mkdir Results\%%k\%%a\%%x
			mkdir Results\%%k\%%a\%%x\Metrics
			mkdir Results\%%k\%%a\%%x\Models

			for %%y in (%pd_variant%) do (

				call clean_environment

				python process_mining.py %%x %%y %exp_type%

				copy Output\PM\Data\* Input\FD\Data

				copy Output\PM\PetriNet\PN.pnml Results\%%k\%%a\%%x\Models
				ren Results\%%k\%%a\%%x\Models\PN.pnml %%y_PN.pnml
				copy Output\PM\Metrics\cc_time.txt Results\%%k\%%a\%%x\Metrics
				ren Results\%%k\%%a\%%x\Metrics\cc_time.txt %%y_cc_time.txt
				copy Output\PM\Data\N.csv Results\%%k\%%a\%%x\Metrics
				ren Results\%%k\%%a\%%x\Metrics\N.csv %%y_F_P.csv

				for %%z in (%fd_methods%) do (

					python fault_detection.py %%z %val_split%
					
					copy Output\FD\Metrics\Metrics.txt Results\%%k\%%a\%%x\Metrics
					ren Results\%%k\%%a\%%x\Metrics\Metrics.txt %%y_%%z_metrics.txt
						
				)
			)
		)
	)
)




