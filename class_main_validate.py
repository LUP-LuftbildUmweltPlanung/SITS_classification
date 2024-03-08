# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: Admin
"""

#os.chdir(r"E:\++++Promotion\Verwaltung\Publikation_1\Workflow_Scripts\final")


#model_path = r"/uge_mount/FORCE/new_struc/process/result/_SITSModels/vgh_30ep_full_tempcnn_5dint_RELU/Archi6/bestmodel-SB_cont-features_vgh_train-noarchi6-norun0.h5"
model_path = r"/uge_mount/FORCE/new_struc/process/result/_SITSModels/vgh_30ep_full_tempcnn_5dint_RELU/Archi2/bestmodel-SB_cont-features_vgh_train-noarchi2-norun0.h5"
feature = "SB_cont"
n_channels = 10
test_file = r"/uge_mount/FORCE/new_struc/process/result/_SITSrefdata/vgh_30ep_full_tempcnn_5dint/features_vgh_test.csv"
result_file = test_file.replace("test","test_result")
proba = False
regression = True
#predict_csv(model_path, feature, n_channels, test_file, result_file, proba, regression)


from sits_classification.utils.validate import validate_main
#### names with space if existing, otherwise without
response_name = "Vegetation Height "
aoi_name = ""
algorithm_name = "TempCNN "
csv_test = test_file.replace("features","prediction_relu_noarchi2/features").replace("test","test_result")
csv_ref = test_file.replace("features","prediction_relu_noarchi2/features").replace("test","ref_result")
strat_validation = True

validate_main(csv_ref, csv_test, response_name, aoi_name, algorithm_name, strat_validation)
