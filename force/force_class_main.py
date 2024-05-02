from sits_classification.force.force_class_utils import *

preprocess_params = {
    "sample" : False,
    #preprocess
    "aois" : None,
    "years" : None,
    "date_ranges" : None,
    #analysis TSS
    #compute
    "NTHREAD_READ" : 7, #4,
    "NTHREAD_COMPUTE" : 7, #11,
    "NTHREAD_WRITE" : 2, #2,
    "BLOCK_SIZE" : 1000,
    #spectral prop
    "Indices" : "BLUE GREEN RED NIR SWIR1 SWIR2 RE1 RE2 RE3 BNIR",#Type: Character list. Valid values: {BLUE,GREEN,RED,NIR,SWIR1,SWIR2,RE1,RE2,RE3,BNIR,NDVI,EVI,NBR,NDTI,ARVI,SAVI,SARVI,TC-BRIGHT,TC-GREEN,TC-WET,TC-DI,NDBI,NDWI,MNDWI,NDMI,NDSI,SMA,kNDVI,NDRE1,NDRE2,CIre,NDVIre1,NDVIre2,NDVIre3,NDVIre1n,NDVIre2n,NDVIre3n,MSRre,MSRren,CCI},
    "Sensors" : "SEN2A SEN2B", #LND04 LND05 LND07 LND08 LND09 SEN2A SEN2B,
    "SPECTRAL_ADJUST" : "FALSE",
    #filter and interpolation
    "ABOVE_NOISE" : "3",
    "BELOW_NOISE" : "1",
    #DATE_RANGE = "2018-01-01 2023-08-24"
    "OUTPUT_TSS" : 'TRUE',
    "INTERPOLATE" : 'RBF', # NONE,LINEAR,MOVING,RBF,HARMONIC
    "INT_DAY" : '5',
    "OUTPUT_TSI" : 'FALSE',
    "column_name": 'class',
    }

if __name__ == '__main__':
    base_params = {
        "force_dir": "/force:/force",
        "local_dir": "/uge_mount:/uge_mount",
        "force_skel": "/uge_mount/FORCE/new_struc/scripts_sits/sits_force/skel/force_cube_sceleton",
        "scripts_skel": "/uge_mount/FORCE/new_struc/scripts_sits/sits_force/skel",
        "temp_folder": "/uge_mount/FORCE/new_struc/process/temp",
        "mask_folder": "/uge_mount/FORCE/new_struc/process/mask",
        "proc_folder": "/uge_mount/FORCE/new_struc/process/result",
        "data_folder": "/uge_mount/FORCE/new_struc/data",
        ###BASIC PARAMS###
        "project_name": "uge_class_training_vghtest_newapproach",
        "hold": False,  # execute cmd
    }

    preprocess_params['aois'] = glob.glob(f"{base_params['data_folder']}/_ReferencePoints/{base_params['project_name']}/*shp")
    preprocess_params['years'] = [int(re.search(r'(\d{4})', os.path.basename(f)).group(1)) for f in preprocess_params['aois'] if re.search(r'(\d{4})', os.path.basename(f))]
    preprocess_params['date_ranges'] = [f"{year - 1}-10-01 {year}-09-30" for year in preprocess_params['years']]
    preprocess_params['sample'] = True
    force_class(**base_params, **preprocess_params)




