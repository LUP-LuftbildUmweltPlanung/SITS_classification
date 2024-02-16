from classification.tempcnn.sits.readingsits import *
from classification.tempcnn.deeplearning.architecture_features import *
from classification.tempcnn.outputfiles.save import *


def predict_csv(model_path, feature, n_channels, test_file, result_file, proba, regression):

    # -- Get the number of classes
    #n_classes = getNoClasses(model_path)

    # -- Read min max values
    minMaxVal_file = '.'.join(model_path.split('.')[0:-1])
    minMaxVal_file = minMaxVal_file + '_minMax.txt'
    if os.path.exists(minMaxVal_file):
        min_per, max_per = read_minMaxVal(minMaxVal_file)
    else:
        assert False, "ERR: min-max values needs to be stored during training"

    X_test, polygon_ids_test, y_test = readSITSData(test_file, regression)
    X_test = addingfeat_reshape_data(X_test, feature, n_channels)
    X_test = normalizingData(X_test, min_per, max_per)
    # ---- Loading the model
    model = load_model(model_path)
    p_test = model.predict(x=X_test)
    if not proba and regression == False:
        p_test = p_test.argmax(axis=1)
    write_predictions_csv(result_file, p_test, y_test, regression)