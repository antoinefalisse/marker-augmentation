import os
import numpy as np
import dataman
import copy
import tensorflow as tf
import platform

from mySettings import get_lstm_settings, get_lstm_tuner_settings

# %% User settings.
# Select case you want to train, see mySettings for case-specific settings.
case = 'reference'
# Set hyperparameterTuning to True if the model was trained while tuning the
# hyperparameters
hyperparameterTuning = False

# %% Paths.
if platform.system() == 'Linux':
    # To use docker.
    pathMain = '/marker-augmentation'
else:
    pathMain = os.getcwd()
    
if hyperparameterTuning:
    pathParameterTuning = os.path.join(
        pathMain, "trained_models_hyperparameter_tuning_LSTM")
    pathParameterTuningModels = os.path.join(pathParameterTuning, 
                                             "Models", str(case))
    pathCModel = os.path.join(pathParameterTuningModels, "")
else:
    pathTrainedModels = os.path.join(pathMain, "trained_models_LSTM")
    pathCModel = os.path.join(pathTrainedModels, "")

# %% Failure example
failure_case_name = 'plantarflexion'
pathFailreDir = os.path.join(pathMain, 'test')
pathFile = os.path.join(pathFailreDir, failure_case_name + ".trc")
subject_height = 1.96
subject_weight = 78.2

# %% Fixed settings (no need to change that).
featureHeight = True
featureWeight = True

# %% Helper indices (no need to change that).
# Get indices features/responses based on augmenter_type and poseDetector.
if hyperparameterTuning:
    settings = get_lstm_tuner_settings(case)
else:
    settings = get_lstm_settings(case)
augmenter_type = settings["augmenter_type"]
poseDetector = settings["poseDetector"]
from utilities import getAllMarkers
feature_markers_all, response_markers_all = getAllMarkers()
if augmenter_type == 'fullBody':
    if poseDetector == 'OpenPose':
        from utilities import getOpenPoseMarkers_fullBody
        feature_markers, response_markers, _, _ = (
            getOpenPoseMarkers_fullBody())
    elif poseDetector == 'mmpose':
        from utilities import getMMposeMarkers_fullBody
        feature_markers, response_markers, _, _ = (
            getMMposeMarkers_fullBody())
    else:
        raise ValueError('poseDetector not recognized')        
elif augmenter_type == 'lowerExtremity':
    if poseDetector == 'OpenPose':
        from utilities import getOpenPoseMarkers_lowerExtremity
        feature_markers, response_markers, _, _ = (
            getOpenPoseMarkers_lowerExtremity())
    elif poseDetector == 'mmpose':
        from utilities import getMMposeMarkers_lowerExtremity
        feature_markers, response_markers, _, _ = (
            getMMposeMarkers_lowerExtremity())
    else:
        raise ValueError('poseDetector not recognized')
elif augmenter_type == 'upperExtremity_pelvis':
    from utilities import getMarkers_upperExtremity_pelvis
    feature_markers, response_markers, _, _ = (
        getMarkers_upperExtremity_pelvis())
elif augmenter_type == 'upperExtremity_noPelvis':
    from utilities import getMarkers_upperExtremity_noPelvis
    feature_markers, response_markers, _, _ = (
        getMarkers_upperExtremity_noPelvis())
else:
    raise ValueError('augmenter_type not recognized')

# %% Prepare inputs.
# Step 1: import .trc file with OpenPose marker trajectories.
from utilities import TRC2numpy
trc_data = TRC2numpy(pathFile, feature_markers)
trc_data_time = trc_data[:,0]
trc_data_data = trc_data[:,1:]

# Step 2: Normalize with reference marker position.
referenceMarker = "midHip"
trc_file = dataman.TRCFile(pathFile)
time = trc_file.time
referenceMarker_data = trc_file.marker(referenceMarker)  
norm_trc_data_data = np.zeros((trc_data_data.shape[0],trc_data_data.shape[1]))
for i in range(0,trc_data_data.shape[1],3):
    norm_trc_data_data[:,i:i+3] = trc_data_data[:,i:i+3] - referenceMarker_data
    
# Step 3: Normalize with subject's height.
norm2_trc_data_data = copy.deepcopy(norm_trc_data_data)
norm2_trc_data_data = norm2_trc_data_data / subject_height

# Step 4: Add remaining features.
inputs = copy.deepcopy(norm2_trc_data_data)
if featureHeight:    
    inputs = np.concatenate((inputs, 
                             subject_height*np.ones((inputs.shape[0],1))), 
                            axis=1)
if featureWeight:    
    inputs = np.concatenate((inputs, 
                             subject_weight*np.ones((inputs.shape[0],1))), 
                            axis=1)
    
# %% Predict outputs
mean_subtraction = settings["mean_subtraction"] 
std_normalization = settings["std_normalization"] 
# Import previously trained model and weights for prediction.
json_file = open(pathCModel + case + "_model.json", 'r')
pretrainedModel_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(pretrainedModel_json)    
# Load weights into new model.
model.load_weights(pathCModel + case + "_weights.h5")  
# Process data
if mean_subtraction:
    trainFeatures_mean = np.load(
        os.path.join(pathCModel, case + "_trainFeatures_mean.npy"),
        allow_pickle=True)
    inputs -= trainFeatures_mean
if std_normalization:
    trainFeatures_std = np.load(
        os.path.join(pathCModel, case + "_trainFeatures_std.npy"), 
        allow_pickle=True)
    inputs /= trainFeatures_std
outputs = model.predict(np.reshape(inputs, 
                                   (1, inputs.shape[0], inputs.shape[1])))
outputs = np.reshape(outputs, (outputs.shape[1], outputs.shape[2]))

# %% Post-process outputs    
# Step 1: Un-normalize with subject's height.
unnorm_outputs = outputs * subject_height

# Step 2: Un-normalize with reference marker position.
unnorm2_outputs = np.zeros((unnorm_outputs.shape[0],unnorm_outputs.shape[1]))
for i in range(0,unnorm_outputs.shape[1],3):
    unnorm2_outputs[:,i:i+3] = unnorm_outputs[:,i:i+3] + referenceMarker_data
    
# %% Return augmented .trc file
for c, marker in enumerate(response_markers):
    x = unnorm2_outputs[:,c*3]
    y = unnorm2_outputs[:,c*3+1]
    z = unnorm2_outputs[:,c*3+2]
    trc_file.add_marker(marker, x, y, z)
pathFileOut = os.path.join(
    pathFailreDir, "augmented_{}_{}.trc".format(failure_case_name, case))
trc_file.write(pathFileOut)
 
