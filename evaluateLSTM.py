import os
import numpy as np
import copy
import tensorflow as tf
import platform

from mySettings import get_lstm_settings, get_lstm_tuner_settings

# %% User inputs.
# Select case you want to train, see mySettings for case-specific settings.
case = 0
# Set hyperparameterTuning to True if the model was trained while tuning the
# hyperparameters
hyperparameterTuning = False

evaluationSet = 'val' # options are 'test' and 'val'
evaluateValidation = True
furtherAnalysis = False
plotResults = True

# %% Paths
if platform.system() == 'Linux':
    # To use docker.
    pathMain = '/augmenter-cs230'
else:
    pathMain = os.getcwd()    
pathData = os.path.join(pathMain, "Data")
pathData_all = os.path.join(pathData, "subset")

if hyperparameterTuning:
    pathParameterTuning = os.path.join(
        pathMain, "trained_models_hyperparameter_tuning_LSTM")
    pathParameterTuningModels = os.path.join(pathParameterTuning, 
                                             "Models", str(case))
    pathCModel = os.path.join(pathParameterTuningModels, "")
else:
    pathTrainedModels = os.path.join(pathMain, "trained_models_LSTM")
    pathCModel = os.path.join(pathTrainedModels, "")

# %% Settings.
print('Evaluating case {} for set {}'.format(case, evaluationSet))

if hyperparameterTuning:
    settings = get_lstm_tuner_settings(case)
else:
    settings = get_lstm_settings(case)
augmenter_type = settings["augmenter_type"]
poseDetector = settings["poseDetector"]
idxDatasets = settings["idxDatasets"]
scaleFactors = settings["scaleFactors"]
idxFold = settings["idxFold"] 
mean_subtraction = settings["mean_subtraction"]
std_normalization = settings["std_normalization"]

noise_bool = False # default
noise_magnitude = 0 # default
noise_type = 'no_noise' # default
if "noise_magnitude" in settings:
    noise_magnitude = settings["noise_magnitude"]
    if noise_magnitude > 0:
        noise_bool = True
    noise_type = 'per_timestep'
    if 'noise_type' in settings:
        noise_type = settings["noise_type"]
    
# %% Fixed settings (no need to change that).
featureHeight = True
featureWeight = True
marker_dimensions = ["x", "y", "z"]
nDim = len(marker_dimensions)

# Get indices in features/responses based on augmenter_type and poseDetector.
from utilities import getAllMarkers
feature_markers_all, response_markers_all = getAllMarkers()
if augmenter_type == 'fullBody':
    if poseDetector == 'OpenPose':
        from utilities import getOpenPoseMarkers_fullBody
        _, response_markers, idx_in_all_feature_markers, idx_in_all_response_markers = (
            getOpenPoseMarkers_fullBody())
    elif poseDetector == 'mmpose':
        from utilities import getMMposeMarkers_fullBody
        _, response_markers, idx_in_all_feature_markers, idx_in_all_response_markers = (
            getMMposeMarkers_fullBody())
    else:
        raise ValueError('poseDetector not recognized')        
elif augmenter_type == 'lowerExtremity':
    if poseDetector == 'OpenPose':
        from utilities import getOpenPoseMarkers_lowerExtremity
        _, response_markers, idx_in_all_feature_markers, idx_in_all_response_markers = (
            getOpenPoseMarkers_lowerExtremity())
    elif poseDetector == 'mmpose':
        from utilities import getMMposeMarkers_lowerExtremity
        _, response_markers, idx_in_all_feature_markers, idx_in_all_response_markers = (
            getMMposeMarkers_lowerExtremity())
    else:
        raise ValueError('poseDetector not recognized')
elif augmenter_type == 'upperExtremity_pelvis':
    from utilities import getMarkers_upperExtremity_pelvis
    _, response_markers, idx_in_all_feature_markers, idx_in_all_response_markers = (
        getMarkers_upperExtremity_pelvis())
elif augmenter_type == 'upperExtremity_noPelvis':
    from utilities import getMarkers_upperExtremity_noPelvis
    _, response_markers, idx_in_all_feature_markers, idx_in_all_response_markers = (
        getMarkers_upperExtremity_noPelvis())
else:
    raise ValueError('augmenter_type not recognized')

# Each marker has 3 dimensions.
idx_in_all_features = []
for idx in idx_in_all_feature_markers:
    idx_in_all_features.append(idx*3)
    idx_in_all_features.append(idx*3+1)
    idx_in_all_features.append(idx*3+2)
idx_in_all_responses = []
for idx in idx_in_all_response_markers:
    idx_in_all_responses.append(idx*3)
    idx_in_all_responses.append(idx*3+1)
    idx_in_all_responses.append(idx*3+2)
    
marker_names_responses = []
for marker in response_markers:
    for marker_dimension in marker_dimensions:
        marker_names_responses.append(marker + "_" + marker_dimension)      

# Additional features (height and weight).
nAddFeatures = 0
nFeature_markers = len(idx_in_all_feature_markers)*nDim
nResponse_markers = len(idx_in_all_response_markers)*nDim
if featureHeight:
    nAddFeatures += 1
    idxFeatureHeight = len(feature_markers_all)*nDim
    idx_in_all_features.append(idxFeatureHeight)
if featureWeight:
    nAddFeatures += 1 
    if featureHeight:
        idxFeatureWeight = len(feature_markers_all)*nDim + 1
    else:
        idxFeatureWeight = len(feature_markers_all)*nDim
    idx_in_all_features.append(idxFeatureWeight)
        
fc = 60 # sampling frequency (Hz)
desired_duration = 0.5
desired_nFrames = int(desired_duration * fc)

# %% Partition.
datasetName = ' '.join([str(elem) for elem in idxDatasets])
datasetName = datasetName.replace(' ', '_')

scaleFactorName = ' '.join([str(elem) for elem in scaleFactors])
scaleFactorName = scaleFactorName.replace(' ', '_')
scaleFactorName = scaleFactorName.replace('.', '')

partitionName = ('{}_{}_{}_{}'.format(augmenter_type, poseDetector, 
                                      datasetName, scaleFactorName))

# Load infoData dict.
infoData = np.load(os.path.join(pathData_all, 'infoData.npy'),
                   allow_pickle=True).item()
# Load partition.
partition = np.load(
    os.path.join(pathData_all, 'partition_{}.npy'.format(partitionName)),
    allow_pickle=True).item()
        
# %% Data processing: add noise and compute mean and std.
features_mean = np.load(
    pathCModel + str(case) + "_trainFeatures_mean.npy", allow_pickle=True)
features_std = np.load(
    pathCModel + str(case) + "_trainFeatures_std.npy", allow_pickle=True)

# %% Import trained model and weights for prediction.
json_file = open(pathCModel + str(case) + "_model.json", 'r')
pretrainedModel_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(pretrainedModel_json)
model.load_weights(pathCModel + str(case) + "_weights.h5")

# %% Analysis
from utilities import getMetrics
from utilities import getMetrics_unnorm_lstm
from utilities import getMetrics_ind_unnorm_lstm
from utilities import getMPME_unnorm_lstm
  
if evaluateValidation:
    
    valFeatures_all = np.zeros((partition[evaluationSet].shape[0], 
                               desired_nFrames, nFeature_markers+nAddFeatures))
    valFeatures_noise_all = np.zeros((partition[evaluationSet].shape[0], 
                               desired_nFrames, nFeature_markers+nAddFeatures))
    valResponses_all = np.zeros((partition[evaluationSet].shape[0], 
                               desired_nFrames, nResponse_markers))
    heights_all = np.zeros((partition[evaluationSet].shape[0], 
                               desired_nFrames, 1))    
    rmse_ind_max_all = np.zeros((partition[evaluationSet].shape[0],))
    rmse_ind_max_marker_all = []
    rmse_ind_max_d = {}
    rmse_noise_ind_max_all = np.zeros((partition[evaluationSet].shape[0],))
    rmse_noise_ind_max_marker_all = []
    rmse_noise_ind_max_d = {}
    for count, idxVal in enumerate(partition[evaluationSet]):
        
        c_feature_all = np.load(os.path.join(
            pathData_all, "feature_{}.npy".format(idxVal)))
        c_feature_sel = c_feature_all[:,idx_in_all_features]
        
        c_response_all = np.load(os.path.join(
            pathData_all, "responses_{}.npy".format(idxVal)))
        c_response_sel = c_response_all[:,idx_in_all_responses]
        
        c_height = copy.deepcopy(c_feature_all[:, idxFeatureHeight])
        
        c_feature_noise = copy.deepcopy(c_feature_sel)
        if noise_magnitude > 0:            
            if noise_type == "per_timestep": 
                np.random.seed(idxVal)                
                
                noise = np.zeros((desired_nFrames,
                                  nFeature_markers+nAddFeatures))
                noise[:,:nFeature_markers] = np.random.normal(
                    0, noise_magnitude, (desired_nFrames, nFeature_markers))
                    
                c_feature_noise += noise
                
        # Process data
        # Mean subtraction
        if mean_subtraction:                
            c_feature_sel -= features_mean
            c_feature_noise -= features_mean
                
        # Std division
        if std_normalization:
            c_feature_sel /= features_std
            c_feature_noise /= features_std
        
        valFeatures_all[count, :, :] = c_feature_sel
        valFeatures_noise_all[count, :, :] = c_feature_noise
        valResponses_all[count, :, :] = c_response_sel
        heights_all[count, :, :] = np.reshape(c_height, (c_height.shape[0], 1))
        
        if furtherAnalysis:
            # Per trial RMSE
            _, rmse = getMetrics_unnorm_lstm(
                np.reshape(c_feature_sel, (1, desired_nFrames, c_feature_sel.shape[1])),
                np.reshape(c_response_sel, (1, desired_nFrames, c_response_sel.shape[1])), 
                model,                                                     
                np.reshape(c_height, (1, desired_nFrames, 1)))
            
            _, rmse_noise = getMetrics_unnorm_lstm(
                np.reshape(c_feature_noise, (1, desired_nFrames, c_feature_noise.shape[1])),
                np.reshape(c_response_sel, (1, desired_nFrames, c_response_sel.shape[1])), 
                model,                                                     
                np.reshape(c_height, (1, desired_nFrames, 1))) 
            
            # Per marker RMSE
            _, rmse_ind = getMetrics_ind_unnorm_lstm(
                np.reshape(c_feature_sel, (1, desired_nFrames, c_feature_sel.shape[1])),
                np.reshape(c_response_sel, (1, desired_nFrames, c_response_sel.shape[1])), 
                model,                                                     
                np.reshape(c_height, (1, desired_nFrames, 1)))
            
            _, rmse_noise_ind = getMetrics_ind_unnorm_lstm(
                np.reshape(c_feature_noise, (1, desired_nFrames, c_feature_noise.shape[1])),
                np.reshape(c_response_sel, (1, desired_nFrames, c_response_sel.shape[1])), 
                model,                                                     
                np.reshape(c_height, (1, desired_nFrames, 1)))
            
            rmse_ind_max = np.max(rmse_ind)
            rmse_ind_max_marker = marker_names_responses[np.argmax(rmse_ind)]
            
            rmse_noise_ind_max = np.max(rmse_noise_ind)
            rmse_noise_ind_max_marker = marker_names_responses[np.argmax(rmse_noise_ind)]
            
            c_idx = int(np.argwhere(infoData['indexes'] == idxVal))
            
            rmse_ind_max_all[count,] = rmse_ind_max
            rmse_ind_max_marker_all.append(rmse_ind_max_marker)
            
            rmse_noise_ind_max_all[count,] = rmse_noise_ind_max
            rmse_noise_ind_max_marker_all.append(rmse_noise_ind_max_marker) 
            
            rmse_ind_max_d[count] = {"dataset": infoData['datasets'][c_idx],
                                     "subject": infoData['subjects'][c_idx],
                                     "scale_factor": infoData['scale_factors'][c_idx],
                                     "trial": infoData['trials'][c_idx],
                                     "subtrial": infoData['subtrials'][c_idx],
                                     "rmse_ind_max": rmse_ind_max,
                                     "rmse_ind_max_marker": rmse_ind_max_marker}
            
            rmse_noise_ind_max_d[count] = {"dataset": infoData['datasets'][c_idx],
                                           "subject": infoData['subjects'][c_idx],
                                           "scale_factor": infoData['scale_factors'][c_idx],
                                           "trial": infoData['trials'][c_idx],
                                           "subtrial": infoData['subtrials'][c_idx],
                                           "rmse_ind_max": rmse_noise_ind_max,
                                           "rmse_ind_max_marker": rmse_noise_ind_max_marker}
        
    # Get various metrics for validation set.    
    _, val_rmse = getMetrics(valFeatures_all, valResponses_all, model)
    # To get the real rmse, ie in m, we need to un-normalize by height.
    _, val_rmse_unnorm = (
        getMetrics_unnorm_lstm(valFeatures_all, valResponses_all, model,
                               heights_all))
    
    # Mean per-marker error (mean Euclidian distance)
    MPME, MPMEvec = getMPME_unnorm_lstm(valFeatures_all, valResponses_all,
                                        model, heights_all)    
    
    if furtherAnalysis:
        rmse_ind_max_max = np.max(rmse_ind_max_all)
        rmse_ind_max_max_marker = rmse_ind_max_marker_all[np.argmax(rmse_ind_max_all)]
    
    print("RMSE (adjusted for height): {} (mm).".format(np.round(val_rmse_unnorm,4)*1000))
    print("MPME (adjusted for height): {} (mm).".format(np.round(MPME,4)*1000))
    if furtherAnalysis:
        print("max RMSE per marker - no noise (adjusted for height): {} (mm) for {}  in dataset{}-subject{}-trial{}".format(
            np.round(rmse_ind_max_max,4)*1000, rmse_ind_max_max_marker, rmse_ind_max_d[int(np.argmax(rmse_ind_max_all))]["dataset"], 
            rmse_ind_max_d[int(np.argmax(rmse_ind_max_all))]["subject"], rmse_ind_max_d[int(np.argmax(rmse_ind_max_all))]["trial"]))
        
if plotResults:
    from utilities import plotLossOverEpochs
    import pickle
    history = pickle.load(open(pathCModel + str(case) + "_history", "rb"))
    plotLossOverEpochs(history)
