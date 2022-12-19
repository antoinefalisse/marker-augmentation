import os
import platform
import numpy as np

# %% Paths.
if platform.system() == 'Linux':
    # To use docker.
    pathMain = '/augmenter-cs230'
else:
    pathMain = os.getcwd()
pathData = os.path.join(pathMain, "Data")
pathData_all = os.path.join(pathData, "subset")

# %%
augmenter_type = 'lowerExtremity'
poseDetector = 'OpenPose'
featureHeight = True
featureWeight = True

# %% Helper indices
from utilities import getAllMarkers
feature_markers_all, response_markers_all = getAllMarkers()
if augmenter_type == 'lowerExtremity':
    if poseDetector == 'OpenPose':
        from utilities import getOpenPoseMarkers_lowerExtremity
        _, _, idx_in_all_feature_markers, idx_in_all_response_markers = (
            getOpenPoseMarkers_lowerExtremity())
        
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
    
# Additional features (height and weight). 
marker_dimensions = ["x", "y", "z"]
nDim = len(marker_dimensions)  
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

# %% Load example feature
pathFeature0 = os.path.join(pathData_all, 'feature_0.npy')
# feature_0 dimension is 30*47
# 30 is time dimension (0.5s at 60Hz)
# 47 is the feature dimension
#   - 3D coordinates of 15 markers => 45
#   - Height and weight => 2
feature_0 = np.load(pathFeature0)[:, idx_in_all_features]

# %% Load example response/label
pathResponse0 = os.path.join(pathData_all, 'responses_0.npy')
# feature_0 dimension is 30*105
# 30 is time dimension (0.5s at 60Hz)
# 105 is the label dimension
#   - 3D coordinates of 35 markers => 105
reponse_0 = np.load(pathResponse0)[:, idx_in_all_responses]

# %% Info Data
# Indices of the feature and response npy files for the different datasets,
# scale factors, subjects, etc.
pathInfoData = os.path.join(pathData_all, 'infoData.npy')
infoData = np.load(pathInfoData, allow_pickle=True).item()

# %% Partition
# Indices of the feature and response npy files for training, validation, test.
pathPartition = os.path.join(pathData_all,
                             'partition_lowerExtremity_OpenPose_0_09_095_10_105_11.npy')
partition = np.load(pathPartition, allow_pickle=True).item()