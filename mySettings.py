# Dense.
def get_dense_settings(a):
    
    settings = {"0":
                {"idxDatasets": [idx for idx in range(9)],
                 "nFirstUnits": 1000,
                 "nHUnits": 0,
                 "nHLayers": 0,
                 "nEpochs": 20,
                 "batchSize": 64,
                 "idxFold": 0,
                 "dropout_p": 1.39e-6,
                 "L2_p": 1.56e-6,
                 "learning_r": 6.76e-5,
                 "mean_subtraction": True,
                 "std_normalization": True,
                 "noise_magnitude": 0.018,
                 "noise_type": "per_subject_per_marker"}}
        
    return settings[str(a)]

# LSTM.
def get_lstm_settings(a):

    settings = {
        "reference":
            {'augmenter_type': 'lowerExtremity',
             "poseDetector": 'OpenPose',
             "mean_subtraction": True,
             "std_normalization": True,},
        "0":                
            {'augmenter_type': 'lowerExtremity',
             "poseDetector": 'OpenPose',
             "idxDatasets": [idx for idx in range(0,1)],
             "scaleFactors": [0.9, 0.95, 1., 1.05, 1.1],
             "nHUnits": 96,
             "nHLayers": 2,
             "nEpochs": 50,
             "batchSize": 64,
             "idxFold": 0,
             'learning_r': 5e-05,
             "mean_subtraction": True,
             "std_normalization": True,
             "noise_magnitude": 0.018,
             "noise_type": "per_timestep",
             'nRotations': 1,
             'bidirectional': False},
        "1":                
            {'augmenter_type': 'lowerExtremity',
             "poseDetector": 'OpenPose',
             "idxDatasets": [idx for idx in range(0,10)], # Includes 10 datasets
             "scaleFactors": [0.9, 0.95, 1., 1.05, 1.1], # Each dataset is scaled 5 times with these scale factors
             "nHUnits": 96,
             "nHLayers": 2,
             "nEpochs": 50,
             "batchSize": 64,
             "idxFold": 0,
             'learning_r': 5e-05,
             "mean_subtraction": True,
             "std_normalization": True,
             "noise_magnitude": 0.018,
             "noise_type": "per_timestep",
             'nRotations': 8, # Data rotated about vertical axis to represent motions in all directions, each time sequence is rotated 8 (nRotations) times.
             'bidirectional': False},
            
        "2":                
            {'augmenter_type': 'lowerExtremity',
             "poseDetector": 'OpenPose',
             "idxDatasets": [idx for idx in range(0,1)],
             "scaleFactors": [0.9, 0.95, 1., 1.05, 1.1],
             "nHUnits": 96,
             "nHLayers": 2,
             "nEpochs": 50,
             "batchSize": 64,
             "idxFold": 0,
             'learning_r': 5e-05,
             "mean_subtraction": True,
             "std_normalization": True,
             "noise_magnitude": 0.018,
             "noise_type": "per_timestep",
             'nRotations': 10,
             'bidirectional': False,
             'rotation_type': 'sphereRotation'},

         "3":                
            {'augmenter_type': 'lowerExtremity',
             "poseDetector": 'OpenPose',
             "idxDatasets": [idx for idx in range(0,1)],
             "scaleFactors": [0.9, 0.95, 1., 1.05, 1.1],
             "nHUnits": 96,
             "nHLayers": 2,
             "nEpochs": 50,
             "batchSize": 64,
             "idxFold": 0,
             'learning_r': 5e-05,
             "mean_subtraction": True,
             "std_normalization": True,
             "noise_magnitude": 0.018,
             "noise_type": "per_timestep",
             'nRotations': 1,
             'bidirectional': False,
             'rotation_type': 'circleRotation'}}
        
    return settings[str(a)]

# LSTM - Hyperparameters tuning
def get_lstm_tuner_settings(a):

    settings = {
        "0":
         {'augmenter_type': 'lowerExtremity',
          "poseDetector": 'OpenPose',
          "idxDatasets": [idx for idx in range(0,10)],
          "scaleFactors": [0.9, 0.95, 1., 1.05, 1.1],
          "nEpochs": 10,
          "batchSize": 64,
          "idxFold": 0,
          "mean_subtraction": True,
          "std_normalization": True,
          "max_trials": 10,
          "executions_per_trial": 1,
          "nEpochsBest": 15,
          "learning_r": {
              "name": 'learning_r', "min": 1e-5, "max": 1e-4, 
              "sampling": 'LOG', "default": 5e-5},
          "units_h": {
              "name": 'units_h', "min": 64, "max": 160, 
              "step": 16, "default": 96},
          "layer_h": {
              "name": 'layerh', "min": 1, "max": 4, 
              "step": 1, "default": 2},
          "noise_magnitude": 0.018,
          "noise_type": "per_timestep",
          'nRotations': 8,
          'bidirectional': False}}
        
    return settings[str(a)]
