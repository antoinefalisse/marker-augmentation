import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import dataman

# %% Metrics.
def getMetrics(features, responses, model):
    y_pred2 = model.predict(features)
    mse = np.mean(np.square(y_pred2-responses))
    rmse = np.sqrt(mse)
    return mse,rmse

def getMetrics_ind(features, responses, model):
    y_pred2 = model.predict(features)
    mse = np.mean(np.square(y_pred2-responses),axis=0)
    rmse = np.sqrt(mse)
    return mse,rmse

def getMetrics_unnorm(features, responses, model, heights):
    heights_sh = np.tile(heights, (responses.shape[1], 1)).T
    responses_unnorm = responses * heights_sh   
    y_pred2_unnorm = model.predict(features) * heights_sh
    
    mse_unnorm = np.mean(np.square(y_pred2_unnorm-responses_unnorm))
    rmse_unnorm = np.sqrt(mse_unnorm)
    return mse_unnorm,rmse_unnorm

def getMetrics_unnorm_lstm(features, responses, model, heights):
    responses_unnorm = responses * heights   
    y_pred2_unnorm = model.predict(features) * heights
    
    mse_unnorm = np.mean(np.square(y_pred2_unnorm-responses_unnorm))
    rmse_unnorm = np.sqrt(mse_unnorm)
    return mse_unnorm,rmse_unnorm

def getMPME_unnorm_lstm(features, responses, model, heights):
    responses_unnorm = responses * heights   
    y_pred2_unnorm = model.predict(features) * heights
    
    # We know there are three dimensions (x,y,z).
    MPMEvec = np.zeros((int(responses_unnorm.shape[2]/3),))
    for i in range(int(responses_unnorm.shape[2]/3)):
        MPMEvec[i] = np.mean(np.linalg.norm(
            y_pred2_unnorm[:,:,i*3:i*3+3] - 
            responses_unnorm[:,:,i*3:i*3+3],axis = 2))
    MPME = np.mean(MPMEvec)
    
    return MPME, MPMEvec

def getMetrics_ind_unnorm_lstm(features, responses, model, heights):
    responses_unnorm = responses * heights       
    responses_unnorm_2D = np.reshape(responses_unnorm, 
                                      (responses_unnorm.shape[1], 
                                      responses_unnorm.shape[2]))    
    y_pred2_unnorm = model.predict(features) * heights
    y_pred2_unnorm_2D = np.reshape(y_pred2_unnorm, 
                                    (y_pred2_unnorm.shape[1], 
                                    y_pred2_unnorm.shape[2])) 
    
    mse = np.mean(np.square(y_pred2_unnorm_2D-responses_unnorm_2D),axis=0)
    rmse = np.sqrt(mse)
    return mse,rmse
    
def plotLossOverEpochs(history):
    plt.figure()
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.ylabel('loss (mean squared error)')
    plt.xlabel('Epoch Number');
    plt.legend(('Training','Evaluation'))
    plt.show()
    
# Partition dataset.
def getPartition(idxDatasets, scaleFactors, infoData, subjectSplit, idxFold):
    
    idxSubject = {'train': {}, 'val': {}, 'test': {}}
    partition  = {'train': np.array([], dtype=int), 
                  'val': np.array([], dtype=int), 
                  'test': np.array([], dtype=int)}
    count_s_train = 0
    count_s_val = 0
    count_s_test = 0
    acc_val = 0
    for idxDataset in idxDatasets:
        c_dataset = "dataset" + str(idxDataset)
        
        for scaleFactor in scaleFactors:
            
            if idxDataset == 9 and scaleFactor == 0.9:
                scaleFactor = 1.15
        
            # Train
            count_s = 0
            for c_idx in subjectSplit[c_dataset]["training_" + str(idxFold)]:
                c_where = np.argwhere(np.logical_and(
                    infoData["scale_factors"]==scaleFactor,
                    np.logical_and(infoData["datasets"]==idxDataset, 
                                    infoData["subjects"]==c_idx)))
                partition["train"] = np.append(partition["train"], c_where)
                idxSubject["train"][count_s_train + count_s] = c_where  
                count_s += 1
            count_s_train += count_s
            
            # Val
            count_s = 0
            for c_idx in subjectSplit[c_dataset]["validation_" + str(idxFold)]:
                c_where = np.argwhere(np.logical_and(
                    infoData["scale_factors"]==scaleFactor,
                    np.logical_and(infoData["datasets"]==idxDataset, 
                                    infoData["subjects"]==c_idx)))
                partition["val"] = np.append(partition["val"], c_where)
                idxSubject["val"][count_s_val + count_s] = c_where
                count_s += 1
            count_s_val += count_s
            
            # Test
            count_s = 0
            for c_idx in subjectSplit[c_dataset]["test"]:
                c_where = np.argwhere(np.logical_and(
                    infoData["scale_factors"]==scaleFactor,
                    np.logical_and(infoData["datasets"]==idxDataset, 
                                    infoData["subjects"]==c_idx)))
                partition["test"] = np.append(partition["test"], c_where)
                idxSubject["test"][count_s_test + count_s] = c_where
                count_s += 1
            count_s_test += count_s
        
            acc_val += np.sum(np.logical_and(
                infoData["datasets"] == idxDataset,
                infoData["scale_factors"]==scaleFactor))
        
    # Make sure the all the data has been split in the three sets.
    assert (acc_val == partition["train"].shape[0] + 
            partition["val"].shape[0] + partition["test"].shape[0]), (
                "missing data")
    return partition

# %% Welford's online algorithm: 
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)
  
# %% Markers
def getAllMarkers():
    
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist",
        "RSmallToe_mmpose", "LSmallToe_mmpose"]
    
    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]
    
    return feature_markers, response_markers

def getOpenPoseMarkers_fullBody():
    
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist"]
    
    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]
    
    all_feature_markers, all_response_markers = getAllMarkers()
    
    idx_in_all_feature_markers = []
    for marker in feature_markers:
        idx_in_all_feature_markers.append(
            all_feature_markers.index(marker))
        
    idx_in_all_response_markers = []
    for marker in response_markers:
        idx_in_all_response_markers.append(
            all_response_markers.index(marker))
        
    return (feature_markers, response_markers, idx_in_all_feature_markers, 
            idx_in_all_response_markers)

def getMMposeMarkers_fullBody():
    
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe_mmpose", 
        "LSmallToe_mmpose", "RElbow", "LElbow", "RWrist", "LWrist"]
    
    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]
    
    all_feature_markers, all_response_markers = getAllMarkers()
    
    idx_in_all_feature_markers = []
    for marker in feature_markers:
        idx_in_all_feature_markers.append(
            all_feature_markers.index(marker))
        
    idx_in_all_response_markers = []
    for marker in response_markers:
        idx_in_all_response_markers.append(
            all_response_markers.index(marker))
        
    return (feature_markers, response_markers, idx_in_all_feature_markers, 
            idx_in_all_response_markers)        
    
def getOpenPoseMarkers_lowerExtremity():
    
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]
    
    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]
    
    all_feature_markers, all_response_markers = getAllMarkers()
    
    idx_in_all_feature_markers = []
    for marker in feature_markers:
        idx_in_all_feature_markers.append(
            all_feature_markers.index(marker))
        
    idx_in_all_response_markers = []
    for marker in response_markers:
        idx_in_all_response_markers.append(
            all_response_markers.index(marker))
        
    return (feature_markers, response_markers, idx_in_all_feature_markers, 
            idx_in_all_response_markers)

def getMMposeMarkers_lowerExtremity():
    
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe_mmpose", 
        "LSmallToe_mmpose"]
    
    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]
    
    all_feature_markers, all_response_markers = getAllMarkers()
    
    idx_in_all_feature_markers = []
    for marker in feature_markers:
        idx_in_all_feature_markers.append(
            all_feature_markers.index(marker))
        
    idx_in_all_response_markers = []
    for marker in response_markers:
        idx_in_all_response_markers.append(
            all_response_markers.index(marker))
        
    return (feature_markers, response_markers, idx_in_all_feature_markers, 
            idx_in_all_response_markers)
    
def getMarkers_upperExtremity_pelvis():
    
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RElbow", "LElbow",
        "RWrist", "LWrist"]
    
    response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study"]
    
    all_feature_markers, all_response_markers = getAllMarkers()
    
    idx_in_all_feature_markers = []
    for marker in feature_markers:
        idx_in_all_feature_markers.append(
            all_feature_markers.index(marker))
        
    idx_in_all_response_markers = []
    for marker in response_markers:
        idx_in_all_response_markers.append(
            all_response_markers.index(marker))
        
    return (feature_markers, response_markers, idx_in_all_feature_markers,
            idx_in_all_response_markers)

def getMarkers_upperExtremity_noPelvis():
    
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", 
        "RWrist", "LWrist"]
    
    response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study"]
    
    all_feature_markers, all_response_markers = getAllMarkers()
    
    idx_in_all_feature_markers = []
    for marker in feature_markers:
        idx_in_all_feature_markers.append(
            all_feature_markers.index(marker))
        
    idx_in_all_response_markers = []
    for marker in response_markers:
        idx_in_all_response_markers.append(
            all_response_markers.index(marker))
        
    return (feature_markers, response_markers, idx_in_all_feature_markers, 
            idx_in_all_response_markers)

# %% Rotate data.
def rotateArray(data, axis, value, inDegrees=True):
    
    assert np.mod(data.shape[1],3) == 0, 'wrong dimension rotateArray'
    r = R.from_euler(axis, value, degrees=inDegrees)     
    
    data_out = np.zeros((data.shape[0], data.shape[1]))
    for i in range(int(data.shape[1]/3)):
        c = data[:,i*3:(i+1)*3]        
        data_out[:,i*3:(i+1)*3] = r.apply(c)
        
    return data_out

# %% Rotate data (sphere).
# Based on first approach from:
# http://paulbourke.net/geometry/randomvector/
def rotateArraySphere1(data, theta_x, theta_y, theta_z, inDegrees=False):    
    assert np.mod(data.shape[1],3) == 0, 'wrong dimension rotateArray'
    
    r1 = R.from_euler('x', theta_x, degrees=False)
    r2 = R.from_euler('y', theta_z, degrees=False)
    r3 = R.from_euler('z', theta_z, degrees=False)
    
    data_out = np.zeros((data.shape[0], data.shape[1]))
    for i in range(int(data.shape[1]/3)):
        c = data[:,i*3:(i+1)*3]        
        unit_vec_r1 = r1.apply(c)
        unit_vec_r2 = r2.apply(unit_vec_r1)
        data_out[:,i*3:(i+1)*3] = r3.apply(unit_vec_r2)
        
    return data_out

# %% Rotate data (sphere).
# Based on second approach from:
# http://paulbourke.net/geometry/randomvector/
def rotateArraySphere2(data, theta_x, theta_z, inDegrees=False):    
    assert np.mod(data.shape[1],3) == 0, 'wrong dimension rotateArray'
    
    r1 = R.from_euler('x', theta_x, degrees=inDegrees)
    r2 = R.from_euler('z', theta_z, degrees=inDegrees)
    
    data_out = np.zeros((data.shape[0], data.shape[1]))
    for i in range(int(data.shape[1]/3)):
        c = data[:,i*3:(i+1)*3]
        unit_vec_r1 = r1.apply(c)
        data_out[:,i*3:(i+1)*3] = r2.apply(unit_vec_r1)
        
    return data_out

# %% Rotate data (sphere).
# Based on third approach from:
# http://paulbourke.net/geometry/randomvector/
def get_rotation_matrix(vec2, vec1=np.array([1, 0, 0])):
    """get rotation matrix between two vectors using scipy"""
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = R.align_vectors(vec2, vec1)
    return r[0]

def rotateArraySphere3(data, ref_vec, theta_x, theta_z, unit_vec=np.array([0,0,0]),
                       inDegrees=False):    
    assert np.mod(data.shape[1],3) == 0, 'wrong dimension rotateArray'
    
    r1 = R.from_euler('x', theta_x, degrees=inDegrees)
    r2 = R.from_euler('z', theta_z, degrees=inDegrees)
    
    if not np.any(unit_vec):
        unit_vec = data[0, 0:3]
    r_align = get_rotation_matrix(vec1=unit_vec, vec2=ref_vec)
    
    data_out = np.zeros((data.shape[0], data.shape[1]))
    for i in range(int(data.shape[1]/3)):
        for j in range(data.shape[0]):
            c = data[j,i*3:(i+1)*3]        
            unit_vec_align = r_align.apply(c)            
            unit_vec_r1 = r1.apply(unit_vec_align)
            data_out[j,i*3:(i+1)*3] = r2.apply(unit_vec_r1)
        
    return data_out, unit_vec

# %% TRC format to numpy format.
def TRC2numpy(pathFile, markers):
    
    trc_file = dataman.TRCFile(pathFile)
    time = trc_file.time
    num_frames = time.shape[0]
    data = np.zeros((num_frames, len(markers)*3))
    for count, marker in enumerate(markers):
        data[:,3*count:3*count+3] = trc_file.marker(marker)    
    this_dat = np.empty((num_frames, 1))
    this_dat[:, 0] = time
    data_out = np.concatenate((this_dat, data), axis=1)
    
    return data_out