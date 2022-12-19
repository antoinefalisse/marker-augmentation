"""
    This script splits the training data into training, evaluation, and test
    sets, "while making sure that all of a participant’s data resides in only
    one set". Distribution is about 80%, 10%, and 10%.
"""

import os
import numpy as np
from infoDatasets import getInfodataset

saveData = True
# TODO: not really splitting for k-fold cross-validation, it is just splitting
# the training and validation sets nFold different times.
nFold = 1

pathMain = os.getcwd()
pathData = os.path.join(pathMain, "Data")
os.makedirs(pathData, exist_ok=True)

idxDatasets = [0]

# subjectSplit_t is only used to give a proportion reference. The actual
# splitting for the validation and training sets is made in the next block.
# This code isn't great but has evolved when including for cross-validation.
subjectSplit = {}
subjectSplit_t = {}
for idxDataset in idxDatasets:
    c_dataset = "dataset" + str(idxDataset)
    subjectSplit[c_dataset] = {}
    subjectSplit_t[c_dataset] = {}
    infoDataset = getInfodataset(idxDataset)
    nSubjects = infoDataset["nSubjects"]-len(infoDataset["idxSubjectsToExclude"])
    if idxDataset == 5:
        np.random.seed(5)
    else:
        np.random.seed(0)
    x = np.random.uniform(0,1,infoDataset["nSubjects"]-len(infoDataset["idxSubjectsToExclude"]))
    # This handles cases where data from certain subjects are not included.
    # We do want to include the 'index' of those subjects.
    idxX = np.linspace(0, infoDataset["nSubjects"]-1, infoDataset["nSubjects"])
    for i in infoDataset["idxSubjectsToExclude"]:
        if i in idxX:
            idxX = idxX[idxX != i]             
    idxTraining = np.argwhere(x<=0.8).flatten()
    idxEvaluate = np.argwhere(np.logical_and(x>0.8,x<= 0.9)).flatten()
    idxTest = np.argwhere(x>0.9).flatten()    
    subjectSplit_t[c_dataset]["training_t"] = np.array([int(idxX[i]) for i in idxTraining])
    subjectSplit_t[c_dataset]["validation_t"] = np.array([int(idxX[i]) for i in idxEvaluate])
    subjectSplit[c_dataset]["test"] = np.array([int(idxX[i]) for i in idxTest])
    # make sure the sets are not empty
    assert (not subjectSplit_t[c_dataset]["training_t"].size == 0), 'empty training set'
    assert (not subjectSplit_t[c_dataset]["validation_t"].size == 0), 'empty validation set'
    assert (not subjectSplit[c_dataset]["test"].size == 0), 'empty validation set'    

# Not convinced this piece of code is great but it should be okay for now.
# This should return k validation and training sets with, when possible,
# all different validation sets, ie one subject will only be in one validation
# set (even when a validation set contains multiple subjects).
for idxDataset in idxDatasets:
    c_dataset = "dataset" + str(idxDataset)
    nVal = subjectSplit_t[c_dataset]["validation_t"].shape[0]
    train_val = np.concatenate((subjectSplit_t[c_dataset]["training_t"], subjectSplit_t[c_dataset]["validation_t"]), axis=0) 
    
    np.random.seed(1)
    if nVal*(nFold) <= train_val.shape[0]:
        test = np.random.choice(train_val, nVal*(nFold), replace=False)
    else:
        test = np.random.choice(train_val, nVal*(nFold), replace=True)
        
    for nf in range(nFold):
        subjectSplit[c_dataset]["validation_" + str(nf)] = test[nf*nVal:nf*nVal+nVal,]
        idx_toDel = [np.where(train_val==a) for a in test[nf*nVal:nf*nVal+nVal,]]
        subjectSplit[c_dataset]["training_" + str(nf)] = np.delete(train_val, idx_toDel)
        
        # Make sure the sets are not empty.
        assert (not subjectSplit[c_dataset]["training_" + str(nf)].size == 0), 'empty training set'
        assert (not subjectSplit[c_dataset]["validation_" + str(nf)].size == 0), 'empty validation set'
        
        # Make sure values of the validation set aren't in the training set.
        idx_val_in_train = [np.where(subjectSplit[c_dataset]["training_" + str(nf)]==a) for a in subjectSplit[c_dataset]["validation_" + str(nf)]]
        for nElm in range(len(idx_val_in_train)):
            assert (idx_val_in_train[nElm][0].shape[0] == 0), "mix validation and training"
            
        # Make sure that validation and training sets together haves same values as train_val set.
        train_val_test = np.concatenate(
            (subjectSplit[c_dataset]["training_" + str(nf)], subjectSplit[c_dataset]["validation_" + str(nf)]), axis=0)    
        assert (np.sum(np.abs(np.sort(train_val_test)-np.sort(train_val)))==0), "missing values"   
    
if saveData:
    np.save(os.path.join(pathData, 'subjectSplit.npy'), subjectSplit)
