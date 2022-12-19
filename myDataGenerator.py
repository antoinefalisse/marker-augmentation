import numpy as np
from tensorflow import keras
import os
from utilities import rotateArray, rotateArraySphere3

# Inspired from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class lstmDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, pathData, batch_size=64, dim_f=(30,59), 
                 dim_r=(30,87), shuffle=True, noise_bool=False, noise_type='',
                 noise_magnitude=0, mean_subtraction=False, 
                 std_normalization=False, features_mean=0, features_std=0,                 
                 idx_features=[], idx_responses=[], 
                 rotation_type='circleRotation', rotations=[0], 
                 ref_vec=np.array([0,0,1]) ):
        'Initialization'
        self.dim_f = dim_f
        self.dim_r = dim_r
        
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        
        self.shuffle = shuffle
        self.on_epoch_end()
        
        self.pathData = pathData
        
        self.noise_bool = noise_bool
        self.noise_magnitude = noise_magnitude 
        
        self.mean_subtraction = mean_subtraction
        self.std_normalization = std_normalization
        self.features_mean = features_mean
        self.features_std = features_std
        
        self.idx_features = idx_features
        self.idx_responses = idx_responses
        
        self.rotation_type = rotation_type
        self.rotations = rotations
        self.ref_vec = ref_vec
        
        if rotation_type == 'circleRotation':
            self.nRotations = len(rotations)
        elif rotation_type == 'sphereRotation':
            self.nRotations = rotations.shape[1]
        else:
            raise ValueError('Unknown rotation type')
            

    def __len__(self):
        'Denotes the number of batches per epoch'        
        # We multiply the number of samples by the number of rotations we want
        # to apply to each sample, and then divide by the batch size.
        return int(np.floor(
            (len(self.list_IDs)*(self.nRotations+1)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch        
        # Take into account the rotation: we have nRotations x the same sample.
        # Eg,   list_IDs = [0,1,2,3,4,5,6,7,8,9,10,11]
        #       batch_size = 3, n_rot = 4, n_batch = (12*4)/3=16
        #       1st 4 indices correspond to list_IDs0, rot0-4
        #       next 4 indices correspond to list_IDs1, rot0-4
        #       ...
        #       index => floor(index/n_rot)
        #           idx0=0, idx1=0, idx2=0, idx3=0, idx4=1, idx5=1, idx6=1, ...
        index_rot = int(np.floor(index/(self.nRotations+1)))
        indexes = self.indexes[index_rot*self.batch_size:(index_rot+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]        

        #       ...
        #       index => mod(index, n_rot)
        #           idx0=0, idx1=1, idx2=2, idx3=3, idx4=0, idx5=1, idx6=2, ...
        idx_rot = np.mod(index, (self.nRotations+1))

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, idx_rot)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, idx_rot):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim_f))
        y = np.empty((self.batch_size, *self.dim_r))       
        
        # Pick a rotation between
        if idx_rot != 0:
            if self.rotation_type == 'circleRotation':
                rotations_all = self.rotations + [360.]
                step = 360/self.nRotations/4
                rotation = np.random.choice(np.arange(rotations_all[idx_rot-1], rotations_all[idx_rot], step), size=1)
            elif self.rotation_type == 'sphereRotation':
                rotation = self.rotations[:, idx_rot-1]                
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X_temp = np.load(os.path.join(self.pathData, 'feature_{}.npy'.format(ID)))[:, self.idx_features]
            # Apply rotation
            if idx_rot == 0:
                X[i,] = X_temp
            else:
                if self.rotation_type == 'circleRotation':
                    X_temp_xyz_rot = rotateArray(X_temp[:,:-2], 'y', rotation)
                elif self.rotation_type == 'sphereRotation':
                    theta_x = rotation.flatten()[0,]
                    theta_z = rotation.flatten()[1,]                
                    X_temp_xyz_rot, unit_vec = rotateArraySphere3(
                        X_temp[:,:-2], self.ref_vec, theta_x, theta_z)                
                X_temp_rot = np.concatenate((X_temp_xyz_rot, X_temp[:,-2:]), axis=1)
                X[i,] = X_temp_rot
            # Add noise
            if self.noise_bool:                
                np.random.seed(ID)     
                noise = np.zeros((self.dim_f[0], self.dim_f[1]))
                noise[:,:self.dim_f[1]-2] = np.random.normal(
                    0, self.noise_magnitude, (self.dim_f[0], self.dim_f[1]-2))           
                X[i,] += noise
                
            # Process data
            # Mean subtraction
            if self.mean_subtraction:                
                X[i,] -= self.features_mean
            # Std division
            if self.std_normalization:
                X[i,] /= self.features_std

            # Store class
            y_temp = np.load(os.path.join(self.pathData, 'responses_{}.npy'.format(ID)))[:, self.idx_responses]
            # Apply rotation
            if idx_rot == 0:
                y[i,] = y_temp
            else:
                if self.rotation_type == 'circleRotation':
                    y_temp_xyz_rot = rotateArray(y_temp, 'y', rotation)
                elif self.rotation_type == 'sphereRotation':
                    theta_x = rotation.flatten()[0,]
                    theta_z = rotation.flatten()[1,]
                    # Use alignment used for features to make sure the same
                    # rotation is applied.
                    y_temp_xyz_rot, _ = rotateArraySphere3(y_temp, self.ref_vec, theta_x, theta_z, unit_vec=unit_vec)
                y[i,] = y_temp_xyz_rot                

        return X, y