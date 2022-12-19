import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError

# %% Dense model.
def get_dense_model(nFirstUnits, nHiddenUnits, nHiddenLayers, input_dim,
                    output_dim, L2_p, dropout_p, learning_r, loss_f,
                    marker_weights):
    
    # For reproducibility.
    np.random.seed(1)
    tf.random.set_seed(1)

    # Model.
    model = Sequential()
    # First layer.
    if L2_p > 0:
        model.add(Dense(nFirstUnits, input_shape=(input_dim,), 
                        kernel_initializer=glorot_normal(seed=None), 
                        activation='relu',
                        activity_regularizer=L2(L2_p)))
    else:
        model.add(Dense(nFirstUnits, input_shape=(input_dim,), 
                        kernel_initializer=glorot_normal(seed=None), 
                        activation='relu'))
    # Hidden layers.
    if nHiddenLayers > 0:
        for i in range(nHiddenLayers):
            if dropout_p > 0:
                model.add(Dropout(dropout_p))
            if L2_p > 0:
                model.add(Dense(nHiddenUnits, 
                                kernel_initializer=glorot_normal(seed=None), 
                                activation='relu',
                                kernel_regularizer=L2(L2_p)))            
            else:
                model.add(Dense(nHiddenUnits, 
                                kernel_initializer=glorot_normal(seed=None), 
                                activation='relu'))
    if dropout_p > 0:
        model.add(Dropout(dropout_p))
    # Last layer.
    model.add(Dense(output_dim, kernel_initializer=glorot_normal(seed=None), 
                    activation='linear'))
    
    # Optimizer.
    opt=Adam(learning_rate=learning_r)
    
    # Loss function.
    if loss_f == "weighted_mean_squared_error":
        model.compile(
            optimizer=opt,
            loss=weighted_mean_squared_error(marker_weights),
            metrics=[MeanSquaredError(), RootMeanSquaredError()])    
    else:
        model.compile(
            optimizer=opt,
            loss=loss_f,
            metrics=[MeanSquaredError(), RootMeanSquaredError()])
    
    return model

# %% LSTM model.
def get_lstm_model(input_dim, output_dim, nHiddenLayers, nHUnits, learning_r,
                   loss_f, bidirectional=False):
    
    # For reproducibility.
    np.random.seed(1)
    tf.random.set_seed(1)

    # Model.
    model = Sequential()
    # First layer.
    if bidirectional:
        model.add(Bidirectional(LSTM(units=nHUnits, 
                                     input_shape=(None, input_dim),
                                     return_sequences=True)))
    else:
        model.add(LSTM(units=nHUnits, input_shape = (None, input_dim),
                       return_sequences=True))
    # Hidden layers.
    if nHiddenLayers > 0:
        for i in range(nHiddenLayers):
            if bidirectional:
                model.add(Bidirectional(LSTM(units=nHUnits, 
                                             return_sequences=True)))
            else:
                model.add(LSTM(units=nHUnits, return_sequences=True))
    # Last layer.    
    model.add(TimeDistributed(Dense(output_dim, activation='linear')))
    
    # Optimizer.
    opt=Adam(learning_rate=learning_r)
    
    # Loss function.
    model.compile(
            optimizer=opt,
            loss=loss_f,
            metrics=[MeanSquaredError(), RootMeanSquaredError()])
    
    return model

# %% Helper functions.
def get_callback():
    callback =  tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=1e-08, patience=10, verbose=0,
        mode='min', baseline=None, restore_best_weights=True )
    
    return callback

def weighted_mean_squared_error(weights):
    def loss(y_true, y_pred):      
        squared_difference = tf.square(y_true - y_pred)        
        weighted_squared_difference = weights * squared_difference  
        return tf.reduce_mean(weighted_squared_difference, axis=-1)
    return loss