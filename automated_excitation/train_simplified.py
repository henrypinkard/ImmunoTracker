import numpy as np
import h5py
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from automated_excitation.lami_helper import bin_surface_distance

def load_data():

    def readIntoArray(structgroup, name):
        arr = np.zeros(structgroup[name].shape)
        structgroup[name].read_direct(arr)
        return arr.T

    f = h5py.File('/Users/henrypinkard/Desktop/Lymphosight/2017 data/2017-4-4 medLN/C_40_600_70_1--Positions as time_GFPCandidates.mat', mode='r')
    # f = h5py.File('/Users/henrypinkard/Desktop/Lymphosight/2017 data/2017-4-4 medLN/MT_55_600_30_2--Positions as time_e670Candidates.mat', mode='r')

    struct = f.get('excitationNNData')
    #read variables
    normalized_tile_position = readIntoArray(struct, 'normalizedTilePosition')
    normalized_tile_position = np.reshape(normalized_tile_position, (-1, 2))
    normalized_brightness = readIntoArray(struct, 'normalizedBrightness')
    # normalizedBrightnessMedian = readIntoArray(struct, 'normalizedBrightnessMedian')
    # timeIndex = readIntoArray(struct, 'timeIndices')
    distances_to_surface = readIntoArray(struct, 'distancesToInterpolation')
    # distancesToInterpolationSP = readIntoArray(struct, 'distancesToInterpolarionSP') #centered at local stage position
    excitations = readIntoArray(struct, 'excitations')
    f.close()

    return normalized_tile_position, normalized_brightness, distances_to_surface, excitations


model_export_name = 'GFP_LAMI_model'
normalized_tile_position, normalized_brightness, distances_to_surface, excitations = load_data()

distance_histograms = bin_surface_distance(distances_to_surface)
# save distance normalization params
distance_histograms_means = np.mean(distance_histograms, axis=0)
distance_histograms_stddev = np.std(distance_histograms, axis=0)

design_matrix = np.concatenate((distance_histograms, normalized_tile_position, normalized_brightness), axis=1)


def normalize_histograms(design_mat, distance_histograms_means=distance_histograms_means,
                         distance_histograms_stddev=distance_histograms_stddev):
    import tensorflow as tf
    normalized = (design_mat[:, :-3] - distance_histograms_means) / distance_histograms_stddev
    return tf.concat([normalized, design_mat[:, -3:]], axis=1)

model = keras.Sequential()
model.add(keras.Input(shape=design_matrix.shape[1]))
model.add(layers.Lambda(normalize_histograms))
model.add(layers.Dense(200, activation='tanh'))
model.add(layers.Dense(1, activation=None))

def excitation_power_loss(y_true, y_pred):
    voltage_to_power = lambda x: (tf.cos(3.1415 + 2 * 3.1415 / 510 * x) + 1) / 2
    # dont overly penalize values outside of physical range eom can apply range
    clampedy = tf.minimum(255.0, tf.maximum(0.0, y_pred))
    totalSqError = tf.square(voltage_to_power(clampedy) - voltage_to_power(y_true))
    return tf.sqrt(tf.reduce_mean(totalSqError))


model.compile(optimizer='adam', loss=excitation_power_loss)
model.fit(x=design_matrix, y=excitations, batch_size=1000, validation_split=0.2, epochs=100000,
          verbose=1, steps_per_epoch=10, shuffle=True,
          callbacks=keras.callbacks.EarlyStopping(patience=50))

model.save(model_export_name, overwrite=True)

