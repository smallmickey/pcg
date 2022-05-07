#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np
import librosa
from scipy.fftpack import next_fast_len
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

def __get_norm(norm):
    if norm == 0 or norm is None:
        return None, None
    else:
        try:
            norm1, norm2 = norm
        except TypeError:
            norm1 = norm2 = norm
        return norm1, norm2


def __freq_ind(freq, f0):
    try:
        return [np.argmin(np.abs(freq - f)) for f in f0]
    except TypeError:
        return np.argmin(np.abs(freq - f0))


def __product_other_freqs(spec, indices, synthetic=(), t=None):
    p1 = np.prod([amplitude * np.exp(2j * np.pi * freq * t + phase)
                  for (freq, amplitude, phase) in synthetic], axis=0)
    p2 = np.prod(spec[:, indices[len(synthetic):]], axis=1)
    return p1 * p2


def _polycoherence_0d(data, fs, *freqs, norm=2, synthetic=(), **kwargs):
    """Polycoherence between freqs and sum of freqs"""
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode='complex', **kwargs)
    ind = __freq_ind(freq, freqs)
    sum_ind = __freq_ind(freq, np.sum(freqs))
    spec = np.transpose(spec, [1, 0])
    p1 = __product_other_freqs(spec, ind, synthetic, t)
    p2 = np.conjugate(spec[:, sum_ind])
    coh = np.mean(p1 * p2, axis=0)
    if norm is not None:
        coh = np.abs(coh)
        coh **= 2
        temp2 = np.mean(np.abs(p1) ** norm1 * np.abs(p2) ** norm2, axis=0)
        coh /= temp2
        coh **= 0.5
    return coh


def _polycoherence_1d(data, fs, *freqs, norm=2, synthetic=(), **kwargs):
    """
    Polycoherence between f1 given freqs and their sum as a function of f1
    """
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode='complex', **kwargs)
    spec = np.transpose(spec, [1, 0])
    ind2 = __freq_ind(freq, freqs)
    ind1 = np.arange(len(freq) - sum(ind2))
    sumind = ind1 + sum(ind2)
    otemp = __product_other_freqs(spec, ind2, synthetic, t)[:, None]
    temp = spec[:, ind1] * otemp
    temp2 = np.mean(np.abs(temp) ** 2, axis=0)
    temp *= np.conjugate(spec[:, sumind])
    coh = np.mean(temp, axis=0)
    if norm is not None:
        coh = np.abs(coh)
        coh **= 2
        temp2 *= np.mean(np.abs(spec[:, sumind]) ** 2, axis=0)
        coh /= temp2
        coh **= 0.5
    return freq[ind1], coh


def _polycoherence_1d_sum(data, fs, f0, *ofreqs, norm=2,
                          synthetic=(), **kwargs):
    """Polycoherence with fixed frequency sum f0 as a function of f1"""
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode='complex', **kwargs)
    spec = np.transpose(spec, [1, 0])
    ind3 = __freq_ind(freq, ofreqs)
    otemp = __product_other_freqs(spec, ind3, synthetic, t)[:, None]
    sumind = __freq_ind(freq, f0)
    ind1 = np.arange(np.searchsorted(freq, f0 - np.sum(ofreqs)))
    ind2 = sumind - ind1 - sum(ind3)
    temp = spec[:, ind1] * spec[:, ind2] * otemp
    if norm is not None:
        temp2 = np.mean(np.abs(temp) ** 2, axis=0)
    temp *= np.conjugate(spec[:, sumind, None])
    coh = np.mean(temp, axis=0)
    if norm is not None:
        coh = np.abs(coh)
        coh **= 2
        temp2 *= np.mean(np.abs(spec[:, sumind]) ** 2, axis=0)
        coh /= temp2
        coh **= 0.5
    return freq[ind1], coh


def _polycoherence_2d(data, fs, *ofreqs, norm=2, flim1=None, flim2=None,
                      synthetic=(), **kwargs):
    """
    Polycoherence between freqs and their sum as a function of f1 and f2
    """
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode='complex', **kwargs)
    spec = np.require(spec, 'complex64')
    spec = np.transpose(spec, [1, 0])  # transpose (f, t) -> (t, f)
    if flim1 is None:
        flim1 = (0, (np.max(freq) - np.sum(ofreqs)) / 2)
    if flim2 is None:
        flim2 = (0, (np.max(freq) - np.sum(ofreqs)) / 2)
    ind1 = np.arange(*np.searchsorted(freq, flim1))
    ind2 = np.arange(*np.searchsorted(freq, flim2))
    ind3 = __freq_ind(freq, ofreqs)
    otemp = __product_other_freqs(spec, ind3, synthetic, t)[:, None, None]
    sumind = ind1[:, None] + ind2[None, :] + sum(ind3)
    temp = spec[:, ind1, None] * spec[:, None, ind2] * otemp
    if norm is not None:
        temp2 = np.mean(np.abs(temp) ** norm1, axis=0)
    temp *= np.conjugate(spec[:, sumind])
    coh = np.mean(temp, axis=0)
    del temp
    if norm is not None:
        coh = np.abs(coh, out=coh)
        coh **= 2
        temp2 *= np.mean(np.abs(spec[:, sumind]) ** norm2, axis=0)
        coh /= temp2
        coh **= 0.5
    return freq[ind1], freq[ind2], coh


def polycoherence(data, *args, dim=2, **kwargs):
    """
    Polycoherence between frequencies and their sum frequency

    Polycoherence as a function of two frequencies.

    |<prod(spec(fi)) * conj(spec(sum(fi)))>| ** n0 /
        <|prod(spec(fi))|> ** n1 * <|spec(sum(fi))|> ** n2

    i ... 1 - N: N=2 bicoherence, N>2 polycoherence
    < > ... averaging
    | | ... absolute value

    data: 1d data
    fs: sampling rate
    ofreqs: further positional arguments are fixed frequencies

    dim:
        2 - 2D polycoherence as a function of f1 and f2, ofreqs are additional
            fixed frequencies (default)
        1 - 1D polycoherence as a function of f1, at least one fixed frequency
            (ofreq) is expected
        'sum' - 1D polycoherence with fixed frequency sum. The first argument
            after fs is the frequency sum. Other fixed frequencies possible.
        0 - polycoherence for fixed frequencies
    norm:
        2 - return polycoherence, n0 = n1 = n2 = 2 (default)
        0 - return polyspectrum, <prod(spec(fi)) * conj(spec(sum(fi)))>
        tuple (n1, n2): general case with n0=2
    synthetic:
        used for synthetic signal for some frequencies,
        list of 3-item tuples (freq, amplitude, phase), freq must coincide
        with the first fixed frequencies (ofreq, except for dim='sum')
    flim1, flim2: for 2D case, frequency limits can be set
    **kwargs: are passed to scipy.signal.spectrogram. Important are the
        parameters nperseg, noverlap, nfft.
    """
    N = len(data)
    kwargs.setdefault('nperseg', N // 20)
    kwargs.setdefault('nfft', next_fast_len(N // 10))
    if dim == 0:
        f = _polycoherence_0d
    elif dim == 1:
        f = _polycoherence_1d
    elif dim == 'sum':
        f = _polycoherence_1d_sum
    elif dim == 2:
        f = _polycoherence_2d
    else:
        raise
    return f(data, *args, **kwargs)

def get_all_filenames(file_dir):
    all_files = [file for file in os.listdir(file_dir)]
    # print(all_files)
    return all_files


def get_bi_spectrum(file_folder, class_list, data_num=2000):
    dataset = np.zeros((256,256,1))
    for class_nam in class_list:
        path = os.path.join(file_folder, class_nam)
        all_files = get_all_filenames(path)
        index = 0
        for name in all_files[:data_num]:
            file_path = os.path.join(path, name)
            sig, sr = librosa.load(file_path, sr=1000) # load heart sound data
            freq1, freq2, bi_spectrum = polycoherence(sig,nfft=1024, fs = 1000, norm=None,noverlap = 100, nperseg=256)
            bi_spectrum = np.array(abs(bi_spectrum))  # calculate bi_spectrum
            bi_spectrum = bi_spectrum.reshape((256, 256, 1))
            bi_spectrum = 255 * (bi_spectrum - np.min(bi_spectrum)) / (np.max(bi_spectrum) - np.min(bi_spectrum))
            dataset = np.vstack((dataset, np.array([bi_spectrum])))  # concat the dataset
            index+=1
            print(class_nam,'num:',index)
        # remove the first one of the dataset, due to initialization
    dataset = np.delete(dataset, 0, 0)
    return dataset




def plot_polycoherence(freq1, freq2, bicoh):
    df1 = freq1[1] - freq1[0]
    df2 = freq2[1] - freq2[0]
    freq1 = np.append(freq1, freq1[-1] + df1) - 0.5 * df1
    freq2 = np.append(freq2, freq2[-1] + df2) - 0.5 * df2
    plt.figure(figsize=(4,4),dpi=300)
    # fig, ax = plt.subplots()
    plt.pcolormesh(freq2, freq1, np.abs(bicoh),cmap=plt.cm.jet)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    plt.figure(figsize=(4,4),dpi=300)
    ax = plt.subplot(1, 1, 1)
    plt.contour(freq2[:-1], freq1[:-1], np.abs(bicoh),10,linewidths=1,cmap=plt.cm.jet)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################
def build_model_1():
    inputdata = keras.Input(shape=(1024,256,1))

    final = keras.layers.Conv2D(128,(3,3), padding="same",activation='relu')(inputdata)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2,2), strides=(2,2))(final)
    final = keras.layers.Dropout(rate=0.2)(final)

    final = keras.layers.Conv2D(64,(3,3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2,2), strides=(2,2) )(final)
    final = keras.layers.Dropout(rate=0.2)(final)

    final = keras.layers.Conv2D(16,(3,3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2,2), strides=(2,2))(final)
    final = keras.layers.Dropout(rate=0.2)(final)

    final = keras.layers.Conv2D(16, (3,3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)


    final = keras.layers.Flatten()(final)
    final = keras.layers.Dense(3)(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.Softmax()(final)

    model = keras.Model(inputs=inputdata, outputs=final)
    optimizer = keras.optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    # print(model.summary())
    return model
def build_model_2():
    inputdata = keras.Input(shape=(1024,256,1))

    final = keras.layers.Conv2D(128,(3,3), padding="same",activation='relu')(inputdata)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2,2), strides=(2,2))(final)


    final = keras.layers.Conv2D(64,(3,3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2,2), strides=(2,2) )(final)
    final = keras.layers.Dropout(rate=0.2)(final)

    final = keras.layers.Conv2D(16,(3,3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2,2), strides=(2,2))(final)
    final = keras.layers.Dropout(rate=0.2)(final)

    final = keras.layers.Conv2D(16, (3,3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)


    final = keras.layers.Flatten()(final)
    final = keras.layers.Dense(2)(final)
    #final = keras.layers.BatchNormalization()(final)
    final = keras.layers.Softmax()(final)

    model = keras.Model(inputs=inputdata, outputs=final)
    optimizer = keras.optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    # print(model.summary())
    return model
# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    features = list()
    murmurs = list()
    outcomes = list()

    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)

        # Extract features.
        current_features = get_features(current_patient_data, current_recordings)
        features.append(current_features)

        # Extract labels and use one-hot encoding.
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)

        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)

    features = np.array(features)
    murmurs = np.array(murmurs)
    outcomes = np.array(outcomes)
    print(murmurs.shape)
    # Train the model.
    if verbose >= 1:
        print('Training model...')
    imputer=0
    num_epochs = 10
    model_1 = build_model_1()

    history1 = model_1.fit(features, murmurs,
                           epochs=num_epochs,
                           batch_size=32,
                           validation_split=0.2,
                           verbose=1)

    model_1.summary()

    model_2 = build_model_2()

    history2 = model_2.fit(features, outcomes,
                           epochs=num_epochs,
                           batch_size=32,
                           validation_split=0.2,
                           verbose=1)

    model_2.summary()


    # Save the model.
    save_challenge_model(model_folder, imputer, murmur_classes, model_1, outcome_classes, model_2)

    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename1 = os.path.join(model_folder, 'model_1.h5')
    filename3 = os.path.join(model_folder, 'model_2.h5')
    filename2= os.path.join(model_folder, 'model.sav')
    return  joblib.load(filename2),load_model(filename1),load_model(filename3)

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
   # imputer = model[0]['imputer']
    murmur_classes = model[0]['murmur_classes']
    murmur_classifier = model[1]
    outcome_classes = model[0]['outcome_classes']
    outcome_classifier = model[2]


    pre_features = list()
    # Load features.
    features = get_features(data, recordings)
    pre_features.append(features)
    pre_features = np.array(pre_features)

    # Impute missing data.
    #features = features.reshape(1, -1)
   # features = imputer.transform(features)

    # Get classifier probabilities.
    murmur_probabilities = murmur_classifier.predict(pre_features)
    murmur_probabilities = np.asarray(murmur_probabilities, dtype=np.float32)[0, :]
    outcome_probabilities = outcome_classifier.predict(pre_features)
    outcome_probabilities = np.asarray(outcome_probabilities, dtype=np.float32)[0, :]

    # Choose label with highest probability.
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    idx = np.argmax(murmur_probabilities)
    murmur_labels[idx] = 1
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1

    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier):
    d = {'imputer': imputer, 'murmur_classes': murmur_classes,  'outcome_classes': outcome_classes, }
    filename = os.path.join(model_folder, 'model.sav')
    model_1=murmur_classifier
    model_1.save(model_folder+'/'+'model_1.h5')
    model_2 = outcome_classifier
    model_2.save(model_folder + '/' + 'model_2.h5')

    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(data, recordings):
    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    dataset = np.zeros((1024, 256, 1))
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV']
    # recording_locations = ['MV']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations == num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i]) > 0:
                    freq1, freq2, bi_spectrum = polycoherence(
                        recordings[i],

                        nfft=1024,
                        nperseg=256,
                        noverlap=100,
                        fs=1000,
                        norm=None)
                    bi_spectrum = np.array(abs(bi_spectrum))  # calculate bi_spectrum
                    bi_spectrum = bi_spectrum.reshape((256, 256, 1))  # 修改尺寸以便于投入神经网络
                    bi_spectrum = 255 * (bi_spectrum - np.min(bi_spectrum)) / (
                                np.max(bi_spectrum) - np.min(bi_spectrum))
                    # temp=temp.append(bi_spectrum.tolist())
                    # dataset=np.array(temp)
                    if i>=4:
                        break
                    for a in range(256):
                        for b in range(256):
                            dataset[a + i * 256, b] = bi_spectrum[a, b]
                            # print(dataset.shape)
    # dataset = np.delete(dataset, 0, 0)

    # recording_features = bi_spectrum.flatten()

    # features = np.hstack(( recording_features))

    return np.asarray(dataset, dtype=np.float32)
