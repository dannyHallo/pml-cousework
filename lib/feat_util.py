import librosa
import os
import skimage.util
import numpy as np
import config
import pickle
import math
import collections
import pandas as pd
from sklearn.utils import shuffle
# Sound type unique values: {'background', 'mosquito', 'audio'}; class labels: {1, 0, 0}

# Extract features from wave files with id corresponding to dataframe data_df.


def get_feat(data_df, data_dir, rate, min_duration, n_feat):
    ''' Returns features extracted with Librosa. A list of features, with the number of items equal to the number of input recordings'''

    X = []
    y = []
    bugs = []
    idx = 0
    skipped_files = []
    for row_idx_series in data_df.iterrows():
        idx += 1
        if idx % 100 == 0:
            print('Completed', idx, 'of', len(data_df))
        row = row_idx_series[1]
        label_duration = row['length']
        if label_duration > min_duration:
            _, file_format = os.path.splitext(row['name'])
            filename = os.path.join(data_dir, str(row['id']) + file_format)
            length = librosa.get_duration(filename=filename)
#             assert math.isclose(length,label_duration, rel_tol=0.01), "File: %s label duration (%.4f) does not match audio length (%.4f)" % (row['path'], label_duration, length)

            if math.isclose(length, label_duration, rel_tol=0.01):
                signal, rate = librosa.load(filename, sr=rate)
                feat = librosa.feature.melspectrogram(
                    signal, sr=rate, n_mels=n_feat)
                feat = librosa.power_to_db(feat, ref=np.max)
                if config.norm_per_sample:
                    feat = (feat-np.mean(feat))/np.std(feat)
                X.append(feat)
                if row['sound_type'] == 'mosquito':
                    y.append(1)
                # Condition to check we are not adding empty (or unexpected) labels as 0
                elif row['sound_type']:
                    y.append(0)
            else:
                print("File: %s label duration (%.4f) does not match audio length (%.4f)" % (
                    row['name'], label_duration, length))
                bugs.append([row['name'], label_duration, length])

        else:
            skipped_files.append([row['id'], row['name'], label_duration])
    return X, y, skipped_files, bugs


def get_feat_multi_class(df_all, label_recordings_dict):
    '''Extract features for multi-class classification.'''
    X = []
    y = []

    for class_label in label_recordings_dict.keys():  # Loop over classes
        # Loop over recordings in class
        for i in label_recordings_dict[class_label]:
            row = df_all[df_all.id == i].iloc[0]

            _, file_format = os.path.splitext(row['name'])
            filename = os.path.join(
                config.data_dir, str(row['id']) + file_format)
            signal, rate = librosa.load(filename, sr=config.rate)
            feat = librosa.feature.melspectrogram(
                y=signal, sr=rate, n_mels=config.n_feat)
            feat = librosa.power_to_db(feat, ref=np.max)
            if config.norm_per_sample:
                feat = (feat-np.mean(feat))/np.std(feat)
            X.append(feat)
            y.append(class_label)

    return X, y


def get_train_test_with_selector(df_all, column_name, class_names, random_seed,  train_fraction=0.75):
    '''Extract features for multi-class classification.'''

    pickle_name_train = f'feat_{column_name}_sd_{random_seed}_train.pickle'
    pickle_name_test = f'feat_{column_name}_sd_{random_seed}_test.pickle'

    if not os.path.isfile(os.path.join(config.feature_save_dir, pickle_name_train)):
        print('Extracting train features...')
        _recordings = collections.OrderedDict()
        for class_name in class_names:
            rows = df_all[df_all[column_name] == class_name]
            ids = pd.unique(rows.id)
            print(class_name, len(ids))
            _recordings[class_name] = ids

        # Divide recordings into train and test, with recording shuffling fixed by random_state
        train_recordings = {}
        test_recordings = {}

        print('class name, train count, test count')

        for i in range(len(class_names)):
            class_name = class_names[i]
            total_id_count = len(_recordings[class_name])
            n_train = int(total_id_count * train_fraction)
            n_test = total_id_count - n_train
            print(class_name, n_train, n_test)

            train_recordings[i] = shuffle(
                _recordings[class_name], random_state=random_seed)[:n_train]
            test_recordings[i] = shuffle(
                _recordings[class_name], random_state=random_seed)[n_train:]

        X_train, y_train = get_feat_multi_class(df_all, train_recordings)

        feat_train = {"X_train": X_train, "y_train": y_train}

        with open(os.path.join(config.feature_save_dir, pickle_name_train), 'wb') as f:
            pickle.dump(feat_train, f, protocol=4)
            print('Saved features to:', os.path.join(
                config.feature_save_dir, pickle_name_train))
    else:
        with open(os.path.join(config.feature_save_dir, pickle_name_train), 'rb') as input_file:
            log_mel_feat = pickle.load(input_file)
            X_train = log_mel_feat["X_train"]
            y_train = log_mel_feat["y_train"]

    if not os.path.isfile(os.path.join(config.feature_save_dir, pickle_name_test)):

        print('Extracting test features...')
        X_test, y_test = get_feat_multi_class(df_all, test_recordings)

        feat_test = {"X_test": X_test, "y_test": y_test}
        with open(os.path.join(config.feature_save_dir, pickle_name_test), 'wb') as f:
            pickle.dump(feat_test, f, protocol=4)
            print('Saved features to:', os.path.join(
                config.feature_save_dir, pickle_name_test))
    else:
        with open(os.path.join(config.feature_save_dir, pickle_name_test), 'rb') as input_file:
            log_mel_feat = pickle.load(input_file)
            X_test = log_mel_feat["X_test"]
            y_test = log_mel_feat["y_test"]

    return X_train, y_train, X_test, y_test


def reshape_feat(feats, labels, win_size, step_size):
    '''Reshaping features from get_feat to be compatible for classifiers expecting a 2D slice as input. Parameter `win_size` is 
    given in number of feature windows (in librosa this is the hop length divided by the sample rate.)
    Can code to be a function of time and hop length instead in future.'''

    feats_windowed_array = []
    labels_windowed_array = []
    for idx, feat in enumerate(feats):
        if np.shape(feat)[1] < win_size:
            print('Length of recording shorter than supplied window size.')
            pass
        else:
            feats_windowed = skimage.util.view_as_windows(
                feat.T, (win_size, np.shape(feat)[0]), step=step_size)
            labels_windowed = np.full(len(feats_windowed), labels[idx])
            feats_windowed_array.append(feats_windowed)
            labels_windowed_array.append(labels_windowed)
    return np.vstack(feats_windowed_array), np.hstack(labels_windowed_array)
