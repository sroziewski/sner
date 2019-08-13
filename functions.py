import pickle

import numpy as np

_dir = 'E:\challenge\SNER\\'


def split_train_test(_df, _ratio):
    msk = np.random.rand(len(_df)) < _ratio
    return _df[msk], _df[~msk]


def get_split_train_valid_indexes(_positives, _negatives, _ratio):
    _train_pos, _valid_pos = split_train_test(np.array(_positives), _ratio)
    _train_neg, _valid_neg = split_train_test(np.array(_negatives), _ratio)
    _concatenated_train = np.concatenate([_train_pos, _train_neg])
    _concatenated_valid = np.concatenate([_valid_pos, _valid_neg])
    np.random.shuffle(_concatenated_train)
    np.random.shuffle(_concatenated_valid)
    return _concatenated_train, _concatenated_valid


def generate_random_train_valid_indexes(_data_training_raw):
    _pos_indexes = _data_training_raw.index[_data_training_raw['notified'] == True].tolist()
    _neg_indexes = _data_training_raw.index[_data_training_raw['notified'] == False].tolist()

    _train_indexes, _valid_indexes = get_split_train_valid_indexes(_pos_indexes, _neg_indexes, 0.8)

    save_to_file(_dir, "train_valid_shuffled_indexes", (_train_indexes, _valid_indexes))


def save_to_file(_dir, filename, obj):
    with open(_dir + filename + '.pkl', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
