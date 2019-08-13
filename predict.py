import os
import time
from random import randint, random, sample

import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from sklearn import metrics
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder

_dir = 'E:\challenge\SNER\\'
data_features_prefix = _dir + "features/"
features_storm_out = 'features_storm_out'
features_storm_set = 'features_storm_set'

features_to_remove = [6, 8, 12, 14, 20, 22, 23, 24, 29, 34, 37, 41, 49, 51, 53, 55]

i = 1


def save_to_file(_dir, filename, obj):
    with open(_dir + filename + '.pkl', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def get_pickled(_dir, filename):
    try:
        with open(_dir + filename + '.pkl', 'rb') as handle:
            data = pickle.load(handle)
            handle.close()
            return data
    except FileNotFoundError:
        return None


def get_rows(_df, _column_name, _value):
    return _df.loc[_df[_column_name] == _value]


def write_solution(solution_path, data):
    with open(solution_path, 'w') as file:
        for line in data:
            file.write("{}\n".format(line))


def get_split_train_valid(_positives, _negatives, _ratio):
    _train_pos, _test_pos = split_train_test(_positives, _ratio)
    _train_neg, _test_neg = split_train_test(_negatives, _ratio)
    _concatenated_train = pd.concat([_train_pos, _train_neg])
    _concatenated_test = pd.concat([_test_pos, _test_neg])
    return _concatenated_train.sample(frac=1), _concatenated_test.sample(frac=1)


def split_train_test(_df, _ratio):
    msk = np.random.rand(len(_df)) < _ratio
    return _df[msk], _df[~msk]


def get_columns_containing_na(_df):
    #  data_training_raw.isnull().sum().values -- sum of NaN among all columns
    #  positives.groupby('n3').groups
    #  df.info(verbose=True)
    return _df.columns[_df.isna().any()].tolist()


def refine_ip(_df, _i):
    _df["ip{}".format(_i)] = _df.apply(lambda x: x["ip"].split('.')[-5 + _i], axis=1)


def get_learning_rates(_n):
    _a = -9.9 * 10 ** -7
    _b = 0.001
    return list(map(lambda x: x * _a + _b, range(_n)))


def get_features(_list, _importance):
    return list(filter(lambda x: x[1] <= _importance, _list))


def get_features_ids(_features_to_remove):
    return list(map(lambda x: int(x[0].replace('f', '')), _features_to_remove))


def remove_columns(_df, _cols):
    _df.drop(_df.columns[_cols], axis=1, inplace=True)


def strip_dim(data, indexes):
    indexes.sort(reverse=True)
    for i in indexes:
        data = np.concatenate((data[:, 0:i], data[:, i + 1:len(data[1, :])]), axis=1)
    return data


def take_sample_dim(train, ratio):
    _dim = train.shape[1]
    _range2strip = int(_dim / ratio)
    _nb2strip = randint(1, _range2strip)
    _sample = sample(range(1, _dim), _nb2strip)
    _sample.sort()
    return _sample


def persist_sample(_train, _sample, _sample_set, _ratio):
    while frozenset(_sample) in _sample_set:
        _sample = take_sample_dim(_train, _ratio)
    _sample_set.add(frozenset(_sample))
    save_to_file(data_features_prefix, features_storm_set, _sample_set)
    return _sample


def strip_dim_sets(x_train, x_test, _sample):
    x_train_str = strip_dim(x_train, _sample)
    x_test_str = strip_dim(x_test, _sample)
    return x_train_str, x_test_str


def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return metrics.roc_auc_score(y_test, predicted)


def write_chosen_solution(_sample, _auc):
    filepath = data_features_prefix + features_storm_out+'.pkl'
    if os.path.isfile(filepath):
        sample_f, auc_f = get_pickled(data_features_prefix, features_storm_out)
        if sample_f is not None:
            if _auc > float(auc_f):
                save_to_file(data_features_prefix, features_storm_out, (_sample, _auc))
    else:
        save_to_file(data_features_prefix, features_storm_out, (_sample, _auc))


def make_forced_step(_df_in, _it, _features_to_remove):
    _df = _df_in.copy()
    _ratio = 4
    # _x_train, _y_train, _x_val, _y_val = _dataframe
    _y_train = _df.pop('notified')

    remove_columns(_df, _features_to_remove)

    # _cat_df = pd.DataFrame({col: _df[col].astype('category').cat.codes for col in _df},
    #                   index=_df.index)

    _dummies_df = pd.get_dummies(_df, drop_first=True)
    _dummies_df['notified'] = _y_train

    _df_positives = get_rows(_dummies_df, 'notified', 1)
    _df_negatives = get_rows(_dummies_df, 'notified', 0)

    _train_df, _valid_df = get_split_train_valid(_df_positives, _df_negatives, 0.8)
    _y_train = _train_df.pop('notified')
    _y_valid = _valid_df.pop('notified')

    _tmp = time.time()
    _gpu_res = {}
    _dtrain = xgb.DMatrix(_train_df.values, label=_y_train.values, missing=-99)
    _dval = xgb.DMatrix(_valid_df.values, label=_y_valid.values, missing=-99)
    _evals = [(_dval, 'valid')]

    _param = {'objective': 'binary:logistic',  # Specify multiclass classification
              'num_class': 1,  # Number of possible output classes
              'tree_method': 'gpu_exact',  # Use GPU accelerated algorithm
              'scale_pos_weight': 16.32,
              # 'scale_pos_weight': 1,
              # 'scale_pos_weight': 10,
              'eval_metric': 'auc',
              'subsample': 0.8,
              'colsample_bytree': 0.9,
              # 'n_estimators': 5000,
              # 'eta_decay': 0.5,
              'seed': 1,
              # 'min_child_weight': 0.8,
              }

    _rounds = 10000
    _bst = xgb.train(_param, _dtrain, _rounds, early_stopping_rounds=300, evals=_evals, evals_result=_gpu_res, verbose_eval=True)

    print("GPU Training Time: %s seconds" % (str(time.time() - _tmp)))

    _predicted = _bst.predict(_dval)
    _auc = metrics.roc_auc_score(_y_valid, _predicted)
    # auc = trainClassifier(clf, xtrain, ytrain, xtest, ytest)
    # write_chosen_solution(_sample, _auc)
    print("Iteration: {} AUC: {} sample: {}".format(_it, _auc, _features_to_remove))
    i = 1
    # return auc, sample


# data_test_raw = pd.read_csv(_dir+'\cybersecurity_test.csv',sep='|')

# data_training_raw = pd.read_csv(_dir+'\cybersecurity_training.csv',sep='|')
# data_localized_alerts_raw = pd.read_csv(_dir+'\localized_alerts_data.csv',sep='|')
#
# save_to_file(_dir, "cybersecurity_training", data_training_raw)
# save_to_file(_dir, "localized_alerts_data", data_localized_alerts_raw)
# save_to_file(_dir, "cybersecurity_test", data_test_raw)

data_training_raw = get_pickled(_dir, "cybersecurity_training")
data_test_raw = get_pickled(_dir, "cybersecurity_test")
# data_localized_alerts_raw = get_pickled(_dir, "localized_alerts_data")

# refine_ip(data_training_raw, 2)
# refine_ip(data_test_raw, 2)
#
# refine_ip(data_training_raw, 4)
# refine_ip(data_test_raw, 4)

columns_to_drop = ['alert_ids', 'client_code', 'dstipcategory_dominate', 'start_minute',
                   'timestamp_dist']

columns_to_drop = ['alert_ids', 'start_minute', 'timestamp_dist', 'ip']

data_training_raw.drop(columns=columns_to_drop, inplace=True)
data_test_raw.drop(columns=columns_to_drop, inplace=True)

data_training_raw.fillna((-999), inplace=True)
data_test_raw.fillna((-999), inplace=True)

remove_columns(data_training_raw, features_to_remove)
remove_columns(data_test_raw, features_to_remove)

# make_forced_step(data_training_raw, 0, features_to_remove)


_y_train = data_training_raw.pop('notified')

con_df = pd.concat([data_training_raw, data_test_raw])
con_dummies_df = pd.get_dummies(con_df, drop_first=True)

# feature_ids = get_pickled(_dir, "features_to_remove_10")
# remove_columns(con_dummies_df, feature_ids)

train_size = data_training_raw.shape[0]

_train_dummies_df = con_dummies_df[:train_size]
_train_dummies_df['notified'] = _y_train
_test_dummies_df = con_dummies_df[train_size:]

cat_df = pd.DataFrame({col: data_training_raw[col].astype('category').cat.codes for col in data_training_raw},
                      index=data_training_raw.index)
cat_test_df = pd.DataFrame({col: data_test_raw[col].astype('category').cat.codes for col in data_test_raw},
                           index=data_test_raw.index)

categorical_columns = ['client_code', 'categoryname', 'ip', 'ipcategory_name', 'ipcategory_scope',
                       'grandparent_category',
                       'weekday', 'dstipcategory_dominate', 'srcipcategory_dominate', 'parent_category']
numerical_columns = ['overallseverity', 'timestamp_dist', 'start_hour', 'start_minute', 'correlatedcount',
                     'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'score', 'srcip_cd', 'dstip_cd',
                     'srcport_cd', 'dstport_cd', 'alerttype_cd', 'direction_cd', 'eventname_cd', 'severity_cd',
                     'reportingdevice_cd', 'devicetype_cd',
                     'devicevendor_cd', 'domain_cd', 'protocol_cd', 'username_cd', 'srcipcategory_cd',
                     'dstipcategory_cd', 'isiptrusted', 'untrustscore', 'flowscore',
                     'trustscore', 'enforcementscore', 'dstportcategory_dominate', 'srcportcategory_dominate',
                     'thrcnt_month', 'thrcnt_week', 'thrcnt_day',
                     'p6', 'p9', 'p5m', 'p5w', 'p5d', 'p8m', 'p8w', 'p8d']
# column_trans = make_column_transformer(
#     (categorical_columns, OneHotEncoder(handle_unknown='ignore')),
#     (numerical_columns, RobustScaler())
# )


# data_training_transformed = column_trans.fit_transform(data_training_raw)

# g = data_training_raw.columns.to_series().groupby(data_training_raw.dtypes).groups
# print(g)

# print(data_training_raw.select_dtypes(['object']))
_df_positives = get_rows(_train_dummies_df, 'notified', 1)
_df_negatives = get_rows(_train_dummies_df, 'notified', 0)

_train_df, _valid_df = get_split_train_valid(_df_positives, _df_negatives, 0.8)
_y_train = _train_df.pop('notified')
_y_valid = _valid_df.pop('notified')

_tmp = time.time()
_gpu_res = {}
_dtrain = xgb.DMatrix(_train_df.values, label=_y_train.values, missing=np.NaN)
_dval = xgb.DMatrix(_valid_df.values, label=_y_valid.values, missing=np.NaN)
_dtest = xgb.DMatrix(_test_dummies_df.values, missing=np.NaN)
_evals = [(_dval, 'valid')]

_param = {'objective': 'binary:logistic',  # Specify multiclass classification
          'num_class': 1,  # Number of possible output classes
          'tree_method': 'gpu_exact',  # Use GPU accelerated algorithm
          'scale_pos_weight': 16.32,
          # 'scale_pos_weight': 1,
          # 'scale_pos_weight': 10,
          'eval_metric': 'auc',
          'subsample': 0.8,
          'colsample_bytree': 0.9,
          # 'n_estimators': 5000,
          # 'eta_decay': 0.5,
          'seed': 1,
          # 'min_child_weight': 0.8,
          }

_rounds = 10000
bst = xgb.train(_param, _dtrain, _rounds, early_stopping_rounds=300, evals=_evals, evals_result=_gpu_res,
                 verbose_eval=True)

# pickle.dump(bst, open(_dir+"/models/model.1", "wb"))

# bst = get_pickled(_dir, "models/model.1")

# trained = xgb.train(param, dtrain, 782)
# xgb.train(param, dtrain)

# imp = xgb.plot_importance(bst)
# scores = bst.get_score()
#
# scores_sort = sorted(scores.items(), key=lambda kv: kv[1])
#
# features_to_remove = get_features(scores_sort, 10)
#
#
# feature_ids = get_features_ids(features_to_remove)
#
# save_to_file(_dir, "features_to_remove_10", feature_ids)


print("GPU Training Time: %s seconds" % (str(time.time() - _tmp)))

solution_path = _dir + 'solutions/solutions_s6.txt'

write_solution(solution_path, bst.predict(_dtest))

i = 1
