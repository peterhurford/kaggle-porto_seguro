import random
import time
import os

from datetime import datetime
from copy import copy
from pprint import pprint

import pathos.multiprocessing as mp

import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler, MinMaxScaler, \
                                  OneHotEncoder, scale, minmax_scale
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import BernoulliRBM, MLPClassifier

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
                             ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


SHOULD_SUBMIT = True
KFOLDS = 5


def ginic(actual, pred):
    actual = np.asarray(actual) 
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalized(a, p):
    if p.ndim == 2:
        p = p[:,1] 
    return ginic(a, p) / ginic(a, a)

def gini_lgb(labels, preds):
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score, True

def gini_xgb(preds, labels):
    labels = labels.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score


def cv_pipeline(label, pipeline, index, eval_fn, kfolds=5, fold_seed=42):
    tag = label + ' - Fold %d/%d -' % (index + 1, kfolds)
    path = '/dev/cache/' + label + '-fold-' + str(index + 1) + '.csv'
    sub_path = '/dev/cache/' + label + '-sub-fold-' + str(index + 1) + '.csv'

    has_cache = os.path.isfile(path)
    if has_cache:
        print('[%s Already has a cache file... skipping!]' % tag)
    else:
        print('[%s Accepted job]' % tag)

    if has_cache:
        train = pd.read_csv(path)
    elif 'Lvl3' in label:
        train = pd.read_csv('level_2_df.csv')
    elif 'Lvl2' in label:
        train = pd.read_csv('base_df.csv')
    else:
        train = pd.read_csv('train_clean.csv')
    if has_cache:
        test = copy(train)
    train_id = train['id']
    target = train['target']
    train.drop(['target', 'id'], axis=1, inplace=True)
    if 'Lvl3' in label:
        test = pd.read_csv('level_2_submissions.csv')
    elif 'Lvl2' in label:
        test = pd.read_csv('base_submissions.csv')
    elif not has_cache:
        test = pd.read_csv('test_clean.csv')
    test_id = test['id']
    test.drop(['id'], axis=1, inplace=True)
    if 'Unnamed: 0' in train.columns.values:
        train.drop(['Unnamed: 0'], axis=1, inplace=True)
        test.drop(['Unnamed: 0'], axis=1, inplace=True)

    model_seed = sum([ord(s) * (i + 1) for i, s in enumerate(label)])

    model = pipeline['model']
    pipeline = pipeline.get('munge')
    if has_cache:
        train = train[train.columns.values[0]]
        upsample = False
    elif pipeline:
        if 'upsample' in pipeline:
            pipeline = [p for p in pipeline if p != 'upsample']
            upsample = True
        else:
            upsample = False
        for step in pipeline:
            print('[%s Running %s]' % (tag, str(step)))
            col_names = train.columns.values
            if 'ApplyEncoder' in str(step):
                train, test = step.encode(train, test)
            elif 'TargetEncoder' in str(step):
                train, test = step.encode(train, test, target)
            else:
                fit_step = step.fit(train)
                train = fit_step.transform(train)
                test = fit_step.transform(test)
            if isinstance(train, np.ndarray):
                train = pd.DataFrame(train)
                test = pd.DataFrame(test)
                train.columns = col_names
                test.columns = col_names
        print('[%s Finished pipeline, train shape is %s]' % (tag, str(train.shape)))
        print('[%s Finished pipeline, test shape is %s]' % (tag, str(test.shape)))
    else:
        upsample = False

    y = target if isinstance(target, np.ndarray) else target.values 
    X = train.values
    Z = test.values

    result = np.zeros(len(train))
    fold = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)
    train_index, test_index = list(fold.split(X, y))[index]
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]

    if upsample:
        X_train_df = pd.DataFrame(X_train)
        ones = X_train_df[y_train == 1]
        for _ in range(2):
            X_train_df = pd.concat([X_train_df, ones], axis=0)
            y_train = np.append(y_train, [1 for _ in range(len(ones))])
        X_train = resample(X_train_df.values, random_state=model_seed)
        y_train = resample(y_train, random_state=model_seed)
        print('[%s Upsample completed - now have %s target distribution]' % (tag, str(np.bincount(y_train))))

    if not has_cache:
        if 'XGB' == str(model) or 'LightGBM' == str(model):
            fit_model = model.fit(X_train, y_train, X_valid, y_valid, seed=model_seed, label=label)
        elif 'ClusterDistanceClassifier' == str(model) or 'RBMClassifier' == str(model):
            fit_model = model.fit(X_train, y_train, X_valid, y_valid)
        else:
            fit_model = model.fit(X_train, y_train)
        result[test_index] = fit_model.predict_proba(X_valid)[:, 1]
        submission = fit_model.predict_proba(Z)[:, 1]
    else:
        result[test_index] = X_valid
    fold_eval = eval_fn(y_valid, result[test_index])
    if not has_cache:
        print('[%s OOF eval %f]' % (tag, fold_eval))
        pred_df = pd.DataFrame({'id': train_id, 'target': y, label: result})
        pred_df.to_csv(path, index=False)
        sub_df = pd.DataFrame({'id': test_id, 'target': submission})
        sub_df.to_csv(sub_path, index=False)
    return(label, index + 1, path, sub_path, fold_eval)


class ApplyEncoder:
    def __repr__(self):
        return 'ApplyEncoder(%s)' % str(self.encoder)

    def __init__(self, encoder):
        self.encoder = encoder
    
    def encode(self, train, test):
        if isinstance(train, np.ndarray):
            train = pd.DataFrame(train)
        if isinstance(test, np.ndarray):
            test = pd.DataFrame(test)
        for col in train.columns:
            if not np.issubdtype(train[col].dtype, np.number):
                train[col] = train[col].fillna(-1)
                test[col] = test[col].fillna(-1)
                fit_encoder = self.encoder.fit(train[col])
                train[col] = fit_encoder.transform(train[col])
                test[col] = fit_encoder.transform(test[col])
        return train.values, test.values


class TargetEncoder:
    # https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    def __repr__(self):
        return 'TargetEncoder'

    def __init__(self, cols, smoothing=1, min_samples_leaf=1, noise_level=0, keep_original=False):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.keep_original = keep_original
        
    @staticmethod
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def encode(self, train, test, target):
        for col in self.cols:
            if self.keep_original:
                train[col + '_te'], test[col + '_te'] = self.encode_column(train[col], test[col], target)
            else:
                train[col], test[col] = self.encode_column(train[col], test[col], target)
        return train, test

    def encode_column(self, trn_series, tst_series, target):
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean 
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - self.min_samples_leaf) / self.smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(['mean', 'count'], axis=1, inplace=True)
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index 
        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
        return self.add_noise(ft_trn_series, self.noise_level), self.add_noise(ft_tst_series, self.noise_level)


class ReplaceMissing:
    def __repr__(self):
        return 'ReplaceMissing'
    
    def fit(self, data):
        return self

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        data = data.fillna(-99999, axis=1)
        return data.values


class LightGBM:
    def __repr__(self):
        return 'LightGBM'

    def __init__(self, params=None):
        self.params = copy(params)
        self.feval = self.params.pop('feval')
        if self.params.get('early_stop'):
            self.early_stop = self.params.pop('early_stop')
        else:
            self.early_stop = None
        if self.params.get('meta_bags'):
            self.meta_bags = self.params.pop('meta_bags')
        else:
            self.meta_bags = 1
        if not self.params.get('objective'):
            params['objective'] = 'binary'
        if not self.params.get('metric'):
            params['metric'] = 'auc'
        if self.params.get('verbose'):
            self.verbose = self.params.pop('verbose')
        else:
            self.verbose = False
        if not self.params.get('silent'):
            params['silent'] = True
        if not self.params.get('n_jobs'):
            self.params['n_jobs'] = 1

    def fit(self, X_train, y_train, X_valid, y_valid, seed=None, label=None):
        self.label = label if label else 'LGB'
        self.models = []
        for bag in range(self.meta_bags):
            self.params['random_state'] = seed * (bag + 1)
            model = LGBMClassifier(**self.params)
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)],
                      eval_metric=self.feval,
                      early_stopping_rounds=self.early_stop,
                      verbose=self.verbose)
            # TODO: Get printout
            model.lgb_eval = model.evals_result_["valid_0"]["gini"]
            model.best_round = np.argsort(model.lgb_eval)[::-1][0]
            model.importances = model.feature_importances_
            self.models.append(model)
            print('[%s Bag %d] Found %.6f @ %d with %d estimators' 
                  % (self.label,
                     bag, 
                     model.lgb_eval[model.best_round],
                     model.best_round,
                     self.params.get('n_estimators', 100)))
        return self
    
    def predict_proba(self, Z):
        preds = np.zeros((len(Z), 2))
        for bag in range(self.meta_bags):
            model = self.models[bag]
            preds += model.predict_proba(Z, num_iteration=model.best_round)
        preds /= float(self.meta_bags)
        return preds



class XGB:
    def __repr__(self):
        return 'XGB'

    def __init__(self, params=None):
        self.params = copy(params)
        self.feval = self.params.pop('feval')
        self.feval_maximize = self.params.pop('feval_maximize')
        if self.params.get('early_stop'):
            self.early_stop = self.params.pop('early_stop')
        else:
            self.early_stop = None
        if self.params.get('meta_bags'):
            self.meta_bags = self.params.pop('meta_bags')
        else:
            self.meta_bags = 1
        if not self.params.get('objective'):
            self.params['objective'] = 'binary:logistic'
        if not self.params.get('silent'):
            self.params['silent'] = True
        if not self.params.get('nthread'):
            self.params['nthread'] = 2

    def fit(self, X_train, y_train, X_valid, y_valid, seed=42, label=None):
        self.label = label if label else 'LGB'
        self.models = []
        for bag in range(self.meta_bags):
            self.params['seed'] = seed * (bag + 1)
            model = XGBClassifier(**self.params)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=self.feval,
                      early_stopping_rounds=self.early_stop,
                      verbose=False)
            model.xgb_eval = model.evals_result_['validation_1']['gini']
            model.best_round = np.argsort(model.xgb_eval)[::-1][0]
            model.importances = model.feature_importances_
            self.models.append(model)
            print('[%s Bag %d] Found %.6f @ %d with %d estimators' 
                  % (self.label,
                     bag,
                     model.xgb_eval[model.best_round],
                     model.best_round,
                     self.params.get('n_estimators', 100)))
        return self
    
    def predict_proba(self, Z):
        preds = np.zeros((len(Z), 2))
        for bag in range(self.meta_bags):
            model = self.models[bag]
            preds += model.predict_proba(Z, ntree_limit=model.best_round)
        preds /= float(self.meta_bags)
        return preds


class DistanceClassifier:
    def __repr__(self):
        return 'DistanceClassifier'
    
    def __init__(self, strategy='min', invert=False):
        if strategy not in ['min', 'max', 'average', 'median', 'std']:
            raise ValueError('Only `min`, `max`, `average`, `median`, and `std` strategies are allowed.')
        self.strategy = strategy
        self.invert = invert

    def fit(self, X_train, y_train):
        training_data = pd.DataFrame(X_train)
        training_data['target'] = y_train
        ones = training_data[training_data['target'] == 1]
        self.ones = ones.drop('target', axis=1).values
        return self
    
    def predict_proba(self, Z):
        distances = []
        if self.strategy == 'min':
            for zi in range(Z.shape[0]):
                distances.append(min(np.sum((Z[zi, :] - self.ones) ** 2, axis=0)))
        elif self.strategy == 'max':
            for zi in range(Z.shape[0]):
                distances.append(max(np.sum((Z[zi, :] - self.ones) ** 2, axis=0)))
        elif self.strategy == 'median':
            for zi in range(Z.shape[0]):
                distances.append(np.median(np.sum((Z[zi, :] - self.ones) ** 2, axis=0)))
        elif self.strategy == 'average':
            for zi in range(Z.shape[0]):
                distances.append(np.linalg.norm(Z[zi, :] - self.ones))
        elif self.strategy == 'std':
            for zi in range(Z.shape[0]):
                distances.append(np.std(np.sum((Z[zi, :] - self.ones) ** 2, axis=0)))
        preds_class_1 = minmax_scale(scale(distances, axis=0), axis=0)
        preds = []
        for pred in preds_class_1:
            preds += [[1 - pred, pred]]
        preds = np.array(preds)
        return preds if not self.invert else -preds


class ClusterDistanceClassifier:
    def __repr__(self):
        return 'ClusterDistanceClassifier'

    def __init__(self, num_clusters, strategy, invert=False, random_state=42):
        self.kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=random_state)
        if strategy not in ['min', 'max', 'average', 'median', 'std']:
            raise ValueError('Only `min`, `max`, `average`, `median`, and `std` strategies are allowed.')
        self.strategy = strategy
        self.invert = invert

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.distances = self.kmeans.fit_transform(pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid)]))
        if self.strategy == 'min':
            self.distances = np.min(self.distances, axis=1)
        elif self.strategy == 'max':
            self.distances = np.max(self.distances, axis=1)
        elif self.strategy == 'average':
            self.distances = np.average(self.distances, axis=1)
        elif self.strategy == 'median':
            self.distances = np.median(self.distances, axis=1)
        elif self.strategy == 'std':
            self.distances = np.std(self.distances, axis=1)
        self.predict = scale(self.distances[len(X_train):])
        return self

    def predict_proba(self, Z):
        if self.invert:
            self.predict = -self.predict
        preds_class_1 = minmax_scale(self.predict)
        preds = []
        for pred in preds_class_1:
            preds += [[1 - pred, pred]]
        return np.array(preds)


class RBMClassifier:
    def __repr__(self):
        return 'RBMClassifier'

    def __init__(self, random_state=42, class_weight='balanced', verbose=True):
        self.rbm = BernoulliRBM(random_state=random_state, verbose=verbose)
        self.lr = LogisticRegression(class_weight=class_weight)

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.data = self.rbm.fit_transform(pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_valid)]))
        self.train = self.data[:len(X_train)]
        self.fit_lr = self.lr.fit(self.train, y_train)
        self.predict = self.data[len(X_train):]
        return self

    def predict_proba(self, Z):
        return self.fit_lr.predict_proba(self.predict)


class AverageColumns:
    def __repr__(self):
        return 'AverageColumns'

    def fit(self, X_train, y_train):
        return self

    def predict_proba(self, Z):
        preds_class_1 = np.mean(Z, axis=1)
        preds = []
        for pred in preds_class_1:
            preds += [[1 - pred, pred]]
        return np.array(preds)


class Ridge:
    def __repr__(self):
        return 'Ridge'

    def __init__(self, alpha, class_weight, random_state):
        self.ridge = RidgeClassifier(alpha, class_weight=class_weight, fit_intercept=False, random_state=random_state)

    def fit(self, X_train, y_train):
        self.ridge.fit(X_train, y_train)
        return self

    def predict_proba(self, Z):
        preds_class_1 = self.ridge.decision_function(Z)
        preds = []
        for pred in preds_class_1:
            preds += [[1 - pred, pred]]
        return np.array(preds)



# Start
start = datetime.now()
print('...Starting at ' + str(start))


# Read clean train data
print('Reading...')
if not os.path.isfile('train_clean.csv'):
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    # Drop most _calc features
    train.drop([c for c in train.columns if c.startswith('ps_calc_') and c not in ['ps_calc_09', 'ps_calc_05']], axis=1, inplace=True)
    test.drop([c for c in test.columns if c.startswith('ps_calc_') and c not in ['ps_calc_09', 'ps_calc_05']], axis=1, inplace=True)
    # Drop empty features
    train.drop(['ps_ind_10_bin', 'ps_ind_13_bin'], axis=1, inplace=True)
    test.drop(['ps_ind_10_bin', 'ps_ind_13_bin'], axis=1, inplace=True)
    # Drop boring features
    train.drop(['ps_car_10_cat', 'ps_ind_11_bin'], axis=1, inplace=True)
    test.drop(['ps_car_10_cat', 'ps_ind_11_bin'], axis=1, inplace=True)
    # Convert -1 to NA
    train.replace(-1, np.NaN)
    test.replace(-1, np.NaN)
    # Convert categorical to factor
    for col in [c for c in train.columns if 'cat' in c]:
        train[col] = train[col].astype('str')
        test[col] = test[col].astype('str')
    # Create combinations
    train['ps_reg_01_plus_ps_car_02_cat'] = train['ps_reg_01'].astype('str') + '_' + train['ps_car_02_cat']
    train['ps_reg_01_plus_ps_car_04_cat'] = train['ps_reg_01'].astype('str') + '_' + train['ps_car_04_cat']
    test['ps_reg_01_plus_ps_car_02_cat'] = test['ps_reg_01'].astype('str') + '_' + test['ps_car_02_cat']
    test['ps_reg_01_plus_ps_car_04_cat'] = test['ps_reg_01'].astype('str') + '_' + test['ps_car_04_cat']
    # Label Encode
    for col in train.columns:
        if not np.issubdtype(train[col].dtype, np.number):
            train[col] = train[col].fillna(-1)
            test[col] = test[col].fillna(-1)
            fit_encoder = LabelEncoder().fit(list(train[col]) + list(test[col]))
            train[col] = fit_encoder.transform(train[col])
            test[col] = fit_encoder.transform(test[col])
    # Add KMeans
    imputer = Imputer()
    train_ = train.drop(['target', 'id'], axis=1)
    cat_cols = [c for c, col in enumerate(train_.columns) if 'cat' in col]
    test_ = test.drop('id', axis=1)
    imputer = imputer.fit(train_)
    train_ = imputer.transform(train_)
    test_ = imputer.transform(test_)
    ohe = OneHotEncoder(categorical_features=cat_cols, sparse=False)
    ohe = ohe.fit(train_)
    train_ = ohe.transform(train_)
    test_ = ohe.transform(test_)
    scaler = StandardScaler()
    scaler = scaler.fit(train_)
    train_ = scaler.transform(train_)
    test_ = scaler.transform(test_)
    kmeans = MiniBatchKMeans(n_clusters=200, random_state=42)
    kmeans = kmeans.fit(train_)
    train['kmeans_cat'] = kmeans.predict(train_)
    test['kmeans_cat'] = kmeans.predict(test_)
    # Save
    train.to_csv('train_clean.csv', index=False)
    test.to_csv('test_clean.csv', index=False)
else:
    train = pd.read_csv('train_clean.csv')
    test = pd.read_csv('test_clean.csv')
train_id = train['id']
test_id = test['id']
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)


# Models
lgbm_params3 = {'n_estimators': 900, 'is_unbalance': 'true', 'learning_rate': 0.05, 'max_depth': 3, 'num_leaves': 8, 'subsample': 0.8, 'colsample_bytree': 0.2, 'min_child_samples': 300, 'feval': gini_lgb}
lgbm_params_3_2 = {'n_estimators': 2500, 'is_unbalance': 'false', 'learning_rate': 0.01, 'max_depth': 3, 'num_leaves': 6, 'subsample': 0.8, 'colsample_bytree': 0.2, 'min_child_samples': 10, 'feval': gini_lgb}
lgbm_params4 = {'n_estimators': 900, 'is_unbalance': 'true', 'learning_rate': 0.05, 'max_depth': 4, 'num_leaves': 14, 'subsample': 0.8, 'colsample_bytree': 0.2, 'min_child_samples': 500, 'feval': gini_lgb}
lgbm_params5 = {'n_estimators': 500, 'is_unbalance': 'true', 'learning_rate': 0.04, 'max_depth': 5, 'num_leaves': 31, 'subsample': 1.0, 'colsample_bytree': 0.2, 'min_child_samples': 2000, 'feval': gini_lgb}
lgbm_params6 = {'n_estimators': 500, 'is_unbalance': 'false', 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 64, 'subsample': 1.0, 'colsample_bytree': 0.2, 'min_child_samples': 2000, 'feval': gini_lgb}
lgbm_params_6_2 = {'n_estimators': 500, 'is_unbalance': 'false', 'learning_rate': 0.04, 'max_depth': 6, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.2, 'min_child_samples': 1000, 'feval': gini_lgb}
lgbm_params_6_3 = {'n_estimators': 1500, 'is_unbalance': 'false', 'learning_rate': 0.01, 'max_depth': 6, 'num_leaves': 31, 'subsample': 0.9, 'colsample_bytree': 0.3, 'min_child_samples': 1000, 'feval': gini_lgb}
lgbm_params_6_4 = {'n_estimators': 1500, 'is_unbalance': 'true', 'learning_rate': 0.01, 'max_depth': 6, 'num_leaves': 31, 'subsample': 0.9, 'colsample_bytree': 0.3, 'min_child_samples': 1000, 'feval': gini_lgb}
lgbm_params7 = {'n_estimators': 900, 'is_unbalance': 'false', 'learning_rate': 0.03, 'max_depth': 7, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.2, 'min_child_samples': 500, 'scale_pos_weight': 3, 'feval': gini_lgb}
lgbm_params8 = {'n_estimators': 2200, 'is_unbalance': 'false', 'learning_rate': 0.01, 'max_depth': 8, 'num_leaves': 31, 'subsample': 0.9, 'colsample_bytree': 0.3, 'min_child_samples': 10, 'feval': gini_lgb}
lgbm_params_8_2 = {'n_estimators': 1500, 'is_unbalance': 'false', 'learning_rate': 0.01, 'max_depth': 8, 'num_leaves': 51, 'subsample': 0.9, 'colsample_bytree': 0.2, 'min_child_samples': 1000, 'scale_pos_weight': 5, 'feval': gini_lgb}
lgbm_params9 = {'n_estimators': 1200, 'is_unbalance': 'false', 'learning_rate': 0.03, 'max_depth': 9, 'num_leaves': 31, 'subsample': 1.0, 'colsample_bytree': 0.2, 'min_child_samples': 1000, 'scale_pos_weight': 3, 'feval': gini_lgb}
lgbm_params10 = {'n_estimators': 800, 'is_unbalance': 'false', 'learning_rate': 0.02, 'max_depth': 10, 'num_leaves': 31, 'subsample': 0.9, 'colsample_bytree': 0.4, 'min_child_samples': 1000, 'scale_pos_weight': 3, 'feval': gini_lgb}
lgbm_params11 = {'n_estimators': 800, 'is_unbalance': 'false', 'learning_rate': 0.02, 'max_depth': 11, 'num_leaves': 31, 'subsample': 0.9, 'colsample_bytree': 0.4, 'min_child_samples': 1000, 'scale_pos_weight': 3, 'feval': gini_lgb}

xgb_params3 = {'n_estimators': 2500, 'colsample_bytree': 0.5, 'learning_rate': 0.01, 'gamma': 1, 'max_depth': 3, 'scale_pos_weight': 2, 'min_child_weight': 12, 'subsample': 0.8, 'feval': gini_xgb, 'feval_maximize': True}
xgb_params4 = {'n_estimators': 2000, 'colsample_bytree': 0.3, 'learning_rate': 0.01, 'gamma': 30, 'max_depth': 4, 'scale_pos_weight': 7, 'min_child_weight': 5, 'subsample': 0.8 , 'feval': gini_xgb, 'feval_maximize': True}
xgb_params_4_2 = {'n_estimators': 2400, 'colsample_bytree': 0.5, 'learning_rate': 0.005, 'gamma': 30, 'max_depth': 4, 'scale_pos_weight': 13, 'min_child_weight': 12, 'subsample': 0.9 , 'feval': gini_xgb, 'feval_maximize': True}
xgb_params_4_3 = {'n_estimators': 1400, 'colsample_bytree': 0.5, 'learning_rate': 0.02, 'gamma': 30, 'max_depth': 4, 'scale_pos_weight': 13, 'min_child_weight': 12, 'subsample': 0.9 , 'feval': gini_xgb, 'feval_maximize': True}
xgb_params5 = {'n_estimators': 1500, 'colsample_bytree': 0.8, 'learning_rate': 0.01, 'gamma': 5, 'max_depth': 5, 'scale_pos_weight': 3, 'min_child_weight': 5, 'subsample': 0.8, 'feval': gini_xgb, 'feval_maximize': True}
xgb_params6 = {'n_estimators': 1600, 'colsample_bytree': 0.3, 'learning_rate': 0.02, 'gamma': 20, 'max_depth': 6, 'scale_pos_weight': 3, 'min_child_weight': 5, 'subsample': 0.8, 'feval': gini_xgb, 'feval_maximize': True}
xgb_params7 = {'n_estimators': 800, 'colsample_bytree': 0.4, 'learning_rate': 0.03, 'gamma': 15, 'max_depth': 7, 'scale_pos_weight': 2, 'min_child_weight': 6, 'subsample': 0.9, 'feval': gini_xgb, 'feval_maximize': True}
xgb_params_7_2 = {'n_estimators': 1500, 'colsample_bytree': 0.4, 'learning_rate': 0.01, 'gamma': 20, 'max_depth': 7, 'scale_pos_weight': 4, 'min_child_weight': 12, 'subsample': 0.8, 'feval': gini_xgb, 'feval_maximize': True}
xgb_params8 = {'n_estimators': 300, 'colsample_bytree': 0.8, 'learning_rate': 0.04, 'gamma': 15, 'max_depth': 8, 'scale_pos_weight': 2, 'min_child_weight': 10, 'subsample': 0.8, 'feval': gini_xgb, 'feval_maximize': True}
xgb_params9 = {'n_estimators': 800, 'colsample_bytree': 0.6, 'learning_rate': 0.04, 'gamma': 20, 'max_depth': 9, 'scale_pos_weight': 3, 'min_child_weight': 10, 'subsample': 0.8, 'feval': gini_xgb, 'feval_maximize': True}
xgb_params10 = {'n_estimators': 600, 'colsample_bytree': 0.5, 'learning_rate': 0.03, 'gamma': 25, 'max_depth': 10, 'scale_pos_weight': 3, 'min_child_weight': 10, 'subsample': 0.8, 'feval': gini_xgb, 'feval_maximize': True}
xgb_params11 = {'n_estimators': 600, 'colsample_bytree': 0.5, 'learning_rate': 0.03, 'gamma': 25, 'max_depth': 11, 'scale_pos_weight': 3, 'min_child_weight': 10, 'subsample': 0.8, 'feval': gini_xgb, 'feval_maximize': True}

rf_params = {'n_estimators': 400, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 0.2,
             'max_leaf_nodes': 500, 'n_jobs': 3, 'random_state': 43, 'class_weight': {0: 1, 1: 3}}

catboost_params7 = {'iterations': 900, 'depth': 7, 'learning_rate': 0.02, 'random_seed': 47, 'thread_count': 1}

catboost_params8 = {'iterations': 1000, 'depth': 8, 'rsm': 0.95, 'learning_rate': 0.03,
                   'l2_leaf_reg': 3.5, 'gradient_iterations': 4, 'random_seed': 46,
                   'thread_count': 1}

# rfg_params = {'max_leaf': 2000, 'learning_rate': 0.5, 'algorithm': 'RGF_Sib', 'test_interval': 100,
#               'min_samples_leaf': 3, 'reg_depth': 1.0, 'l2': 0.5, 'sl2': 0.005}


cat_cols = [c for c, col in enumerate(train.columns) if 'cat' in col]
f_cats = [f for f in train.columns if "_cat" in f]
ohe = OneHotEncoder(categorical_features=cat_cols, sparse=False)
target_encode = TargetEncoder(min_samples_leaf=200, smoothing=10, noise_level=0,
                              keep_original=True, cols=f_cats)

base_models = {'LGBM3': {'munge': [target_encode], 'model': LightGBM(lgbm_params3)},
               'LGBM-3-2': {'munge': [target_encode], 'model': LightGBM(lgbm_params_3_2)},
               'LGBM4': {'munge': [target_encode], 'model': LightGBM(lgbm_params4)},
               'LGBM5': {'munge': [target_encode], 'model': LightGBM(lgbm_params5)},
               'LGBM6': {'munge': [target_encode], 'model': LightGBM(lgbm_params6)},
               'LGBM-6-2': {'munge': [target_encode], 'model': LightGBM(lgbm_params_6_2)},
               'LGBM-6-3': {'munge': [target_encode], 'model': LightGBM(lgbm_params_6_3)},
               'LGBM-6-4': {'munge': [target_encode], 'model': LightGBM(lgbm_params_6_4)},
               'LGBM7': {'munge': [target_encode], 'model': LightGBM(lgbm_params7)},
               'LGBM8': {'munge': [target_encode], 'model': LightGBM(lgbm_params8)},
               'LGBM-8-2': {'munge': [target_encode], 'model': LightGBM(lgbm_params_8_2)},
               'LGBM9': {'munge': [target_encode], 'model': LightGBM(lgbm_params9)},
               'LGBM10': {'munge': [target_encode], 'model': LightGBM(lgbm_params10)},
               'LGBM11': {'munge': [target_encode], 'model': LightGBM(lgbm_params11)},
               'XGB3': {'munge': [target_encode], 'model': XGB(xgb_params3)},
               'XGB4': {'munge': [target_encode], 'model': XGB(xgb_params4)},
               'XGB-4-2': {'munge': [target_encode], 'model': XGB(xgb_params_4_2)},
               'XGB5': {'munge': [target_encode], 'model': XGB(xgb_params5)},
               'XGB6': {'munge': [target_encode], 'model': XGB(xgb_params6)},
               'XGB7': {'munge': [target_encode], 'model': XGB(xgb_params7)},
               'XGB-7-2': {'munge': [target_encode], 'model': XGB(xgb_params_7_2)},
               'XGB8': {'munge': [target_encode], 'model': XGB(xgb_params8)},
               'XGB9': {'munge': [target_encode], 'model': XGB(xgb_params9)},
               'XGB10': {'munge': [target_encode], 'model': XGB(xgb_params10)},
               'XGB11': {'munge': [target_encode], 'model': XGB(xgb_params11)},
               # 'LR': {'munge': [target_encode, Imputer(), StandardScaler()],
               #        'model': LogisticRegression(class_weight={0:1, 1:3}, C=1)},
               # 'RF': {'munge': [target_encode, Imputer()],
               #        'model': RandomForestClassifier(**rf_params)},
               # 'ET7': {'munge': [target_encode, ReplaceMissing(), 'upsample'],
               #         'model': ExtraTreesClassifier(300, max_features=0.8, max_depth=7)},
               # 'ET8': {'munge': [target_encode, ReplaceMissing(), 'upsample'],
               #         'model': ExtraTreesClassifier(200, max_features=0.8, max_depth=8)},
               # 'ET9': {'munge': [target_encode, ReplaceMissing(), 'upsample'],
               #         'model': ExtraTreesClassifier(100, max_features=0.8, max_depth=9)},
               'Cat7': {'munge': [ReplaceMissing(), 'upsample'],
                       'model': CatBoostClassifier(**catboost_params7)},
               'Cat8': {'munge': [ReplaceMissing()],
                        'model': CatBoostClassifier(**catboost_params8)},
               'GBM4': {'munge': [target_encode, ReplaceMissing(), 'upsample'],
                        'model': GradientBoostingClassifier(max_depth=4, random_state=244, max_features=0.9, subsample=0.9)},
               'GBM5': {'munge': [target_encode, ReplaceMissing(), 'upsample'],
                        'model': GradientBoostingClassifier(max_depth=5, random_state=242)},
               'GBM6': {'munge': [target_encode, ReplaceMissing(), 'upsample'],
                        'model': GradientBoostingClassifier(max_depth=6, random_state=243, max_features=0.7, subsample=0.9)},
               'Adaboost': {'munge': [ReplaceMissing(), 'upsample'],
                            'model': AdaBoostClassifier(n_estimators=300, random_state=245)},
               'AdaboostT': {'munge': [target_encode, ReplaceMissing(), 'upsample'],
                             'model': AdaBoostClassifier(n_estimators=300, random_state=246)},
               # 'AvgDist': {'munge': [Imputer(), ohe, StandardScaler()],
               #             'model': DistanceClassifier(strategy='average')},
               # 'MedDist': {'munge': [Imputer(), ohe, StandardScaler()],
               #             'model': DistanceClassifier(strategy='median', invert=True)},
               # 'StdDist': {'munge': [Imputer(), ohe, StandardScaler()],
               #             'model': DistanceClassifier(strategy='std')},
               # 'MaxClusterDist': {'munge': [Imputer(), ohe, StandardScaler()],
               #                    'model': ClusterDistanceClassifier(200, strategy='max')},
               # 'AvgClusterDist': {'munge': [Imputer(), ohe, StandardScaler()],
               #                    'model': ClusterDistanceClassifier(200, strategy='average')},
               # 'MedClusterDist': {'munge': [Imputer(), ohe, StandardScaler()],
               #                    'model': ClusterDistanceClassifier(200, strategy='median')},
               # 'StdClusterDist': {'munge': [Imputer(), ohe, StandardScaler()],
               #                    'model': ClusterDistanceClassifier(200, strategy='std', invert=True)},
               # 'MinClusterDist': {'munge': [Imputer(), ohe, StandardScaler()],
               #                    'model': ClusterDistanceClassifier(200, strategy='min')},
               # 'MaxClusterDist': {'munge': [Imputer(), ohe, StandardScaler()],
               #                    'model': ClusterDistanceClassifier(200, strategy='max')},
               # 'RBM': {'munge': [Imputer(), ohe, StandardScaler()],
               #        'model': RBMClassifier()},
               'MLP6-Bag': {'munge': [target_encode, Imputer(), StandardScaler(), 'upsample'],
                            'model': BaggingClassifier(MLPClassifier((6,), random_state=344), random_state=345)},
               'MLP7-Bag': {'munge': [target_encode, Imputer(), StandardScaler(), 'upsample'],
                            'model': BaggingClassifier(MLPClassifier((7,), alpha = 0.01, random_state=345), random_state=346)},
               'MLP5-Bag': {'munge': [target_encode, Imputer(), StandardScaler(), 'upsample'],
                            'model': BaggingClassifier(MLPClassifier((5,), random_state=346), random_state=347)}}


lgbm_params = {'n_estimators': 400, 'is_unbalance': 'true', 'learning_rate': 0.01, 'max_depth': 3, 'num_leaves': 31, 'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_samples': 10, 'meta_bags': 10, 'feval': gini_lgb}
xgb_params3 = {'n_estimators': 300, 'colsample_bytree': 0.9, 'learning_rate': 0.005, 'gamma': 1, 'max_depth': 3, 'scale_pos_weight': 3, 'subsample': 1.0, 'meta_bags': 10, 'feval': gini_xgb, 'feval_maximize': True}
xgb_params4 = {'n_estimators': 300, 'colsample_bytree': 0.9, 'learning_rate': 0.01, 'gamma': 1, 'max_depth': 4, 'scale_pos_weight': 3, 'subsample': 1.0, 'meta_bags': 10, 'feval': gini_xgb, 'feval_maximize': True}
xgb_params5 = {'n_estimators': 300, 'colsample_bytree': 0.9, 'learning_rate': 0.01, 'gamma': 1, 'max_depth': 5, 'scale_pos_weight': 3, 'subsample': 1.0, 'meta_bags': 10, 'feval': gini_xgb, 'feval_maximize': True}
xgb_params6 = {'n_estimators': 300, 'colsample_bytree': 0.9, 'learning_rate': 0.01, 'gamma': 1, 'max_depth': 6, 'scale_pos_weight': 3, 'subsample': 1.0, 'meta_bags': 10, 'feval': gini_xgb, 'feval_maximize': True}
rf_params = {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_features': 0.9, 'max_leaf_nodes': 500, 'n_jobs': 2, 'random_state': 42, 'class_weight': {0: 1, 1: 3}}

level_2_models = {'Lvl2-LGBM': {'munge': [StandardScaler()], 'model': LightGBM(lgbm_params)},
                  'Lvl2-XGB3': {'munge': [StandardScaler()], 'model': XGB(xgb_params3)},
                  'Lvl2-XGB4': {'munge': [StandardScaler()], 'model': XGB(xgb_params4)},
                  'Lvl2-XGB5': {'munge': [StandardScaler()], 'model': XGB(xgb_params5)},
                  'Lvl2-XGB6': {'munge': [StandardScaler()], 'model': XGB(xgb_params6)},
                  # 'Lvl2-RF': {'model': RandomForestClassifier(**rf_params)},
                  # 'Lvl2-ET': {'model': ExtraTreesClassifier(100, max_features=0.8, max_depth=10, class_weight={0: 1, 1: 3})},
                  # 'Lvl2-GBM': {'munge': ['upsample'],
                  #              'model': BaggingClassifier(GradientBoostingClassifier(max_depth=4, random_state=642), n_estimators=5)},
                  'Lvl2-MLP4-Bag': {'munge': [StandardScaler(), 'upsample'],
                                    'model': BaggingClassifier(MLPClassifier((4,), random_state=644), random_state=645, n_estimators=80)},
                  'Lvl2-MLP7-Bag': {'munge': [StandardScaler(), 'upsample'],
                                    'model': BaggingClassifier(MLPClassifier((7,), alpha=0.01, random_state=644), random_state=645, n_estimators=80)},
                  'Lvl2-MLP11-Bag': {'munge': [StandardScaler(), 'upsample'],
                                     'model': BaggingClassifier(MLPClassifier((11,), alpha=0.01, random_state=644), random_state=645, n_estimators=80)},
                  'Lvl2-LR': {'munge': [StandardScaler()],
                              'model': LogisticRegression(class_weight={0:1, 1:3}, C=0.011)}} #,
                  # 'Lvl2-RankAverage': {'munge': [StandardScaler()], 'model': AverageColumns()}}
                  # 'Lvl2-Ridge': {'munge': [StandardScaler()],
                  #                'model': Ridge(90, class_weight={0:1, 1:3}, random_state=44)}}

level_3_models = {'Lvl3-LR': {'munge': [StandardScaler()],
                              'model': LogisticRegression(class_weight={0:1, 1:3}, C=0.011)},
                  'Lvl3-MLP2-Bag': {'munge': [StandardScaler(), 'upsample'],
                                    'model': BaggingClassifier(MLPClassifier((2,), random_state=846), random_state=845, n_estimators=80)},
                  'Lvl3-MLP3-Bag': {'munge': [StandardScaler(), 'upsample'],
                                    'model': BaggingClassifier(MLPClassifier((3,), random_state=846), random_state=845, n_estimators=80)},
                  'Lvl3-MLP4-Bag': {'munge': [StandardScaler(), 'upsample'],
                                    'model': BaggingClassifier(MLPClassifier((4,), random_state=846), random_state=845, n_estimators=80)},
                  'Lvl3-RankAverage': {'munge': [StandardScaler()], 'model': AverageColumns()},
                  'Lvl3-Average': {'model': AverageColumns()}}


n_cpu = mp.cpu_count()
n_nodes = max(int(n_cpu * 0.7), 1)
print('Starting a jobs server with %d nodes' % n_nodes)
pool = mp.ProcessingPool(n_nodes, maxtasksperchild=500)

jobs = [{'index': i, 'label': label, 'pipeline': pipeline} for i in range(KFOLDS) for (j, (label, pipeline)) in enumerate(base_models.items())]
jobs = sorted(jobs, key=lambda x: x['label'])

print('Queueing %d jobs' % len(jobs))
base = pool.map(lambda dat: cv_pipeline(label=dat['label'], pipeline=dat['pipeline'], index=dat['index'], eval_fn=gini_normalized, kfolds=KFOLDS), jobs)
print('Initial base models trained! Calculating results...')
models = np.unique(map(lambda x: x[0], base))
base_results = [(model, sum([pd.read_csv(x[2])[model] for x in base if x[0] == model])) for model in models]
base_df = pd.DataFrame({s[0]: s[1] for s in base_results})
mean_preds = np.mean(base_df, axis=1)
for model in models:
    ginis = [x[4] for x in base if x[0] == model and x[4]]
    print('%s: %.5f (SD: %.5f, [%s]), DV %.4f' % (model, np.mean(ginis), np.std(ginis), ', '.join(['%.4f' % g for g in ginis]), 1 - np.corrcoef(base_df[model], mean_preds)[1,0]))
base_df['target'] = train['target']
base_df['id'] = train_id
base_df.to_csv('base_df.csv', index=False)
if SHOULD_SUBMIT:
    base_submissions = [(model, np.mean([pd.read_csv(x[3])['target'] for x in base if x[0] == model], axis=0)) for model in models]
    for s in base_submissions:
        pd.DataFrame({'id': test_id, 'target': s[1]}).to_csv('/dev/cache/submission-' + s[0] + '.csv', index=False)
    base_submissions = pd.DataFrame({s[0]: s[1] for s in base_submissions})
    base_submissions['id'] = test_id
    base_submissions.to_csv('base_submissions.csv', index=False)
pool.close()
pool.join()
pool.terminate()
pool.restart()


jobs2 = [{'index': i, 'label': label, 'pipeline': pipeline} for i in range(KFOLDS) for (j, (label, pipeline)) in enumerate(sorted(level_2_models.items()))]
print('Queueing %d jobs' % len(jobs2))

level_2 = pool.map(lambda dat: cv_pipeline(label=dat['label'], pipeline=dat['pipeline'], index=dat['index'], eval_fn=gini_normalized, kfolds=KFOLDS), jobs2)
print('Initial level_2 models trained! Calculating results...')
models = np.unique(map(lambda x: x[0], level_2))
level_2_results = [(model, sum([pd.read_csv(x[2])[model] for x in level_2 if x[0] == model])) for model in models]
level_2_df = pd.DataFrame({s[0]: s[1] for s in level_2_results})
mean_preds = np.mean(level_2_df, axis=1)
for model in models:
    ginis = [x[4] for x in level_2 if x[0] == model and x[4]]
    print('%s: %.5f (SD: %.5f, [%s]), DV %.4f' % (model, np.mean(ginis), np.std(ginis), ', '.join(['%.4f' % g for g in ginis]), 1 - np.corrcoef(level_2_df[model], mean_preds)[1,0]))
level_2_df['target'] = train['target']
level_2_df['id'] = train_id
level_2_df.to_csv('level_2_df.csv', index=False)
if SHOULD_SUBMIT:
    level_2_submissions = [(model, np.mean([pd.read_csv(x[3])['target'] for x in level_2 if x[0] == model], axis=0)) for model in models]
    for s in level_2_submissions:
        pd.DataFrame({'id': test_id, 'target': s[1]}).to_csv('/dev/cache/submission-' + s[0] + '.csv', index=False)
    level_2_submissions = pd.DataFrame({s[0]: s[1] for s in level_2_submissions})
    level_2_submissions['id'] = test_id
    level_2_submissions.to_csv('level_2_submissions.csv', index=False)
pool.close()
pool.join()
pool.terminate()
pool.restart()


jobs3 = [{'index': i, 'label': label, 'pipeline': pipeline} for i in range(KFOLDS) for (j, (label, pipeline)) in enumerate(sorted(level_3_models.items()))]
print('Queueing %d jobs' % len(jobs3))

level_3 = pool.map(lambda dat: cv_pipeline(label=dat['label'], pipeline=dat['pipeline'], index=dat['index'], eval_fn=gini_normalized, kfolds=KFOLDS), jobs3)
print('Initial level_3 models trained! Calculating results..')
models = np.unique(map(lambda x: x[0], level_3))
level_3_results = [(model, np.mean([pd.read_csv(x[2])[model] for x in level_3 if x[0] == model], axis=0)) for model in models]
level_3_df = pd.DataFrame({s[0]: s[1] for s in level_3_results})
mean_preds = np.mean(level_3_df, axis=1)
for model in models:
    ginis = [x[4] for x in level_3 if x[0] == model and x[4]]
    print('%s: %.5f (SD: %.5f, [%s]), DV %.4f' % (model, np.mean(ginis), np.std(ginis), ', '.join(['%.4f' % g for g in ginis]), 1 - np.corrcoef(level_3_df[model], mean_preds)[1,0]))
if SHOULD_SUBMIT:
    level_3_submissions = [(model, sum([pd.read_csv(x[3])['target'] for x in level_3 if x[0] == model])) for model in models]
    for s in level_3_submissions:
        pd.DataFrame({'id': test_id, 'target': s[1]}).to_csv('/dev/cache/submission-' + s[0] + '.csv', index=False)
    level_3_submissions = pd.DataFrame({s[0]: s[1] for s in level_3_submissions})
    level_3_submissions['id'] = test_id
    level_3_submissions.to_csv('level_3_submissions.csv', index=False)
pool.close()
pool.join()
pool.terminate()

end = datetime.now()
print('...Ending at ' + str(end) + '... Total time: ' + str((end - start).total_seconds()) + 'sec')
import pdb
pdb.set_trace()
