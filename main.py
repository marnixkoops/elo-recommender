import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import itertools
from datetime import datetime
import time
from contextlib import contextmanager

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')
np.random.seed(4590)

############################################################################################
#                                                                      [+]  PIPELINE MANAGER
############################################################################################


def pipeline_manager():
    """Initiate a function to process the pipeline
    """
    global timer

    @contextmanager
    def timer(title):
        t0 = time.time()
        yield
        print('\n[✓] Finished {} in {:.0f}s at {}'.format(
            title, time.time() - t0, datetime.now().replace(second=0, microsecond=0)))


pipeline_manager()


############################################################################################
#                                                                      [+]  LOAD & PREP DATA
############################################################################################

path = '/Users/marnix/Desktop/ELO Recommender'


def downcast_datatypes(df):
    """Reduce memory footprint of dataframes by downcasting the datatype of each column.

    :param pandas df: pandas dataframe `df`.
    :return: Pandas dataframe with optimized datatypes.
    :rtype: pandas dataframe

    """
    float_cols = df.select_dtypes(include=['float'])
    int_cols = df.select_dtypes(include=['int'])

    for cols in float_cols.columns:
        df[cols] = pd.to_numeric(df[cols], downcast='float')
    for cols in int_cols.columns:
        df[cols] = pd.to_numeric(df[cols], downcast='integer')


def load_raw_data():
    """Load raw competition data.
    """
    df_train = pd.read_csv(path + '/data/train.csv')
    df_test = pd.read_csv(path + '/data/test.csv')
    df_hist_trans = pd.read_csv(path + '/data/historical_transactions.csv')
    df_new_merchant_trans = pd.read_csv(path +
                                        '/data/new_merchant_transactions.csv')

    downcast_datatypes(df_train)
    downcast_datatypes(df_test)
    downcast_datatypes(df_hist_trans)
    downcast_datatypes(df_new_merchant_trans)

    return df_train, df_test, df_hist_trans, df_new_merchant_trans

############################################################################################
#                                                                   [+]  FEATURE ENGINEERING
############################################################################################


def process_features(df_train, df_test, df_hist_trans, df_new_merchant_trans):
    """Generate and aggregate features.

    :param type df_train: Description of parameter `df_train`.
    :param type df_test: Description of parameter `df_test`.
    :param type df_hist_trans: Description of parameter `df_hist_trans`.
    :param type df_new_merchant_trans: Description of parameter `df_new_merchant_trans`.
    :return: Description of returned object.
    :rtype: type

    """
    for df in [df_hist_trans, df_new_merchant_trans]:
        df['category_2'].fillna(1.0, inplace=True)
        df['category_3'].fillna('A', inplace=True)
        df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)

    def get_new_columns(name, aggs):
        return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

    for df in [df_hist_trans, df_new_merchant_trans]:
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df['year'] = df['purchase_date'].dt.year
        df['weekofyear'] = df['purchase_date'].dt.weekofyear
        df['month'] = df['purchase_date'].dt.month
        df['dayofweek'] = df['purchase_date'].dt.dayofweek
        df['weekend'] = (df.purchase_date.dt.weekday >= 5).astype(int)
        df['hour'] = df['purchase_date'].dt.hour
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})
        df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
        df['month_diff'] = (
            (datetime.today() - df['purchase_date']).dt.days) // 30
        df['month_diff'] += df['month_lag']

    aggs = {}
    for col in [
            'month', 'hour', 'weekofyear', 'dayofweek', 'year', 'subsector_id',
            'merchant_id', 'merchant_category_id']:
        aggs[col] = ['nunique']

    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var']
    aggs['installments'] = ['sum', 'max', 'min', 'mean', 'var']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var']
    aggs['month_diff'] = ['mean']
    aggs['authorized_flag'] = ['sum', 'mean']
    aggs['weekend'] = ['sum', 'mean']
    aggs['category_1'] = ['sum', 'mean']
    aggs['card_id'] = ['size']

    for col in ['category_2', 'category_3']:
        df_hist_trans[col + '_mean'] = df_hist_trans.groupby(
            [col])['purchase_amount'].transform('mean')
        aggs[col + '_mean'] = ['mean']

    new_columns = get_new_columns('hist', aggs)
    df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs)
    df_hist_trans_group.columns = new_columns
    df_hist_trans_group.reset_index(drop=False, inplace=True)
    df_hist_trans_group['hist_purchase_date_diff'] = (
        df_hist_trans_group['hist_purchase_date_max'] -
        df_hist_trans_group['hist_purchase_date_min']).dt.days
    df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff'] / \
        df_hist_trans_group['hist_card_id_size']
    df_hist_trans_group['hist_purchase_date_uptonow'] = (
        datetime.today() -
        df_hist_trans_group['hist_purchase_date_max']).dt.days
    df_train = df_train.merge(df_hist_trans_group, on='card_id', how='left')
    df_test = df_test.merge(df_hist_trans_group, on='card_id', how='left')
    del df_hist_trans_group
    gc.collect()

    aggs = {}
    for col in [
            'month', 'hour', 'weekofyear', 'dayofweek', 'year', 'subsector_id',
            'merchant_id', 'merchant_category_id']:
        aggs[col] = ['nunique']
        aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var']
        aggs['installments'] = ['sum', 'max', 'min', 'mean', 'var']
        aggs['purchase_date'] = ['max', 'min']
        aggs['month_lag'] = ['max', 'min', 'mean', 'var']
        aggs['month_diff'] = ['mean']
        aggs['weekend'] = ['sum', 'mean']
        aggs['category_1'] = ['sum', 'mean']
        aggs['card_id'] = ['size']

    for col in ['category_2', 'category_3']:
        df_new_merchant_trans[col + '_mean'] = df_new_merchant_trans.groupby(
            [col])['purchase_amount'].transform('mean')
        aggs[col + '_mean'] = ['mean']

    new_columns = get_new_columns('new_hist', aggs)
    df_hist_trans_group = df_new_merchant_trans.groupby('card_id').agg(aggs)
    df_hist_trans_group.columns = new_columns
    df_hist_trans_group.reset_index(drop=False, inplace=True)
    df_hist_trans_group['new_hist_purchase_date_diff'] = (
        df_hist_trans_group['new_hist_purchase_date_max'] -
        df_hist_trans_group['new_hist_purchase_date_min']).dt.days
    df_hist_trans_group['new_hist_purchase_date_average'] = df_hist_trans_group['new_hist_purchase_date_diff'] / \
        df_hist_trans_group['new_hist_card_id_size']
    df_hist_trans_group['new_hist_purchase_date_uptonow'] = (
        datetime.today() -
        df_hist_trans_group['new_hist_purchase_date_max']).dt.days
    df_train = df_train.merge(df_hist_trans_group, on='card_id', how='left')
    df_test = df_test.merge(df_hist_trans_group, on='card_id', how='left')
    del df_hist_trans_group
    gc.collect()

    del df_hist_trans
    gc.collect()
    del df_new_merchant_trans
    gc.collect()
    df_train.head(5)

    df_train['bucket_target'] = 0
    df_train.loc[df_train['target'] <= -10, 'bucket_target'] = 1
    df_train.loc[(df_train['target'] > -10) & (df_train['target'] <= -5), 'bucket_target'] = 2
    df_train.loc[(df_train['target'] > -5) & (df_train['target'] < 0), 'bucket_target'] = 3
    df_train.loc[(df_train['target'] >= 0) & (df_train['target'] <= 5), 'bucket_target'] = 4
    df_train.loc[df_train['target'] > 5, 'bucket_target'] = 5
    df_train['bucket_target'].value_counts()

    for df in [df_train, df_test]:
        df['first_active_month'] = pd.to_datetime(df['first_active_month'])
        df['dayofweek'] = df['first_active_month'].dt.dayofweek
        df['weekofyear'] = df['first_active_month'].dt.weekofyear
        df['month'] = df['first_active_month'].dt.month
        df['elapsed_time'] = (
            datetime.today() - df['first_active_month']).dt.days
        df['hist_first_buy'] = (
            df['hist_purchase_date_min'] - df['first_active_month']).dt.days
        df['new_hist_first_buy'] = (
            df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
        for f in [
                'hist_purchase_date_max', 'hist_purchase_date_min',
                'new_hist_purchase_date_max', 'new_hist_purchase_date_min'
        ]:
            df[f] = df[f].astype(np.int64) * 1e-9
        df['card_id_total'] = df['new_hist_card_id_size'] + df['hist_card_id_size']
        df['purchase_amount_total'] = df['new_hist_purchase_amount_sum'] + df[
            'hist_purchase_amount_sum']

    for f in ['feature_1', 'feature_2', 'feature_3']:
        order_label = df_train.groupby([f])['bucket_target'].mean()
        df_train[f] = df_train[f].map(order_label)
        df_test[f] = df_test[f].map(order_label)

    top_feats = ['hist_month_diff_mean',
                 'hist_authorized_flag_mean',
                 'new_hist_purchase_amount_max',
                 'new_hist_purchase_date_uptonow',
                 'hist_category_1_sum',
                 'hist_month_lag_mean',
                 'hist_purchase_date_min',
                 'hist_purchase_date_max',
                 'hist_purchase_amount_mean',
                 'hist_category_1_mean',
                 'new_hist_purchase_amount_mean',
                 'hist_installments_sum',
                 'hist_merchant_id_nunique',
                 'hist_purchase_amount_min',
                 'hist_month_nunique']

    for i, j in itertools.combinations(top_feats, 2):
        df_train['{}_{}_sum'.format(i, j)] = df_train['{}'.format(i)] + df_train['{}'.format(j)]
        df_train['{}_{}_diff'.format(i, j)] = df_train['{}'.format(i)] - df_train['{}'.format(j)]
        df_test['{}_{}_sum'.format(i, j)] = df_test['{}'.format(i)] + df_test['{}'.format(j)]
        df_test['{}_{}_diff'.format(i, j)] = df_test['{}'.format(i)] - df_test['{}'.format(j)]

    for i, j in itertools.combinations_with_replacement(top_feats, 2):
        df_train['{}_{}_product'.format(i, j)] = df_train['{}'.format(i)
                                                          ] * df_train['{}'.format(j)]
        df_train['{}_{}_ratio'.format(i, j)] = df_train['{}'.format(i)] / df_train['{}'.format(j)]
        df_test['{}_{}_product'.format(i, j)] = df_test['{}'.format(i)] * df_test['{}'.format(j)]
        df_test['{}_{}_ratio'.format(i, j)] = df_test['{}'.format(i)] / df_test['{}'.format(j)]

    downcast_datatypes(df_train)
    downcast_datatypes(df_test)

    # Write prepared dataframes to disk (as HDF for speedy I/O)
    df_train.to_hdf(path + '/data/df_train_prepd.h5', 'df_train_prepd', index=False)
    df_test.to_hdf(path + '/data/df_test_prepd.h5', 'df_test_prepd', index=False)

    return df_train, df_test

############################################################################################
#                                                                     [+]  LIGHTGBM BOOSTING
############################################################################################


def load_prepared_data():
    """Load prepared dataframes with processed features from disk.


    :return: Description of returned object.
    :rtype: type

    """

    df_train = pd.read_hdf(path + '/data/df_train_prepd.h5')
    df_test = pd.read_hdf(path + '/data/df_test_prepd.h5')

    return df_train, df_test


def boost(df_train, df_test):
    """Define a LightGBM model to train on the data and make predictions.

    :param type df_train: Description of parameter `df_train`.
    :param type df_test: Description of parameter `df_test`.
    :return: Description of returned object.
    :rtype: type

    """

    param = {
        'num_leaves': 31,
        'min_data_in_leaf': 30,
        'objective': 'regression',
        'max_depth': -1,
        'learning_rate': 0.01,
        'min_child_weight': 20,
        'boosting': 'gbdt',
        'feature_fraction': 0.9,
        'bagging_freq': 1,
        'bagging_fraction': 0.9,
        'bagging_seed': 11,
        'metric': 'rmse',
        'lambda_l1': 0.1,
        'verbosity': -1,
        'nthread': 4,
        'random_state': 4590
    }

    df_train_columns = [
        c for c in df_train.columns if c not in
        ['card_id', 'first_active_month', 'target', 'bucket_target']]
    target = df_train['target']
    del df_train['target']

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
    oof = np.zeros(len(df_train))
    predictions = np.zeros(len(df_test))
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(
            folds.split(df_train, df_train['bucket_target'].values)):
        print('Started training fold {} on {}'.format(
            fold_, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        # , categorical_feature=categorical_feats)
        trn_data = lgb.Dataset(
            df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])
        # , categorical_feature=categorical_feats)
        val_data = lgb.Dataset(
            df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])

        num_round = 10000
        clf = lgb.train(
            param,
            trn_data,
            num_round,
            valid_sets=[trn_data, val_data],
            verbose_eval=100,
            early_stopping_rounds=150)
        oof[val_idx] = clf.predict(
            df_train.iloc[val_idx][df_train_columns],
            num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = df_train_columns
        fold_importance_df['importance'] = clf.feature_importance()
        fold_importance_df['fold'] = fold_ + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(
            df_test[df_train_columns],
            num_iteration=clf.best_iteration) / folds.n_splits

    cv_score = np.round(np.sqrt(mean_squared_error(oof, target)), 5)
    print('CV RMSE: {}'.format(cv_score))

    return predictions, target, feature_importance_df, cv_score

############################################################################################
#                                                                       [+]  PROCESS RESULTS
############################################################################################


def display_importances(runtag, feature_importance_df):
    """Plot the most important features averaged over all folds.

    :param type feature_importance_df: Description of parameter `feature_importance_df`.
    :return: Description of returned object.
    :rtype: type

    """
    cols = feature_importance_df[[
        'feature', 'importance'
    ]].groupby('feature').mean().sort_values(
        by='importance', ascending=False)[0:75].index
    best_features = feature_importance_df.loc[feature_importance_df.feature.
                                              isin(cols)]
    plt.figure(figsize=(12, 20))
    sns.set_style('darkgrid')
    sns.barplot(
        x='importance',
        y='feature',
        palette='viridis',
        data=best_features.sort_values(by='importance', ascending=False))
    plt.title('LightGBM {} \n RMSE: {}'.format(runtag, cv_score))
    plt.tight_layout()


def write_submission(runtag, predictions, df_test, cv_score):
    """Prepare a file to submist to Kaggle.

    :param type predictions: Description of parameter `predictions`.
    :param type df_test: Description of parameter `df_test`.
    :param type cv_score: Description of parameter `cv_score`.
    :return: Description of returned object.
    :rtype: type

    """
    sub_df = pd.DataFrame({'card_id': df_test['card_id'].values})
    sub_df['target'] = predictions
    sub_df.to_csv(path + '/sub/runtag.csv.gz', index=False, compression='gzip')


############################################################################################
#                                                                        [+] lightGBM RUNNER
############################################################################################

runtag = 'ELO_{}_LGBM_{}'.format(cv_score, datetime.now().replace(second=0, microsecond=0))


# Process features and save prepared data to disk
with timer('loading raw data'):
    df_train, df_test, df_hist_trans, df_new_merchant_trans = load_raw_data()
with timer('processing features'):
    df_train, df_test = process_features(df_train, df_test, df_hist_trans, df_new_merchant_trans)
gc.collect()


# Load prepared data and train model
with timer('loading prepared data'):
    df_train, df_test = load_prepared_data()
    gc.collect()
with timer('training model'):
    predictions, target, feature_importance_df, cv_score = boost(df_train, df_test)
    gc.collect()
with timer('writing submission'):
    write_submission(runtag, predictions, df_test, cv_score)


display_importances(feature_importance_df)
