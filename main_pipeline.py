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

from sklearn.linear_model import LogisticRegressionCV, ElasticNet
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

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

    # Categorize target var into buckets
    # for perc in [1.1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
    #     print(np.round(np.percentile(df_train['target'].values, perc), 3))

    df_train['bucket_target'] = 0
    df_train.loc[df_train['target'] <= -14.206, 'bucket_target'] = 1
    df_train.loc[(df_train['target'] > -14.206) &
                 (df_train['target'] <= -5.016), 'bucket_target'] = 2
    df_train.loc[(df_train['target'] > -5.016) &
                 (df_train['target'] <= -3.108), 'bucket_target'] = 3
    df_train.loc[(df_train['target'] > -3.108) &
                 (df_train['target'] <= -2.042), 'bucket_target'] = 4
    df_train.loc[(df_train['target'] > -2.042) &
                 (df_train['target'] <= -1.146), 'bucket_target'] = 5
    df_train.loc[(df_train['target'] > -1.146) &
                 (df_train['target'] <= -0.664), 'bucket_target'] = 6
    df_train.loc[(df_train['target'] > -0.664) &
                 (df_train['target'] <= -0.312), 'bucket_target'] = 7
    df_train.loc[(df_train['target'] > -0.312) &
                 (df_train['target'] <= -0.023), 'bucket_target'] = 8
    df_train.loc[(df_train['target'] > -0.023) &
                 (df_train['target'] <= 0.236), 'bucket_target'] = 9
    df_train.loc[(df_train['target'] > 0.236) & (
        df_train['target'] <= 0.565), 'bucket_target'] = 10
    df_train.loc[(df_train['target'] > 0.565) & (
        df_train['target'] <= 1.014), 'bucket_target'] = 11
    df_train.loc[(df_train['target'] > 1.014) & (df_train['target'] <= 1.83), 'bucket_target'] = 12
    df_train.loc[(df_train['target'] > 1.83) & (df_train['target'] <= 2.703), 'bucket_target'] = 13
    df_train.loc[(df_train['target'] > 2.703) & (
        df_train['target'] <= 4.813), 'bucket_target'] = 14
    df_train.loc[df_train['target'] > 4.813, 'bucket_target'] = 15
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

    # top_feats = ['hist_month_diff_mean',
    #              'hist_authorized_flag_mean',
    #              'new_hist_purchase_amount_max',
    #              'new_hist_purchase_date_uptonow',
    #              'hist_category_1_sum',
    #              'hist_month_lag_mean',
    #              'hist_purchase_date_min',
    #              'hist_purchase_date_max',
    #              'hist_purchase_amount_mean',
    #              'hist_category_1_mean',
    #              'new_hist_purchase_amount_mean',
    #              'hist_installments_sum',
    #              'hist_merchant_id_nunique',
    #              'hist_purchase_amount_min',
    #              'hist_month_nunique']

    top_feats = ['hist_month_diff_mean',
                 'new_hist_purchase_amount_max',
                 'hist_authorized_flag_mean',
                 'new_hist_purchase_date_uptonow',
                 'hist_category_1_sum',
                 'hist_month_lag_mean',
                 'hist_purchase_date_min',
                 'hist_purchase_date_max',
                 'new_hist_purchase_amount_mean',
                 'hist_purchase_amount_mean',
                 'hist_category_1_mean',
                 'hist_month_lag_var',
                 'hist_purchase_amount_min',
                 'hist_merchant_id_nunique',
                 'hist_installments_sum',
                 'hist_month_nunique',
                 'hist_first_buy',
                 'hist_purchase_date_diff',
                 'new_hist_month_lag_mean',
                 'hist_weekend_mean',
                 'hist_purchase_date_average',
                 'hist_purchase_amount_max',
                 'hist_category_3_mean_mean',
                 'hist_weekofyear_nunique',
                 'hist_purchase_date_uptonow']

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

    # Selected feature interactions
    # for df in [df_train, df_test]:
    #     df['hist_month_diff_mean_new_hist_purchase_date_uptonow_ratio'] = df['hist_month_diff_mean'] / df['new_hist_purchase_date_uptonow']
    #     df['new_hist_purchase_date_uptonow_hist_month_nunique_product'] = df['new_hist_purchase_date_uptonow'] * df['hist_month_nunique']
    #     df['hist_merchant_id_nunique_hist_month_nunique_ratio'] = df['hist_merchant_id_nunique'] * df['hist_month_nunique']
    #     df['new_hist_purchase_amount_max_hist_purchase_amount_mean_diff'] = df['new_hist_purchase_amount_max'] - df['hist_purchase_amount_mean']
    #     df['hist_month_diff_mean_hist_month_nunique_product'] = df['hist_month_diff_mean'] * df['hist_month_nunique']
    #     df['hist_month_lag_mean_hist_month_nunique_sum'] = df['hist_month_lag_mean'] + df['hist_month_nunique']
    #     df['hist_month_lag_mean_hist_month_nunique_ratio'] = df['hist_month_lag_mean'] * df['hist_month_nunique']
    #     df['hist_purchase_amount_mean_new_hist_purchase_amount_mean_diff'] = df['hist_purchase_amount_mean'] - df['new_hist_purchase_amount_mean']
    #     df['hist_purchase_date_max_hist_purchase_amount_min_ratio'] = df['hist_purchase_date_max'] * df['hist_purchase_amount_min']
    #     df['hist_month_diff_mean_hist_month_nunique_sum'] = df['hist_month_diff_mean'] + df['hist_month_nunique']
    #     df['hist_month_diff_mean_hist_authorized_flag_mean_ratio'] = df['hist_month_diff_mean'] * df['hist_authorized_flag_mean']
    #     df['hist_category_1_sum_hist_month_nunique_product'] = df['hist_category_1_sum'] * df['hist_month_nunique']
    #     df['hist_month_lag_mean_hist_merchant_id_nunique_ratio'] = df['hist_month_lag_mean'] / df['hist_merchant_id_nunique']
    #     df['hist_month_diff_mean_hist_category_1_mean_sum'] = df['hist_month_diff_mean'] + df['hist_category_1_mean']
    #     df['hist_authorized_flag_mean_hist_month_nunique_ratio'] = df['hist_authorized_flag_mean'] * df['hist_month_nunique']

    downcast_datatypes(df_train)
    downcast_datatypes(df_test)

    # Write prepared dataframes to disk (as HDF for speedy I/O)
    df_train.to_hdf(path + '/data/df_train_prepd.h5', 'df_train_prepd', index=False)
    df_test.to_hdf(path + '/data/df_test_prepd.h5', 'df_test_prepd', index=False)
    gc.collect()

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

    # Drop unimportant features
    drop_feats = ['hist_purchase_date_diff_hist_weekofyear_nunique_sum',
                  'hist_month_lag_var_hist_purchase_date_uptonow_ratio',
                  'hist_purchase_amount_mean_hist_purchase_amount_max_diff',
                  'hist_category_1_sum_hist_weekend_mean_sum',
                  'hist_purchase_amount_mean_hist_purchase_amount_mean_product',
                  'hist_merchant_id_nunique_hist_purchase_amount_max_diff',
                  'hist_month_diff_mean_hist_first_buy_ratio',
                  'hist_month_lag_mean_hist_weekend_mean_sum',
                  'hist_purchase_date_min_hist_category_3_mean_mean_product',
                  'hist_purchase_date_average_hist_category_3_mean_mean_sum',
                  'hist_first_buy_hist_purchase_amount_max_sum',
                  'hist_category_1_sum_new_hist_purchase_amount_mean_sum',
                  'hist_category_1_sum_hist_purchase_date_uptonow_ratio',
                  'hist_month_lag_mean_hist_purchase_amount_min_ratio',
                  'hist_category_1_mean_hist_purchase_amount_min_diff',
                  'hist_category_1_sum_hist_purchase_date_diff_ratio',
                  'hist_category_1_mean_hist_month_nunique_ratio',
                  'hist_purchase_date_max_hist_purchase_amount_mean_product',
                  'new_hist_purchase_amount_max_hist_category_1_sum_ratio',
                  'new_hist_purchase_amount_max_hist_month_lag_var_sum',
                  'hist_merchant_id_nunique_hist_purchase_date_uptonow_ratio',
                  'hist_category_1_sum_hist_purchase_amount_mean_sum',
                  'hist_purchase_amount_mean_hist_weekofyear_nunique_sum',
                  'hist_month_diff_mean_hist_merchant_id_nunique_ratio',
                  'new_hist_purchase_amount_max_hist_weekofyear_nunique_diff',
                  'hist_month_diff_mean_hist_month_lag_var_ratio',
                  'hist_category_1_sum_hist_month_nunique_sum',
                  'hist_authorized_flag_mean_hist_installments_sum_sum',
                  'hist_purchase_amount_min_hist_purchase_date_diff_product',
                  'hist_month_diff_mean_hist_first_buy_product',
                  'hist_merchant_id_nunique_hist_weekofyear_nunique_sum',
                  'hist_category_1_mean_new_hist_month_lag_mean_product',
                  'hist_installments_sum_hist_purchase_date_uptonow_ratio',
                  'new_hist_purchase_date_uptonow_hist_installments_sum_ratio',
                  'hist_month_diff_mean_hist_purchase_amount_min_sum',
                  'hist_purchase_amount_min_hist_purchase_date_diff_ratio',
                  'hist_purchase_date_diff_hist_purchase_date_average_diff',
                  'hist_category_1_sum_hist_purchase_amount_min_diff',
                  'hist_month_diff_mean_hist_installments_sum_sum',
                  'new_hist_month_lag_mean_hist_purchase_date_average_diff',
                  'hist_month_lag_mean_hist_merchant_id_nunique_diff',
                  'hist_category_1_sum_hist_purchase_amount_mean_diff',
                  'hist_month_diff_mean_hist_category_1_mean_ratio',
                  'new_hist_purchase_amount_max_hist_weekofyear_nunique_sum',
                  'hist_purchase_amount_max_hist_purchase_date_uptonow_ratio',
                  'hist_month_diff_mean_hist_purchase_date_max_product',
                  'new_hist_purchase_amount_mean_hist_category_1_mean_ratio',
                  'hist_purchase_date_min_hist_purchase_amount_max_ratio',
                  'hist_month_lag_var_new_hist_month_lag_mean_sum',
                  'hist_category_1_mean_hist_purchase_date_uptonow_ratio',
                  'hist_month_lag_mean_hist_purchase_date_uptonow_sum',
                  'hist_purchase_date_diff_hist_purchase_date_average_sum',
                  'hist_purchase_date_min_hist_category_1_mean_ratio',
                  'hist_purchase_date_min_hist_purchase_amount_mean_product',
                  'hist_authorized_flag_mean_hist_category_1_sum_diff',
                  'hist_category_1_mean_hist_purchase_date_average_diff',
                  'new_hist_purchase_amount_max_hist_month_lag_var_diff',
                  'new_hist_purchase_amount_mean_hist_installments_sum_sum',
                  'hist_authorized_flag_mean_hist_weekofyear_nunique_diff',
                  'hist_installments_sum_hist_weekofyear_nunique_sum',
                  'new_hist_purchase_date_uptonow_hist_month_lag_mean_sum',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_diff_ratio',
                  'hist_category_1_sum_hist_weekend_mean_diff',
                  'hist_purchase_date_min_hist_first_buy_ratio',
                  'hist_month_lag_var_hist_purchase_date_diff_product',
                  'new_hist_month_lag_mean_hist_purchase_amount_max_product',
                  'hist_installments_sum_hist_purchase_amount_max_sum',
                  'hist_month_diff_mean_hist_purchase_date_max_ratio',
                  'hist_month_nunique_hist_first_buy_sum',
                  'hist_purchase_amount_mean_hist_month_lag_var_sum',
                  'hist_purchase_date_min_hist_first_buy_sum',
                  'hist_month_nunique_hist_weekofyear_nunique_diff',
                  'new_hist_purchase_date_uptonow_hist_category_1_mean_ratio',
                  'hist_purchase_amount_min_hist_weekend_mean_ratio',
                  'new_hist_purchase_amount_max_hist_installments_sum_diff',
                  'hist_month_lag_var_hist_installments_sum_diff',
                  'hist_purchase_date_min_hist_purchase_date_min_product',
                  'hist_month_lag_var_hist_category_3_mean_mean_diff',
                  'hist_authorized_flag_mean_hist_purchase_date_average_sum',
                  'hist_month_lag_var_hist_installments_sum_sum',
                  'hist_purchase_amount_mean_hist_purchase_date_average_diff',
                  'hist_category_1_mean_hist_purchase_amount_min_ratio',
                  'hist_purchase_date_min_hist_first_buy_diff',
                  'hist_purchase_amount_mean_hist_merchant_id_nunique_sum',
                  'new_hist_purchase_date_uptonow_hist_month_lag_mean_diff',
                  'hist_category_1_mean_hist_purchase_date_average_sum',
                  'hist_purchase_amount_min_hist_weekofyear_nunique_product',
                  'hist_first_buy_hist_first_buy_product',
                  'hist_merchant_id_nunique_hist_purchase_date_average_diff',
                  'hist_month_diff_mean_hist_category_1_sum_ratio',
                  'hist_first_buy_hist_weekofyear_nunique_sum',
                  'new_hist_purchase_amount_max_hist_purchase_date_uptonow_sum',
                  'hist_purchase_amount_min_hist_weekofyear_nunique_sum',
                  'hist_month_lag_var_hist_category_3_mean_mean_sum',
                  'hist_purchase_date_min_hist_purchase_date_uptonow_product',
                  'hist_category_1_sum_new_hist_month_lag_mean_sum',
                  'hist_purchase_amount_min_hist_purchase_amount_max_ratio',
                  'hist_month_lag_var_new_hist_month_lag_mean_diff',
                  'hist_category_1_mean_hist_month_lag_var_sum',
                  'hist_weekend_mean_hist_weekofyear_nunique_diff',
                  'hist_category_1_sum_hist_purchase_date_min_diff',
                  'hist_first_buy_hist_purchase_date_average_diff',
                  'hist_purchase_date_average_hist_category_3_mean_mean_diff',
                  'new_hist_purchase_amount_max_hist_merchant_id_nunique_sum',
                  'hist_purchase_amount_mean_hist_weekofyear_nunique_diff',
                  'new_hist_purchase_amount_mean_hist_purchase_date_average_diff',
                  'hist_authorized_flag_mean_hist_purchase_date_average_diff',
                  'hist_purchase_amount_min_hist_weekend_mean_product',
                  'hist_weekend_mean_hist_purchase_date_average_sum',
                  'hist_month_lag_var_hist_first_buy_diff',
                  'hist_purchase_amount_mean_hist_month_lag_var_diff',
                  'new_hist_purchase_amount_mean_hist_month_lag_var_sum',
                  'hist_category_3_mean_mean_hist_weekofyear_nunique_sum',
                  'hist_purchase_amount_min_hist_first_buy_ratio',
                  'hist_purchase_amount_mean_hist_purchase_amount_max_sum',
                  'hist_category_1_mean_hist_purchase_date_diff_product',
                  'new_hist_purchase_amount_max_hist_first_buy_sum',
                  'new_hist_purchase_amount_max_hist_merchant_id_nunique_diff',
                  'hist_purchase_date_max_hist_purchase_date_uptonow_product',
                  'hist_month_lag_var_hist_purchase_amount_min_product',
                  'new_hist_month_lag_mean_hist_purchase_date_uptonow_sum',
                  'hist_month_lag_mean_hist_installments_sum_diff',
                  'hist_purchase_amount_mean_hist_purchase_date_average_sum',
                  'hist_category_1_sum_new_hist_month_lag_mean_diff',
                  'hist_purchase_date_diff_hist_purchase_amount_max_sum',
                  'hist_authorized_flag_mean_hist_month_lag_var_sum',
                  'hist_month_lag_var_hist_purchase_date_diff_diff',
                  'hist_month_lag_var_hist_purchase_amount_min_ratio',
                  'new_hist_purchase_date_uptonow_hist_purchase_amount_mean_sum',
                  'hist_category_3_mean_mean_hist_weekofyear_nunique_diff',
                  'hist_category_1_mean_hist_weekofyear_nunique_sum',
                  'hist_purchase_date_min_hist_purchase_amount_max_diff',
                  'hist_purchase_amount_min_hist_purchase_date_uptonow_diff',
                  'hist_category_1_sum_hist_month_nunique_diff',
                  'new_hist_purchase_amount_max_hist_purchase_date_max_diff',
                  'hist_purchase_amount_min_hist_installments_sum_diff',
                  'hist_month_diff_mean_new_hist_purchase_date_uptonow_sum',
                  'hist_month_lag_var_hist_weekend_mean_sum',
                  'hist_purchase_date_min_hist_weekofyear_nunique_ratio',
                  'hist_month_nunique_hist_weekofyear_nunique_sum',
                  'hist_month_lag_mean_hist_purchase_date_min_diff',
                  'hist_first_buy_hist_purchase_date_average_sum',
                  'new_hist_purchase_amount_mean_hist_month_lag_var_diff',
                  'hist_weekend_mean_hist_purchase_date_average_diff',
                  'hist_merchant_id_nunique_hist_month_nunique_sum',
                  'hist_purchase_amount_min_hist_weekofyear_nunique_diff',
                  'hist_installments_sum_hist_purchase_date_average_diff',
                  'hist_purchase_amount_mean_hist_purchase_date_uptonow_sum',
                  'hist_installments_sum_hist_weekend_mean_sum',
                  'hist_installments_sum_hist_weekend_mean_diff',
                  'hist_authorized_flag_mean_hist_authorized_flag_mean_product',
                  'hist_purchase_date_max_hist_weekofyear_nunique_ratio',
                  'hist_purchase_date_max_hist_purchase_amount_max_ratio',
                  'hist_purchase_amount_min_hist_installments_sum_ratio',
                  'hist_category_1_sum_hist_category_3_mean_mean_sum',
                  'hist_month_diff_mean_hist_purchase_date_uptonow_diff',
                  'hist_category_1_mean_hist_month_lag_var_diff',
                  'hist_category_1_mean_hist_purchase_amount_min_product',
                  'hist_purchase_date_max_hist_category_1_mean_ratio',
                  'hist_installments_sum_hist_month_nunique_diff',
                  'new_hist_purchase_amount_mean_hist_purchase_date_average_sum',
                  'hist_first_buy_hist_purchase_amount_max_diff',
                  'hist_month_diff_mean_hist_purchase_date_uptonow_sum',
                  'hist_authorized_flag_mean_hist_weekofyear_nunique_sum',
                  'hist_purchase_amount_mean_hist_purchase_date_uptonow_diff',
                  'hist_purchase_amount_min_hist_merchant_id_nunique_diff',
                  'hist_purchase_date_max_hist_first_buy_ratio',
                  'hist_purchase_amount_mean_hist_merchant_id_nunique_diff',
                  'hist_purchase_date_min_hist_purchase_date_uptonow_ratio',
                  'new_hist_purchase_amount_max_new_hist_purchase_date_uptonow_sum',
                  'hist_purchase_amount_mean_hist_installments_sum_diff',
                  'hist_purchase_date_diff_hist_purchase_amount_max_diff',
                  'hist_purchase_amount_min_hist_installments_sum_sum',
                  'new_hist_purchase_amount_max_hist_category_1_mean_ratio',
                  'hist_purchase_date_max_hist_weekend_mean_product',
                  'hist_authorized_flag_mean_hist_merchant_id_nunique_diff',
                  'hist_purchase_date_min_hist_merchant_id_nunique_ratio',
                  'hist_category_1_mean_hist_purchase_date_uptonow_sum',
                  'hist_month_lag_mean_hist_first_buy_sum',
                  'hist_month_lag_mean_hist_first_buy_diff',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_min_diff',
                  'new_hist_purchase_amount_mean_hist_weekofyear_nunique_diff',
                  'hist_month_diff_mean_hist_first_buy_sum',
                  'new_hist_purchase_date_uptonow_hist_weekend_mean_sum',
                  'hist_authorized_flag_mean_hist_installments_sum_diff',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_min_sum',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_min_ratio',
                  'hist_month_nunique_hist_first_buy_diff',
                  'hist_first_buy_new_hist_month_lag_mean_diff',
                  'hist_purchase_amount_min_hist_purchase_amount_max_product',
                  'new_hist_purchase_amount_mean_hist_installments_sum_diff',
                  'hist_weekend_mean_hist_weekofyear_nunique_sum',
                  'hist_purchase_amount_mean_hist_installments_sum_sum',
                  'hist_purchase_amount_min_hist_merchant_id_nunique_ratio',
                  'new_hist_purchase_date_uptonow_hist_purchase_amount_mean_diff',
                  'new_hist_purchase_amount_mean_hist_merchant_id_nunique_diff',
                  'hist_month_lag_mean_hist_purchase_amount_min_sum',
                  'hist_purchase_amount_min_hist_first_buy_product',
                  'hist_purchase_date_min_hist_weekend_mean_product',
                  'new_hist_purchase_amount_max_hist_purchase_date_uptonow_diff',
                  'hist_month_diff_mean_hist_purchase_date_diff_sum',
                  'hist_month_diff_mean_hist_purchase_date_max_diff',
                  'hist_weekend_mean_hist_purchase_date_uptonow_sum',
                  'hist_purchase_amount_min_hist_purchase_date_average_product',
                  'hist_category_1_mean_hist_purchase_date_average_ratio',
                  'new_hist_purchase_amount_mean_hist_weekofyear_nunique_sum',
                  'hist_authorized_flag_mean_hist_purchase_date_min_sum',
                  'hist_authorized_flag_mean_hist_month_lag_var_diff',
                  'hist_category_1_sum_hist_category_3_mean_mean_diff',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_max_product',
                  'hist_authorized_flag_mean_hist_first_buy_sum',
                  'new_hist_purchase_date_uptonow_hist_category_1_mean_sum',
                  'hist_authorized_flag_mean_new_hist_purchase_date_uptonow_diff',
                  'hist_month_lag_var_hist_weekend_mean_diff',
                  'hist_purchase_amount_min_hist_purchase_amount_max_sum',
                  'hist_purchase_date_min_hist_weekofyear_nunique_product',
                  'hist_purchase_date_min_hist_weekend_mean_ratio',
                  'hist_purchase_date_max_hist_weekofyear_nunique_product',
                  'hist_purchase_date_min_hist_category_1_mean_product',
                  'hist_merchant_id_nunique_hist_weekend_mean_diff',
                  'new_hist_purchase_amount_max_hist_purchase_date_diff_sum',
                  'new_hist_purchase_amount_mean_hist_purchase_date_uptonow_diff',
                  'hist_month_diff_mean_new_hist_purchase_date_uptonow_diff',
                  'hist_purchase_amount_min_hist_purchase_date_uptonow_sum',
                  'hist_category_1_mean_hist_weekofyear_nunique_diff',
                  'hist_purchase_amount_min_hist_purchase_date_average_ratio',
                  'new_hist_purchase_amount_max_hist_first_buy_diff',
                  'new_hist_purchase_amount_mean_hist_merchant_id_nunique_sum',
                  'new_hist_month_lag_mean_hist_weekofyear_nunique_diff',
                  'new_hist_purchase_amount_max_new_hist_purchase_date_uptonow_diff',
                  'new_hist_purchase_date_uptonow_new_hist_month_lag_mean_diff',
                  'hist_month_lag_mean_hist_purchase_amount_min_diff',
                  'hist_authorized_flag_mean_hist_merchant_id_nunique_sum',
                  'hist_purchase_amount_min_hist_purchase_amount_max_diff',
                  'new_hist_purchase_date_uptonow_hist_purchase_amount_min_sum',
                  'hist_authorized_flag_mean_new_hist_purchase_date_uptonow_sum',
                  'hist_installments_sum_new_hist_month_lag_mean_sum',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_max_diff',
                  'hist_month_lag_var_hist_purchase_date_diff_sum',
                  'hist_purchase_date_max_hist_weekend_mean_ratio',
                  'hist_purchase_date_diff_hist_purchase_date_average_ratio',
                  'new_hist_purchase_amount_mean_hist_purchase_date_uptonow_sum',
                  'new_hist_month_lag_mean_hist_weekofyear_nunique_sum',
                  'hist_purchase_date_min_hist_month_lag_var_sum',
                  'hist_authorized_flag_mean_hist_purchase_date_uptonow_diff',
                  'hist_month_diff_mean_hist_purchase_date_diff_diff',
                  'hist_category_1_mean_hist_purchase_date_uptonow_diff',
                  'hist_purchase_amount_min_hist_merchant_id_nunique_sum',
                  'hist_purchase_date_min_hist_purchase_date_diff_sum',
                  'hist_category_1_sum_hist_purchase_date_min_ratio',
                  'new_hist_purchase_amount_mean_hist_first_buy_diff',
                  'hist_purchase_date_min_hist_merchant_id_nunique_sum',
                  'hist_month_lag_mean_hist_purchase_date_diff_diff',
                  'hist_purchase_amount_min_hist_merchant_id_nunique_product',
                  'hist_purchase_date_min_hist_merchant_id_nunique_product',
                  'new_hist_month_lag_mean_hist_purchase_date_uptonow_diff',
                  'hist_merchant_id_nunique_new_hist_month_lag_mean_sum',
                  'hist_purchase_date_min_hist_merchant_id_nunique_diff',
                  'hist_installments_sum_hist_month_nunique_sum',
                  'hist_purchase_date_min_hist_installments_sum_sum',
                  'hist_month_nunique_hist_purchase_date_diff_sum',
                  'hist_category_1_sum_hist_purchase_amount_min_ratio',
                  'hist_category_1_mean_hist_merchant_id_nunique_sum',
                  'hist_category_1_sum_hist_category_1_mean_diff',
                  'new_hist_purchase_amount_max_hist_purchase_date_max_sum',
                  'hist_installments_sum_new_hist_month_lag_mean_diff',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_max_ratio',
                  'hist_merchant_id_nunique_hist_weekend_mean_sum',
                  'hist_month_lag_mean_hist_purchase_date_diff_sum',
                  'new_hist_purchase_date_uptonow_hist_purchase_amount_min_diff',
                  'hist_category_3_mean_mean_hist_purchase_date_uptonow_diff',
                  'hist_weekend_mean_hist_purchase_date_uptonow_diff',
                  'new_hist_purchase_date_uptonow_new_hist_purchase_amount_mean_diff',
                  'hist_purchase_date_min_hist_month_lag_var_diff',
                  'hist_merchant_id_nunique_hist_category_3_mean_mean_sum',
                  'hist_purchase_date_min_hist_purchase_amount_max_sum',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_min_product',
                  'hist_purchase_date_min_hist_installments_sum_diff',
                  'hist_first_buy_hist_weekend_mean_diff',
                  'hist_month_lag_mean_hist_purchase_date_min_ratio',
                  'hist_month_nunique_hist_purchase_date_diff_diff',
                  'hist_month_lag_mean_hist_purchase_date_max_product',
                  'hist_purchase_amount_mean_hist_first_buy_sum',
                  'hist_purchase_date_max_hist_purchase_amount_max_product',
                  'hist_purchase_date_min_new_hist_purchase_amount_mean_sum',
                  'hist_category_1_sum_hist_purchase_amount_min_product',
                  'new_hist_purchase_amount_max_hist_purchase_date_diff_diff',
                  'hist_category_1_sum_hist_purchase_date_min_product',
                  'hist_category_3_mean_mean_hist_purchase_date_uptonow_sum',
                  'hist_authorized_flag_mean_hist_first_buy_diff',
                  'hist_month_diff_mean_hist_purchase_date_max_sum',
                  'hist_purchase_amount_mean_hist_purchase_date_diff_sum',
                  'new_hist_purchase_amount_mean_hist_purchase_date_diff_diff',
                  'new_hist_purchase_amount_mean_hist_purchase_date_diff_sum',
                  'new_hist_purchase_date_uptonow_hist_category_3_mean_mean_sum',
                  'hist_category_1_sum_hist_purchase_date_min_sum',
                  'hist_purchase_date_max_hist_merchant_id_nunique_product',
                  'hist_purchase_date_min_hist_purchase_date_uptonow_diff',
                  'hist_month_lag_mean_hist_purchase_date_min_product',
                  'hist_purchase_amount_mean_hist_first_buy_diff',
                  'new_hist_purchase_date_uptonow_hist_category_3_mean_mean_diff',
                  'hist_category_1_mean_hist_merchant_id_nunique_diff',
                  'new_hist_purchase_date_uptonow_new_hist_month_lag_mean_sum',
                  'new_hist_purchase_date_uptonow_hist_category_1_mean_diff',
                  'hist_category_1_mean_hist_first_buy_diff',
                  'hist_purchase_date_max_hist_category_1_mean_product',
                  'hist_category_1_sum_hist_category_1_mean_sum',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_max_sum',
                  'new_hist_purchase_amount_mean_hist_first_buy_sum',
                  'hist_authorized_flag_mean_hist_purchase_date_max_diff',
                  'hist_purchase_amount_min_hist_purchase_date_average_sum',
                  'hist_purchase_date_max_hist_installments_sum_ratio',
                  'hist_month_diff_mean_hist_month_diff_mean_product',
                  'hist_merchant_id_nunique_new_hist_month_lag_mean_diff',
                  'hist_month_lag_mean_hist_purchase_date_min_sum',
                  'hist_purchase_amount_mean_hist_purchase_date_diff_diff',
                  'hist_month_lag_var_hist_purchase_amount_min_diff',
                  'hist_purchase_date_min_hist_purchase_date_max_ratio',
                  'hist_month_lag_mean_hist_purchase_date_max_ratio',
                  'hist_purchase_date_min_hist_first_buy_product',
                  'hist_purchase_date_max_hist_month_lag_var_product',
                  'hist_purchase_date_max_hist_first_buy_product',
                  'hist_purchase_amount_min_hist_installments_sum_product',
                  'hist_authorized_flag_mean_hist_purchase_date_uptonow_sum',
                  'hist_purchase_date_diff_new_hist_month_lag_mean_sum',
                  'hist_month_lag_mean_hist_purchase_date_max_diff',
                  'new_hist_purchase_date_uptonow_hist_weekend_mean_diff',
                  'hist_purchase_date_diff_hist_category_3_mean_mean_diff',
                  'hist_category_1_mean_hist_installments_sum_sum',
                  'new_hist_month_lag_mean_new_hist_month_lag_mean_product',
                  'hist_purchase_date_min_hist_purchase_amount_max_product',
                  'hist_merchant_id_nunique_hist_category_3_mean_mean_diff',
                  'hist_purchase_date_max_hist_merchant_id_nunique_ratio',
                  'hist_category_1_mean_hist_purchase_date_diff_diff',
                  'hist_purchase_date_max_hist_purchase_date_max_product',
                  'hist_purchase_date_max_hist_purchase_date_average_diff',
                  'hist_purchase_date_max_hist_purchase_date_average_product',
                  'hist_first_buy_hist_category_3_mean_mean_sum',
                  'hist_category_1_mean_hist_purchase_date_diff_sum',
                  'hist_purchase_date_min_hist_weekofyear_nunique_sum',
                  'hist_purchase_date_max_hist_month_lag_var_ratio',
                  'hist_purchase_date_min_hist_purchase_date_max_diff',
                  'hist_purchase_date_min_hist_purchase_date_diff_diff',
                  'hist_purchase_date_max_hist_merchant_id_nunique_sum',
                  'hist_purchase_date_min_hist_purchase_date_diff_ratio',
                  'hist_purchase_date_min_hist_purchase_date_diff_product',
                  'hist_purchase_date_min_hist_purchase_date_uptonow_sum',
                  'hist_purchase_date_max_hist_purchase_date_diff_product',
                  'hist_purchase_date_max_new_hist_purchase_amount_mean_diff',
                  'hist_first_buy_new_hist_month_lag_mean_sum',
                  'hist_purchase_date_min_hist_installments_sum_product',
                  'hist_installments_sum_hist_category_3_mean_mean_sum',
                  'hist_purchase_date_diff_hist_weekend_mean_sum',
                  'hist_month_lag_mean_hist_month_lag_mean_product',
                  'hist_purchase_date_max_hist_first_buy_sum',
                  'hist_first_buy_hist_weekend_mean_sum',
                  'hist_purchase_date_min_hist_month_lag_var_product',
                  'hist_purchase_date_min_hist_purchase_date_average_ratio',
                  'hist_purchase_date_min_hist_installments_sum_ratio',
                  'new_hist_purchase_date_uptonow_new_hist_purchase_date_uptonow_product',
                  'new_hist_purchase_date_uptonow_new_hist_purchase_amount_mean_sum',
                  'hist_purchase_date_min_new_hist_purchase_amount_mean_diff',
                  'hist_purchase_date_min_new_hist_month_lag_mean_diff',
                  'hist_category_1_mean_hist_installments_sum_diff',
                  'hist_category_1_mean_hist_first_buy_sum',
                  'hist_category_1_mean_hist_category_1_mean_product',
                  'hist_first_buy_hist_category_3_mean_mean_diff',
                  'hist_purchase_date_min_hist_month_nunique_sum',
                  'hist_purchase_date_max_hist_purchase_date_diff_diff',
                  'hist_purchase_amount_min_hist_first_buy_sum',
                  'hist_purchase_date_uptonow_hist_purchase_date_uptonow_product',
                  'hist_purchase_date_diff_new_hist_month_lag_mean_diff',
                  'hist_purchase_amount_min_hist_first_buy_diff',
                  'hist_purchase_date_max_hist_first_buy_diff',
                  'hist_authorized_flag_mean_hist_purchase_date_diff_diff',
                  'hist_purchase_date_min_hist_month_nunique_diff',
                  'hist_purchase_amount_min_hist_purchase_date_diff_sum',
                  'hist_purchase_amount_min_hist_purchase_date_diff_diff',
                  'hist_category_1_sum_hist_purchase_date_max_diff',
                  'hist_category_1_sum_hist_purchase_date_max_sum',
                  'hist_purchase_date_max_hist_weekofyear_nunique_sum',
                  'hist_purchase_date_min_hist_purchase_date_average_product',
                  'hist_purchase_date_max_hist_purchase_date_uptonow_diff',
                  'hist_purchase_date_max_hist_purchase_date_diff_sum',
                  'hist_category_1_sum_hist_purchase_date_max_ratio',
                  'hist_purchase_date_max_hist_purchase_date_diff_ratio',
                  'hist_purchase_date_min_hist_month_lag_var_ratio',
                  'hist_category_1_sum_hist_purchase_date_max_product',
                  'hist_merchant_id_nunique_hist_merchant_id_nunique_ratio',
                  'hist_purchase_date_min_hist_category_1_mean_sum',
                  'hist_month_lag_mean_hist_purchase_date_max_sum',
                  'hist_purchase_date_max_hist_category_1_mean_diff',
                  'hist_installments_sum_hist_category_3_mean_mean_diff',
                  'hist_purchase_date_min_hist_category_3_mean_mean_diff',
                  'hist_purchase_date_max_hist_purchase_date_average_sum',
                  'hist_purchase_date_max_hist_category_1_mean_sum',
                  'hist_purchase_date_min_hist_purchase_amount_min_diff',
                  'hist_purchase_date_max_hist_category_3_mean_mean_diff',
                  'hist_purchase_date_min_hist_category_1_mean_diff',
                  'hist_month_lag_var_hist_month_lag_var_product',
                  'hist_purchase_date_max_hist_category_3_mean_mean_sum',
                  'hist_category_1_sum_hist_category_1_sum_product',
                  'hist_purchase_date_min_hist_purchase_amount_min_sum',
                  'hist_category_1_sum_hist_category_1_sum_ratio',
                  'hist_month_nunique_hist_month_nunique_product',
                  'hist_purchase_date_min_hist_category_3_mean_mean_sum',
                  'hist_purchase_amount_max_hist_purchase_amount_max_ratio',
                  'new_hist_month_lag_mean_new_hist_month_lag_mean_ratio',
                  'hist_authorized_flag_mean_hist_authorized_flag_mean_ratio',
                  'hist_purchase_date_diff_hist_category_3_mean_mean_sum',
                  'new_hist_purchase_date_uptonow_new_hist_purchase_date_uptonow_ratio',
                  'hist_purchase_date_min_hist_purchase_amount_mean_diff',
                  'hist_month_lag_mean_hist_month_lag_mean_ratio',
                  'hist_purchase_date_min_hist_purchase_amount_mean_sum',
                  'new_hist_purchase_amount_max_new_hist_purchase_amount_max_ratio',
                  'hist_purchase_date_diff_hist_weekend_mean_diff',
                  'hist_installments_sum_hist_installments_sum_ratio',
                  'hist_merchant_id_nunique_hist_merchant_id_nunique_product',
                  'hist_purchase_date_diff_hist_purchase_date_diff_product',
                  'hist_purchase_date_diff_hist_purchase_date_diff_ratio',
                  'hist_purchase_date_average_hist_purchase_date_average_ratio',
                  'hist_purchase_date_average_hist_purchase_date_average_product',
                  'hist_purchase_date_min_hist_purchase_date_average_diff',
                  'hist_month_nunique_hist_month_nunique_ratio',
                  'hist_purchase_date_max_new_hist_purchase_amount_mean_sum',
                  'hist_purchase_date_max_hist_installments_sum_product',
                  'hist_purchase_date_max_hist_weekend_mean_diff',
                  'hist_first_buy_hist_first_buy_ratio',
                  'hist_purchase_date_max_hist_purchase_amount_max_sum',
                  'hist_purchase_date_min_hist_purchase_date_min_ratio',
                  'hist_purchase_date_max_hist_purchase_amount_mean_diff',
                  'hist_purchase_amount_min_hist_purchase_date_average_diff',
                  'hist_purchase_date_max_hist_purchase_amount_mean_sum',
                  'hist_purchase_date_min_hist_weekend_mean_diff',
                  'hist_purchase_date_min_hist_weekend_mean_sum',
                  'hist_purchase_amount_min_hist_purchase_amount_min_ratio',
                  'hist_purchase_date_max_hist_purchase_date_uptonow_sum',
                  'hist_purchase_date_max_hist_installments_sum_diff',
                  'hist_purchase_date_min_new_hist_month_lag_mean_sum',
                  'hist_purchase_date_max_hist_purchase_amount_min_diff',
                  'hist_purchase_date_max_hist_purchase_date_uptonow_ratio',
                  'hist_month_lag_var_hist_month_lag_var_ratio',
                  'hist_purchase_date_max_hist_purchase_amount_min_sum',
                  'hist_purchase_date_max_hist_purchase_date_max_ratio',
                  'hist_purchase_date_min_hist_weekofyear_nunique_diff',
                  'hist_category_3_mean_mean_hist_category_3_mean_mean_ratio',
                  'new_hist_purchase_amount_mean_new_hist_purchase_amount_mean_ratio',
                  'hist_purchase_date_max_hist_purchase_amount_max_diff',
                  'hist_purchase_date_max_hist_month_nunique_sum',
                  'hist_purchase_date_max_hist_weekend_mean_sum',
                  'hist_purchase_date_uptonow_hist_purchase_date_uptonow_ratio',
                  'hist_weekofyear_nunique_hist_weekofyear_nunique_ratio',
                  'hist_purchase_date_max_hist_purchase_date_average_ratio',
                  'hist_authorized_flag_mean_hist_purchase_date_diff_sum',
                  'hist_weekofyear_nunique_hist_weekofyear_nunique_product',
                  'hist_purchase_date_min_hist_purchase_date_average_sum',
                  'hist_purchase_date_max_hist_installments_sum_sum',
                  'hist_weekend_mean_hist_weekend_mean_ratio',
                  'hist_weekend_mean_hist_weekend_mean_product',
                  'hist_purchase_date_max_hist_merchant_id_nunique_diff',
                  'hist_month_lag_var_hist_purchase_amount_min_sum',
                  'hist_authorized_flag_mean_hist_purchase_date_max_sum',
                  'hist_purchase_amount_mean_hist_purchase_amount_mean_ratio',
                  'hist_purchase_date_max_new_hist_month_lag_mean_sum',
                  'hist_purchase_date_max_new_hist_month_lag_mean_diff',
                  'hist_purchase_date_max_hist_month_lag_var_diff',
                  'hist_purchase_date_max_hist_weekofyear_nunique_diff',
                  'hist_purchase_date_max_hist_month_lag_var_sum',
                  'hist_purchase_date_max_hist_month_nunique_diff',
                  'hist_month_diff_mean_hist_month_diff_mean_ratio',
                  'hist_category_1_mean_hist_category_1_mean_ratio',
                  'hist_installments_sum_hist_weekofyear_nunique_ratio',
                  'hist_purchase_amount_min_hist_category_3_mean_mean_diff',
                  'hist_purchase_amount_mean_hist_purchase_date_uptonow_ratio',
                  'new_hist_purchase_amount_mean_hist_category_3_mean_mean_ratio',
                  'hist_installments_sum_hist_first_buy_ratio',
                  'hist_month_diff_mean_hist_category_3_mean_mean_sum',
                  'hist_purchase_amount_mean_hist_purchase_date_uptonow_product',
                  'hist_category_3_mean_mean_hist_weekofyear_nunique_ratio',
                  'hist_authorized_flag_mean_new_hist_purchase_amount_mean_diff',
                  'hist_purchase_amount_min_hist_category_3_mean_mean_sum',
                  'hist_month_lag_var_hist_purchase_date_uptonow_sum',
                  'hist_month_lag_mean_hist_installments_sum_sum',
                  'hist_purchase_date_average_hist_purchase_amount_max_sum',
                  'hist_month_diff_mean_hist_purchase_amount_min_ratio',
                  'new_hist_purchase_date_uptonow_hist_weekofyear_nunique_diff',
                  'hist_purchase_date_min_hist_month_nunique_product',
                  'hist_category_1_sum_new_hist_purchase_amount_mean_product',
                  'hist_purchase_amount_min_hist_weekend_mean_sum',
                  'hist_purchase_amount_mean_hist_weekend_mean_ratio',
                  'new_hist_purchase_amount_mean_new_hist_month_lag_mean_ratio',
                  'hist_purchase_amount_mean_hist_first_buy_ratio',
                  'hist_month_lag_var_hist_purchase_amount_max_sum',
                  'hist_purchase_date_diff_hist_weekend_mean_product',
                  'hist_category_1_mean_hist_weekend_mean_diff',
                  'new_hist_purchase_amount_max_hist_category_1_mean_sum',
                  'new_hist_purchase_date_uptonow_hist_month_lag_var_sum',
                  'hist_merchant_id_nunique_hist_purchase_date_diff_sum',
                  'hist_month_lag_mean_hist_first_buy_ratio',
                  'hist_merchant_id_nunique_hist_purchase_date_diff_product',
                  'hist_first_buy_hist_weekofyear_nunique_product',
                  'hist_purchase_date_average_hist_purchase_date_uptonow_sum',
                  'hist_authorized_flag_mean_hist_category_3_mean_mean_product',
                  'hist_weekend_mean_hist_purchase_date_average_product',
                  'hist_category_1_mean_hist_category_3_mean_mean_product',
                  'new_hist_purchase_amount_mean_hist_purchase_amount_mean_ratio',
                  'hist_installments_sum_hist_first_buy_sum',
                  'hist_month_diff_mean_hist_category_1_sum_diff',
                  'hist_authorized_flag_mean_hist_purchase_amount_max_sum',
                  'new_hist_purchase_date_uptonow_hist_first_buy_product',
                  'hist_installments_sum_hist_month_nunique_product',
                  'hist_category_1_sum_hist_purchase_amount_mean_ratio',
                  'hist_authorized_flag_mean_hist_purchase_amount_min_diff',
                  'new_hist_purchase_date_uptonow_hist_installments_sum_product',
                  'hist_purchase_amount_mean_hist_weekend_mean_product',
                  'hist_installments_sum_hist_first_buy_diff',
                  'hist_month_diff_mean_hist_weekend_mean_product',
                  'new_hist_purchase_amount_max_new_hist_purchase_amount_mean_ratio',
                  'hist_month_lag_mean_hist_purchase_date_average_ratio',
                  'new_hist_purchase_amount_max_hist_category_1_mean_diff',
                  'hist_authorized_flag_mean_hist_first_buy_ratio',
                  'hist_category_1_sum_hist_weekend_mean_ratio',
                  'hist_purchase_date_diff_hist_weekend_mean_ratio',
                  'new_hist_purchase_amount_max_hist_category_1_sum_sum',
                  'new_hist_purchase_amount_max_hist_purchase_amount_max_product',
                  'hist_month_lag_var_hist_purchase_date_diff_ratio',
                  'hist_purchase_amount_min_hist_category_3_mean_mean_ratio',
                  'hist_month_diff_mean_new_hist_purchase_amount_max_product',
                  'new_hist_purchase_date_uptonow_hist_category_1_mean_product',
                  'new_hist_purchase_amount_max_hist_installments_sum_ratio',
                  'hist_purchase_amount_mean_new_hist_month_lag_mean_diff',
                  'hist_month_lag_mean_hist_purchase_date_average_product',
                  'hist_installments_sum_new_hist_month_lag_mean_ratio',
                  'hist_weekofyear_nunique_hist_purchase_date_uptonow_diff',
                  'new_hist_purchase_amount_mean_hist_weekend_mean_sum',
                  'new_hist_purchase_amount_max_hist_purchase_amount_mean_product',
                  'hist_month_nunique_hist_purchase_amount_max_product',
                  'new_hist_purchase_amount_mean_new_hist_month_lag_mean_product',
                  'hist_installments_sum_hist_category_3_mean_mean_product',
                  'hist_merchant_id_nunique_hist_purchase_date_average_sum',
                  'hist_installments_sum_hist_category_3_mean_mean_ratio',
                  'new_hist_purchase_amount_max_hist_category_3_mean_mean_ratio',
                  'hist_purchase_amount_min_new_hist_month_lag_mean_product',
                  'hist_month_lag_mean_hist_purchase_amount_max_ratio',
                  'hist_month_nunique_hist_weekend_mean_ratio',
                  'new_hist_purchase_amount_max_hist_month_lag_var_product',
                  'new_hist_purchase_amount_max_hist_first_buy_product',
                  'new_hist_month_lag_mean_hist_purchase_date_average_ratio',
                  'hist_month_lag_var_new_hist_month_lag_mean_ratio',
                  'hist_purchase_amount_mean_hist_month_lag_var_product',
                  'new_hist_purchase_amount_max_hist_purchase_amount_min_ratio',
                  'hist_authorized_flag_mean_hist_purchase_amount_mean_product',
                  'new_hist_purchase_amount_mean_hist_purchase_amount_min_sum',
                  'hist_weekend_mean_hist_purchase_date_average_ratio',
                  'hist_category_1_sum_hist_purchase_date_average_diff',
                  'hist_category_3_mean_mean_hist_purchase_date_uptonow_ratio',
                  'hist_category_1_mean_hist_weekofyear_nunique_ratio',
                  'hist_month_nunique_new_hist_month_lag_mean_product',
                  'hist_merchant_id_nunique_hist_first_buy_diff',
                  'hist_merchant_id_nunique_hist_purchase_date_diff_diff',
                  'hist_purchase_date_diff_new_hist_month_lag_mean_product',
                  'hist_category_1_mean_hist_month_nunique_product',
                  'new_hist_purchase_amount_mean_hist_category_1_mean_sum',
                  'new_hist_purchase_date_uptonow_hist_purchase_amount_max_product',
                  'hist_month_nunique_hist_purchase_amount_max_ratio',
                  'hist_month_lag_mean_hist_weekofyear_nunique_sum',
                  'hist_weekend_mean_hist_purchase_date_uptonow_product',
                  'hist_month_diff_mean_hist_merchant_id_nunique_product',
                  'hist_weekend_mean_hist_purchase_amount_max_product',
                  'hist_month_diff_mean_new_hist_purchase_amount_mean_ratio',
                  'hist_month_diff_mean_new_hist_month_lag_mean_product',
                  'new_hist_month_lag_mean_hist_purchase_date_average_product',
                  'hist_category_1_sum_hist_purchase_amount_max_diff',
                  'hist_category_1_sum_hist_purchase_amount_mean_product',
                  'hist_authorized_flag_mean_new_hist_purchase_amount_mean_sum',
                  'hist_installments_sum_hist_purchase_date_diff_sum',
                  'hist_authorized_flag_mean_hist_purchase_amount_min_product',
                  'hist_authorized_flag_mean_hist_purchase_amount_max_ratio',
                  'hist_category_1_sum_hist_category_3_mean_mean_product',
                  'new_hist_purchase_amount_max_hist_first_buy_ratio',
                  'hist_month_lag_mean_hist_weekofyear_nunique_diff',
                  'hist_installments_sum_hist_purchase_date_diff_product',
                  'new_hist_month_lag_mean_hist_weekend_mean_ratio',
                  'hist_installments_sum_hist_weekofyear_nunique_product',
                  'new_hist_purchase_amount_max_hist_purchase_date_diff_product',
                  'new_hist_purchase_amount_mean_hist_month_lag_var_product',
                  'hist_category_1_sum_hist_month_lag_var_ratio',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_uptonow_sum',
                  'hist_month_lag_mean_hist_category_1_mean_ratio',
                  'hist_category_1_mean_hist_purchase_amount_min_sum',
                  'hist_installments_sum_hist_purchase_date_diff_diff',
                  'hist_category_1_sum_hist_weekofyear_nunique_sum',
                  'new_hist_purchase_amount_max_new_hist_month_lag_mean_product',
                  'hist_month_nunique_hist_purchase_date_uptonow_ratio',
                  'hist_purchase_amount_max_hist_weekofyear_nunique_sum',
                  'hist_category_1_sum_hist_category_3_mean_mean_ratio',
                  'new_hist_purchase_amount_max_hist_month_nunique_diff',
                  'hist_month_lag_mean_hist_purchase_amount_max_product',
                  'hist_purchase_amount_mean_hist_weekofyear_nunique_ratio',
                  'new_hist_purchase_amount_mean_hist_weekofyear_nunique_product',
                  'hist_purchase_date_diff_hist_purchase_amount_max_product',
                  'hist_first_buy_hist_purchase_date_uptonow_sum',
                  'new_hist_purchase_amount_max_hist_purchase_amount_mean_ratio',
                  'new_hist_purchase_date_uptonow_hist_first_buy_sum',
                  'new_hist_purchase_amount_max_hist_month_lag_mean_product',
                  'hist_purchase_amount_min_new_hist_month_lag_mean_ratio',
                  'hist_month_nunique_hist_first_buy_ratio',
                  'hist_month_lag_mean_hist_month_nunique_product',
                  'new_hist_purchase_date_uptonow_hist_month_lag_var_product',
                  'new_hist_purchase_amount_mean_hist_first_buy_ratio',
                  'hist_first_buy_hist_purchase_date_diff_sum',
                  'new_hist_purchase_amount_max_hist_month_nunique_product',
                  'hist_month_nunique_hist_weekofyear_nunique_product',
                  'hist_month_lag_mean_hist_category_1_mean_diff',
                  'new_hist_purchase_amount_max_hist_weekend_mean_diff',
                  'hist_merchant_id_nunique_hist_purchase_date_uptonow_product',
                  'hist_authorized_flag_mean_hist_purchase_date_max_ratio',
                  'hist_category_1_mean_hist_month_lag_var_ratio',
                  'hist_purchase_date_average_hist_category_3_mean_mean_ratio',
                  'new_hist_purchase_amount_mean_hist_installments_sum_product',
                  'new_hist_month_lag_mean_hist_weekofyear_nunique_product',
                  'hist_month_lag_mean_new_hist_purchase_amount_mean_product',
                  'new_hist_purchase_amount_mean_hist_weekend_mean_product',
                  'hist_merchant_id_nunique_new_hist_month_lag_mean_product',
                  'hist_month_lag_mean_hist_category_3_mean_mean_diff',
                  'hist_authorized_flag_mean_hist_category_1_sum_product',
                  'new_hist_purchase_date_uptonow_new_hist_purchase_amount_mean_ratio',
                  'hist_purchase_amount_max_hist_weekofyear_nunique_diff',
                  'new_hist_purchase_amount_mean_new_hist_purchase_amount_mean_product',
                  'hist_month_diff_mean_hist_month_lag_mean_ratio',
                  'hist_month_lag_mean_hist_purchase_date_uptonow_ratio',
                  'hist_purchase_amount_mean_hist_purchase_date_diff_ratio',
                  'hist_month_lag_mean_hist_category_1_mean_product',
                  'hist_authorized_flag_mean_new_hist_purchase_amount_mean_product',
                  'hist_category_1_sum_hist_purchase_date_average_sum',
                  'hist_authorized_flag_mean_hist_month_nunique_sum',
                  'hist_purchase_date_min_hist_purchase_date_max_product',
                  'new_hist_month_lag_mean_hist_category_3_mean_mean_ratio',
                  'new_hist_purchase_amount_max_hist_category_1_mean_product',
                  'hist_purchase_date_average_hist_purchase_date_uptonow_diff',
                  'new_hist_purchase_amount_max_hist_purchase_date_diff_ratio',
                  'hist_purchase_date_diff_hist_purchase_amount_max_ratio',
                  'hist_category_1_sum_hist_month_lag_var_sum',
                  'hist_month_lag_var_hist_purchase_amount_max_product',
                  'hist_month_lag_var_hist_merchant_id_nunique_sum',
                  'hist_category_1_sum_hist_purchase_date_diff_sum',
                  'hist_purchase_amount_min_hist_month_nunique_sum',
                  'hist_first_buy_new_hist_month_lag_mean_product',
                  'hist_purchase_amount_mean_hist_month_nunique_ratio',
                  'hist_month_nunique_new_hist_month_lag_mean_sum',
                  'hist_purchase_date_min_hist_month_nunique_ratio',
                  'new_hist_purchase_amount_mean_hist_first_buy_product',
                  'hist_month_nunique_hist_purchase_date_average_sum',
                  'new_hist_purchase_amount_max_hist_authorized_flag_mean_sum',
                  'hist_category_1_sum_hist_month_lag_var_diff',
                  'hist_category_1_sum_hist_merchant_id_nunique_ratio',
                  'hist_purchase_amount_mean_hist_month_lag_var_ratio',
                  'hist_month_diff_mean_hist_purchase_amount_max_ratio',
                  'hist_authorized_flag_mean_hist_category_1_mean_product',
                  'hist_category_1_sum_new_hist_month_lag_mean_ratio',
                  'hist_authorized_flag_mean_hist_merchant_id_nunique_product',
                  'hist_category_1_sum_hist_merchant_id_nunique_sum',
                  'new_hist_purchase_amount_max_hist_weekend_mean_product',
                  'hist_authorized_flag_mean_hist_purchase_date_average_product',
                  'hist_month_diff_mean_hist_purchase_date_diff_ratio',
                  'hist_category_1_mean_hist_category_3_mean_mean_ratio',
                  'hist_month_diff_mean_hist_purchase_date_min_ratio',
                  'hist_month_lag_var_hist_purchase_date_average_product',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_average_product',
                  'new_hist_purchase_date_uptonow_hist_merchant_id_nunique_ratio',
                  'hist_category_1_sum_hist_weekofyear_nunique_diff',
                  'hist_month_lag_mean_hist_purchase_amount_mean_diff',
                  'hist_category_1_mean_hist_weekend_mean_ratio',
                  'new_hist_month_lag_mean_hist_purchase_amount_max_sum',
                  'new_hist_purchase_amount_mean_hist_purchase_date_average_product',
                  'hist_purchase_amount_mean_hist_purchase_amount_min_sum',
                  'new_hist_purchase_amount_mean_hist_merchant_id_nunique_product',
                  'hist_category_1_sum_hist_installments_sum_diff',
                  'hist_first_buy_hist_purchase_date_diff_product',
                  'hist_category_1_sum_hist_purchase_date_average_product',
                  'new_hist_purchase_amount_mean_hist_purchase_date_diff_product',
                  'hist_purchase_date_diff_hist_purchase_date_uptonow_diff',
                  'hist_installments_sum_hist_month_nunique_ratio',
                  'hist_category_1_mean_hist_purchase_date_uptonow_product',
                  'new_hist_purchase_amount_mean_hist_category_3_mean_mean_product',
                  'new_hist_purchase_amount_mean_hist_month_nunique_diff',
                  'hist_category_1_sum_hist_purchase_amount_max_sum',
                  'new_hist_month_lag_mean_hist_weekend_mean_product',
                  'hist_purchase_date_average_hist_weekofyear_nunique_diff',
                  'hist_month_diff_mean_hist_month_lag_var_diff',
                  'hist_authorized_flag_mean_hist_first_buy_product',
                  'new_hist_purchase_date_uptonow_hist_month_nunique_sum',
                  'new_hist_purchase_amount_mean_hist_category_3_mean_mean_diff',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_average_sum',
                  'hist_first_buy_hist_purchase_date_uptonow_product',
                  'hist_month_lag_var_hist_purchase_date_average_sum',
                  'new_hist_purchase_date_uptonow_hist_category_1_sum_product',
                  'new_hist_purchase_amount_max_hist_weekofyear_nunique_product',
                  'hist_purchase_date_average_hist_purchase_amount_max_product',
                  'new_hist_purchase_amount_max_hist_authorized_flag_mean_product',
                  'new_hist_purchase_amount_mean_hist_weekofyear_nunique_ratio',
                  'new_hist_purchase_amount_max_hist_purchase_date_min_ratio',
                  'hist_month_diff_mean_new_hist_month_lag_mean_sum',
                  'new_hist_purchase_date_uptonow_hist_weekend_mean_product',
                  'hist_month_diff_mean_hist_month_lag_var_product',
                  'hist_category_1_mean_hist_purchase_amount_max_sum',
                  'hist_month_lag_var_new_hist_month_lag_mean_product',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_average_ratio',
                  'hist_month_lag_var_hist_installments_sum_ratio',
                  'hist_category_1_sum_hist_month_lag_mean_diff',
                  'hist_purchase_date_average_hist_purchase_amount_max_diff',
                  'hist_category_1_sum_hist_month_lag_mean_ratio',
                  'hist_merchant_id_nunique_hist_purchase_amount_max_sum',
                  'hist_category_1_mean_hist_month_nunique_diff',
                  'new_hist_purchase_date_uptonow_hist_month_nunique_diff',
                  'hist_month_lag_mean_hist_month_nunique_diff',
                  'hist_category_1_sum_hist_purchase_date_average_ratio',
                  'hist_month_diff_mean_hist_weekofyear_nunique_sum',
                  'hist_purchase_date_min_hist_category_3_mean_mean_ratio',
                  'hist_month_diff_mean_new_hist_purchase_date_uptonow_product',
                  'hist_authorized_flag_mean_hist_purchase_date_min_product',
                  'new_hist_purchase_amount_max_hist_purchase_amount_mean_sum',
                  'new_hist_purchase_amount_max_hist_purchase_date_uptonow_ratio',
                  'hist_month_diff_mean_hist_purchase_date_average_ratio',
                  'hist_merchant_id_nunique_hist_first_buy_sum',
                  'hist_month_lag_mean_hist_purchase_date_diff_product',
                  'hist_month_diff_mean_hist_weekend_mean_ratio',
                  'hist_weekend_mean_hist_purchase_date_uptonow_ratio',
                  'hist_month_lag_var_hist_month_nunique_ratio',
                  'hist_month_diff_mean_new_hist_purchase_amount_mean_diff',
                  'new_hist_purchase_date_uptonow_hist_purchase_amount_max_ratio',
                  'hist_purchase_amount_max_hist_purchase_date_uptonow_sum',
                  'new_hist_purchase_amount_mean_hist_purchase_amount_max_product',
                  'hist_purchase_date_min_new_hist_purchase_amount_mean_ratio',
                  'hist_category_3_mean_mean_hist_category_3_mean_mean_product',
                  'hist_merchant_id_nunique_hist_installments_sum_product',
                  'hist_month_diff_mean_hist_category_1_sum_sum',
                  'hist_category_1_mean_new_hist_month_lag_mean_ratio',
                  'new_hist_purchase_amount_max_new_hist_month_lag_mean_ratio',
                  'hist_month_diff_mean_hist_purchase_date_average_product',
                  'hist_category_1_sum_new_hist_month_lag_mean_product',
                  'hist_purchase_amount_min_hist_weekend_mean_diff',
                  'new_hist_purchase_amount_mean_hist_weekend_mean_diff',
                  'new_hist_purchase_amount_max_hist_weekend_mean_ratio',
                  'new_hist_month_lag_mean_hist_purchase_amount_max_ratio',
                  'hist_month_lag_mean_hist_purchase_amount_mean_sum',
                  'hist_category_1_mean_hist_purchase_date_average_product',
                  'hist_merchant_id_nunique_hist_month_nunique_diff',
                  'hist_month_lag_mean_hist_category_1_mean_sum',
                  'hist_month_diff_mean_hist_weekofyear_nunique_ratio',
                  'new_hist_purchase_amount_mean_hist_month_nunique_ratio',
                  'new_hist_purchase_amount_mean_hist_purchase_amount_max_ratio',
                  'hist_category_1_sum_hist_weekofyear_nunique_ratio',
                  'hist_purchase_amount_max_hist_purchase_date_uptonow_diff',
                  'hist_category_1_mean_hist_category_3_mean_mean_diff',
                  'new_hist_purchase_amount_max_hist_merchant_id_nunique_ratio',
                  'hist_purchase_amount_mean_new_hist_month_lag_mean_product',
                  'new_hist_purchase_date_uptonow_hist_month_nunique_ratio',
                  'new_hist_purchase_amount_max_hist_purchase_date_average_ratio',
                  'hist_purchase_date_min_hist_purchase_amount_mean_ratio',
                  'hist_category_1_sum_hist_purchase_date_diff_diff',
                  'hist_category_1_mean_hist_purchase_date_diff_ratio',
                  'hist_month_diff_mean_hist_purchase_date_min_sum',
                  'new_hist_purchase_amount_mean_hist_purchase_amount_mean_sum',
                  'hist_month_nunique_hist_purchase_date_uptonow_diff',
                  'new_hist_purchase_date_uptonow_hist_weekend_mean_ratio',
                  'hist_month_diff_mean_new_hist_month_lag_mean_diff',
                  'new_hist_purchase_amount_max_hist_month_lag_mean_ratio',
                  'hist_authorized_flag_mean_hist_purchase_date_average_ratio',
                  'new_hist_month_lag_mean_hist_category_3_mean_mean_product',
                  'hist_purchase_amount_max_hist_category_3_mean_mean_sum',
                  'hist_authorized_flag_mean_hist_purchase_date_min_ratio',
                  'new_hist_purchase_amount_max_hist_authorized_flag_mean_ratio',
                  'new_hist_purchase_amount_max_hist_purchase_amount_min_sum',
                  'hist_first_buy_hist_purchase_date_diff_ratio',
                  'hist_installments_sum_hist_purchase_date_uptonow_product',
                  'hist_purchase_date_max_hist_category_3_mean_mean_ratio',
                  'hist_purchase_amount_min_hist_category_3_mean_mean_product',
                  'hist_category_1_sum_hist_month_lag_mean_sum',
                  'hist_month_diff_mean_hist_installments_sum_product',
                  'new_hist_purchase_amount_mean_hist_weekend_mean_ratio',
                  'hist_authorized_flag_mean_hist_installments_sum_product',
                  'hist_purchase_amount_max_hist_purchase_date_uptonow_product',
                  'hist_month_lag_mean_hist_month_lag_var_diff',
                  'new_hist_purchase_date_uptonow_hist_month_lag_mean_ratio',
                  'hist_category_1_sum_hist_installments_sum_sum',
                  'hist_merchant_id_nunique_hist_installments_sum_diff',
                  'hist_purchase_amount_min_hist_weekofyear_nunique_ratio',
                  'hist_purchase_date_max_hist_month_nunique_product',
                  'hist_month_diff_mean_hist_purchase_date_min_product',
                  'hist_month_lag_mean_hist_category_3_mean_mean_sum',
                  'hist_month_lag_var_hist_purchase_date_uptonow_product',
                  'hist_purchase_date_max_new_hist_purchase_amount_mean_ratio',
                  'new_hist_purchase_date_uptonow_hist_purchase_amount_max_sum',
                  'hist_month_nunique_hist_purchase_date_diff_product',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_diff_diff',
                  'hist_month_lag_mean_hist_month_lag_var_product',
                  'hist_installments_sum_hist_purchase_date_average_ratio',
                  'hist_month_lag_mean_new_hist_month_lag_mean_diff',
                  'new_hist_purchase_amount_mean_hist_category_1_mean_product',
                  'new_hist_purchase_amount_max_hist_purchase_date_min_sum',
                  'hist_month_diff_mean_hist_month_nunique_ratio',
                  'hist_month_lag_var_hist_purchase_date_uptonow_diff',
                  'hist_month_diff_mean_hist_merchant_id_nunique_diff',
                  'new_hist_purchase_amount_mean_hist_month_lag_var_ratio',
                  'hist_authorized_flag_mean_hist_category_1_sum_sum',
                  'hist_merchant_id_nunique_hist_installments_sum_sum',
                  'new_hist_purchase_amount_max_hist_month_lag_var_ratio',
                  'hist_category_1_sum_hist_first_buy_diff',
                  'new_hist_purchase_date_uptonow_hist_weekofyear_nunique_ratio',
                  'new_hist_purchase_date_uptonow_hist_purchase_amount_min_ratio',
                  'hist_weekofyear_nunique_hist_purchase_date_uptonow_ratio',
                  'new_hist_purchase_amount_max_hist_purchase_date_average_diff',
                  'hist_installments_sum_hist_purchase_amount_max_diff',
                  'hist_merchant_id_nunique_hist_purchase_date_average_ratio',
                  'hist_category_1_mean_hist_purchase_amount_max_diff',
                  'hist_purchase_amount_mean_hist_purchase_amount_max_ratio',
                  'hist_month_nunique_new_hist_month_lag_mean_diff',
                  'hist_purchase_amount_min_hist_month_nunique_ratio',
                  'hist_installments_sum_new_hist_month_lag_mean_product',
                  'new_hist_purchase_amount_max_hist_month_lag_mean_sum',
                  'new_hist_purchase_amount_max_new_hist_purchase_amount_mean_product',
                  'hist_category_1_mean_hist_merchant_id_nunique_ratio',
                  'new_hist_purchase_amount_max_hist_purchase_amount_min_product',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_average_diff',
                  'hist_month_diff_mean_hist_weekofyear_nunique_diff',
                  'new_hist_purchase_amount_mean_hist_month_nunique_sum',
                  'new_hist_purchase_date_uptonow_hist_purchase_amount_max_diff',
                  'new_hist_purchase_date_uptonow_hist_purchase_date_uptonow_product',
                  'hist_month_nunique_hist_purchase_date_uptonow_sum',
                  'hist_installments_sum_hist_purchase_date_average_sum',
                  'hist_authorized_flag_mean_hist_month_lag_mean_diff',
                  'hist_authorized_flag_mean_hist_month_lag_mean_sum',
                  'hist_purchase_date_diff_hist_purchase_date_uptonow_sum',
                  'new_hist_purchase_amount_max_new_hist_purchase_amount_mean_sum',
                  'hist_purchase_amount_mean_hist_purchase_amount_min_diff',
                  'new_hist_purchase_amount_max_hist_category_1_sum_diff',
                  'hist_month_nunique_hist_purchase_date_average_ratio',
                  'hist_month_diff_mean_hist_merchant_id_nunique_sum',
                  'hist_month_diff_mean_hist_purchase_amount_max_product',
                  'new_hist_purchase_amount_mean_hist_purchase_date_diff_ratio',
                  'hist_purchase_amount_mean_hist_installments_sum_ratio',
                  'hist_authorized_flag_mean_hist_purchase_date_max_product',
                  'new_hist_purchase_amount_max_hist_installments_sum_sum',
                  'hist_purchase_amount_mean_hist_purchase_amount_min_product',
                  'hist_category_1_sum_hist_category_1_mean_product',
                  'hist_purchase_amount_max_hist_weekofyear_nunique_ratio',
                  'new_hist_purchase_amount_max_hist_month_lag_mean_diff',
                  'new_hist_purchase_amount_max_new_hist_purchase_date_uptonow_ratio',
                  'hist_purchase_date_max_hist_category_3_mean_mean_product',
                  'new_hist_purchase_amount_mean_hist_purchase_amount_min_ratio',
                  'new_hist_month_lag_mean_hist_purchase_date_average_sum',
                  'hist_weekend_mean_hist_purchase_amount_max_sum',
                  'hist_purchase_date_average_hist_weekofyear_nunique_ratio',
                  'new_hist_purchase_amount_mean_hist_purchase_amount_mean_product',
                  'hist_month_lag_mean_new_hist_month_lag_mean_sum',
                  'hist_month_diff_mean_hist_purchase_amount_min_diff',
                  'hist_purchase_amount_mean_hist_purchase_date_average_ratio',
                  'hist_authorized_flag_mean_hist_category_1_mean_ratio',
                  'hist_category_1_sum_hist_month_nunique_ratio',
                  'hist_month_diff_mean_hist_category_1_sum_product',
                  'hist_month_lag_var_hist_month_nunique_sum',
                  'hist_month_lag_mean_hist_purchase_date_uptonow_diff',
                  'hist_weekend_mean_hist_purchase_amount_max_diff',
                  'hist_authorized_flag_mean_hist_purchase_amount_max_diff',
                  'hist_purchase_date_max_hist_purchase_amount_mean_ratio',
                  'new_hist_month_lag_mean_hist_purchase_amount_max_diff',
                  'hist_category_1_sum_hist_purchase_amount_min_sum',
                  'new_hist_purchase_amount_mean_hist_purchase_date_uptonow_ratio',
                  'new_hist_purchase_amount_mean_hist_merchant_id_nunique_ratio',
                  'hist_month_lag_mean_hist_weekend_mean_diff',
                  'hist_purchase_date_diff_hist_weekofyear_nunique_diff',
                  'hist_category_1_sum_hist_purchase_date_uptonow_product',
                  'hist_installments_sum_hist_weekofyear_nunique_diff',
                  'hist_installments_sum_hist_purchase_date_diff_ratio',
                  'new_hist_purchase_amount_mean_hist_purchase_date_average_ratio',
                  'hist_purchase_date_max_new_hist_purchase_amount_mean_product',
                  'hist_month_lag_var_hist_month_nunique_product',
                  'hist_purchase_date_average_hist_purchase_date_uptonow_ratio',
                  'hist_purchase_amount_mean_hist_category_1_mean_ratio',
                  'new_hist_purchase_date_uptonow_hist_month_lag_var_ratio',
                  'hist_month_lag_mean_new_hist_purchase_amount_mean_diff',
                  'hist_month_diff_mean_hist_first_buy_diff',
                  'hist_first_buy_hist_weekofyear_nunique_diff',
                  'new_hist_purchase_date_uptonow_hist_month_lag_var_diff',
                  'hist_first_buy_hist_purchase_date_uptonow_ratio',
                  'hist_month_lag_mean_hist_purchase_amount_min_product',
                  'hist_authorized_flag_mean_hist_purchase_date_min_diff',
                  'new_hist_purchase_date_uptonow_hist_first_buy_ratio',
                  'hist_purchase_amount_max_hist_category_3_mean_mean_diff',
                  'new_hist_purchase_amount_mean_hist_installments_sum_ratio',
                  'hist_month_diff_mean_hist_installments_sum_ratio',
                  'hist_month_diff_mean_hist_installments_sum_diff',
                  'hist_purchase_amount_min_hist_purchase_amount_min_product',
                  'new_hist_purchase_amount_max_hist_purchase_amount_max_sum',
                  'new_hist_purchase_amount_max_hist_purchase_date_min_diff',
                  'hist_purchase_date_average_hist_purchase_date_uptonow_product',
                  'new_hist_purchase_amount_mean_hist_purchase_amount_max_sum',
                  'new_hist_purchase_amount_mean_hist_purchase_amount_min_product',
                  'hist_purchase_date_min_new_hist_purchase_amount_mean_product',
                  'hist_category_1_sum_new_hist_purchase_amount_mean_diff',
                  'hist_purchase_date_diff_hist_purchase_date_uptonow_ratio',
                  'new_hist_purchase_amount_max_hist_purchase_date_min_product',
                  'hist_month_lag_mean_new_hist_purchase_amount_mean_sum',
                  'hist_month_lag_mean_hist_merchant_id_nunique_sum',
                  'new_hist_purchase_date_uptonow_hist_category_1_sum_ratio',
                  'hist_authorized_flag_mean_hist_purchase_amount_max_product',
                  'hist_purchase_amount_mean_hist_purchase_amount_min_ratio',
                  'hist_month_lag_var_hist_first_buy_sum',
                  'new_hist_purchase_amount_max_hist_purchase_date_max_product',
                  'hist_authorized_flag_mean_hist_weekend_mean_diff',
                  'hist_purchase_amount_max_hist_category_3_mean_mean_product',
                  'hist_month_diff_mean_hist_purchase_amount_max_diff',
                  'hist_month_lag_var_hist_first_buy_ratio',
                  'hist_purchase_amount_mean_hist_category_3_mean_mean_diff',
                  'hist_month_lag_mean_new_hist_purchase_amount_mean_ratio',
                  'new_hist_purchase_amount_mean_hist_purchase_amount_max_diff',
                  'hist_merchant_id_nunique_hist_purchase_amount_max_ratio',
                  'hist_category_1_sum_hist_installments_sum_product',
                  'hist_category_1_mean_hist_purchase_amount_max_product',
                  'hist_authorized_flag_mean_new_hist_month_lag_mean_sum',
                  'hist_authorized_flag_mean_hist_month_nunique_diff',
                  'hist_purchase_amount_mean_new_hist_month_lag_mean_sum',
                  'hist_category_1_sum_hist_purchase_date_uptonow_sum',
                  'hist_installments_sum_hist_weekend_mean_product',
                  'new_hist_purchase_amount_max_hist_category_1_sum_product',
                  'hist_category_1_sum_hist_purchase_date_diff_product',
                  'hist_purchase_amount_mean_hist_merchant_id_nunique_ratio',
                  'hist_authorized_flag_mean_hist_purchase_date_uptonow_product',
                  'new_hist_purchase_date_uptonow_hist_first_buy_diff',
                  'hist_installments_sum_hist_purchase_date_uptonow_sum',
                  'new_hist_purchase_amount_mean_hist_month_nunique_product',
                  'hist_category_1_sum_hist_month_lag_mean_product',
                  'hist_weekofyear_nunique_hist_purchase_date_uptonow_sum',
                  'hist_merchant_id_nunique_hist_first_buy_ratio',
                  'hist_purchase_amount_mean_hist_weekofyear_nunique_product',
                  'new_hist_purchase_amount_max_hist_month_nunique_ratio',
                  'hist_month_diff_mean_hist_category_1_mean_diff',
                  'new_hist_purchase_amount_mean_hist_category_3_mean_mean_sum',
                  'hist_month_diff_mean_new_hist_purchase_amount_max_ratio',
                  'hist_purchase_amount_max_hist_category_3_mean_mean_ratio',
                  'hist_authorized_flag_mean_hist_weekend_mean_sum',
                  'hist_merchant_id_nunique_hist_purchase_amount_max_product',
                  'hist_category_1_mean_hist_month_nunique_sum',
                  'new_hist_purchase_amount_max_hist_installments_sum_product',
                  'hist_category_1_sum_hist_purchase_amount_max_ratio',
                  'hist_month_lag_mean_hist_merchant_id_nunique_ratio',
                  'hist_purchase_date_average_hist_weekofyear_nunique_sum',
                  'hist_category_1_mean_hist_weekofyear_nunique_product',
                  'hist_category_1_mean_hist_weekend_mean_product',
                  'hist_merchant_id_nunique_hist_installments_sum_ratio',
                  'hist_month_lag_var_hist_weekofyear_nunique_diff',
                  'new_hist_purchase_amount_max_hist_authorized_flag_mean_diff',
                  'hist_purchase_amount_min_hist_purchase_date_uptonow_product',
                  'new_hist_purchase_amount_max_hist_purchase_date_average_sum',
                  'hist_category_1_sum_hist_first_buy_ratio',
                  'hist_category_1_mean_hist_installments_sum_ratio',
                  'hist_authorized_flag_mean_hist_category_3_mean_mean_diff',
                  'hist_weekend_mean_hist_weekofyear_nunique_ratio',
                  'hist_month_diff_mean_hist_category_3_mean_mean_ratio',
                  'hist_category_1_mean_hist_purchase_amount_max_ratio',
                  'new_hist_purchase_date_uptonow_hist_merchant_id_nunique_sum',
                  'hist_month_diff_mean_hist_purchase_date_min_diff',
                  'hist_authorized_flag_mean_hist_purchase_amount_min_sum',
                  'new_hist_purchase_amount_max_hist_purchase_date_max_ratio',
                  'hist_authorized_flag_mean_hist_merchant_id_nunique_ratio',
                  'new_hist_purchase_date_uptonow_hist_weekofyear_nunique_sum',
                  'hist_month_lag_mean_hist_purchase_amount_max_sum',
                  'hist_installments_sum_hist_purchase_date_average_product',
                  'hist_purchase_amount_mean_hist_category_1_mean_sum',
                  'hist_category_1_mean_hist_month_lag_var_product',
                  'hist_first_buy_hist_purchase_date_uptonow_diff',
                  'new_hist_purchase_amount_max_new_hist_month_lag_mean_sum',
                  'hist_category_1_mean_new_hist_month_lag_mean_sum',
                  'hist_month_nunique_new_hist_month_lag_mean_ratio',
                  'hist_purchase_amount_mean_hist_purchase_date_average_product',
                  'hist_month_lag_var_hist_weekend_mean_ratio',
                  'hist_month_lag_mean_new_hist_month_lag_mean_ratio',
                  'hist_purchase_date_diff_new_hist_month_lag_mean_ratio',
                  'hist_month_nunique_hist_purchase_date_average_diff',
                  'hist_category_1_mean_hist_category_3_mean_mean_sum',
                  'hist_month_lag_mean_hist_weekend_mean_product',
                  'hist_category_1_sum_hist_purchase_date_uptonow_diff',
                  'hist_purchase_date_min_hist_purchase_date_max_sum',
                  'hist_month_diff_mean_hist_purchase_date_average_sum',
                  'hist_month_lag_var_hist_merchant_id_nunique_ratio',
                  'hist_authorized_flag_mean_hist_category_1_sum_ratio',
                  'new_hist_purchase_date_uptonow_hist_category_1_sum_sum',
                  'new_hist_purchase_date_uptonow_hist_merchant_id_nunique_product',
                  'hist_merchant_id_nunique_hist_weekend_mean_product',
                  'new_hist_purchase_amount_max_new_hist_purchase_amount_max_product',
                  'new_hist_month_lag_mean_hist_category_3_mean_mean_sum',
                  'new_hist_purchase_date_uptonow_hist_installments_sum_sum',
                  'hist_category_1_mean_hist_weekend_mean_sum',
                  'new_hist_purchase_amount_max_hist_merchant_id_nunique_product',
                  'hist_purchase_amount_min_hist_month_nunique_diff',
                  'hist_category_1_mean_hist_first_buy_ratio',
                  'new_hist_purchase_date_uptonow_hist_purchase_amount_min_product',
                  'hist_weekend_mean_hist_category_3_mean_mean_sum',
                  'hist_merchant_id_nunique_hist_category_3_mean_mean_product',
                  'hist_month_diff_mean_hist_purchase_date_uptonow_product',
                  'hist_purchase_date_min_hist_purchase_amount_min_product',
                  'hist_month_lag_mean_hist_purchase_amount_max_diff',
                  'hist_merchant_id_nunique_hist_first_buy_product',
                  'hist_merchant_id_nunique_hist_category_3_mean_mean_ratio',
                  'hist_installments_sum_hist_purchase_date_uptonow_diff',
                  'hist_purchase_amount_max_hist_weekofyear_nunique_product',
                  'hist_merchant_id_nunique_hist_purchase_date_diff_ratio',
                  'hist_installments_sum_hist_purchase_amount_max_ratio',
                  'hist_month_diff_mean_new_hist_month_lag_mean_ratio',
                  'hist_month_lag_var_hist_purchase_amount_max_ratio',
                  'hist_month_lag_var_hist_category_3_mean_mean_product',
                  'hist_month_lag_var_hist_purchase_date_average_ratio',
                  'hist_month_lag_mean_new_hist_month_lag_mean_product',
                  'hist_category_3_mean_mean_hist_purchase_date_uptonow_product',
                  'hist_month_lag_mean_hist_merchant_id_nunique_product',
                  'hist_month_diff_mean_hist_authorized_flag_mean_product',
                  'hist_first_buy_hist_weekofyear_nunique_ratio',
                  'new_hist_month_lag_mean_hist_weekofyear_nunique_ratio',
                  'hist_authorized_flag_mean_hist_category_3_mean_mean_ratio',
                  'hist_month_lag_mean_hist_installments_sum_product',
                  'hist_purchase_amount_mean_hist_month_nunique_sum',
                  'hist_purchase_amount_mean_hist_month_nunique_product',
                  'hist_month_lag_mean_hist_weekofyear_nunique_product',
                  'hist_purchase_amount_mean_hist_purchase_date_diff_product',
                  'hist_weekend_mean_hist_purchase_amount_max_ratio',
                  'hist_month_lag_var_hist_weekend_mean_product',
                  'hist_month_diff_mean_hist_purchase_amount_mean_ratio',
                  'hist_authorized_flag_mean_new_hist_month_lag_mean_diff',
                  'hist_first_buy_hist_purchase_date_diff_diff',
                  'hist_purchase_amount_mean_hist_purchase_amount_max_product',
                  'new_hist_purchase_date_uptonow_hist_category_1_sum_diff',
                  'hist_month_lag_var_hist_installments_sum_product',
                  'hist_purchase_date_average_hist_purchase_amount_max_ratio',
                  'hist_month_lag_mean_hist_purchase_amount_mean_product',
                  'hist_merchant_id_nunique_hist_purchase_date_uptonow_diff',
                  'hist_month_diff_mean_hist_purchase_amount_max_sum',
                  'hist_purchase_amount_mean_hist_category_1_mean_diff',
                  'new_hist_purchase_amount_max_hist_purchase_amount_max_ratio',
                  'hist_authorized_flag_mean_new_hist_purchase_amount_mean_ratio',
                  'hist_category_1_sum_hist_month_nunique_product',
                  'new_hist_purchase_amount_max_new_hist_purchase_date_uptonow_product',
                  'hist_purchase_amount_mean_hist_category_3_mean_mean_product',
                  'new_hist_purchase_amount_mean_hist_category_1_mean_diff',
                  'hist_purchase_date_diff_hist_category_3_mean_mean_ratio',
                  'hist_merchant_id_nunique_new_hist_month_lag_mean_ratio',
                  'hist_purchase_amount_min_new_hist_month_lag_mean_diff',
                  'hist_first_buy_hist_purchase_date_average_ratio',
                  'hist_purchase_amount_mean_hist_weekend_mean_diff',
                  'hist_month_nunique_hist_purchase_date_average_product',
                  'hist_purchase_date_min_new_hist_month_lag_mean_ratio',
                  'hist_month_lag_mean_hist_purchase_amount_mean_ratio',
                  'hist_month_diff_mean_hist_weekend_mean_diff',
                  'hist_weekend_mean_hist_weekofyear_nunique_product',
                  'hist_month_lag_mean_hist_installments_sum_ratio',
                  'hist_month_diff_mean_hist_month_lag_var_sum',
                  'hist_month_lag_mean_hist_purchase_date_uptonow_product',
                  'hist_purchase_amount_mean_hist_category_3_mean_mean_sum',
                  'new_hist_purchase_date_uptonow_hist_month_lag_mean_product',
                  'hist_category_1_mean_hist_installments_sum_product',
                  'hist_month_lag_var_hist_weekofyear_nunique_sum',
                  'hist_purchase_date_max_new_hist_month_lag_mean_ratio',
                  'hist_category_1_sum_new_hist_purchase_amount_mean_ratio',
                  'hist_purchase_amount_mean_hist_month_nunique_diff',
                  'new_hist_month_lag_mean_hist_weekend_mean_sum',
                  'new_hist_purchase_date_uptonow_hist_weekofyear_nunique_product',
                  'hist_month_nunique_hist_weekend_mean_product',
                  'hist_purchase_date_diff_hist_purchase_date_average_product',
                  'hist_category_1_sum_hist_installments_sum_ratio',
                  'hist_month_diff_mean_new_hist_purchase_amount_max_diff',
                  'hist_month_nunique_hist_purchase_amount_max_sum',
                  'hist_purchase_date_diff_hist_weekofyear_nunique_product',
                  'hist_month_diff_mean_hist_purchase_amount_mean_diff',
                  'new_hist_month_lag_mean_hist_purchase_date_uptonow_product',
                  'new_hist_month_lag_mean_hist_category_3_mean_mean_diff',
                  'hist_authorized_flag_mean_new_hist_purchase_date_uptonow_product',
                  'hist_month_lag_var_hist_category_3_mean_mean_ratio',
                  'hist_month_nunique_hist_category_3_mean_mean_sum',
                  'hist_month_lag_mean_hist_category_3_mean_mean_ratio',
                  'new_hist_purchase_date_uptonow_new_hist_month_lag_mean_product',
                  'hist_purchase_amount_mean_hist_first_buy_product',
                  'hist_category_3_mean_mean_hist_weekofyear_nunique_product',
                  'hist_month_nunique_hist_weekofyear_nunique_ratio',
                  'hist_merchant_id_nunique_hist_weekend_mean_ratio',
                  'new_hist_month_lag_mean_hist_weekend_mean_diff',
                  'hist_month_nunique_hist_weekend_mean_sum',
                  'hist_first_buy_new_hist_month_lag_mean_ratio',
                  'hist_month_lag_var_hist_merchant_id_nunique_diff',
                  'hist_authorized_flag_mean_hist_weekend_mean_ratio',
                  'hist_installments_sum_hist_weekend_mean_ratio',
                  'hist_authorized_flag_mean_hist_purchase_amount_mean_ratio',
                  'hist_first_buy_hist_category_3_mean_mean_product',
                  'hist_month_diff_mean_hist_month_nunique_diff',
                  'hist_authorized_flag_mean_hist_weekend_mean_product',
                  'hist_authorized_flag_mean_hist_installments_sum_ratio',
                  'new_hist_purchase_date_uptonow_hist_merchant_id_nunique_diff',
                  'hist_month_lag_var_hist_weekofyear_nunique_ratio',
                  'hist_month_diff_mean_hist_month_lag_mean_product',
                  'new_hist_purchase_date_uptonow_hist_category_3_mean_mean_ratio',
                  'hist_weekend_mean_hist_category_3_mean_mean_ratio',
                  'hist_category_1_sum_hist_merchant_id_nunique_diff',
                  'hist_purchase_amount_min_hist_month_nunique_product',
                  'hist_authorized_flag_mean_hist_purchase_amount_mean_diff',
                  'hist_month_lag_mean_hist_weekend_mean_ratio',
                  'hist_month_lag_var_hist_first_buy_product',
                  'hist_first_buy_hist_weekend_mean_product',
                  'hist_month_diff_mean_new_hist_purchase_amount_mean_product',
                  'hist_authorized_flag_mean_hist_weekofyear_nunique_product',
                  'hist_month_diff_mean_hist_weekend_mean_sum',
                  'hist_authorized_flag_mean_new_hist_purchase_date_uptonow_ratio',
                  'hist_purchase_date_average_hist_category_3_mean_mean_product',
                  'hist_authorized_flag_mean_hist_month_lag_var_product',
                  'new_hist_purchase_date_uptonow_hist_purchase_amount_mean_ratio',
                  'hist_purchase_amount_max_hist_purchase_amount_max_product',
                  'hist_month_lag_var_hist_purchase_amount_max_diff',
                  'hist_month_lag_mean_hist_purchase_date_average_diff',
                  'hist_month_lag_var_hist_purchase_date_average_diff',
                  'hist_purchase_amount_mean_hist_merchant_id_nunique_product',
                  'hist_purchase_amount_mean_hist_category_1_mean_product',
                  'new_hist_purchase_amount_max_hist_category_3_mean_mean_product',
                  'hist_month_lag_mean_hist_purchase_date_average_sum',
                  'hist_month_lag_var_hist_weekofyear_nunique_product',
                  'hist_purchase_date_diff_hist_category_3_mean_mean_product',
                  'hist_authorized_flag_mean_hist_month_lag_var_ratio',
                  'new_hist_month_lag_mean_hist_purchase_date_uptonow_ratio',
                  'new_hist_purchase_amount_mean_hist_purchase_date_uptonow_product',
                  'hist_weekend_mean_hist_category_3_mean_mean_product',
                  'hist_weekend_mean_hist_category_3_mean_mean_diff',
                  'hist_merchant_id_nunique_hist_weekofyear_nunique_product',
                  'new_hist_purchase_date_uptonow_new_hist_purchase_amount_mean_product',
                  'hist_month_diff_mean_hist_purchase_amount_mean_sum',
                  'hist_purchase_amount_min_hist_purchase_date_uptonow_ratio',
                  'hist_authorized_flag_mean_new_hist_month_lag_mean_ratio',
                  'hist_month_diff_mean_new_hist_purchase_amount_mean_sum',
                  'hist_purchase_amount_mean_new_hist_month_lag_mean_ratio',
                  'hist_month_diff_mean_hist_authorized_flag_mean_sum',
                  'hist_merchant_id_nunique_hist_month_nunique_product',
                  'hist_month_diff_mean_hist_weekofyear_nunique_product',
                  'hist_purchase_date_diff_hist_purchase_date_uptonow_product',
                  'new_hist_purchase_amount_max_new_hist_month_lag_mean_diff',
                  'hist_authorized_flag_mean_hist_month_nunique_product',
                  'hist_month_lag_mean_hist_month_lag_var_sum',
                  'hist_purchase_date_max_new_hist_month_lag_mean_product',
                  'hist_month_diff_mean_hist_purchase_date_average_diff',
                  'hist_merchant_id_nunique_hist_weekofyear_nunique_diff',
                  'hist_category_1_mean_new_hist_month_lag_mean_diff',
                  'hist_month_lag_mean_hist_first_buy_product',
                  'new_hist_purchase_date_uptonow_hist_purchase_amount_mean_product',
                  'hist_month_diff_mean_new_hist_purchase_amount_max_sum',
                  'hist_month_diff_mean_hist_month_lag_mean_diff',
                  'new_hist_purchase_date_uptonow_hist_category_3_mean_mean_product',
                  'new_hist_purchase_amount_max_hist_weekofyear_nunique_ratio',
                  'new_hist_purchase_amount_max_hist_weekend_mean_sum',
                  'new_hist_purchase_amount_max_hist_purchase_amount_min_diff',
                  'new_hist_purchase_amount_max_hist_category_3_mean_mean_diff',
                  'hist_authorized_flag_mean_hist_purchase_date_uptonow_ratio',
                  'new_hist_purchase_amount_max_new_hist_purchase_amount_mean_diff',
                  'hist_authorized_flag_mean_hist_purchase_date_diff_product',
                  'new_hist_purchase_amount_mean_hist_purchase_amount_min_diff',
                  'hist_first_buy_hist_weekend_mean_ratio',
                  'hist_category_1_sum_hist_first_buy_sum']

    df_train.drop(drop_feats, axis=1, inplace=True)
    df_test.drop(drop_feats, axis=1, inplace=True)

    print('[+] {} features included'.format(df_train.shape[1]))

    return df_train, df_test


def elnet(df_train, df_test):

    print('[+] Preparing to train Elastic Net \n')

    target = df_train['target']
    del df_train['target']

    features_with_na = df_train.columns[df_train.isna().any()].tolist() + df_test.columns[df_test.isna().any()].tolist()

    elnet_features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month', 'target', 'bucket_target', 'bin_target'] + features_with_na]

    # feature_scaler = StandardScaler()
    # df_train_std = feature_scaler.fit_transform(df_train[elnet_features].dropna(axis=1))
    # df_test_std = feature_scaler.fit_transform(df_test[elnet_features].dropna(axis=1))

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
    oof = np.zeros(len(df_train))
    elnet_test_pred = np.zeros(len(df_test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train[elnet_features], df_train['bucket_target'].values)):

        print('[+] Started training Elastic Net fold {} on {}'.format(fold_, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        trn_data = df_train.iloc[trn_idx][elnet_features]
        # val_data = df_train.iloc[val_idx][elnet_features]

        elnet = ElasticNet(alpha=1, l1_ratio=0.5, max_iter=100000, tol=1e-8, normalize=True, random_state=4590)
        elnet = elnet.fit(trn_data, target.iloc[trn_idx])

        oof[val_idx] = elnet.predict(df_train.iloc[val_idx][elnet_features])
        elnet_test_pred += elnet.predict(df_test[elnet_features]) / folds.n_splits

    cv_score = np.round(np.sqrt(mean_squared_error(oof, target)), 5)
    print('\n[+] Cross-Validated Elastic Net RMSE: {}'.format(cv_score))
    gc.collect()

    df_train['elnet_prediction'] = oof
    df_test['elnet_prediction'] = elnet_test_pred

    return df_train, df_test, target




def boost(df_train, df_test, target):
    """Define a LightGBM model to train on the data and make predictions.

    :param type df_train: Description of parameter `df_train`.
    :param type df_test: Description of parameter `df_test`.
    :return: Description of returned object.
    :rtype: type

    """

    print('[+] Preparing to train LightGBM \n')

    param = {
        'num_leaves': 31,
        'min_data_in_leaf': 30,
        'objective': 'regression',
        'max_depth': -1,
        'learning_rate': 0.007,
        'min_child_weight': 20,
        'boosting': 'gbdt',
        'feature_fraction': 0.8,
        'bagging_freq': 1,
        'bagging_fraction': 0.9,
        'bagging_seed': 11,
        'metric': 'rmse',
        'lambda_l1': 0.15,
        'verbosity': -1,
        'nthread': 4,
        'random_state': 4590
    }

    boost_features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month', 'target', 'bucket_target']]

    # # Initialize weighting vector to focus on large abs(target) samples
    # sample_weights = pd.Series(index=range(len(target)), data=1, dtype=np.int8)
    # sample_weights.iloc[np.where(target < -10)] = 2
    # sample_weights.iloc[np.where(target > 10)] = 2

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
    oof = np.zeros(len(df_train))
    predictions = np.zeros(len(df_test))
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(
            folds.split(df_train, df_train['bucket_target'].values)):
        print('[+] Started training LightGBM fold {} on {}'.format(
            fold_, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        # , categorical_feature=categorical_feats)
        trn_data = lgb.Dataset(
            df_train.iloc[trn_idx][boost_features], label=target.iloc[trn_idx])
        # , categorical_feature=categorical_feats)
        val_data = lgb.Dataset(
            df_train.iloc[val_idx][boost_features], label=target.iloc[val_idx])

        num_round = 10000
        clf = lgb.train(
            param,
            trn_data,
            num_round,
            valid_sets=[trn_data, val_data],
            verbose_eval=200,
            early_stopping_rounds=150)

        oof[val_idx] = clf.predict(df_train.iloc[val_idx][boost_features],
                                   num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = boost_features
        fold_importance_df['importance'] = clf.feature_importance()
        fold_importance_df['fold'] = fold_ + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(
            df_test[boost_features],
            num_iteration=clf.best_iteration) / folds.n_splits

    cv_score = np.round(np.sqrt(mean_squared_error(oof, target)), 5)
    print('\n[+] Cross-Validated RMSE: {}'.format(cv_score))
    gc.collect()

    return predictions, target, feature_importance_df, cv_score, param, folds

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
    plt.title('{} \n RMSE: {}'.format(runtag, cv_score))
    plt.tight_layout()


def write_logs(runtag, predictions, df_test, cv_score):
    pass


def write_submission(adjust, runtag, predictions, df_test, cv_score):
    """Prepare a file to submit to Kaggle.

    :param type predictions: Description of parameter `predictions`.
    :param type df_test: Description of parameter `df_test`.
    :param type cv_score: Description of parameter `cv_score`.
    :return: Description of returned object.
    :rtype: type

    """
    sub_df = pd.DataFrame({'card_id': df_test['card_id'].values})
    sub_df['target'] = predictions

    if adjust:
        sub_df.loc[sub_df['target'] < -20, 'target'] = -33.21928

    sub_df.to_csv(path + '/sub/{}.csv.gz'.format(runtag), index=False, compression='gzip')


############################################################################################
#                                                                        [+] lightGBM RUNNER
############################################################################################

# Settings
np.random.seed(4950)
adjust = True

# Process features and save prepared data to disk
with timer('loading raw data'):
    df_train, df_test, df_hist_trans, df_new_merchant_trans = load_raw_data()
with timer('processing features'):
    df_train, df_test = process_features(df_train, df_test, df_hist_trans, df_new_merchant_trans)


# Load prepared data and train model
with timer('loading prepared data'):
    df_train, df_test = load_prepared_data()
    gc.collect()

with timer('training model'):
    df_train, df_test, target = elnet(df_train, df_test)
    predictions, target, feature_importance_df, cv_score, param, folds = boost(df_train, df_test, target)
    gc.collect()

runtag = 'ELO_{}_LGBM_{}_ADJ_{}'.format(cv_score, datetime.now().replace(
    second=0, microsecond=0).strftime('%m-%d_%H-%M-%S'), adjust)
display_importances(runtag, feature_importance_df)

with timer('writing submission'):
    write_submission(adjust, runtag, predictions, df_test, cv_score)
