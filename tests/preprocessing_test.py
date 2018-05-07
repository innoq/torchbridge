import pandas as pd
import numpy as np

from torchbridge.preprocessing import DataframeScaler, static_split_df

def test_create_scaler():
    df_test = pd.DataFrame({'A': [14.00, 90.20, 90.95, 96.27, 91.21],
                           'B': [103.02, 107.26, 110.35, 114.23, 114.68],
                           'C': ['big', 'small', 'big', 'small', 'small']})

    df_orig = df_test.copy()
    print("")
    print(df_orig)

    scaler = DataframeScaler(df_test)
    scaler.fit_transform(df_test)

    print(df_test)

    assert(df_test['A'].abs().max() < 2)
    assert(df_test['B'].abs().max() < 2)

    df_restore = scaler.inverse_transform(df_test, inplace=False)

    print(df_restore)

    assert((df_orig == df_restore).all().all())

def test_static_split_len():
    df = pd.DataFrame(np.arange(208).reshape(-1, 4))

    test_ratio = 0.1
    splits = static_split_df(df, test_ratio=test_ratio)
    train_df = splits['train']
    test_df = splits['test']

    expected_train_len = df.shape[0] * (1-test_ratio)
    assert (expected_train_len - 1 < train_df.shape[0] < expected_train_len + 1)

    expected_test_len = df.shape[0] * test_ratio
    assert (expected_test_len - 1 < test_df.shape[0] < expected_test_len + 1)

def test_static_split_unique():
    df = pd.DataFrame(np.arange(408).reshape(-1, 4))

    splits = static_split_df(df)
    train_df = splits['train']
    test_df = splits['test']

    # values from np.arange just count up, so they should be unique
    assert(np.unique(np.array(train_df.iloc[:,0])).shape[0] == train_df.shape[0])
    assert(np.unique(np.array(test_df.iloc[:,0])).shape[0] == test_df.shape[0])

    assert(np.intersect1d(np.array(train_df), np.array(test_df)).shape[0] == 0)
