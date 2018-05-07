import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def static_split_df(big_df, test_ratio=0.2, target=None):
    """ splits the dataframe based on a random selection and given ratio into training and test dataframe.
    If target is given, the coresponding column in the splitted dataframes will be extracted into separate ones

    return a dict with following entries
    'train' training dataframe
    'test' test dataframe
    'train_t' target values for training dataframe (only if target was given)
    'test_t'  target values for test dataframe (only if target was given)"""
    (itrain, itest) = train_test_split(big_df.index, test_size=test_ratio)

    df_train = big_df.iloc[itrain]
    df_test = big_df.iloc[itest]

    result = {'train': df_train, 'test': df_test}

    if target is not None:
        target_train = df_train.loc[target]
        df_train.drop(target, inplace=True)
        target_test = df_test.loc[target]
        df_train.drop(target, inplace=True)
        result['train_t'] = target_train
        result['test_t'] = target_test

    return result


def static_split_csv(in_file, test_ratio=0.2, write_path=None):
    """ splits a csv file into training and test data and writes 2 separate files"""
    if write_path is None:
        write_path = os.path.dirname(in_file)

    prefix = os.path.splitext(os.path.basename(in_file))[0]

    bdf = pd.read_csv(in_file)
    splits = static_split_df(bdf, test_ratio=test_ratio)

    filenames = (f'{write_path}/{prefix}_train.csv', f'{write_path}/{prefix}_test.csv')
    df_train = splits['train']
    df_train.to_csv(filenames[0], index=False)

    df_test = splits['test']
    df_test.to_csv(filenames[1], index=False)

    return filenames


class DataframeScaler():
    """
    applies a sklearn scaler to a dataframe

    remembers which columns were scaled and only applies to these columns
    """

    def __init__(self, df, method=StandardScaler, leave=[]):
        self.scaler = method()
        self.cols = [c for c in df.columns if (is_numeric_dtype(df[c].dtype) and c not in leave)]

    def fit(self, df):
        self.scaler.fit(df[self.cols])

    def fit_transform(self, df):
        df[self.cols] = self.scaler.fit_transform(df[self.cols])

    def transform(self, df, inplace=True):
        if not inplace:
            df = df.copy()
        df[self.cols] = self.scaler.transform(df[self.cols])
        return df

    def inverse_transform(self, df, inplace=True):
        if not inplace:
            df = df.copy()
        df[self.cols] = self.scaler.inverse_transform(df[self.cols])
        return df


