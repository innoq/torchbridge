import pandas as pd
import numpy as np

from torchbridge.datahandler import *

def test_DFDataSet_len_and_content():
    df = pd.DataFrame(np.arange(20).reshape(-1, 4))
    df_y = pd.DataFrame(np.arange(df.shape[0]))
    ds = DFDataSet(df, df_y)

    assert len(ds) == df.shape[0]

    for i in range(len(ds)):
        (d_in, d_target) = ds.__getitem__(i)
        assert((df.iloc[i] == d_in).all())
        assert((df_y.iloc[i] == d_target).all())

def test_ProxyDataset():
    df = pd.DataFrame(np.arange(20).reshape(-1, 4))
    df_y = pd.DataFrame(np.arange(df.shape[0]))
    ds = DFDataSet(df, df_y)

    proxy = ProxyDataSet(ds.__len__, ds.__getitem__)

    assert len(proxy) == df.shape[0]

    for i in range(len(proxy)):
        (d_in, d_target) = proxy.__getitem__(i)
        assert((df.iloc[i] == d_in).all())
        assert((df_y.iloc[i] == d_target).all())

def test_CrossValidationData_len():
    df = pd.DataFrame(np.arange(200).reshape(-1, 4))
    df_y = pd.DataFrame(np.arange(df.shape[0]))
    ds = DFDataSet(df, df_y)

    cv = CrossValidationData(ds)
    cv.prepare()
    cv.next_split()

    expected_train_len = df.shape[0]*0.9
    assert(expected_train_len-1 < len(cv) < expected_train_len+1)

    expected_val_len = df.shape[0]*0.1
    assert(expected_val_len-1 < len(cv.get_validation_set()) < expected_val_len+1)

    assert(len(cv.get_validation_set()) + len(cv) == df.shape[0])

def test_CrossValidationData_unique():
    df = pd.DataFrame(np.arange(208).reshape(-1, 4))
    df_y = pd.DataFrame(np.arange(df.shape[0]))
    ds = DFDataSet(df, df_y)

    cv = CrossValidationData(ds)
    cv.prepare()
    cv.next_split()

    train = []
    train_y = []
    for i in range(len(cv)):
        (d_in, d_target) = cv.__getitem__(i)
        train.append(d_in[0])
        train_y.append(d_target)

    # values from np.arange just count up, so they should be unique
    assert(np.unique(np.array(train)).shape[0] == len(cv))
    assert(np.unique(np.array(train_y)).shape[0] == len(cv))

    cv_val = cv.get_validation_set()
    val = []
    val_y = []
    for i in range(len(cv_val)):
        (d_in, d_target) = cv_val.__getitem__(i)
        val.append(d_in[0])
        val_y.append(d_target)

    assert(np.unique(np.array(val)).shape[0] == len(cv_val))
    assert(np.unique(np.array(val_y)).shape[0] == len(cv_val))

    # train and validation set are distinct
    assert(np.intersect1d(np.array(train), np.array(val)).shape[0] == 0)

    cv.next_split()

    val2 = []
    val2_y = []
    for i in range(len(cv_val)):
        (d_in, d_target) = cv_val.__getitem__(i)
        val2.append(d_in[0])
        val2_y.append(d_target)

    # at least not the complete validation set is same
    assert(np.intersect1d(np.array(val), np.array(val2)).shape[0] < len(cv_val))
