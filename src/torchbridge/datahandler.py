import numpy as np
import torch.utils.data
from sklearn.model_selection._split import ShuffleSplit


class DFDataSet(torch.utils.data.Dataset):
    
    def __init__(self, data_x, data_y, cat_cols=None):
        self.data_x = data_x
        self.data_y = data_y
        self.len = data_x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.data_x.iloc[idx].as_matrix().astype(np.float32), self.data_y.iloc[idx].astype(np.float32))

class ProxyDataSet(torch.utils.data.Dataset):

    def __init__(self, len_func, get_func):
        self.len_func = len_func
        self.get_func = get_func

    def __len__(self):
        return self.len_func()

    def __getitem__(self, item):
        return self.get_func(item)

class CrossValidationData:
    """
    uses by default a sklearn ShuffleSplit with given number of splits

    when calling prepare() the data split is executed, which means the splits are computed. necessary once.
    with next_split() is called, one of the calculated splits becomes effective

    acts itself as an implementation of torch.utils.data.Dataset and presents *just the train data* on this interface
    the validation data for the current split can be reached by a separate Dataset that can be aquired with get_validation_set

    Usage:

    data = CrossValidationData(DFDataSet(df_in, df_target))
    loader = torch.utils.data.DataLoader(data, bs, shuffle=False)
    data.prepare()
    val_data = data.get_validation_set()
    val_loader = torch.utils.data.DataLoader(val_data, bs, shuffle=False)
    for i in range(n_epoch):
        data.next_split()
        for x, y_true in loader:
            # do training
        for x, y_true in val_loader:
            # do validation

    Attention: this way you run into an error if n_epoch is larger than n_splits

    """

    def __init__(self, dataset, splitter=None, n_splits=5, test_size=0.1):
        """ splitter can be either KFold or ShuffleSplit, just needs split method"""
        self.dataset = dataset
        if splitter is None:
            self.splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size)
        else:
            self.splitter = splitter

    def prepare(self):
        self.split_gen = self.splitter.split(self.dataset)
        self.train_index = None
        self.validation_index = None

    def next_split(self):
        """
        it is *NOT* safe to call next_split() while consuming an iterator!
        """
        try:
            self.train_index, self.validation_index = next(self.split_gen)
        except StopIteration as e:
            print(f'number of splits exhausted, prepare again')
            raise e

    def __len__(self):
        return len(self.train_index)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.train_index[idx])

    def validation_len(self):
        return len(self.validation_index)

    def get_validation(self, idx):
        return self.dataset.__getitem__(self.validation_index[idx])

    def get_validation_set(self):
        """returns a proxy object that presents closures for accessing the current validation set
        is safe with respect to next_split, means that it's just a proxy always representing the current state of CrossValidationData

        it is *NOT* safe to call next_split() while consuming an iterator!
        """
        return ProxyDataSet(self.validation_len, self.get_validation)
