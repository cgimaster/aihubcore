from aihub.core.common import settings
import os
import keras.datasets
from functools import reduce

class Dataset:
    name = None
    version = None
    description = {}
    train_dataset_path = None
    test_dataset = None

    def load_data(self):
        raise NotImplementedError()


class DenseDataset(Dataset):
    ds = None
    def __init__(self,dataset):
        self.ds = dataset

    def load_data(self):
        (xtr,ytr), (xt,yt) = self.ds.load_data()
        item_size = reduce(lambda x, y: x * y, xtr.shape[1:])
        return (xtr.reshape(-1,item_size),ytr),(xt.reshape(-1,item_size),yt)

class KerasDataset(Dataset):
    loader = None
    params = {}
    def __init__(self, loader, params={}):
        self.loader = loader
        self.params = params

    def load_data(self):
        return self.loader(**self.params)

class KerasDatasetCollection:
    datasets = {
        'mnist':KerasDataset(keras.datasets.mnist.load_data),
        'cifar10':KerasDataset(keras.datasets.cifar10.load_data),
        'cifar100/fine': KerasDataset(keras.datasets.cifar100.load_data,{'label_mode':'fine'}),
        'cifar100/coarse': KerasDataset(keras.datasets.cifar100.load_data,{'label_mode':'coarse'}),
        'imdb':KerasDataset(keras.datasets.imdb.load_data, {'path':"imdb.npz",
                                                      'num_words':None,
                                                      'skip_top':0,
                                                      'maxlen':None,
                                                      'seed':113,
                                                      'start_char':1,
                                                      'oov_char':2,
                                                      'index_from':3})
    }


class ReservedDatasetCollections:
    users = {'keras':KerasDatasetCollection()}


class DataClient:
    reserved_users = ReservedDatasetCollections()

    def fetch_dataset(self,username,datasetname, params=None):
        raise NotImplementedError() #settings.FSCACHE_DATASETS

    def push_dataset(self, param, zip_path):
        raise NotImplementedError()

    def get_dataset(self, username, datasetname, params=None):
        if self.reserved_users.users.get(username, None):
            dsname = datasetname.split('/')[0]
            return self.reserved_users.users[username].datasets.get(dsname)

        # TODO cleanup username and datasetname from special chars
        local_path = os.path.join(settings.FSLOCAL_DATA, username, datasetname)
        if os.path.exists(local_path):
            raise NotImplementedError()

dataclient = DataClient()