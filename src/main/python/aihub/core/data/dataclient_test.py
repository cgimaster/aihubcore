import os
from unittest import TestCase
from .dataclient import DataClient as DC
import aihub.core.common.settings as settings
class DataLakeClientTest(TestCase):
    dc = DC()

    def test_get_dataset(self):
        (x_train, y_train), (x_test, y_test) = self.dc.get_dataset('keras','mnist').load_data()
        print(x_train.shape)
        print(y_train.shape)
        assert x_train.shape == (60000, 28, 28)
        assert x_test.shape == (10000, 28, 28)
        assert y_train.shape == (60000,)
        assert y_test.shape == (10000,)


    def ignore_test_fetchdataset(self):
        #zp = self.dlc.create_dataset(dir_path)
        zp = os.path.join(os.path.expanduser('~'),'/aihub/data/temp/mnist.zip')
        self.dlc.push_dataset('datauser/mnist:1',zip_path=zp) # username/datasetname : tag => internally converted into digest ordered with dates
        dataset = self.dc.fetch_dataset('datauser/mnist:latest') #assume there is only one
        assert dataset.path == os.path.join(settings.FSLOCAL_DATA, 'datauser/mnist/1')
        assert os.path.exists(os.path.join(settings.FSLOCAL_DATA, 'datauser/mnist/1'))
        assert os.path.isdir(os.path.join(settings.FSLOCAL_DATA, 'datauser/mnist/1'))