from unittest import TestCase

from aihub.core.data.dataclient import DenseDataset
from .model import AEConv
from unittest import TestCase
from aihub.core.data.dataclient import dataclient
from aihub.core.data.dataclient import DenseDataset
from aihub.core.models.aihubmodel import ModelRepo
import numpy as np
import matplotlib.pyplot as plt



class AEMLPTest(TestCase):

    def test_fit_save_load_predict(self):
        ds = dataclient.get_dataset('keras', 'mnist')
        m = AEConv().fit(ds)
        ModelRepo.save('ae-conv-mnist-2:1',model=m)
        ml = ModelRepo.load('ae-conv-mnist-2:1')
        (xtr, ytr), (xt, yt) = ds.load_data()
        i = 1
        x = xt[i]
        y = ml.predict(x)
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title('Original')

        plt.subplot(1, 2, 2)
        plt.imshow(y.reshape(28, 28), cmap='gray')
        plt.title('Reconstructed')

        plt.show()
        # (x_train, y_train), (x_test, y_test) = dc.get_dataset('keras','mnist')

