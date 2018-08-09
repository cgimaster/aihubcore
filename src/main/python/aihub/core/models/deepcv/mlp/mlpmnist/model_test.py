from unittest import TestCase
from aihub.core.data.dataclient import dataclient
from aihub.core.data.dataclient import DenseDataset
from aihub.core.models.aihubmodel import ModelRepo
from .model import MLPMnist
import numpy as np
import matplotlib.pyplot as plt

class ModelTest(TestCase):

    def test_fit_save_load_predict(self):
        ds = DenseDataset(dataclient.get_dataset('keras', 'mnist'))
        m = MLPMnist().fit(ds,params={'batch_size':128, 'epochs':10})
        ModelRepo.save('mlp-keras-mnist:1',m)

        lm = ModelRepo.load('mlp-keras-mnist:1')
        (_, _), (xt, yt) = DenseDataset(dataclient.get_dataset('keras', 'mnist')).load_data()
        idx = 333
        assert(np.argmax(lm.predict(xt[idx])[0]) == yt[idx])
        # plt.subplot(1, 2, 1)
        # plt.imshow(x.reshape(28, 28), cmap='gray')
        # plt.title('Original')
        # plt.show()
        # (x_train, y_train), (x_test, y_test) = dc.get_dataset('keras','mnist')