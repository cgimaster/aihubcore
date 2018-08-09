from unittest import TestCase
from aihub.core.data.dataclient import dataclient
from aihub.core.data.dataclient import DenseDataset
from aihub.core.data.datasets.dsrandom import DSRandom
from aihub.core.models.aihubmodel import ModelRepo
from .model import MLPRandomMulticlass
import numpy as np
import matplotlib.pyplot as plt

class ModelTest(TestCase):

    def test_fit_save_load_predict(self):
        ds = DSRandom()
        m = MLPRandomMulticlass().fit(ds)
        ModelRepo.save('mlp-3h-10sm-random:1',m)
        lm = ModelRepo.load('mlp-3h-10sm-random:1')
        (xtr, ytr), (xt, yt) = ds.load_data()
        idx = 333
        #Assume normal distribution confused network
        assert(np.max(lm.predict(xtr[idx])[0])<0.5)
        for i in range(100):
            print(lm.predict(xtr[idx])[0],ytr[i])