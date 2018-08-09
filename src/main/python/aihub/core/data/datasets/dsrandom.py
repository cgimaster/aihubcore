from aihub.core.data.dataclient import Dataset
import numpy as np
import keras

class DSRandom(Dataset):
    x_train, y_train, x_test, y_test = None, None, None, None

    def __init__(self,mtrain=10000,mtest=1000,fn=20,cn=10):
        self.mtrain, self.mtest, self.fn, self.cn = mtrain, mtest, fn,cn


    def load_data(self,force=False):
        if self.x_train is None or force:
            self.x_train = np.random.random((self.mtrain, self.fn))
            self.y_train = keras.utils.to_categorical(np.random.randint(self.cn, size=(self.mtrain, 1)), num_classes=self.cn)
            self.x_test = np.random.random((self.mtest, self.fn))
            self.y_test = keras.utils.to_categorical(np.random.randint(self.cn, size=(self.mtest, 1)), num_classes=self.cn)
        return (self.x_train,self.y_train), (self.x_test,self.y_test)