from aihub.core.common import settings
from aihub.core.models.aihubmodel import AIModel
from keras.layers import Input, Dense, Conv2D, Deconvolution2D, Convolution2D
from keras.models import Model
from aihub.core.data.dataclient import DenseDataset
import matplotlib.pyplot as plt
import numpy as np


class AEConv(AIModel):
    model = None

    def build(self):
        inputs = Input(shape=(784,))
        x = Convolution2D(10, 3, 3, border_mode='same', input_shape=(28, 28))(inputs)
        predictions = Deconvolution2D(1, 3, 3, output_shape=(28, 28))(x)
        #predictions = Dense(784, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        return self.model

    def fit(self, ds, rebuild=True, params={}):
        (xtrain,ytr),(xt,yt) = ds.load_data()
        if rebuild or self.model == None:
            self.model = self.build()
        self.meta['usage']['input_shape'] = [-1] + list(xtrain.shape[1:])
        self.model.fit(x=xtrain, y=xtrain, batch_size=128, epochs=10, callbacks=settings.KERAS_CALLBACKS)
        return self