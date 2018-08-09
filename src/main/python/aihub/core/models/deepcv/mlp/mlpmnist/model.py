import keras
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential

from aihub.core.common import settings
from aihub.core.models.aihubmodel import AIModel


class MLPMnist(AIModel):

    def __init__(self):
        self.meta['name']='mlp-keras-mnist'
        self.meta['tag'] = '1'
        self.meta['usage']['input_shape']=[-1,784]
        self.meta['usage']['output_number_classes'] = 10

    def build(self):
        self.model = Sequential([
            Dense(32, input_shape=(784,)),
            Activation('relu'),
            Dense(10),
            Activation('softmax'),
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def fit(self, ds, rebuild=True, params={}):
        (xtr,ytr),(xt,yt) = ds.load_data()
        if rebuild or not self.model: self.model = self.build()
        ytr_ohe = keras.utils.to_categorical(ytr, num_classes=self.meta['usage']['output_number_classes'])
        self.meta['training_params'] = {
            'batch_size':params.get('batch_size',128),
            'epochs':params.get('epochs',10)
        }
        self.model.fit(x=xtr, y=ytr_ohe,
                       batch_size=self.meta['training_params']['batch_size'],
                       epochs=self.meta['training_params']['epochs'],
                       callbacks=settings.KERAS_CALLBACKS)
        return self
