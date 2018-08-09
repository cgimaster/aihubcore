from keras.callbacks import TensorBoard
import shutil
from aihub.core.common import settings
from aihub.core.models.aihubmodel import AIModel
from keras.layers import Input, Dense
from keras.models import Model
from aihub.core.data.dataclient import DenseDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid

class AutoencoderMLP(AIModel):
    model = None

    def __init__(self):
        self.meta['name']='ae-mlp-mnist-2h-300-20'
        self.meta['tag'] = '1' #Must be generated automatically

    def build(self):
        inputs = Input(shape=(784,))
        x = Dense(300, activation='sigmoid',input_dim=784)(inputs)
        x = Dense(20, activation='sigmoid')(x)
        predictions = Dense(784, activation='sigmoid')(x)
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def fit(self, ds, rebuild=True, params={}):
        (xtrain,ytr),(xt,yt) = ds.load_data()
        if rebuild or self.model == None:
            self.model = self.build()
        self.meta['usage']['input_shape'] = [-1] + list(xtrain.shape[1:])
        logdir = os.path.join(settings.TF_LOGDIR,self.meta.get('name'),self.meta.get('tag'),str(uuid.uuid4()))
        if os.path.exists(logdir): shutil.rmtree(logdir)
        os.makedirs(logdir)

        self.model.fit(x=xtrain, y=xtrain, batch_size=128, epochs=100, callbacks=[TensorBoard(log_dir=logdir)])
        return self

