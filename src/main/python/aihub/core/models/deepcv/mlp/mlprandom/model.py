import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import os

from aihub.core.models.aihubmodel import AIModel



class MLPRandomMulticlass(AIModel):

    def build(self):
        model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        model.add(Dense(64, activation='relu', input_dim=20))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return self.model

    def fit(self, dataset, rebuild=False, params=None):
        (xtrain , ytrain), (xtest,ytest)  = dataset.load_data()
        self.meta['usage']['input_shape']=[-1]+list(xtrain.shape[1:])
        if not self.model or rebuild: self.build()
        self.model.fit(xtrain, ytrain, epochs=20, batch_size=128)
        score = self.model.evaluate(xtest, ytest, batch_size=128)
        self.meta['score'] = score
        return self
