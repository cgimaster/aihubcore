# Default settings file
import os
import keras
from keras.callbacks import EarlyStopping, TensorBoard

AICOREHUB_VERSION = '0.1a'

AIHUB_HOME = os.path.join(os.path.expanduser('~'), 'aihub')
FSLOCAL_DATA = os.path.join(AIHUB_HOME, 'local/data/')
FSLOCAL_MODELS = os.path.join(AIHUB_HOME, 'local/models/')

FSOUTPUT_ENV_VIDEO = os.path.join(AIHUB_HOME, 'output/env/video')

DATALAKE_AINODE_PATH = os.path.join(AIHUB_HOME, 'datalake/')

TF_LOGDIR = os.path.join(AIHUB_HOME, 'local/tflogs')
PATHS = [FSOUTPUT_ENV_VIDEO,TF_LOGDIR]
for p in PATHS:
    if not os.path.exists(p): os.makedirs(p)

CURRENT_USER = 'cgimaster'

KERAS_CALLBACKS = []
KERAS_CALLBACK_TENSORBOARD = TensorBoard(log_dir=TF_LOGDIR)
if keras.backend._BACKEND == 'tensorflow': KERAS_CALLBACKS.append(KERAS_CALLBACK_TENSORBOARD)

