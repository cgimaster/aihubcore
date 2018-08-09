from keras.models import model_from_json
import aihub.core.common.settings as settings
import os
import uuid
import json
import keras

# if stop_early:
#     callbacks.append(EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto'))


class AIModel:
    model = None

    meta = {
        'aihubcore-version':settings.AICOREHUB_VERSION,
        'name':'root',
        'version':'base',
        'type': 'root',  # e.g. mlp or autoencoder (ae), vae, gan, mlp, rbm, etc
        'framework': 'keras',
        'usage':{
            'input_shape':[], #e.g. [28,28,1]
            'output_shape':[], #e.g. [10]
            'input_type':'',#e.g. image with 1 channel
            'output_type':'', #e.g. softmax probabilities of number 0-9
        },
        'benchmarks':{}, #computed automatically
        'trainhist':{}, #list of entities {user/dataset & digests, epochs, datetime}
        'whitepapers':{}, #list of hyperlinks to papers
        'architecture':{} #keras json
    }

    def build(self):
        raise NotImplementedError()

    def fit(self, dataset, rebuild=False, params=None):
        raise NotImplementedError()

    def predict(self, x, batch_size=1):
        if not self.model: raise Exception('Model is not created. It must be either loaded or built')
        if self.meta.get('usage',None) is not None and self.meta['usage'].get('input_shape', None) is not None:
            x = x.reshape(*self.meta['usage']['input_shape'])
        return self.model.predict(x,batch_size=batch_size)


class ModelRepoRWKeras():
    @staticmethod
    def save(nametag,model):
        if not model.model: return #TODO raise exception
        model.meta['digest'] = str(uuid.uuid4()) #TODO GENERALIZE
        (model.meta['name'], model.meta['tag']) = tuple(nametag.split(':'))
        model_path = os.path.join(settings.FSLOCAL_MODELS, settings.CURRENT_USER, model.meta['name'], model.meta['tag'])
        if not os.path.exists(model_path): os.makedirs(model_path)
        weights_path = os.path.join(model_path, 'weights.h5')
        architecture_path = os.path.join(model_path, 'architecture.json')
        meta_path = os.path.join(model_path, 'meta.json')
        with open(meta_path,'w+') as f: f.write(json.dumps(model.meta))
        with open(architecture_path, 'w+') as f: f.write(model.model.to_json())
        model.model.save_weights(weights_path, overwrite=True)

    @staticmethod
    def load(nametag):
        (name, tag) = tuple(nametag.split(':'))
        model_path = os.path.join(settings.FSLOCAL_MODELS, settings.CURRENT_USER, name, tag)
        if not os.path.exists(model_path): return None
        weights_path = os.path.join(model_path, 'weights.h5')
        architecture_path = os.path.join(model_path, 'architecture.json')
        meta_path = os.path.join(model_path, 'meta.json')
        model = AIModel()
        model.meta = json.loads(open(meta_path).read())
        model.model = model_from_json(open(architecture_path).read())
        model.model.load_weights(weights_path)
        return model

class ModelRepo():
    @staticmethod
    def rwrepos():
        return  {
            'keras':ModelRepoRWKeras
        }


    @staticmethod
    def save(nametag,model):
        ModelRepo.rwrepos().get(model.meta['framework']).save(nametag,model)

    @staticmethod
    def load(nametag):
        (name, tag) = tuple(nametag.split(':'))
        model_path = os.path.join(settings.FSLOCAL_MODELS, settings.CURRENT_USER, name, tag)
        if not os.path.exists(model_path): return None
        meta = json.loads(open(os.path.join(model_path, 'meta.json')).read())
        return ModelRepo.rwrepos().get(meta['framework']).load(nametag)

modelrepo = ModelRepo()
