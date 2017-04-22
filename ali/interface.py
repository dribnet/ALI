import theano
from theano import tensor
from blocks.serialization import load
from blocks.utils import shared_floatx
import numpy as np

class AliModel:
    def __init__(self, filename=None, model=None):
        if model is not None:
            self.model, = model.top_bricks
        else:
            with open(filename, 'rb') as src:
                main_loop = load(src)
                self.model, = main_loop.model.top_bricks

    def encode_images(self, images):
        x = tensor.tensor4('features')
        latents = theano.function([x], self.model.encode(x))(images)
        num_samples, z_dim, _x, _y = latents.shape
        # print("SO", num_samples, z_dim, _x, _y)
        # print("AND", images.shape)
        return latents.reshape(num_samples, z_dim)

    def get_zdim(self):
        input_shape = self.model.encoder.get_dim('output')
        return input_shape[0]

    def sample_at(self, z):
        num_samples, z_dim = z.shape
        sz = shared_floatx(z.reshape(num_samples, z_dim, 1, 1))
        x = self.model.sample(sz)
        samples = theano.function([], x)()
        return samples

class AliCondModel:
    def __init__(self, filename=None, model=None):
        if model is not None:
            self.model, = model.top_bricks
        else:
            with open(filename, 'rb') as src:
                main_loop = load(src)
                self.model, = main_loop.model.top_bricks

    def encode_images(self, images, ignored=None):
        x = tensor.tensor4('features')
        y = tensor.matrix('y')
        labels = np.zeros(shape=(images.shape[0], 128), dtype=np.uint8)
        labels[:,0] = 1
        latents = theano.function([x, y], self.model.encode(x, y))(images, labels)
        num_samples, z_dim, _x, _y = latents.shape
        # print("SO", num_samples, z_dim, _x, _y)
        # print("AND", images.shape)
        return latents.reshape(num_samples, z_dim)

    def get_zdim(self):
        # input_shape = self.model.encoder.get_dim('output')
        # try:
        #     print("IS THIS THE SIZE: {}".format(self.model.n_emb))
        # except AttributeError:
        #     print("Let's go fishing: {}".format(dir(self.model)))
        return self.model.n_emb
        # return input_shape[0]

    def sample_at(self, z, ignored=None):
        num_samples, z_dim = z.shape
        sz = shared_floatx(z.reshape(num_samples, z_dim, 1, 1))
        # y = tensor.matrix('y')
        labels = np.zeros(shape=(num_samples, 128), dtype=np.uint8)
        labels[:,0] = 1
        x = self.model.sample(sz, labels)
        samples = theano.function([], x)()
        return samples

class AliCondModel2:
    def __init__(self, filename=None, model=None):
        if model is not None:
            self.model, = model.top_bricks
        else:
            with open(filename, 'rb') as src:
                main_loop = load(src)
                self.model, = main_loop.model.top_bricks
        self.null_label = None
        self.embedded_null_label = None

    def encode_images(self, images, labels=None):
        # TODO: allow mix of None and values
        if labels is None or all(v is None for v in labels):
            labels = self.get_null_hypothesis(images.shape[0])
        x = tensor.tensor4('features')
        y = tensor.matrix('y')
        # labels = np.zeros(shape=(images.shape[0], 128), dtype=np.uint8)
        latents = theano.function([x, y], self.model.encode(x, y))(images, labels)
        num_samples, z_dim, _x, _y = latents.shape
        # print("SO", num_samples, z_dim, _x, _y)
        # print("AND", images.shape)
        return latents.reshape(num_samples, z_dim)

    # TODO: THIS IS WRONG
    def get_zdim(self):
        return self.model.n_emb

    def get_emb_dim(self):
        return self.model.n_emb

    def get_cond_dim(self):
        return self.model.n_cond

    def init_null_label(self):
        c_dim = self.get_cond_dim()
        self.null_label = np.zeros(shape=(c_dim,), dtype=np.uint8)
        self.null_label[0] = 1

    def init_embedded_null_label(self):
        if self.null_label is None:
            self.init_null_label()
        one_null_label = np.array([self.null_label])
        one_embedded_label = self.embed_labels(one_null_label)
        self.embedded_null_label = one_embedded_label[0]
        # print("IE: {}".format(self.embedded_null_label))

    def get_null_hypothesis(self, batch_size):
        if self.null_label is None:
            self.init_null_label()
        c_dim = self.get_cond_dim()
        labels = np.zeros(shape=(batch_size,c_dim), dtype=np.uint8)
        # print("SO BEFORE {} and {}".format(labels.shape, self.null_label.shape))
        labels[:] = self.null_label
        # print("SO AFTER {} and {}".format(labels[0], labels[1]))
        return labels

    def get_embedded_null_hypothesis(self, batch_size):
        if self.embedded_null_label is None:
            self.init_embedded_null_label()
        e_dim = self.get_emb_dim()
        embeddings = np.zeros(shape=(batch_size,e_dim))
        embeddings[:] = self.embedded_null_label
        # print("NOTE IE: {} {} and {}".format(self.embedded_null_label, self.embedded_null_label.shape, embeddings.shape))
        # print("SO EAFTER {} and {}".format(embeddings[0], embeddings[1]))
        return embeddings

    def sample_at(self, z, labels=None):
        num_samples, z_dim = z.shape
        sz = shared_floatx(z.reshape(num_samples, z_dim, 1, 1))
        # TODO: allow mix of None and values
        if labels is None or all(v is None for v in labels):
            labels = self.get_null_hypothesis(num_samples)
        # labels = np.zeros(shape=(num_samples, 128), dtype=np.uint8)
        x = self.model.decode(sz, labels)
        samples = theano.function([], x)()
        return samples

    def embed_labels(self, labels):
        if all(v is None for v in labels):
            emb_l = self.get_embedded_null_hypothesis(len(labels))
            return emb_l

        y = tensor.matrix('y')
        # print("L SHAPE {}".format(labels.shape))
        embeddings = theano.function([y], self.model.embed(y))(labels)
        # print("E SHAPE {}".format(embeddings.shape))
        shape = embeddings.shape
        embeddings = embeddings.reshape(shape[0], shape[1])
        # print("ES2 SHAPE {}".format(embeddings.shape))
        return embeddings

    def decode_embedded(self, z, emb_l=None):
        # TODO: THIS looks like a good way to cache compiled functions
        z_t = tensor.tensor4('z')
        e_t = tensor.tensor4('e')
        model_decode_embedded = theano.function([z_t, e_t], self.model.decode_embedded(z_t, e_t), allow_input_downcast=True)

        num_z_samples, z_dim = z.shape

        # TODO: allow mix of None and values
        if emb_l is None or all(v is None for v in emb_l):
            emb_l = self.get_embedded_null_hypothesis(num_z_samples)

        num_e_samples, e_dim = emb_l.shape

        # sz = shared_floatx(z.reshape(num_z_samples, z_dim, 1, 1))
        # se = shared_floatx(emb_l.reshape(num_e_samples, e_dim, 1, 1))
        sz = z.reshape(num_z_samples, z_dim, 1, 1).astype('float32')
        se = emb_l.reshape(num_e_samples, e_dim, 1, 1).astype('float32')
        decodes = model_decode_embedded(sz, se)
        # print("RESULT: {}".format(decodes.shape))
        return decodes
        # labels = np.zeros(shape=(num_samples, 128), dtype=np.uint8)
        # x = self.model.decode_embedded(sz, se)
        # samples = theano.function([], x)()
        # return samples
