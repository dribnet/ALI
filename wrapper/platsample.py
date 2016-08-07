#!/usr/bin/env python
import theano
from theano import tensor
from blocks.serialization import load
from blocks.utils import shared_floatx
from utils import sample
import sys

class AliModel:
    def __init__(self, filename):
        with open(filename, 'rb') as src:
            main_loop = load(src)
            self.model, = main_loop.model.top_bricks

    def encode_images(self, images):
        x = tensor.tensor4('features')
        latents = theano.function([x], self.model.encode(x))(images)
        num_samples, z_dim, _, _ = latents.shape
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

if __name__ == '__main__':
    args = ["--model-module", "wrapper.platsample", "--model-class", "AliModel"] + sys.argv[1:]
    sample.main(args)
