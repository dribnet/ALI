## this is a stub kept for older serialized models
## (this now lives in chips.samplecheckpoint)

import os
import shutil
import theano
import theano.tensor as T

from blocks.extensions.saveload import Checkpoint

class SampleCheckpoint(Checkpoint):
    def __init__(self, interface, z_dim, image_size, channels, dataset, split, save_subdir, **kwargs):
        super(SampleCheckpoint, self).__init__(path=None, **kwargs)
        pass

    def do(self, callback_name, *args):
        """Sample the model and save images to disk
        """
        pass
