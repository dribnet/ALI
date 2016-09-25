# this file is kept only so that old models can deserialize correctly
# (interface has moved to ali/interface.py)

class AliModel:
    def __init__(self, filename=None, model=None):
        pass

    def encode_images(self, images):
        pass

    def get_zdim(self):
        pass

    def sample_at(self, z):
        pass
