import torch.nn as nn

from vec_vgg import get_multi_vgg_model
from vgg_extractor import get_extractor

class CAM_V(nn.Module):
    def __init__(self, generator, extractor, args):
        super(CAM_V, self).__init__()
        self.generator = generator
        self.extractor = extractor
        self.args = args

    