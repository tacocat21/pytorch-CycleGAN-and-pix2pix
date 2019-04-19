import torch
import uuid
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.cycle_gan_model import CycleGANModel
import os

class CycleGANModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.generators = []
        self.netD_A  = None
        self.netD_B = None


class GeneratorPair:

    def __init__(self, base_dir='./cache'):
        self.netG_A = None # TODO fill these
        self.netG_B = None
        self.uuid = str(uuid.uuid4())
        self.save_dir = os.path.join(base_dir, self.uuid)
        while os.path.exists(self.save_dir):
            self.uuid = str(uuid.uuid4())
            self.save_dir = os.path.join(base_dir, self.uuid)
            # make sure there isn't a duplicate
        try:
            os.makedirs(self.save_dir)
        except:
            pass

    def save_to_disk(self):
        torch.save(self.netG_A, os.path.join(self.save_dir, 'netG_A.model'))
        torch.save(self.netG_B, os.path.join(self.save_dir, 'netG_B.model'))
        # remove network from memory
        del self.netG_A
        del self.netG_B
        self.netG_A = None
        self.netG_B = None

    def load_from_disk(self):
        self.netG_A = torch.load(os.path.join(self.save_dir, 'netG_A.model'))
        self.netG_B = torch.load(os.path.join(self.save_dir, 'netG_B.model'))

