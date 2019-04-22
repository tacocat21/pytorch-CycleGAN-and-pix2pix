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

   def backward_G(self, real_A, real_B):
       pass

# TODO: check if these mutation costs are correct
# Assuming p_z is uniform distribution
def minimax_mutation_cost(fake_disc_pred):
    """
    Assuming p_z is uniform distribution
    :param fake_disc_pred: tensor of shape (N). Results of D(G(x))
    :return: 1/2 * E[log(1- fake_disc_pred)]
    """
    log_dist = torch.log(torch.ones(fake_disc_pred.shape[0]) - fake_disc_pred)
    return -0.5 * log_dist.mean()

def heuristic_mutation_cost(fake_disc_pred):
    """
    Assuming p_z is uniform distribution
    :param fake_disc_pred: tensor of shape (N). Results of D(G(x))
    :return: -1/2 * E[log(fake_disc_pred)]
    """
    log_dist = torch.log(fake_disc_pred)
    return -0.5 * log_dist.mean()

def least_square_mutation_cost(fake_disc_pred):
    """
    Assuming p_z is uniform distribution
    :param fake_disc_pred: tensor of shape (N). Results of D(G(x))
    :return: E[(fake_disc_pred - 1)^2]
    """

    sq_dist = (fake_disc_pred - torch.ones(fake_disc_pred.shape[0]))**2
    return sq_dist.mean()

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

        

if __name__ == '__main__':
    pass