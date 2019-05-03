import torch
import uuid
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.cycle_gan_model import CycleGANModel
import os
import numpy as np
import copy
import math
import torch.nn as nn
import ipdb


class EvolutionaryCycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        return parser

    def __init__(self, opt):
        """
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        D_A pass image B
        D_B pass image A
        :param opt:
        """
        BaseModel.__init__(self, opt)

        self.generators = [GeneratorPair(opt)]
        self.netD_A  = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                         opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        
        self.mutations = [minimax_mutation_cost, heuristic_mutation_cost, least_square_mutation_cost]

        # parent optimizer for generator
        self.optimizer_G = torch.optim.Adam(itertools.chain.from_iterable([g.parameters() for g in self.generators]),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.opt = opt
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.num_parents = 1
        self.gamma = opt.gamma # used for fitness score
        self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        self.sigmoid = nn.Sigmoid()
        self.model_names = ['D_A', 'D_B']
        self.disc_optimizer = [self.optimizer_D]
        self.set_optimizers()

    def set_optimizers(self):
        self.optimizers = self.disc_optimizer + [self.optimizer_G]


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def add_mutation_func(self, mutation_func):
        self.mutations.append(mutation_func)

    def forward(self):
        # Runs a forward pass for each generator pair

        self.fake_A_list, self.fake_B_list = [], []
        self.rec_A_list, self.rec_B_list = [], []

        for i in range(len(self.generators)):
            gen_pair = self.generators[i]
            f_b = gen_pair.netG_A(self.real_A)    # G_A(A)
            r_a = gen_pair.netG_B(f_b)            # G_B(G_A(A))
            f_a = gen_pair.netG_B(self.real_B)    # G_B(B)
            r_b = gen_pair.netG_A(f_a)            # G_A(G_B(B))

            self.fake_B_list.append(f_b)
            self.rec_A_list.append(r_a)
            self.fake_A_list.append(f_a)
            self.rec_B_list.append(r_b)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake list((tensor array)) -- list of images generated by generators

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.

        loss_D = avg(avg(loss_D_real, loss_D_fake[0]), avg(loss_D_real, loss_D_fake[1])..
                ... avg(loss_D_real, loss_D_fake[N-1]))
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        loss_D = 0

        for i in range(len(fake)):

            # Fake

            pred_fake = netD(fake[i].detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)

            loss_D += (loss_D_real + loss_D_fake) * 0.5

        loss_D = loss_D / len(fake)
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B_queries = [self.fake_B_pool.query(fb) for fb in self.fake_B_list]
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B_queries)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A_queries = [self.fake_A_pool.query(fa) for fa in self.fake_A_list]
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A_queries)

    def optimize_D(self):
        """Forward and backward pass for both discriminators"""
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.forward()  # compute fake images and reconstruction images.
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def optimize_G(self):
        # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.forward() #TODO: added forward here. Not sure if correct
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.netD_A.zero_grad()
        self.netD_B.zero_grad()
        # self.optimizer_G.step()       # update G_A and G_B's weights


    def backward_G(self):
        """
        Calculate the loss function for each generator pair
        Calculate the loss for generators G_A and G_B
        """

        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        generator_list = []
        fitness_scores = []
        optimizer_list = [] # keep optimizer of best child

        #loop over parent generators
        for i in range(len(self.generators)):
            gen_pair = self.generators[i] # parent

            for mut_func in self.mutations:
                gen_pair.zero_grad()
                # GAN loss D_A(G_A(A))
                loss_G_A = self.criterionGAN(self.netD_A(self.fake_B_list[i]), True)
                # GAN loss D_B(G_B(B))
                loss_G_B = self.criterionGAN(self.netD_B(self.fake_A_list[i]), True)
                # Forward cycle loss || G_B(G_A(A)) - A||
                loss_cycle_A = self.criterionCycle(self.rec_A_list[i], self.real_A) * lambda_A
                # Backward cycle loss || G_A(G_B(B)) - B||
                loss_cycle_B = self.criterionCycle(self.rec_B_list[i], self.real_B) * lambda_B

                child_generator = copy.deepcopy(gen_pair)
                generator_list.append(child_generator)
                # Identity loss
                if lambda_idt > 0:
                    # G_A should be identity if real_B is fed: ||G_A(B) - B||
                    idt_A = child_generator.netG_A(self.real_B)
                    loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
                    # G_B should be identity if real_A is fed: ||G_B(A) - A||
                    idt_B = child_generator.netG_B(self.real_A)
                    loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt
                else:
                    loss_idt_A = 0
                    loss_idt_B = 0



                # combined loss and calculate gradients
                loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

                mut_cost = loss_G + mut_func(self.netD_A(self.fake_B_list[i])) + mut_func(self.netD_B(self.fake_A_list[i]))
                # self.optimizer_G.__setstate__({
                #     'param_groups': child_generator.get_parameters()
                # })
                mut_cost.backward(retain_graph=True)
                #TODO: check generator steps
                try:
                    optimizer = self.get_copy_optimizer(child_generator)
                    optimizer.step()
                except RuntimeError as e:
                    ipdb.set_trace()
                    print(e)
                optimizer_list.append(optimizer)
                fitness = self.fitness_score(child_generator)
                fitness_scores.append(fitness)

        # order of the fitness score
        order = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)

        self.generators = [generator_list[i] for i in order[:self.num_parents]]
        self.optimizer_G = optimizer_list[order[0]]
        self.set_optimizers()


    def fitness_score(self, generator):
        """
        Evalute the fitness
        """
        #TODO: check this value is computed correctly!
        self.netD_A.zero_grad()
        self.netD_B.zero_grad()

        pred_real_A = self.netD_A(self.real_B)
        pred_real_B = self.netD_B(self.real_A)

        img_fake_A = generator.netG_B(self.real_B)
        img_fake_B = generator.netG_A(self.real_A)
        pred_fake_A = self.netD_A(img_fake_B)
        pred_fake_B = self.netD_B(img_fake_A)
        fq = pred_fake_A.mean() + pred_fake_B.mean() # quality fitness #TODO: should one of this be negative?
        #        self.pred_real_A = self.netD_A(self.real_B)
        # self.pred_real_B = self.netD_B(self.real_A)
        loss_D_A_real = self.criterionGAN(pred_real_A, True)
        loss_D_A_fake = self.criterionGAN(pred_fake_A, False)

        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A.backward()

        loss_D_B_real = self.criterionGAN(pred_real_B, True)
        loss_D_B_fake = self.criterionGAN(pred_fake_B, False)

        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B.backward()

        # get gradient values
        if self.gamma > 0:
            grad_val = 0
            for param in self.netD_A.parameters():
                grad_val += torch.sum(param.grad**2)
            for param in self.netD_B.parameters():
                grad_val += torch.sum(param.grad**2)
            fd = - math.log(grad_val) # diversity fitness
        else:
            fd = 0
        return fq + self.gamma * fd

    def get_copy_optimizer(self, child_generator):
        optimizer = torch.optim.Adam(child_generator.parameters(), lr=self.opt.lr)
        new_state = self.optimizer_G.state_dict()
        new_state['param_groups'] = optimizer.state_dict()['param_groups']
        # optimizer.load_state_dict(new_state) # TODO: figure out how to properly copy optimizer state
        return optimizer

    def optimize_parameters(self):
        pass

    def save_networks(self, epoch):
        super(EvolutionaryCycleGANModel, self).save_networks(epoch)
        self.generators[0].save_to_disk(self.save_dir, epoch)

# Assuming p_z is uniform distribution
def minimax_mutation_cost(fake_disc_pred, epsilon = 1e-8):
    """
    Assuming p_z is uniform distribution
    :param fake_disc_pred: tensor of shape (N). Results of D(G(x))
    :param epsilon: for numerical stability. (we don't input log(0)
    :return: 1/2 * E[log(1- fake_disc_pred)]
    """
    fake_disc_pred = fake_disc_pred.view(fake_disc_pred.shape[0], -1).mean(1)
    fake_disc_pred = torch.sigmoid(fake_disc_pred)
    log_dist = torch.log((1 + epsilon) - fake_disc_pred)
    return 0.5 * log_dist.mean()


def heuristic_mutation_cost(fake_disc_pred, epsilon = 1e-8):
    """
    Assuming p_z is uniform distribution
    :param fake_disc_pred: tensor of shape (N). Results of D(G(x))
    :param epsilon: for numerical stability. (we don't input log(0)
    :return: -1/2 * E[log(fake_disc_pred)]
    """
    fake_disc_pred = fake_disc_pred.view(fake_disc_pred.shape[0], -1).mean(1)
    fake_disc_pred = torch.sigmoid(fake_disc_pred)
    log_dist = torch.log(fake_disc_pred + epsilon)
    return -0.5 * log_dist.mean()

def least_square_mutation_cost(fake_disc_pred):
    """
    Assuming p_z is uniform distribution
    :param fake_disc_pred: tensor of shape (N). Results of D(G(x))
    :return: E[(fake_disc_pred - 1)^2]
    """
    fake_disc_pred = fake_disc_pred.view(fake_disc_pred.shape[0], -1).mean(1)
    fake_disc_pred = torch.sigmoid(fake_disc_pred)
    sq_dist = (fake_disc_pred - 1)**2
    return sq_dist.mean()

class GeneratorPair:

    def __init__(self, opt, base_dir='./cache'):
        #Generator def lifted from CycleGANModel
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                  not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                          not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        if 'genA_load_path' in opt:
            state_dict = torch.load(opt.genA_load_path)
            self.netG_A.module.load_state_dict(state_dict)
        if 'genB_load_path' in opt:
            state_dict = torch.load(opt.genB_load_path)
            self.netG_B.module.load_state_dict(state_dict)

    def parameters(self):
        return itertools.chain(self.netG_A.parameters(), self.netG_B.parameters())

    def zero_grad(self):
        self.netG_A.zero_grad()
        self.netG_B.zero_grad()

    def save_to_disk(self, save_dir, epoch):
        netG_A_path = os.path.join(save_dir, '{}_net_G_A.pth'.format(epoch))
        netG_B_path = os.path.join(save_dir, '{}_net_G_B.pth'.format(epoch))

        if torch.cuda.is_available():
            torch.save(self.netG_A.module.cpu().state_dict(), netG_A_path)
            self.netG_A = self.netG_A.cuda()
            torch.save(self.netG_B.module.cpu().state_dict(), netG_B_path)
            self.netG_B = self.netG_B.cuda()

        else:
            torch.save(self.netG_A.cpu().state_dict(), netG_A_path)
            torch.save(self.netG_A.cpu().state_dict(), netG_B_path)

        # torch.save(self.netG_A, os.path.join(save_dir, '{}_netG_A.pth'.format(epoch)))
        # torch.save(self.netG_B, os.path.join(save_dir, '{}_netG_B.pth'.format(epoch)))
        # remove network from memory
        # del self.netG_A
        # del self.netG_B
        # self.netG_A = None
        # self.netG_B = None

    # def load_from_disk(self):
    #     self.netG_A = torch.load(os.path.join(self.save_dir, 'netG_A.model'))
    #     self.netG_B = torch.load(os.path.join(self.save_dir, 'netG_B.model'))


if __name__ == '__main__':
    pass
