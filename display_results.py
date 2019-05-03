import torch
from options.display_options import DisplayOptions
from data import create_dataset
from models import create_model
import ipdb

if __name__ == '__main__':
    opt = DisplayOptions().parse()
    opt.lr = 0.001
    opt.beta1 = 0.9
    opt.gan_mode = 'wgangp'
    opt.gamma = 0.01
    opt.pool_size = 32
    opt.load_size = 64
    opt.phase = 'test'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    #if torch.cuda.is_available():
    #    model = model.cuda()

    for i, data in enumerate(dataset):
        ipdb.set_trace()

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
