import os
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
from data import create_dataset
import evaluation.mini_resnet
import ipdb
from options.evaluate_gan_options import EvaluationGANOptions
import numpy as np
from models import create_model

def get_model(model_name):
    """
    get resnet model
    :param model_name:
    :return:
    """
    for name, cls in evaluation.mini_resnet.__dict__.items():
        if name.lower() == model_name.lower():
            model_func = cls
            break
    return model_func(False, num_classes=2)


if __name__ == '__main__':

    opt = EvaluationGANOptions().parse()
    opt.phase = 'test'
    opt.lr = 0.001
    opt.beta1 = 0.9
    opt.gan_mode = 'wgangp'
    opt.gamma = 0.01
    opt.pool_size = 32
    opt.load_size = 64
    opt.crop_size = 64
    opt.batch_size = 64
    save_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.evaluation_checkpoint)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    evaluation_model = torch.load(opt.evaluation_model_filename)
    evaluation_model.eval()
    gan_model = create_model(opt)      # create a model given opt.model and other options
    if opt.evolutionary:
        gan_model.generators[0].netG_A.eval()
        gan_model.generators[0].netG_B.eval()

    label_A = torch.zeros((opt.batch_size,)).type(torch.LongTensor)
    label_B = torch.ones((opt.batch_size,)).type(torch.LongTensor)
    if torch.cuda.is_available():
        label_A = label_A.cuda()
        label_B = label_B.cuda()

    evaluator_accurate = 0
    num_accurate = 0
    num_images = 0
    for i, data in enumerate(dataset):
       # ipdb.set_trace()
        #print(data)
        real_A = data['A']
        real_B = data['B']

        y = evaluation_model(real_A)
        evaluator_accurate += torch.sum(y == 0).item()
        y = evaluation_model(real_B)
        evaluator_accurate += torch.sum(y == 1).item()

        gan_model.set_input(data)  # unpack data from data loader
        with torch.no_grad():
            gan_model.forward()           # run inference

        data_A = gan_model.fake_A_list[0]
        data_B = gan_model.fake_B_list[0]

        if torch.cuda.is_available():
            data_A = data_A.cuda()
            data_B = data_B.cuda()

        y = evaluation_model(data_A)
        _, pred_A = torch.max(y, 1)
        num_accurate += torch.sum(pred_A == 0).item()
        num_images += len(data_A)

        # predictions for dataB
        y = evaluation_model(data_B)
        _, pred_B = torch.max(y, 1)
        num_accurate += torch.sum(pred_B == 1).item()
        num_images += len(data_B)

    accuracy = num_accurate / num_images
    print("Accuracy of evaluator = {}".format(evaluator_accurate/ num_images))
    print("Accuracy of GAN model = {}".format(accuracy))
