import os
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
from data import create_dataset
import evaluation.mini_resnet
import ipdb
from options.evaluation_options import EvaluationOptions
import numpy as np


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

    opt = EvaluationOptions().parse()
    opt.crop_size = 64
    opt.load_size = 75
    opt.name = 'evaluators'
    save_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.evaluation_checkpoint)
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    opt.phase = 'val'
    val_dataset = create_dataset(opt)
    opt.phase = 'train'

    train_dataset_size = len(train_dataset)  # get the number of images in the dataset.
    val_dataset_size = len(val_dataset)
    model = get_model(opt.evaluation_model)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    label_A = torch.zeros((opt.batch_size,)).type(torch.LongTensor)
    label_B = torch.ones((opt.batch_size,)).type(torch.LongTensor)
    if torch.cuda.is_available():
        label_A = label_A.cuda()
        label_B = label_B.cuda()

    best_accuracy = 0
    val_loss_list = []
    train_loss_list = []
    val_accuracy_list = []
    for epoch in range(opt.num_epochs):
        val_size = 0
        train_size = 0
        val_loss = 0
        train_loss = 0
        val_accurate = 0
        model.train()
        for i, data in enumerate(train_dataset):
            optimizer.zero_grad()
            data_A = data['A']
            data_B = data['B']
            if torch.cuda.is_available():
                data_A = data_A.cuda()
                data_B = data_B.cuda()
            y = model(data_A)
            loss = criterion(y, label_A[:y.shape[0]])
            train_size += len(y)
            
            y = model(data_B)
            loss += criterion(y, label_B[:y.shape[0]])
            train_size += len(y)
            train_loss += loss.item() * opt.batch_size
            loss.backward()
            optimizer.step()
        train_loss = train_loss / train_size

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dataset):
                data_A = data['A']
                data_B = data['B']
                if torch.cuda.is_available():
                    data_A = data_A.cuda()
                    data_B = data_B.cuda()
                # predictions for dataA
                y = model(data_A)
                loss = criterion(y, label_A[:y.shape[0]])
                _, pred_A = torch.max(y, 1)
                val_accurate += torch.sum(pred_A == 0).item()
                val_size += len(y)

                # predictions for dataB
                y = model(data_B)
                loss += criterion(y, label_B[:y.shape[0]])
                _, pred_B = torch.max(y, 1)
                val_accurate += torch.sum(pred_B == 1).item()
                val_size += len(y)

                val_loss += loss.item() * opt.batch_size

        val_loss = val_loss / val_size
        val_accuracy = val_accurate / val_size * 100
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)
        print("Epoch = {} validation loss = {}, validation accuracy = {}".format(epoch, val_loss, val_accuracy))
        if val_accuracy > best_accuracy:
            torch.save(model, os.path.join(save_dir, 'best_evaluator.pth'))
            best_accuracy = val_accuracy
        if best_accuracy > opt.validation_accuracy_goal:
            print('Training stopped early because validation accuracy > goal.')
            break
    print("Training evaluator finished")
    print("Best validation accuracy = {}".format(best_accuracy))
    np.save(os.path.join(save_dir, 'validation_loss.npy'), np.asarray(val_loss_list))
    np.save(os.path.join(save_dir, 'train_loss.npy'), np.asarray(train_loss_list))
    np.save(os.path.join(save_dir, 'validation_accuracy.npy'), np.asarray(val_accuracy_list))



