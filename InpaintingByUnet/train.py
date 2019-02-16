import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from torchvision import transforms

import matplotlib.pyplot as plt

from model import UNet
from dataloader import DataLoader

from torch.autograd import Variable
import torch.nn.functional as F

WILL_TRAIN = True
WILL_TEST = False

def train_net(net,
              epochs = 3,
              batch_size = 16,
              data_dir='data/',
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              gpu=True):
    train_list = []
    train_path = join(data_dir, 'train.png')
    for i in range(320):
        train_list.append(train_path)
    train_dataset = DataLoader(train_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)    
    print('Total training items', len(train_dataset), ', Total training mini-batches in one epoch:', len(train_loader))

    optimizer = optim.SGD(net.parameters(),
                            lr=lr,
                            momentum=0.99,
                            weight_decay=0.005)

    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        epoch_loss = 0

        for i, (img, label) in enumerate(train_loader):
            # todo: load image tensor to gpu
            if gpu:
                img = Variable(img.cuda())
                label = label.cuda()
            optimizer.zero_grad()
            
            # todo: get prediction and loss
            pred_label = net(img)            
            criterion = nn.MSELoss()
            loss = criterion(pred_label, label)
            epoch_loss += loss.item()
            print('Epoch: %d Itr: %d - Loss: %.6f' % (epoch, i+1, loss.item()))
            
            # optimize weights
            loss.backward()
            optimizer.step()
            
        # save train results
        # label = label.cpu().detach()
        # mask = img.cpu().detach()[:,3:4,:,:]
        # train_input = (label - label * (1 - mask))
        # train_output = pred_label.cpu().detach()
        # path = join(data_dir, 'samples/')
        # save_img(label, path, epoch, 'train_gt.png')
        # save_img(train_input, path, epoch, 'train_in.png')
        # save_img(train_output, path, epoch, 'train_out.png')

        # perform test and save test results
        # test_net(testNet=net, epoch = epoch, batch_size=1, gpu=args.gpu, data_dir=args.data_dir)
        
        # save net 
        if((epoch + 1) % 20 == 0) :   
            torch.save(net.state_dict(), join(data_dir, 'checkpoints/') + 'CP%d.pth' % (epoch + 1))
            print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / (i+1)))

# displays test images with original and predicted masks 
def test_net(testNet, 
            epoch,
            batch_size,
            gpu=True,
            data_dir='data/'):
    test_list = []
    test_path = join(data_dir, 'test.png')
    test_num = 5
    for i in range(test_num):
        test_list.append(test_path)
    test_dataset = DataLoader(test_list)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print('Total testing items', len(test_dataset), ', Total testing mini-batches in one epoch:', len(test_loader))

    testNet.eval()
    # Open accuracy file
    # f = open(data_dir + net_folder + 'accuracy.txt', 'a')
    # f.write(net_name+'\n')
    with torch.no_grad():
        for i, (img, label) in enumerate(test_loader):
            if gpu:
                img = Variable(img.cuda())
            pred_label = testNet(img)          
            mask = img.cpu().detach()[:,3:4,:,:]
            test_input = (label - label * (1 - mask))
            test_output = pred_label.cpu().detach()
            path = join(data_dir, 'samples/')

            # save test_groundtruth, test_input, test_output
            save_img(label, path, i, 'test_gt.png')
            save_img(test_input, path, i, 'test_in.png')
            save_img(test_output, path, i, 'test_out.png')

            # compute accuracy and write to file
            # N = label.shape[0] * label.shape[1]
            # accuracy = np.sum(label == pred_label) / N
            # f.write(str(i) + ' ' + str(accuracy) + '\n')                
        # f.close()

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=3, type='int', help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=16, type='int', help='batch size')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

def save_img(image, path, epoch, image_name):
    plt.imshow(image[0].permute(1,2,0).numpy())
    plt.savefig(path + '%d_' % (epoch + 1) + image_name)
    plt.close()

if __name__ == '__main__':
    args = get_args()

    net = UNet()

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    if WILL_TRAIN:
        train_net(net=net,
            batch_size=args.batch_size,
            epochs=args.epochs,
            gpu=args.gpu,
            data_dir=args.data_dir)

    if WILL_TEST:
        testNet = UNet()
        net_folder = 'checkpoints/'
        net_name = 'CP80'
        state_dict = torch.load('data/' + net_folder + net_name + '.pth')
        testNet.load_state_dict(state_dict)
        testNet.cuda()
        test_net(testNet=testNet, 
            epoch=0,
            batch_size=1,
            gpu=args.gpu,
            data_dir=args.data_dir)
