from data import config, voc0712, __init__
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
import visdom
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

save_folder = "weights/"
resume = None
visdom_flag = True
viz = visdom.Visdom()
learning_rate = 1e-3
gamma = 0.1
cuda_flag = True
voc_dataset = voc0712.VOC_ROOT

def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(torch.zeros((1, 3)).cpu(), X=torch.zeros((1,)).cpu(), opts=dict(xlabel=_xlabel, ylabel=_ylabel, title=_title, legend=_legend))

def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = learning_rate * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    dataset_root = "data/"
    cfg = config.voc
    dataset = voc0712.VOCDetection(root=dataset_root, transform=SSDAugmentation(cfg["min_dim"], config.MEANS))
    viz = visdom.Visdom()

    ssd_net = build_ssd("train", cfg["min_dim"], cfg["num_classes"])
    net = ssd_net
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
        net.cuda()

    if resume:
        print('Resuming training, loading {}...'.format(resume))
        ssd_net.load_weights(resume)
    else:
        vgg_weights = torch.load(save_folder + 'vgg16_reducedfc.pth')
        ssd_net.vgg.load_state_dict(vgg_weights)
        print("Initializing weights...")
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(cfg["num_classes"], 0.5, True, 0, True, 3, 0.5, False, cuda_flag) # cuda = True

    net.train()

    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print("Loading the dataset...")

    epoch_size = len(dataset) // 32
    print('Training SSD on:', dataset.name)

    step_index = 0

    if visdom_flag:

        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, 32, num_workers=4, shuffle=True, collate_fn=__init__.detection_collate, pin_memory=True)

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(0, cfg["max_iter"]):
        if visdom_flag and iteration != 0 and (iteration % epoch_size ==0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None, "append", epoch_size)

            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg["lr_steps"]:
            step_index += 1
            adjust_learning_rate(optimizer, gamma, step_index)

        # load train data
        images, targets = next(batch_iterator)

        if cuda_flag:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), valatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, valatile=True) for ann in targets]

        # forward
        t0 = time.time()
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')

        if visdom_flag:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                       repr(iteration) + '.pth')
        torch.save(ssd_net.state_dict(),
                   save_folder + '' + voc_dataset + '.pth')


if __name__ == "__main__":
    train()










