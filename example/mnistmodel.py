from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# Loggers
import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass
from .testmodels import Net
from logutils import FlexTooledModel, get_presets
from logutils import functional as Fn
import pprint


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',  help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',  help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',  help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,  help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
#
parser.add_argument('--model_log_cfg', type=str, default='basic',  help='model logging config')
parser.add_argument('--plot_log_cfg', type=str, default='basic',  help='stats logging config')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


model = Net()
if use_cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

################################################################################
# Options for setting up logger

# select logging specifications
meters, plots = Fn.generate_layers(model, targets=['grad_norms', 'snr'])
logger = FlexTooledModel(plots, meters, model)  # Initialize the logger

# add another set of stats
plot, meter = get_presets('loss', 'accuracy')
logger.update_config(plot, meter)               # update the config with new chart
################################################################################


def package(tensor, to_cuda):
    if to_cuda is True:
        return Variable(tensor).cuda()
    else:
        return Variable(tensor)


def predict(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).sum().data[0]


def train(epoch):
    model.train()
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs = package(inputs, use_cuda)
        target = package(target, use_cuda)

        optimizer.zero_grad()
        output = model(inputs)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # add data
        logger(train_loss=loss.data[0], train_acc=predict(output, target))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

            test_step()
            logger.log(reset=True, step=True)   # Step and Log
        else:
            logger.step()  # Step and no log


def test_step():
    model.eval()
    inputs, target = next(iter(test_loader))
    inputs = package(inputs, use_cuda)
    target = package(target, use_cuda)
    output = model(inputs)
    loss = F.nll_loss(output, target)
    pred = predict(output, target)
    logger(test_loss=loss.data[0], test_acc=pred)
    model.train()
    return loss, pred


for epoch in range(1, args.epochs + 1):
    train(epoch)


