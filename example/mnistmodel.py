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
from logutils import FlexTooledModel, get_preset
from logutils import functional as Fn

from utils_plus import show, package

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',  help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',  help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',  help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,  help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='how many batches to wait before logging training status')
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
if use_cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

################################################################################
# Options for setting up logger

plots = {'plt1': # an example config for a plot
           {'type': 'line',
            'opts': {'layout': {'yaxis': {'type':'log', 'autorange':True,},}}}}

meters = Fn.SNR(model, target='plt1')

# select logging specifications
logger = FlexTooledModel(plots, meters, model)    # Initialize the logger

plot, meter = get_preset('loss+accuracy')    # add another set of stats
logger.update_config(plot, meter)            # update the config with new chart
################################################################################


def predict(output, target):
    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
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
            logger.log(reset=True, step=True)   # Step and Log
            test_step()
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


