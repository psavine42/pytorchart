from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import MNIST
from os.path import expanduser
home = expanduser("~")
# Loggers
from testmodels import Net, MnistFC
from pytorchart import FlexTooledModel
from pytorchart import functional as Fn
from utils_plus import show

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 1000)')
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


mnist_xform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True} if use_cuda else {}

train_loader = DataLoader(MNIST(home + '/data/mnist', train=True, download=True, transform=mnist_xform),
                          batch_size=args.batch_size, **kwargs)
test_loader = DataLoader(MNIST(home + '/data/mnist', train=False, transform=mnist_xform),
                         batch_size=args.test_batch_size, **kwargs)


model = MnistFC()   # Net() # MnistFC()
if use_cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

################################################################################
# Options for setting up logger

# select logging specifications
meters, plots = Fn.generate_layers(model, targets=['snr']) # 'grad_norms',
Stat = FlexTooledModel(plots, meters, model)  # Initialize the logger

# add another set of stats
Stat.add_presets('loss', 'acc')   # update the config with new chart
print(Stat)
################################################################################


def package(tensor, to_cuda):
    if to_cuda is True:
        return Variable(tensor).cuda()
    else:
        return Variable(tensor)


def predict(output, target):
    # get the index of the max log-probability
    _, pred = output.max(1)
    return pred.eq(target).sum().data[0] / target.size(0)


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
        pred = predict(output, target)
        # add data
        Stat(train_loss=loss.data[0], train_acc=pred)
        if batch_idx % args.log_interval == 0:
            test_loss, test_acc = test_step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain - Loss: {:.4f}, Acc: {:.3f}\tTest - Loss: {:.4f}, Acc: {:.3f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], pred, test_loss, test_acc))
            # Test and log
            Stat(test_loss=test_loss, test_acc=test_acc)
            Stat.log(reset=True)    # Log
        Stat.step()     # Step


def test_step():
    model.eval()
    inputs, target = next(iter(test_loader))
    inputs = package(inputs, use_cuda)
    target = package(target, use_cuda)
    output = model(inputs)
    loss = F.nll_loss(output, target)
    model.train()
    return loss.data[0], predict(output, target)


for epoch in range(1, args.epochs + 1):
    train(epoch)


