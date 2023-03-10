import argparse
from torch import nn as nn
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy.random as npr
import numpy as np
import sys
from itertools import chain
import copy
import math
from resnet import ResNet18
from torch.utils.data import Dataset
device = torch.device("cuda")

def train_one_epoch(args, model, device, train_dataset, optimizer, criterion, epoch):
    train_loss = 0.
    correct = 0.
    total = 0.

    model.train()

    trainset_permutation_inds = npr.permutation(np.arange(len(train_dataset.targets)))
    
    batch_size = args.batch_size
    
    batch_cnt = 0
    for batch_index, batch_start_idx in  enumerate(range(0, len(train_dataset.targets), batch_size)):
        batch_inds = trainset_permutation_inds[batch_start_idx: batch_start_idx + batch_size]
        batch_cnt += 1

        transformed_train_dataset = []
        for ind in batch_inds:
            transformed_train_dataset.append(train_dataset.__getitem__(ind)[0])
        
        inputs = torch.stack(transformed_train_dataset).to(device)
        targets = torch.LongTensor(np.array(train_dataset.targets)[batch_inds].tolist()).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        
        # update model parameters, accuracy, loss
        loss = loss.mean()
        train_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write(
            '| Train | Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
            (epoch, args.epochs, batch_index + 1,
             (len(train_dataset) // batch_size) + 1, loss.item(),
             100. * correct.item() / total))
        sys.stdout.flush()
    
    if isinstance(correct, float) == False:
        correct = correct.item()
    return correct / total, train_loss / batch_cnt


def test_one_epoch(args, epoch, model, device, test_dataset, criterion):
    test_loss = 0.
    correct = 0.
    total = 0.
    test_batch_size = 32

    model.eval()
    batch_cnt = 0
    for batch_index, batch_start_ind in enumerate(range(0, len(test_dataset.targets), test_batch_size)):
        batch_cnt += 1

        transformed_testset = []
        for ind in range(batch_start_ind, min(len(test_dataset.targets), batch_start_ind + test_batch_size)):
            transformed_testset.append(test_dataset.__getitem__(ind)[0])
        
        inputs = torch.stack(transformed_testset).to(device)
        targets = torch.LongTensor(np.array(test_dataset.targets)[batch_start_ind:batch_start_ind + test_batch_size].tolist()).to(device)

        # Forward propagation, compute loss, get predictions
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss.mean()
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write(
            '| Test | Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
            (epoch, args.epochs, batch_index + 1,
             (len(test_dataset) // test_batch_size) + 1, loss.item(),
             100. * correct.item() / total))
        sys.stdout.flush()
    
    if isinstance(correct, float) == False:
        correct = correct.item()
    return correct / total, test_loss / batch_cnt
    
def main(args):
    # model, optim, criterion, scheduler
    model = ResNet18(num_classes = 10).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().cuda()
    criterion.__init__(reduce=False)
    scheduler = MultiStepLR(
        optimizer, milestones=[60, 120, 160], gamma=0.2)
    
    # load dataset
    normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(
        root='C:\\Users\\GM\\Desktop\\liuzhengchang',
        train=True,
        transform=train_transform,
        download=True)
    test_dataset = datasets.CIFAR10(
        root='C:\\Users\\GM\\Desktop\\liuzhengchang',
        train=False,
        transform=test_transform,
        download=True)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    npr.seed(args.seed)

    train_indx = np.array(range(len(train_dataset.targets)))
    train_dataset.data = train_dataset.data[train_indx, :, :, :]
    train_dataset.targets = np.array(train_dataset.targets)[train_indx].tolist()

    # init
    train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []

    # train test loop
    for epoch in range(args.epochs):
        train_acc, train_loss = train_one_epoch(args, model, device, train_dataset, optimizer, criterion, epoch)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        test_acc, test_loss = test_one_epoch(args, epoch, model, device, test_dataset, criterion)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        print(f"epoch: {epoch}, acc_train: {train_acc_list[-1]}, loss_train: {train_loss_list[-1]}, acc_test: {test_acc_list[-1]}, loss_test: {test_loss_list[-1]}")

        scheduler.step()

    print(f"train_acc_list: {train_acc_list}")
    print(f"train_loss_list: {train_loss_list}")
    print(f"test_acc_list: {test_acc_list}")
    print(f"test_loss_list: {test_loss_list}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument("--epochs", type=int, default = 200, help = "epochs")
    parser.add_argument("--batch_size", type=int, default = 128, help = "batch_size for training")
    parser.add_argument("--lr", type=float, default = 0.1, help = "lr")
    parser.add_argument("--seed", type=int, default = 1, help = "seed")
    args = parser.parse_args()
    main(args)
