import argparse
import os
import datetime

import numpy as np
import torch
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, DataParallel
from torch.nn.init import xavier_uniform, normal
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchnet.logger import VisdomPlotLogger, VisdomLogger

from dataset import data_folder, get_train_val_datasets
from net import ResNet, DenseNet, FocalLoss
from PIL import Image

parser = argparse.ArgumentParser(description='camera')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size(default: 64)')
parser.add_argument('--test_batch_size', type=int, default=2,
                    help='input batch size for test(default: 4)')
parser.add_argument('--n_threads_for_data', type=int, default=8,
                    help='threads for loading data')
parser.add_argument('--epochs', type=int, default=50,
                    help='epochs for train (default: 50)')
parser.add_argument('--num_classes', type=int, default=10,
                    help='number of classes (default: 10)')
parser.add_argument('--training_iterations', type=int, default=50000,
                    help='training iterations (default: 30000)')
parser.add_argument('--test_interval', type=int, default=2000,
                    help='test interval (default: 2000)')
parser.add_argument('--decay_step', type=int, default=5000,
                    help='decay interval (default: 5000)')
parser.add_argument('--seed', type=int, default=1024, metavar='S',
                    help='random seed (default: 1024)')
parser.add_argument('--visdom', type=bool, default=True,
                    help='visdom')
parser.add_argument('--visdom_port', type=int, default=7777,
                    help='visdom port')
parser.add_argument('--phase', type=str, default="train",
                    help="phase:train or test")
parser.add_argument('--data_root', type=str, default="../data")
parser.add_argument('--train_dirs', nargs='+', type=str, default=['train', 'add_train1', 'add_train2'])
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--checkpoint', type=str, default="checkpoint.pth.tar")
parser.add_argument('--crop_size', type=int, default='512')
parser.add_argument('--dropout', type=float, default=0.5)

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

train_folder, test_folder = get_train_val_datasets(args.data_root, args.train_dirs, args.crop_size)
print "training set size: {}".format(len(train_folder))
print "val set size: {}".format(len(test_folder))

# class_sample_weights = [1.0 / label_train.count(i) for i in xrange(10)]
# class_sample_weights = [x / np.sum(class_sample_weights) for x in class_sample_weights]
# class_sample_weights_dict = dict(zip(range(10), class_sample_weights))
# weights = [class_sample_weights_dict[x] for x in label_train]

class_weights = float(len(train_folder)) / (args.num_classes * np.bincount(train_folder.label_list))
img_weights = [class_weights[i] for i in train_folder.label_list]
train_sampler = WeightedRandomSampler(img_weights, len(train_folder), replacement=True)
train_data_loader = DataLoader(train_folder, num_workers=args.n_threads_for_data, batch_size=args.batch_size,
                               drop_last=True, sampler=train_sampler)
test_data_loader = DataLoader(test_folder, num_workers=args.n_threads_for_data, batch_size=args.test_batch_size,
                              drop_last=False)

print "start net"
#net = ResNet(152, args.num_classes).cuda()
#parallel_net = DataParallel(net)
#net = ResNet(152, args.num_classes).cuda()
net = DenseNet(201, args.num_classes, args.dropout).cuda()
#net = ResNet(152, 10).cuda()
parallel_net = DataParallel(net)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # normal(m.weight.data, 0, 0.02)
        xavier_uniform(m.weight.data)
        # xavier_uniform(m.bias.data)


print "net complete"

net.apply(weights_init)
optimizer = optim.Adam(net.parameters(), args.lr)
# optimizer = optim.Adam([
#     {'params': net.features.parameters(), 'lr': args.lr * 0.1},
#     {'params': net.fc.parameters(), 'lr': args.lr}
# ], weight_decay=0.0005)
# optimizer = optim.SGD([
#     {'params': net.features.parameters(), 'lr':args.lr * 0.1},
#     {'params': net.fc.parameters(), 'lr': args.lr}
# ], weight_decay=5e-6, momentum=0.9, nesterov=True)
criterion = FocalLoss(args.num_classes).cuda()
scheduler = StepLR(optimizer, gamma=0.5, step_size=args.decay_step)

train_meter_loss = tnt.meter.AverageValueMeter()
train_meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
train_confusion_meter = tnt.meter.ConfusionMeter(args.num_classes, normalized=True)

test_meter_loss = tnt.meter.AverageValueMeter()
test_meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
test_confusion_meter = tnt.meter.ConfusionMeter(args.num_classes, normalized=True)
print "start train"


def reset_meters(type="train"):
    if type == "train":
        train_meter_accuracy.reset()
        train_meter_loss.reset()
        train_confusion_meter.reset()
    else:
        test_meter_accuracy.reset()
        test_meter_loss.reset()
        test_confusion_meter.reset()


def test():
    net.eval()

    for (img_batch, label_batch) in test_data_loader:
        img_batch = Variable(img_batch, volatile=True).cuda()
        label_batch = Variable(label_batch, volatile=True).cuda()
        pred_batch = net(img_batch)
        loss = criterion(pred_batch, label_batch)
        test_confusion_meter.add(pred_batch.cpu().data, label_batch.cpu().data)
        test_meter_accuracy.add(pred_batch.cpu().data, label_batch.cpu().data)
        test_meter_loss.add(loss.cpu().data[0])
    # reset_meters("test")
    net.train()


def save_checkpoint(state, filename=args.checkpoint):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    torch.save(state, os.path.join('checkpoints', filename))


if args.visdom:
    print "visdom init"
    train_loss_logger = VisdomPlotLogger(
        'line', port=args.visdom_port, opts={'title': 'Train Loss mh, time: {}'.format(datetime.datetime.now())})
    train_err_logger = VisdomPlotLogger(
        'line', port=args.visdom_port, opts={'title': 'Train Class Accuracy mh, time: {}'.format(datetime.datetime.now())})
    train_confusion_logger = VisdomLogger('heatmap', port=args.visdom_port, opts={'title': 'Train Confusion matrixmh, time: {}'.format(datetime.datetime.now()),
                                                                                  'columnnames': list(
                                                                                      range(args.num_classes)),
                                                                                  'rownames': list(
                                                                                      range(args.num_classes))})
    test_loss_logger = VisdomPlotLogger(
        'line', port=args.visdom_port, opts={'title': 'Test Loss mh, time: {}'.format(datetime.datetime.now())})
    test_err_logger = VisdomPlotLogger(
        'line', port=args.visdom_port, opts={'title': 'Test Class Accuracy mh, time: {}'.format(datetime.datetime.now())})
    test_confusion_logger = VisdomLogger('heatmap', port=args.visdom_port, opts={'title': 'Test Confusion matrixmh, time: {}'.format(datetime.datetime.now()),
                                                                                 'columnnames': list(
                                                                                     range(args.num_classes)),

                                                                                 'rownames': list(
                                                                                     range(args.num_classes))})


def main():
    best_acc = None
    net.train()
    test_cnt = 0
    reset_meters("train")
    reset_meters("test")
    for iteration in xrange(args.training_iterations):
        try:
            img_batch, label_batch = train_data.next()
        except:
            train_data = iter(train_data_loader)
            img_batch, label_batch = train_data.next()
        optimizer.zero_grad()
        img_batch = Variable(img_batch).cuda()
        label_batch = Variable(label_batch).cuda()
        pred_batch = parallel_net(img_batch)
        loss = criterion(pred_batch, label_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_meter_loss.add(loss.cpu().data[0])
        train_meter_accuracy.add(pred_batch.cpu().data, label_batch.cpu().data)
        train_confusion_meter.add(pred_batch.cpu().data, label_batch.cpu().data)
        print "Iterations: {}/{}\t loss:{} \t".format(iteration + 1, args.training_iterations,
                                                      loss.cpu().data.numpy()[0])

        if (iteration + 1) % 20 == 0 and args.visdom:
            train_loss_logger.log(iteration / 20, train_meter_loss.value()[0])
            train_err_logger.log(iteration / 20, train_meter_accuracy.value()[0])
            train_confusion_logger.log(train_confusion_meter.value())
            reset_meters("train")
        if (iteration + 1) % args.test_interval == 0:
            print "testing"
            test()
            test_cnt += 1
            if args.visdom:
                test_loss_logger.log(test_cnt, test_meter_loss.value()[0])
                test_err_logger.log(test_cnt, test_meter_accuracy.value()[0])
                test_confusion_logger.log(test_confusion_meter.value())

            #if best_acc is None or best_acc < test_meter_accuracy.value()[0]:
                #best_acc = test_meter_accuracy.value()[0]

            save_checkpoint({
                'state_dict': net.state_dict(),
                'best_acc': test_meter_accuracy.value()[0],
                'optimizer': optimizer.state_dict(),
            }, filename=str(test_cnt)+"_"+args.checkpoint)
            reset_meters("test")
            # for epoch in xrange(args.epochs):
            # train(epoch)
            # test(epoch)


if __name__ == "__main__":
    main()
