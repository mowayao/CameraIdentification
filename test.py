import argparse
import os
import numpy as np

import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from utils import LABELS
from dataset import data_folder_test
from net import ResNet, DenseNet

parser = argparse.ArgumentParser(description='camera')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size(default: 8)')
parser.add_argument('--data_root', type=str, default="../data")
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--train_dir", type=str, default="train")
parser.add_argument("--test_dir", type=str, default="test")
parser.add_argument("--checkpoint", type=str, default="dense.pth.tar")
parser.add_argument('--crop_size', type=int, default='512')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--result_name', type=str, default="Dense201.csv")

args = parser.parse_args()


def preprocess(data_root):
    print data_root
    return map(lambda x: os.path.join(data_root, x), os.listdir(data_root))


img_lists = preprocess(os.path.join(args.data_root, args.test_dir))

# train_img_lists = preprocess(os.path.join(args.data_root, os.path.join(args.train_dir, labels[0])))
# img_train, img_test, label_train, label_test = train_test_split(img_lists, label_lists, test_size=0.2, shuffle=True, random_state=args.seed)

test_folder = data_folder_test(img_lists, args.crop_size)
# train_folder = data_folder_test(train_img_lists)
test_data_loader = DataLoader(test_folder, num_workers=8, batch_size=args.batch_size, shuffle=False, drop_last=False)
# train_data_loader = DataLoader(train_folder, num_workers=2, batch_size=args.batch_size, shuffle=False, drop_last=False)
print "testing set size:", len(test_folder)
# subm = pd.read_csv(os.path.join(args.data_root, 'sample_submission.csv'), index_col='fname')
# img_files = subm['camera']['fname']
# print img_files

# def preprocess(data_root):
# 	labels = os.listdir(data_root)
# 	img_lists = []
# 	label_lists = []
# 	for i, label in enumerate(labels):
# 		imgs = os.listdir(os.path.join(data_root, label))
# 		imgs = map(lambda x: os.path.join(data_root, label, x), imgs)
# 		img_lists.extend(imgs)
# 		label_lists.extend([i] * len(imgs))
# 	return img_lists, label_lists
# preprocess(os.path.join(args.data_root, args.train_dir))
net = DenseNet(201, args.num_classes, args.dropout).cuda()
subm = pd.read_csv(os.path.join('../sample_submission.csv'), index_col='fname')

predict = {}
start = 11
end = 18
def test_all_net(idx):
    filename = str(idx) + "_" + args.checkpoint
    net_dict = torch.load(os.path.join("checkpoints", filename))
    print net_dict['best_acc']
    # print net_dict['']
    net.load_state_dict(net_dict['state_dict'])
    net.eval()
    par_net = nn.DataParallel(net)
    repeats = 10
    pred = {}
    for i in xrange(repeats):
        print("start test epoch %d" % i)
        for img_batch, file_batch in test_data_loader:
            img_batch = Variable(img_batch).cuda()
            pred_batch = par_net(img_batch)
            pred_batch = F.softmax(pred_batch, dim=1)
            pred_batch = pred_batch.cpu().data.numpy()
            for i, fname in enumerate(file_batch):
                if fname not in pred:
                    pred[fname] = pred_batch[i] / repeats
                else:
                    pred[fname] += pred_batch[i] / repeats
    for key in pred:
        if key not in predict:
            predict[key] = pred[key] / (end-start+1)
        else:
            predict[key] += pred[key] / (end-start+1)

def ensemble_model():
    for _ in xrange(start, end+1):
        print "testing {} model".format(_)
        test_all_net(_)
    for file_name in predict:
        pred_batch = predict[file_name]
        output = np.argmax(pred_batch)
        subm['camera'][file_name] = LABELS[output]
    subm.to_csv(args.result_name)
def main():
    net_dict = torch.load(os.path.join("checkpoints", args.checkpoint))
    print net_dict['best_acc']
    # print net_dict['']
    net.load_state_dict(net_dict['state_dict'])
    net.eval()
    predict = {}
    repeats = 15
    for i in xrange(repeats):
        print("start test epoch %d" % i)
        for img_batch, file_batch in test_data_loader:
            img_batch = Variable(img_batch).cuda()
            pred_batch = net(img_batch)
            pred_batch = F.softmax(pred_batch, dim=1)
            pred_batch = pred_batch.cpu().data.numpy()
            for i, fname in enumerate(file_batch):
                if fname not in predict:
                    predict[fname] = pred_batch[i] / repeats
                else:
                    predict[fname] += pred_batch[i] / repeats

    for file_name in predict:
        pred_batch = predict[file_name]
        output = np.argmax(pred_batch)
        subm['camera'][file_name] = LABELS[output]
    subm.to_csv(args.result_name)
        # print file_batch
        # img_batch = Variable(img_batch).cuda()
        # pred_batch = net(img_batch)
        # _, output = torch.max(pred_batch, 1)
        # print output, file_batch


if __name__ == "__main__":
    #main()
    ensemble_model()
    #main()
