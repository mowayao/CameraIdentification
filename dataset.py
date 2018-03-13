import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torchvision.transforms as transforms
from utils import RESOLUTIONS, ORIENTATION_FLIP_ALLOWED, LABEL_DICT, MANIPULATIONS

def get_train_val_datasets(data_root, train_dirs, crop_size):
    img_list, label_list = get_img_list(data_root, train_dirs)
    print "finished get img lists"
    train_img_list, train_label_list, val_img_list, val_label_list = train_val_split(img_list, label_list)
    print "finished split train val set"
    return data_folder(train_img_list, train_label_list, crop_size), \
           data_folder(val_img_list, val_label_list, crop_size, False)

def resolution_check(img_path, label):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    return [h, w] in RESOLUTIONS[LABEL_DICT[label]]

def get_img_list(data_root, train_dirs):
    img_dict = {}
    for train_dir in train_dirs:
        train_data_root = os.path.join(data_root, train_dir)
        for root, _, fnames in os.walk(train_data_root):
            label = root.split(os.path.sep)[-1]
            if train_dir != "train" and label == "Motorola-X":
                continue
            for fname in fnames:
                if ".zip" in fname:
                    continue
                img_path = os.path.join(root, fname)
                #print img_path, label
                img_dict[img_path] = LABEL_DICT[label]
    img_list, label_list = zip(*sorted(img_dict.items()))
    return img_list, label_list

def train_val_split(img_list, label_list, k=0.1):
    class_count = np.bincount(label_list)
    val_cnt = int(k * len(label_list)) // len(class_count)
    class_range = []
    start, end = 0, 0
    for cnt in class_count:
        end += cnt
        class_range.append((start, end))
        start += cnt
    val_idx = [set(random.sample(xrange(s, e), val_cnt)) for s, e in class_range]
    train_idx = [set(xrange(s, e)) - val_idx[i] for i, (s, e) in enumerate(class_range)]
    val_idx = reduce(lambda x, y: x | y, val_idx)
    train_idx = reduce(lambda x, y: x | y, train_idx)
    val_img_list, val_label_list = [img_list[i] for i in val_idx], [label_list[i] for i in val_idx]
    train_img_list, train_label_list = [img_list[i] for i in train_idx], [label_list[i] for i in train_idx]
    return train_img_list, train_label_list, val_img_list, val_label_list


def crop_img(img, crop_size, random=True):
    h, w = img.shape[:2]
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    if pad_h > 0:
        img = np.pad(img, ((pad_h // 2, pad_h - pad_h // 2), (0, 0), (0, 0)), "wrap")
    if pad_w > 0:
        img = np.pad(img, ((0, 0), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), "wrap")
    h, w = img.shape[:2]
    if random:
        sh = np.random.randint(h - crop_size + 1)
        sw = np.random.randint(w - crop_size + 1)
    else:
        sh = (h - crop_size) // 2
        sw = (w - crop_size) // 2
    return img[sh: sh + crop_size, sw: sw + crop_size]

def random_manipulation(img):
    manipulation = random.choice(MANIPULATIONS)

    if manipulation.startswith("jpg"):
        quality = int(manipulation[3:])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buf = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(buf, -1)
    elif manipulation.startswith("gamma"):
        gamma = float(manipulation[5:])
        img = np.uint8(cv2.pow(img / 255., gamma) * 255.)
    elif manipulation.startswith("bicubic"):
        scale = float(manipulation[7:])
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img

def process_img(img, label_id, crop_size, train):
    if np.random.rand() < 0.5 and ORIENTATION_FLIP_ALLOWED[label_id]:
        img = np.rot90(img, 1, (0, 1))

    img = crop_img(img, crop_size * 2, train)

    if np.random.rand() < 0.5:
        img = random_manipulation(img)

    img = crop_img(img, crop_size, train)
    return img


class data_folder(Dataset):
    def __init__(self, img_list, label_list, crop_size, train=True):
        super(data_folder, self).__init__()
        self.img_list, self.label_list = img_list, label_list
        self.crop_size = crop_size
        self.train = train
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        label = self.label_list[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = process_img(img, label, self.crop_size, self.train)
        img = self.transforms(img.copy())
        return img, label

    def __len__(self):
        return len(self.img_list)


class data_folder_test(Dataset):
    def __init__(self, img_lists, crop_size):
        super(data_folder_test, self).__init__()
        self.img_lists = img_lists

        self.transforms = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img = cv2.imread(self.img_lists[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transforms(img)
        return img, self.img_lists[idx].split('/')[-1]

    def __len__(self):
        return len(self.img_lists)
