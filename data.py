from torch.utils.data import Dataset
import PIL.Image as pil_img
import PIL
import cv2
import torchvision.transforms as transforms
import numpy as np
from utils import Config
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class data_folder(Dataset):
	def __init__(self, img_lists, label_lists, phrase="training"):
		super(data_folder, self).__init__()
		self.img_lists = img_lists
		self.label_lists = label_lists
		if phrase == "training":
			self.transforms = transforms.Compose([
				transforms.RandomCrop(Config.img_size),###may be tricks
				transforms.RandomHorizontalFlip(),
				transforms.RandomVerticalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
			])
		else:
			self.transforms = transforms.Compose([
				transforms.CenterCrop(Config.img_size),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
	def __getitem__(self, idx):
		img = cv2.imread(self.img_lists[idx])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = PIL.Image.fromarray(img)


		img = self.transforms(img)
		label = self.label_lists[idx]
		return img, label
	def __len__(self):
		return len(self.img_lists)
class data_folder_test(Dataset):
	def __init__(self, img_lists):
		super(data_folder_test, self).__init__()
		self.img_lists = img_lists

		self.transforms = transforms.Compose([
			transforms.CenterCrop(Config.img_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
		])
	def __getitem__(self, idx):
		img = cv2.imread(self.img_lists[idx])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = PIL.Image.fromarray(img)
		img = self.transforms(img)
		return img, self.img_lists[idx].split('/')[-1]
	def __len__(self):
		return len(self.img_lists)