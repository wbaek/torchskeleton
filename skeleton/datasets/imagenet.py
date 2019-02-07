# -*- coding: utf-8 -*-
import os
import json
import logging

from PIL import Image
import torch
from torchvision import transforms

LOGGER = logging.getLogger(__name__)


class Imagenet(torch.utils.data.Dataset):
    @staticmethod
    def sets(batch_size):
        data_shape = [(batch_size, 3, 224, 224), (batch_size,)]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train_set = Imagenet(train=True, transform=transform_train)
        valid_set = Imagenet(train=False, transform=transform_valid)
        #train_set = torch.utils.data.Subset(train_set, list(range(1000)))
        #valid_set = torch.utils.data.Subset(valid_set, list(range(1000)))

        return train_set, valid_set, data_shape

    @staticmethod
    def loader(batch_size, cv_ratio=0.0, num_workers=32):
        assert cv_ratio < 1.0
        eps = 1e-5

        train_set, test_set, data_shape = Imagenet.sets(batch_size=batch_size)
        if cv_ratio > 0.0:
            num_train_set = int(len(train_set) * (1 - cv_ratio) + eps)
            num_valid_set = len(train_set) - num_train_set
            train_set, valid_set = torch.utils.data.random_split(train_set, [num_train_set, num_valid_set])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

        if cv_ratio > 0.0:
            valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
            return train_loader, valid_loader, test_loader, data_shape
        return train_loader, test_loader, data_shape

    def __init__(self, train=True, transform=None, target_transform=None, root='/data/public/ro/dataset/images/imagenet/ILSVRC/2016/object_localization/ILSVRC'):
        super(Imagenet, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        data_path = os.path.abspath(os.path.dirname(__file__) + '/datas/ILSVRC/classification/')
        with open(data_path + '/imagenet1000_classid_to_text_synsetid.json') as f:
            temp_map = json.load(f)
        self._maps = {
            'idx2synset': {int(key): value['id'] for key, value in iter(temp_map.items())},
            'synset2idx': {value['id']: int(key) for key, value in iter(temp_map.items())},
            'idx2text': {int(key): value['text'] for key, value in iter(temp_map.items())}
        }

        if self.train:
            datalist = [
                line.strip().split(' ')[0]
                for line in open(root + '/ImageSets/CLS-LOC/train_cls.txt').readlines()
                if line.strip()
            ]
            datapoints = [
                (root + '/Data/CLS-LOC/train/' + line + '.JPEG', int(self._maps['synset2idx'][line.split('/')[0]]))
                for line in datalist
            ]
            self.train_data, self.train_labels = list(zip(*datapoints))
        else:
            datalist = [
                line.strip()
                for line in open(data_path + '/imagenet_2012_validation_synset_labels.txt').readlines()
                if line.strip()
            ]
            datapoints = [
                (root + '/Data/CLS-LOC/val/ILSVRC2012_val_%08d.JPEG' % (i + 1), int(self._maps['synset2idx'][synset]))
                for i, synset in enumerate(datalist)
            ]
            self.test_data, self.test_labels = list(zip(*datapoints))

    def __len__(self):
        return len(self.train_labels if self.train else self.test_labels)

    def __getitem__(self, index):
        if self.train:
            filename, target = self.train_data[index], self.train_labels[index]
        else:
            filename, target = self.test_data[index], self.test_labels[index]

        img = self.pil_loader(filename)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
