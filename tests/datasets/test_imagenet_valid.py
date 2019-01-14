# -*- coding: utf-8 -*-
from torchvision import transforms
from skeleton.datasets.imagenet import Imagenet


def test_imagenet_valid_dataset():
    transform_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    valid_set = Imagenet(train=False, transform=transform_valid)

    assert len(valid_set) == 50000
    assert min(valid_set.test_labels) == 0
    assert max(valid_set.test_labels) == 999

    assert valid_set._maps['idx2synset'][0] == 'n01440764'
    assert valid_set._maps['synset2idx']['n01440764'] == 0
    assert valid_set._maps['idx2synset'][10] == 'n01530575'
    assert valid_set._maps['synset2idx']['n01530575'] == 10
    assert valid_set._maps['idx2text'][0] == 'tench, Tinca tinca'
    assert valid_set._maps['idx2text'][10] == 'brambling, Fringilla montifringilla'
