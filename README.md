# PyTorch skeleton
[![KakaoBrain](https://img.shields.io/badge/kakao-brain-ffcd00.svg)](http://kakaobrain.com/)
[![pytorch](https://img.shields.io/badge/pytorch-1.1.0-%23ee4c2c.svg)](https://pytorch.org/)
[![CodeFactor](https://www.codefactor.io/repository/github/wbaek/pytorch_skeleton/badge)](https://www.codefactor.io/repository/github/wbaek/pytorch_skeleton)
[![CircleCI](https://circleci.com/gh/wbaek/pytorch_skeleton.svg?style=svg)](https://circleci.com/gh/wbaek/pytorch_skeleton)

## Basic Utilities for PyTorch


----


## [DAWNBench][] Introduction
#### An End-to-End Deep Learning Benchmark and Competition
> DAWNBench is a benchmark suite for end-to-end deep learning training and inference. Computation time and cost are critical resources in building deep models, yet many existing benchmarks focus solely on model accuracy. DAWNBench provides a reference set of common deep learning workloads for quantifying training time, training cost, inference latency, and inference cost across different optimization strategies, model architectures, software frameworks, clouds, and hardware.

### [DAWNBench Image Classification on CIFAR10][]

In my test, 30 out of 50 runs reached 94% test set accuracy. **Runtime for 25 epochs is roughly 69sec** using [Kakao Brain][] [BrainCloud][] V1.XLARGE Type (V100 1GPU, 14CPU, 122GB).

| | trials | <sub>\> 94% count</sub> | average | median | min | max |
|:---:|---:|---:|---:|---:|---:|---:|
| **metric** | 50 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;30 | 94.057 | 94.045 | 93.700 | 94.330 |


### Environment Setup & Experiments
* pre requirements
```bash
$ apt update
$ apt install -y libsm6 libxext-dev libxrender-dev libcap-dev
$ pip install torch torchvision
```

* clone and init. the repository
```bash
$ git clone {THIS_REPOSITORY} && cd pytorch_skeleton
$ pip install -r requirements.txt
```

* run dawnbench image classification on CIFAR10
```bash
$ python bin/dawnbench_cifar10.py --seed 0xC0FFEE --download > log_dawnbench_cifar10.tsv
```


[Kakao Brain]: https://kakaobrain.com/
[BrainCloud]: https://cloud.kakaobrain.com/
[DAWNBench]: https://dawn.cs.stanford.edu/benchmark/index.html
[DAWNBench Image Classification on CIFAR10]: https://dawn.cs.stanford.edu/benchmark/CIFAR10/train.html