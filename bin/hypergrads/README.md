# TorchSkeleton

## Hypergrads


----


* experiments
```
python bin/hypergrads/cifar.py --debug --dataset-name cifar10 --batch 128 --lr 0.01 --hypergrad --result-path checkpoints/cifar10/resnet18/batch128/hypergrad/lr0.01
python bin/hypergrads/cifar.py --debug --dataset-name cifar10 --batch 128 --lr 0.05 --hypergrad --result-path checkpoints/cifar10/resnet18/batch128/hypergrad/lr0.05
python bin/hypergrads/cifar.py --debug --dataset-name cifar10 --batch 128 --lr 0.01 --result-path checkpoints/cifar10/resnet18/batch128/sgd/lr0.01
python bin/hypergrads/cifar.py --debug --dataset-name cifar10 --batch 128 --lr 0.05 --result-path checkpoints/cifar10/resnet18/batch128/sgd/lr0.05
python bin/hypergrads/cifar.py --debug --dataset-name cifar100 --batch 128 --lr 0.01 --hypergrad --result-path checkpoints/cifar100/resnet18/batch128/hypergrad/lr0.01
python bin/hypergrads/cifar.py --debug --dataset-name cifar100 --batch 128 --lr 0.05 --hypergrad --result-path checkpoints/cifar100/resnet18/batch128/hypergrad/lr0.05
python bin/hypergrads/cifar.py --debug --dataset-name cifar100 --batch 128 --lr 0.01 --result-path checkpoints/cifar100/resnet18/batch128/sgd/lr0.01
python bin/hypergrads/cifar.py --debug --dataset-name cifar100 --batch 128 --lr 0.05 --result-path checkpoints/cifar100/resnet18/batch128/sgd/lr0.05
```
