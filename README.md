# Practise

This is the official Pytorch implementation for [Practical Network Acceleration with Tiny Sets](https://arxiv.org/abs/2202.07861). 

In this project, we also implement MiR which is [Compressing Models with Few Samples: Mimicking then Replacing](https://arxiv.org/abs/2201.02620). 

## Requirements

* Python3
* pytorch
* pandas

## Datasets

Please prepare datasets first, and then modify `path` in `dataset.py`.

## Usage

Compute the recoverability of one block
```
python main.py --num_sample 500 --seed 2021 --epoch 1000 --practise one --rm_blocks layer1.1 --gpu_id 0
```

Compute recoverabilities of all blocks
```
python main.py --num_sample 500 --seed 2021 --epoch 1000 --practise all --rm_blocks 2 --gpu_id 0
```

Finetune the pruned network
```
python main.py --num_sample 500 --seed 2021 --epoch 2000 --FT MiR --rm_blocks layer1.1 --gpu_id 0
```

### ResNet

For ResNet-34 and ResNet-50, the removable blocks are

```
1: layer1.1,layer1.2
2: layer2.1,layer2.2,layer2.3
3: layer3.1,layer3.2,layer3.3,layer3.4,layer3.5
4: layer4.1,layer4.2
```

### MobileNet V2

For MobileNetV2, the removable blocks are
```
1: 24->24: features.3
2: 32->32: features.5,features.6
3: 64->64: features.8,features.9,features.10
4: 96->96: features.12,features.13
5: 160->160: features.15,features.16
```


## Test the latency

```
python speed.py --model mobilenet_v2 --cudnn-benchmark --rm_blocks features.9
```

## Results in our paper

We provide all shells to reproduce all results in our paper. Please check shells in the `exp` folder.

## Citation

If you find the work useful for your research, please cite:

```
@inproceedings{wang2023practical,
  title={Practical Network Acceleration with Tiny Sets},
  author={Wang, Guo-Hua and Wu, Jianxin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}

@inproceedings{wang2022compressing,
  title={Compressing models with few samples: Mimicking then replacing},
  author={Wang, Huanyu and Liu, Junjie and Ma, Xin and Yong, Yang and Chai, Zhenhua and Wu, Jianxin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={701--710},
  year={2022}
}
```