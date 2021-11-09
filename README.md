# MCNN
pytorch implement for MCNN

# Environment
(1) python 3.7<br>
(2) pytorch 1.7

# Dataset
[SHHA and SHHB](https://www.datafountain.cn/datasets/5670). <br>
You need to [prepare the two dataset](https://github.com/svishwa/crowdcount-mcnn/blob/master/README.md) before training.

# Train
```bash
cd mcnn
python train.py --dataset SHHA --use_tensorboard True
python train.py --dataset SHHB --use_tensorboard True
```
use "--resume True" to recover from the latest saved model

# Test
```bash
cd mcnn
python test.py --dataset SHHA
python test.py --dataset SHHB
```
