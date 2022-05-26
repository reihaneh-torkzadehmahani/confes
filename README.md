# CONFES : Label Noise-Robust Learning Using a Confidence-Based Sieving Strategy
**CONFES** is a learning approach that is robust to label noise. It is a novel sieving strategy which detects the samples with label noise and excludes them from the training process. 

# Installation
First, install the dependencies:
```
pip3 install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
```

# Run
Then, you can run CONFES:

```
python3 main.py --dataset cifar100 --model preact-resnet18 --lr 0.02 --weight-decay 5e-4 --batch-size 128  --epochs 300 --noise-rate 0.5
```

