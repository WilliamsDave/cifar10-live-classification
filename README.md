## Overview
This script utilizes the user's webcam to perform live image classification based on the CIFAR-10 dataset. 

Available classifications include: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Uses a pre-trained resnet56 model from [chenyaofo](https://github.com/chenyaofo/pytorch-cifar-models)

## Setup
* ```git clone https://github.com/WilliamsDave/cifar10-live-classification.git```
* ```python3 -m venv <your_venv_name>```
* ```source <your_venv_name>/bin/activate``` (for MAC users)
* ```<your_venv_name>\Scripts\activate.bat``` (for Windows users)
* ```pip install -r .\requirements.txt```
* ```python app.py```

