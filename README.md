# Safe Model-based Control from Signal Temporal Logic Specifications Using Recurrent Neural Networks

This is the implementation of the algorithm in the paper "Safe Model-based Control from Signal Temporal Logic Specifications Using Recurrent Neural Networks", ICRA 2023. This code is for simulating the fire-fighting experiment in the paper. Details can be found in the paper.

## Installation
You need Python3, Numpy, Gurobi, and Pytorch installed

## Usage
- Run initial_dataset.py to generate the initial dataset.
- Run training.py to train the model and policy networks and testing.

## Citation
When citing our work, please use the following BibTex:
'''
@inproceedings{liu2023safe,
  title={Safe model-based control from signal temporal logic specifications using recurrent neural networks},
  author={Liu, Wenliang and Nishioka, Mirai and Belta, Calin},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={12416--12422},
  year={2023},
  organization={IEEE}
}
'''
