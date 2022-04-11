# IGMC-dgl

### About
IGMC is an inductive matrix completion model based on graph neural networks without using any side information.  
This repository refactored IGMC using DGL instead of torch geometric(PyG).  
The basic version of IGMC can be found [here](https://github.com/muhanzhang/IGMC).

### Requirements
```
torch >= 1.9.0
dgl >= 0.8
```
Install [DGL](https://www.dgl.ai/pages/start.html)

### Data
It is being provided for the ml_100k dataset.  
You can receive the ml_100k dataset [here](https://grouplens.org/datasets/movielens/100k/).  

### Run
```
python main.py --file_name 'file_name'
```
The file_name is used to store log and model states.

