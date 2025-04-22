# fair-paper-rec
A fairness aware recommender system for honors thesis



  

## Environment Setup


### Install Anaconda3
Download and install [Anaconda3](https://www.anaconda.com/products/distribution)

### 1. Create and activate environment
```bash

conda create -n fairrec python=3.12.2

conda activate fairrec

```

### 2. Install PyTorch with CUDA 11.7 support
```bash

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.7 -c pytorch -c nvidia

```

### 3. Install additional dependencies
```bash

pip install pandas matplotlib scikit-learn numpy

```
