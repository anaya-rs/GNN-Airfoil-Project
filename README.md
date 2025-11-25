# GNN Airfoil Pressure Distribution Prediction

graph neural network surrogate model for predicting airfoil pressure coefficient (Cp) using the airfrans cfd dataset.

## project overview

this project implements graph neural networks (gnns) to predict pressure coefficient distributions around 2d airfoils. the model serves as a fast surrogate for cfd simulations, enabling rapid design exploration and optimization.

**expected outcomes:**
- cp mse < 1e-3 (normalized)
- lift error < 10%
- r² (cp) > 0.9

## getting started (for beginners)

### prerequisites

before you begin, make sure you have:
- **python 3.8 or higher** installed ([download here](https://www.python.org/downloads/))
- **git** installed ([download here](https://git-scm.com/downloads))
- a terminal/command prompt (command prompt on windows, terminal on mac/linux)

### step 1: clone the repository

open your terminal/command prompt and navigate to where you want to save the project:

```bash
# navigate to your desired folder (example)
cd Desktop
# or
cd Documents
```

then clone the repository:

```bash
git clone https://github.com/anaya-rs/GNN-Airfoil-Project.git
```

this will create a folder called `GNN-Airfoil-Project` with all the project files.

### step 2: navigate to the project folder

```bash
cd GNN-Airfoil-Project
```

### step 3: set up python environment (recommended)

it's best practice to use a virtual environment to avoid conflicts with other projects:

**on windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**on mac/linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

you should see `(venv)` at the beginning of your command prompt, indicating the virtual environment is active.

### step 4: install dependencies

install all required packages:

```bash
pip install -r requirements.txt
```

this will install:
- pytorch
- pytorch geometric
- matplotlib
- numpy
- scikit-learn
- airfrans

**note:** this may take a few minutes depending on your internet connection.

### step 5: run the code

now you're ready to run the project! follow these steps in order:

#### 5.1 preprocess the data

this downloads and prepares the dataset:

```bash
python data_preprocess.py
```

**what it does:**
- downloads the airfrans dataset (if not already downloaded)
- converts data to graph format
- normalizes features
- creates visualizations
- saves preprocessed data to `./data/` folder

**expected time:** 5-15 minutes depending on your computer and internet speed

#### 5.2 train the baseline model

train a simple graph convolutional network:

```bash
python train_gcn.py
```

**what it does:**
- trains a 2-layer gcn model
- saves training plots to `./results/`
- saves the best model

**expected time:** 10-30 minutes depending on your gpu/cpu

#### 5.3 train improved models

train more advanced models (gat and graphsage):

```bash
python improved_gnn.py
```

**what it does:**
- trains both gat and graphsage models
- creates cp contour comparisons
- generates lift vs angle of attack plots
- saves results to `./results_improved/` and `./results_sage/`

**expected time:** 20-60 minutes depending on your hardware

#### 5.4 evaluate the models

test the models on unseen data:

```bash
python evaluate.py
```

**what it does:**
- evaluates models on test set
- generates cp vs chordwise position plots
- creates scatter comparison plots
- calculates final metrics

**expected time:** 5-10 minutes

### viewing results

after running the scripts, you'll find:

- **visualizations:** in the `./data/`, `./results/`, and `./evaluation_results_*/` folders
- **trained models:** saved as `.pt` files in the results folders
- **plots:** saved as `.png` files showing predictions, training history, etc.

### troubleshooting

**problem: "command not found" or "python is not recognized"**
- make sure python is installed and added to your system path
- try using `python3` instead of `python` on mac/linux
- restart your terminal after installing python

**problem: "module not found" error**
- make sure you activated the virtual environment (you should see `(venv)` in your prompt)
- run `pip install -r requirements.txt` again
- check that you're in the correct project directory

**problem: out of memory errors**
- reduce `n_samples` in `data_preprocess.py` (change from 500 to 100 or 50)
- reduce `batch_size` in training scripts (change from 8 to 4 or 2)
- close other applications to free up memory

**problem: dataset download is slow**
- the dataset is large (~several gb), be patient
- check your internet connection
- the download happens automatically on first run

**problem: cuda/gpu errors**
- the code will automatically use cpu if gpu is not available
- if you have gpu issues, the code should fall back to cpu automatically

### quick start (all steps at once)

if you want to run everything sequentially:

```bash
# 1. clone and navigate
git clone https://github.com/anaya-rs/GNN-Airfoil-Project.git
cd GNN-Airfoil-Project

# 2. create virtual environment (windows)
python -m venv venv
venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. run all scripts in order
python data_preprocess.py
python train_gcn.py
python improved_gnn.py
python evaluate.py
```

### need help?

if you encounter any issues:
1. check the error message carefully
2. make sure all prerequisites are installed
3. verify you're in the correct directory
4. ensure the virtual environment is activated

## installation

```bash
pip install -r requirements.txt
```

## project structure

```
.
├── data_preprocess.py      # data preprocessing and visualization
├── train_gcn.py            # baseline gcn model training
├── improved_gnn.py        # improved gat/graphsage models
├── evaluate.py             # evaluation on unseen airfoils
├── requirements.txt        # python dependencies
└── README.md              # this file
```

## usage

### 1. preprocess data

```bash
python data_preprocess.py
```

this will:
- load airfrans dataset
- convert to graph format
- normalize features
- create mesh and cp visualizations
- save preprocessed datasets

### 2. train baseline gcn

```bash
python train_gcn.py
```

trains a 2-layer graph convolutional network and generates training plots.

### 3. train improved models

```bash
python improved_gnn.py
```

trains both gat and graphsage models with graph-level features (angle of attack, reynolds number).

### 4. evaluate on test set

```bash
python evaluate.py
```

evaluates models on unseen airfoils and generates:
- cp vs chordwise position plots
- scatter comparison plots
- lift coefficient calculations


## results

results are saved in:
- `./results/` - baseline gcn results
- `./results_improved/` - gat model results
- `./results_sage/` - graphsage model results
- `./evaluation_results_*/` - final evaluation results

## frameworks

- python
- pytorch
- pytorch geometric (pyg)
- matplotlib

