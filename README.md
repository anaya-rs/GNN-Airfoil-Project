# GNN Airfoil Pressure Distribution Prediction

graph neural network surrogate model for predicting airfoil pressure coefficient (Cp) using the airfrans cfd dataset.

## project overview

this project implements graph neural networks (gnns) to predict pressure coefficient distributions around 2d airfoils. the model serves as a fast surrogate for cfd simulations, enabling rapid design exploration and optimization.

**expected outcomes:**
- cp mse < 1e-3 (normalized)
- lift error < 10%
- r² (cp) > 0.9

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

## code style

- minimal comments (only when necessary)
- lowercase, human-sounding comments
- clean, readable code structure

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

