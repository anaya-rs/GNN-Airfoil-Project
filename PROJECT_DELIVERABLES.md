# GNN Airfoil Project - Deliverables & Guidelines

## Project Overview
Train a Graph Neural Network (GNN) to predict pressure coefficient (Cp) at mesh nodes around 2D airfoils using the AirfRANS CFD dataset.

**Frameworks:** Python • PyTorch • PyTorch Geometric (PyG) • Matplotlib

**Expected Outcomes:**
- Cp MSE < 1e-3 (normalized)
- Lift error < 10%
- R² (Cp) > 0.9

---

## Week 1 — Setup & Dataset Preparation

### Objectives
- Understand the AirfRANS dataset and GNN basics
- Prepare the graph data format (nodes, edges, features)

### Tasks
1. Environment setup (install torch, torch_geometric, matplotlib, airfrans)
2. Explore dataset and visualize airfoil meshes
3. Create PyTorch Geometric `Data` objects
4. Normalize coordinates and Cp values

### Deliverables
- `data_preprocess.py` - Data preprocessing script
- Mesh visualization - Visual representation of airfoil meshes
- Cp heatmap visualization - Heatmap showing pressure coefficient distribution

---

## Week 2 — Baseline GNN Implementation

### Objectives
Implement and train a simple Graph Convolutional Network (GCN) to predict Cp.

### Tasks
1. Build a 2-layer GCN (GCNConv → ReLU → Dropout → GCNConv → Linear)
2. Split dataset into train/val/test sets
3. Train using MSE loss, Adam optimizer
4. Evaluate and visualize predicted vs true Cp

### Deliverables
- `train_gcn.py` - Training script for baseline GCN model
- Plots - Visualization of training results and predictions
- Baseline summary report - Brief report on baseline model performance

---

## Week 3 — Model Improvement & Physics Validation

### Objectives
Refine model performance and validate results with aerodynamic metrics.

### Tasks
1. Implement GAT or GraphSAGE and include AoA/Re inputs
2. Compute lift coefficient (Cl) from predicted Cp
3. Visualize Cp error maps and lift vs AoA curve

### Deliverables
- `improved_gnn.py` - Improved GNN model implementation (GAT or GraphSAGE)
- Cp contour comparison - Visual comparison of predicted vs true Cp contours
- Short validation note - Validation report with aerodynamic metrics

---

## Week 4 — Final Testing, Results & Report

### Objectives
Finalize model testing, create visuals, and prepare the report.

### Tasks
1. Test on unseen airfoils and calculate final metrics
2. Generate Cp vs chordwise plots and scatter comparisons
3. Compile a concise 3–5 page report (Introduction, Dataset, Methodology, Results, Discussion, Conclusion)

### Deliverables
- `report.pdf` - Final project report (3–5 pages)
- `presentation.pptx` - Project presentation slides
- Finalized code folder - Complete, organized codebase with all scripts

---

## Code Style Guidelines

### Comments
- **Minimal comments** - Only add comments when absolutely necessary for clarity
- **Lower case** - All comments should be written in lowercase
- **Human-sounding** - Comments should sound natural and conversational, not formal or technical jargon

### Examples of Good Comments
```python
# normalize the coordinates to help training
# check if the model converged
# quick sanity check before saving
```

### Examples of Bad Comments
```python
# NORMALIZE THE COORDINATES TO HELP TRAINING
# This function normalizes the coordinates to help training
# Normalize coordinates
```

---

## Final Output
A trained GNN surrogate model capable of predicting airfoil Cp with near-CFD accuracy.

**Suggested Title:** "Graph Neural Network Surrogate for Airfoil Pressure Distribution Prediction"

