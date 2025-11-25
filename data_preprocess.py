import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import numpy as np
import airfrans as af
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

def load_airfrans_data(split='train', n_samples=None, root='./airfrans_data', task='full'):
    """load airfrans dataset"""
    # download dataset if not exists
    if not os.path.exists(root) or not os.path.exists(os.path.join(root, 'manifest.json')):
        print(f"downloading airfrans dataset to {root}...")
        os.makedirs(root, exist_ok=True)
        af.dataset.download(root=root)
    
    # load dataset using airfrans api
    train = (split == 'train')
    dataset = af.dataset.load(root=root, task=task, train=train)
    
    if n_samples:
        return dataset[:n_samples]
    return dataset

def create_graph_data(sample, k=6):
    """convert airfrans point cloud sample to pytorch geometric data object"""
    # airfrans provides point clouds with features: position, inlet velocity, distance to airfoil
    # sample is a Simulation object or dict-like
    
    if hasattr(sample, 'pos'):
        pos = sample.pos
        inlet_velocity = sample.inlet_velocity if hasattr(sample, 'inlet_velocity') else None
        distance_to_airfoil = sample.distance_to_airfoil if hasattr(sample, 'distance_to_airfoil') else None
        pressure = sample.pressure if hasattr(sample, 'pressure') else None
        angle_of_attack = getattr(sample, 'angle_of_attack', 0.0)
        reynolds_number = getattr(sample, 'reynolds_number', 1e6)
    else:
        # handle dict-like access
        pos = sample.get('pos', sample.get('mesh_pos', None))
        inlet_velocity = sample.get('inlet_velocity', sample.get('velocity', None))
        distance_to_airfoil = sample.get('distance_to_airfoil', None)
        pressure = sample.get('pressure', None)
        angle_of_attack = sample.get('angle_of_attack', 0.0)
        reynolds_number = sample.get('reynolds_number', 1e6)
    
    # convert to tensors
    pos = torch.tensor(pos, dtype=torch.float)
    
    # build node features: position, inlet velocity, distance to airfoil
    node_features = [pos]
    
    if inlet_velocity is not None:
        inlet_velocity = torch.tensor(inlet_velocity, dtype=torch.float)
        if inlet_velocity.dim() == 1:
            inlet_velocity = inlet_velocity.unsqueeze(1)
        node_features.append(inlet_velocity)
    
    if distance_to_airfoil is not None:
        distance_to_airfoil = torch.tensor(distance_to_airfoil, dtype=torch.float)
        if distance_to_airfoil.dim() == 0:
            distance_to_airfoil = distance_to_airfoil.unsqueeze(0).expand(pos.size(0))
        if distance_to_airfoil.dim() == 1:
            distance_to_airfoil = distance_to_airfoil.unsqueeze(1)
        node_features.append(distance_to_airfoil)
    
    x = torch.cat(node_features, dim=1)
    
    # build edges using k-nn graph
    edge_index = knn_graph(pos, k=k, loop=False)
    
    # target: pressure coefficient (Cp) or pressure
    if pressure is not None:
        y = torch.tensor(pressure, dtype=torch.float)
        if y.dim() == 0:
            y = y.unsqueeze(0).expand(pos.size(0))
        if y.dim() == 1:
            y = y.unsqueeze(1)
    else:
        # if no pressure, use zeros (will need to be filled from dataset)
        y = torch.zeros(pos.size(0), 1, dtype=torch.float)
    
    # graph-level features: angle of attack, reynolds number
    graph_features = torch.tensor([
        angle_of_attack,
        reynolds_number
    ], dtype=torch.float)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        pos=pos,
        graph_features=graph_features
    )
    
    return data

def normalize_data(dataset):
    """normalize node features and targets"""
    all_x = []
    all_y = []
    
    for data in dataset:
        all_x.append(data.x)
        all_y.append(data.y)
    
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    x_mean = all_x.mean(dim=0, keepdim=True)
    x_std = all_x.std(dim=0, keepdim=True) + 1e-8
    
    y_mean = all_y.mean()
    y_std = all_y.std() + 1e-8
    
    normalized_dataset = []
    for data in dataset:
        normalized_x = (data.x - x_mean) / x_std
        normalized_y = (data.y - y_mean) / y_std
        
        data.x = normalized_x
        data.y = normalized_y
        
        normalized_dataset.append(data)
    
    return normalized_dataset, {
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std
    }

def visualize_mesh(data, save_path='mesh_visualization.png'):
    """visualize airfoil mesh"""
    pos = data.pos.numpy()
    edge_index = data.edge_index.numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # plot edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        ax.plot([pos[src, 0], pos[dst, 0]], 
                [pos[src, 1], pos[dst, 1]], 
                'b-', alpha=0.3, linewidth=0.5)
    
    # plot nodes
    ax.scatter(pos[:, 0], pos[:, 1], c='red', s=1, alpha=0.6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('airfoil mesh')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"mesh visualization saved to {save_path}")

def visualize_cp_heatmap(data, save_path='cp_heatmap.png'):
    """visualize pressure coefficient heatmap"""
    pos = data.pos.numpy()
    cp = data.y.numpy()
    
    # handle multi-dimensional y
    if cp.ndim > 1:
        cp = cp.squeeze()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(pos[:, 0], pos[:, 1], c=cp, 
                        cmap='coolwarm', s=2, alpha=0.8)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('pressure coefficient (Cp) distribution')
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cp')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"cp heatmap saved to {save_path}")

def preprocess_dataset(split='train', n_samples=100, save_dir='./data', task='full'):
    """main preprocessing function"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"loading {split} dataset...")
    dataset = load_airfrans_data(split=split, n_samples=n_samples, task=task)
    
    # get dataset length
    try:
        dataset_len = len(dataset)
    except:
        dataset_len = n_samples if n_samples else 100
    
    print(f"converting to graph format...")
    graph_dataset = []
    for i, sample in enumerate(dataset):
        if i % 10 == 0:
            print(f"processing sample {i}/{dataset_len}")
        try:
            graph_data = create_graph_data(sample)
            graph_dataset.append(graph_data)
        except Exception as e:
            print(f"error processing sample {i}: {e}")
            continue
    
    if len(graph_dataset) == 0:
        raise ValueError("no valid samples processed")
    
    print("normalizing data...")
    normalized_dataset, norm_stats = normalize_data(graph_dataset)
    
    # visualize first sample
    print("creating visualizations...")
    visualize_mesh(normalized_dataset[0], 
                   save_path=os.path.join(save_dir, 'mesh_visualization.png'))
    visualize_cp_heatmap(normalized_dataset[0], 
                        save_path=os.path.join(save_dir, 'cp_heatmap.png'))
    
    # save dataset and normalization stats
    torch.save(normalized_dataset, os.path.join(save_dir, f'{split}_dataset.pt'))
    torch.save(norm_stats, os.path.join(save_dir, 'norm_stats.pt'))
    
    print(f"preprocessing complete. dataset saved to {save_dir}")
    return normalized_dataset, norm_stats

if __name__ == '__main__':
    # preprocess training data
    train_dataset, norm_stats = preprocess_dataset(split='train', n_samples=500)
    
    # preprocess validation data
    val_dataset, _ = preprocess_dataset(split='val', n_samples=100)
    
    # preprocess test data
    test_dataset, _ = preprocess_dataset(split='test', n_samples=100)

