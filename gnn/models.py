import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv

class SpeciesGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, 1)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        return torch.sigmoid(self.conv2(h, edge_index))

class SpeciesGAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=2)
        self.conv2 = GATConv(2*hidden, 1, heads=1)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        return torch.sigmoid(self.conv2(h, edge_index))

from torch_geometric.data import Data
import networkx as nx

def graph_to_data(G: nx.DiGraph) -> Data:
    mapping = {n: i for i, n in enumerate(G.nodes)}
    edges = [[mapping[u], mapping[v]] for u, v in G.edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = []
    for n in G.nodes:
        x.append([G.out_degree(n), G.in_degree(n)])
    x = torch.tensor(x, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


def train_dummy_gnn(G: nx.DiGraph, epochs: int = 10) -> SpeciesGCN:
    """Train a simple GCN on random targets to demonstrate workflow."""
    data = graph_to_data(G)
    model = SpeciesGCN(in_dim=2, hidden=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    y = torch.rand(len(G.nodes), 1)
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()
        optimizer.step()
    return model


def predict_scores(model: nn.Module, G: nx.DiGraph) -> dict:
    data = graph_to_data(G)
    with torch.no_grad():
        scores = model(data.x, data.edge_index).squeeze().cpu().numpy()
    return {n: float(s) for n, s in zip(G.nodes, scores)}
