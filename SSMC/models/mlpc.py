import torch
import torch.nn as nn
import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

'''
Basic MLP bases on SoftSRV paper
3-layers in -> in + (out - in)/2 -> out
using GElU bc i read that its superior to RELU
'''
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.a1 = nn.Linear(input_dim, hidden_dim)
        self.z1 = nn.GELU()
        self.a2 = nn.Linear(hidden_dim, hidden_dim)
        self.z2 = nn.GELU()
        self.a3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.a3(self.z2(self.a2(self.z1(self.a1(x)))))

'''
creates num_mlps (64 in my case) and computes the forward pass 
(tweaked 1410 NN code we hadnt covered it yet in 1420)
embeddings are passed through all the mlps which learn the embedding pattern without updating model weights
'''
class MLPC(nn.Module):
    def __init__(self):
        super().__init__()
        config = load_config()
        mlpc = config["softsrv"]["mlpc"]
        self.input_dim = mlpc["input_dim"]
        self.hidden_dim = mlpc["hidden_dim"]
        self.output_dim = mlpc["output_dim"]
        self.num_mlps = mlpc["mlps"]

        self.mlps = nn.ModuleList([
            MLP(self.input_dim, self.hidden_dim, self.output_dim) for _ in range(self.num_mlps)
        ])
    
    def forward(self, x):
        outputs = []
        for mlp in self.mlps:
            outputs.append(mlp(x))
        return torch.stack(outputs, dim=1)