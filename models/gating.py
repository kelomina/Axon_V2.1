import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatingMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        # return logits
        return x

class GatingTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead=4, num_layers=2):
        super(GatingTransformer, self).__init__()
        # Project input to hidden_dim for transformer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Transformer expects sequence, so we might need to reshape or project
        # Here we treat the feature vector as a single token embedding if we just project it, 
        # but to use attention we usually need a sequence. 
        # For simplicity, let's just project and pass through encoder as (batch, 1, hidden)
        
        x = self.embedding(x) # (batch, hidden)
        x = x.unsqueeze(1)    # (batch, 1, hidden)
        x = self.transformer_encoder(x) # (batch, 1, hidden)
        x = x.squeeze(1)      # (batch, hidden)
        x = self.fc(x)        # (batch, output)
        return x

def create_gating_model(model_type, input_dim, hidden_dim, output_dim):
    if model_type == 'mlp':
        return GatingMLP(input_dim, hidden_dim, output_dim)
    elif model_type == 'transformer':
        return GatingTransformer(input_dim, hidden_dim, output_dim)
    else:
        raise ValueError(f"Unknown gating model type: {model_type}")
