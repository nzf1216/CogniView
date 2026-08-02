import torch
import torch.nn as nn

class ADClassifierModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, dropout=0.4):
        super(ADClassifierModel, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1) # Logits output for binary classification (HC vs MCI/AD)
        )
        
    def forward(self, x):
        # Shape: (Batch, Seq_Len, Features) -> Permute to (Batch, Features, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.bn(self.conv1d(x))))
        x = x.permute(0, 2, 1) # Back to (Batch, Seq_Len, Channels)
        
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :] # Capture terminal state dynamics
        
        logits = self.classifier(last_step)
        return logits.squeeze(1)