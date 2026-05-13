# model.py
# Model architecture — must match best_model.pt exactly.
# This is the original nn.LSTM version (not the ManualBiLSTM export version),
# since we're running in PyTorch directly, not converting to TFLite.

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


def build_config():
    config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
    config.activation = "gelu_pytorch_tanh"
    return config


class HybridModel(nn.Module):
    """
    DistilBERT + BiLSTM hybrid — matches the original training architecture.
    Uses nn.LSTM (not ManualBiLSTM) since we don't need TFLite compatibility.
    """
    def __init__(self):
        super().__init__()
        config         = build_config()
        self.bert      = DistilBertModel(config)
        self.bilstm    = nn.LSTM(
                            input_size=768,
                            hidden_size=256,
                            batch_first=True,
                            bidirectional=True,
                         )
        self.dropout   = nn.Dropout(0.3)
        self.fc1       = nn.Linear(1024 + 6, 512)
        self.batchnorm = nn.BatchNorm1d(512)
        self.relu      = nn.ReLU()
        self.fc2       = nn.Linear(512, 2)

    def forward(self, input_ids, attention_mask, extra_features):
        outputs         = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state          # [B, 128, 768]
        lstm_out, _     = self.bilstm(sequence_output)       # [B, 128, 512]
        mean_pool       = torch.mean(lstm_out, dim=1)        # [B, 512]
        max_pool, _     = torch.max(lstm_out,  dim=1)        # [B, 512]
        pooled          = torch.cat((mean_pool, max_pool), dim=1)   # [B, 1024]
        combined        = torch.cat((pooled, extra_features), dim=1) # [B, 1030]
        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return self.fc2(x)                                   # [B, 2]


def load_model(checkpoint_path: str, device: torch.device) -> HybridModel:
    model = HybridModel()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model