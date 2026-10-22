# step1_extract_weights.py
# Run this in your PyTorch environment

import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertModel

print("PyTorch:", torch.__version__)

# ── Model Architecture ────────────────────────────────
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1024 + 6, 512)
        self.batchnorm = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)

    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.bilstm(sequence_output)
        mean_pool = torch.mean(lstm_output, dim=1)
        max_pool, _ = torch.max(lstm_output, dim=1)
        pooled_output = torch.cat((mean_pool, max_pool), dim=1)
        combined = torch.cat((pooled_output, extra_features), dim=1)
        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return self.fc2(x)

# ────────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────────
print("Loading PyTorch model...")
device = torch.device("cpu")
model = HybridModel()
model.load_state_dict(
    torch.load("best_model.pt", map_location=device)
)
model.eval()
print("✅ Model loaded")

# ────────────────────────────────────────────────────
# Extract and save all weights as numpy arrays
# ────────────────────────────────────────────────────
print("\nExtracting weights...")

weights = {}

# FC1
weights["fc1_weight"] = model.fc1.weight.detach().numpy()
weights["fc1_bias"]   = model.fc1.bias.detach().numpy()
print("✅ FC1 weights extracted")

# FC2
weights["fc2_weight"] = model.fc2.weight.detach().numpy()
weights["fc2_bias"]   = model.fc2.bias.detach().numpy()
print("✅ FC2 weights extracted")

# BatchNorm
weights["bn_gamma"]        = model.batchnorm.weight.detach().numpy()
weights["bn_beta"]         = model.batchnorm.bias.detach().numpy()
weights["bn_running_mean"] = model.batchnorm.running_mean.detach().numpy()
weights["bn_running_var"]  = model.batchnorm.running_var.detach().numpy()
print("✅ BatchNorm weights extracted")

# BiLSTM — Forward
weights["lstm_ih_f"]   = model.bilstm.weight_ih_l0.detach().numpy()
weights["lstm_hh_f"]   = model.bilstm.weight_hh_l0.detach().numpy()
weights["lstm_b_ih_f"] = model.bilstm.bias_ih_l0.detach().numpy()
weights["lstm_b_hh_f"] = model.bilstm.bias_hh_l0.detach().numpy()
print("✅ BiLSTM forward weights extracted")

# BiLSTM — Backward
weights["lstm_ih_b"]   = model.bilstm.weight_ih_l0_reverse.detach().numpy()
weights["lstm_hh_b"]   = model.bilstm.weight_hh_l0_reverse.detach().numpy()
weights["lstm_b_ih_b"] = model.bilstm.bias_ih_l0_reverse.detach().numpy()
weights["lstm_b_hh_b"] = model.bilstm.bias_hh_l0_reverse.detach().numpy()
print("✅ BiLSTM backward weights extracted")

# Save all weights to a single file
np.savez("scamshield_weights.npz", **weights)
print("\n✅ All weights saved to: scamshield_weights.npz")
print("   Copy this file to your TensorFlow environment")
print("   Then run: python step2_build_tflite.py")