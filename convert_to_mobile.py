import torch
import torch.nn as nn
from transformers import DistilBertModel
from torch.utils.mobile_optimizer import optimize_for_mobile

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

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state
        # (batch, seq_len, 768)

        lstm_output, _ = self.bilstm(sequence_output)
        # (batch, seq_len, 512)

        mean_pool = torch.mean(lstm_output, dim=1)
        max_pool, _ = torch.max(lstm_output, dim=1)

        pooled_output = torch.cat((mean_pool, max_pool), dim=1)
        # shape: (batch, 1024)

        combined = torch.cat((pooled_output, extra_features), dim=1)

        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        logits = self.fc2(x)

        return logits

device = torch.device("cpu")

# Load trained model
model = HybridModel()
model.load_state_dict(torch.load("best_model.pt", map_location=device))

model.eval()

# Dummy input for tracing
input_ids = torch.randint(0, 1000, (1, 128))
attention_mask = torch.ones((1, 128))
extra_features = torch.randn((1, 6))

# Convert to TorchScript
traced_model = torch.jit.trace(
    model,
    (input_ids, attention_mask, extra_features)
)

optimized_model = optimize_for_mobile(traced_model)

# Save mobile model
optimized_model._save_for_lite_interpreter("scamshield_model.ptl")

print("Mobile model saved as scamshield_model.ptl")