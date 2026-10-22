# convert_to_tflite_clean.py

import os
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertModel
import onnx

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
# STEP 1 — Load PyTorch model
# ────────────────────────────────────────────────────
print("=" * 50)
print("Step 1: Loading PyTorch model...")
device = torch.device("cpu")
model = HybridModel()
model.load_state_dict(
    torch.load("best_model.pt", map_location=device)
)
model.eval()
print("✅ PyTorch model loaded\n")

# ────────────────────────────────────────────────────
# STEP 2 — Export PyTorch → ONNX
# ────────────────────────────────────────────────────
print("=" * 50)
print("Step 2: Exporting PyTorch → ONNX...")

dummy_input_ids      = torch.zeros(1, 128, dtype=torch.long)
dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)
dummy_features       = torch.zeros(1, 6, dtype=torch.float32)

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask, dummy_features),
    "scamshield_model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask", "extra_features"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids":      {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "extra_features": {0: "batch_size"},
        "logits":         {0: "batch_size"},
    }
)

size_mb = os.path.getsize("scamshield_model.onnx") / (1024 * 1024)
print(f"✅ ONNX saved — size: {size_mb:.2f} MB\n")

# ────────────────────────────────────────────────────
# STEP 3 — Verify ONNX
# ────────────────────────────────────────────────────
print("=" * 50)
print("Step 3: Verifying ONNX model...")
onnx_model = onnx.load("scamshield_model.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX model verified\n")

# ────────────────────────────────────────────────────
# STEP 4 — Convert ONNX → TensorFlow using onnx2tf CLI
# CLI is more stable than the Python API
# ────────────────────────────────────────────────────
print("=" * 50)
print("Step 4: Converting ONNX → TensorFlow...")

os.makedirs("scamshield_tf_model", exist_ok=True)

result = os.system(
    "onnx2tf "
    "-i scamshield_model.onnx "
    "-o scamshield_tf_model "
    "--non_verbose"
)

if result != 0:
    print("❌ onnx2tf CLI failed")
    print("Try running manually:")
    print("  onnx2tf -i scamshield_model.onnx -o scamshield_tf_model")
    exit(1)

print("✅ TensorFlow SavedModel saved\n")

# ────────────────────────────────────────────────────
# STEP 5 — Convert TensorFlow → TFLite
# ────────────────────────────────────────────────────
print("=" * 50)
print("Step 5: Converting TensorFlow → TFLite...")
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(
    "scamshield_tf_model"
)

# Dynamic range quantization
# Reduces ~263MB → ~60MB with minimal accuracy loss
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Required for LSTM and complex ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

tflite_path = "scamshield_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"✅ TFLite saved — size: {size_mb:.2f} MB\n")

# ────────────────────────────────────────────────────
# STEP 6 — Verify TFLite with test inference
# ────────────────────────────────────────────────────
print("=" * 50)
print("Step 6: Verifying TFLite model...")

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput tensors:")
for inp in input_details:
    print(f"  [{inp['index']}] name : {inp['name']}")
    print(f"            shape: {inp['shape']}")
    print(f"            dtype: {inp['dtype']}")

print("\nOutput tensors:")
for out in output_details:
    print(f"  [{out['index']}] name : {out['name']}")
    print(f"            shape: {out['shape']}")
    print(f"            dtype: {out['dtype']}")

# Run test inference
print("\nRunning test inference...")
for inp in input_details:
    interpreter.set_tensor(
        inp['index'],
        np.zeros(inp['shape'], dtype=inp['dtype'])
    )

interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"Output logits: {output}")
print("✅ Test inference passed!\n")

print("=" * 50)
print("🎉 CONVERSION COMPLETE!")
print(f"📱 Copy to Flutter:")
print(f"   scamshield_model.tflite ({size_mb:.2f} MB)")
print(f"   → scamshield_app/assets/models/scamshield_model.tflite")
print("=" * 50)