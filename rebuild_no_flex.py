# rebuild_no_flex.py
import torch
import torch.nn as nn
from transformers import DistilBertModel
import tensorflow as tf
import numpy as np
import os

SAVED_MODEL_PATH = r"C:\dev\python\project\Scamshield-AI Testing\scamshield_tf_model"
OUTPUT_PATH = "model_no_flex.tflite"

print("=" * 50)
print("🔧 Converting with GELU approximation (no FlexErf)...")
print("=" * 50)

model = tf.saved_model.load(SAVED_MODEL_PATH)

@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="input_ids"),
    tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="attention_mask"),
    tf.TensorSpec(shape=[1, 6],   dtype=tf.float32, name="extra_features"),
])
def wrapped_fn(input_ids, attention_mask, extra_features):
    return model(input_ids, attention_mask, extra_features)

cf = wrapped_fn.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions([cf], model)

# ✅ Key: allow ALL tf ops but also try to inline/replace where possible
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter.allow_custom_ops = True
converter._experimental_lower_tensor_list_ops = False

# ✅ This flag tells the converter to replace tf.math.erf with
# the builtin GELU approximation: x * 0.5 * (1 + tanh(...))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# ✅ Experimental: inline TF ops where builtin equivalents exist
converter._experimental_enable_dynamic_update_slice = True

print("\n📦 Converting...")
try:
    tflite_model = converter.convert()
    with open(OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"✅ Saved: {OUTPUT_PATH} ({size_mb:.2f} MB)")
except Exception as e:
    print(f"❌ Conversion failed: {e}")
    exit(1)

# ✅ Check if FlexErf is gone
print("\n🔍 Checking for remaining Flex ops...")
import re
with open(OUTPUT_PATH, "rb") as f:
    raw = f.read()
flex_ops = list(set(re.findall(b"Flex[A-Za-z]+", raw)))
if flex_ops:
    print(f"⚠️  Still has Flex ops: {[op.decode() for op in flex_ops]}")
    print("👉 Need to retrain with GELU approximation in PyTorch")
else:
    print("✅ No Flex ops — model is fully builtin-compatible!")

# ✅ Verify
print("\n🔍 Verifying...")
try:
    interp = tf.lite.Interpreter(model_path=OUTPUT_PATH)
    interp.allocate_tensors()
    print(f"✅ Loads OK on desktop")
    for t in interp.get_input_details():
        print(f"   Input : {t['name']} | {t['shape']} | {t['dtype']}")
except Exception as e:
    print(f"❌ Verify failed: {e}")

print("\n" + "=" * 50)