# fixed_export.py
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertModel, DistilBertConfig
import tensorflow as tf
import os
import re
import glob
import shutil

CHECKPOINT_PATH  = r"C:\dev\python\project\Scamshield-AI Testing\best_model.pt"
ONNX_PATH        = "model_fixed.onnx"
TFLITE_OUT_DIR   = "tflite_out"
OUTPUT_PATH      = "model_fixed.tflite"

print("=" * 50)
print("🔧 Fixed export — no FlexErf, no FlexSplitV")
print("=" * 50)


# ── Step 1: Patch DistilBERT GELU → tanh approx ───────────────────────────────
print("\n📦 Patching DistilBERT GELU activation...")
config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
config.activation = "gelu_pytorch_tanh"   # eliminates FlexErf


# ── Step 2: Manual LSTM cell (no SplitV) ──────────────────────────────────────
class ManualLSTMCell(nn.Module):
    """Pure matmul/sigmoid/tanh — no SplitV in ONNX export."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wg = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        i = torch.sigmoid(self.Wi(combined))
        f = torch.sigmoid(self.Wf(combined))
        g = torch.tanh(self.Wg(combined))
        o = torch.sigmoid(self.Wo(combined))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ManualBiLSTM(nn.Module):
    """Full BiLSTM using only builtin-compatible ops."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_fwd = ManualLSTMCell(input_size, hidden_size)
        self.lstm_bwd = ManualLSTMCell(input_size, hidden_size)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        hf = torch.zeros(batch, self.hidden_size)
        cf = torch.zeros(batch, self.hidden_size)
        hb = torch.zeros(batch, self.hidden_size)
        cb = torch.zeros(batch, self.hidden_size)

        fwd_outputs = []
        for t in range(seq_len):
            hf, cf = self.lstm_fwd(x[:, t, :], hf, cf)
            fwd_outputs.append(hf.unsqueeze(1))

        bwd_outputs = []
        for t in reversed(range(seq_len)):
            hb, cb = self.lstm_bwd(x[:, t, :], hb, cb)
            bwd_outputs.insert(0, hb.unsqueeze(1))

        fwd = torch.cat(fwd_outputs, dim=1)
        bwd = torch.cat(bwd_outputs, dim=1)
        return torch.cat([fwd, bwd], dim=2)


# ── Step 3: Fixed model definition ────────────────────────────────────────────
# PatchedEmbeddings replaces DistilBERT's internal torch.arange(seq_len) with
# a registered buffer [0..127]. This is traced as a constant initializer in ONNX
# (not a dynamic Range node), eliminating the GATHER crash on Android.
# IMPORTANT: Do NOT run onnxsim after this — it OOMs trying to fold the embedding
# table. The register_buffer approach makes onnxsim unnecessary.
from transformers.models.distilbert.modeling_distilbert import Embeddings as DistilBertEmbeddings

_FIXED_SEQ_LEN = 128

class PatchedEmbeddings(DistilBertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer(
            'fixed_position_ids',
            torch.arange(_FIXED_SEQ_LEN, dtype=torch.long).unsqueeze(0),  # [1, 128]
            persistent=True,
        )

    def forward(self, input_ids, inputs_embeds=None):
        # fixed_position_ids is a buffer → ONNX initializer → TFLite Const
        # No dynamic Range/arange at runtime → no GATHER out-of-bounds on Android
        seq_len = input_ids.shape[1]
        position_ids    = self.fixed_position_ids[:, :seq_len]
        word_embeddings = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        pos_embeddings  = self.position_embeddings(position_ids)
        embeddings      = word_embeddings + pos_embeddings
        embeddings      = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


class HybridModelFixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert      = DistilBertModel(config)
        self.bert.embeddings = PatchedEmbeddings(config)
        self.bilstm    = ManualBiLSTM(input_size=768, hidden_size=256)
        self.dropout   = nn.Dropout(0.3)
        self.fc1       = nn.Linear(1024 + 6, 512)
        self.batchnorm = nn.BatchNorm1d(512)
        self.relu      = nn.ReLU()
        self.fc2       = nn.Linear(512, 2)

    def forward(self, input_ids, attention_mask, extra_features):
        outputs         = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output     = self.bilstm(sequence_output)
        mean_pool       = torch.mean(lstm_output, dim=1)
        max_pool, _     = torch.max(lstm_output,  dim=1)
        pooled          = torch.cat((mean_pool, max_pool), dim=1)
        combined        = torch.cat((pooled, extra_features), dim=1)
        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return self.fc2(x)


# ── Step 4: Load and map weights ──────────────────────────────────────────────
print("\n📦 Loading checkpoint weights...")

fixed_model    = HybridModelFixed()
fixed_model.eval()
original_state = torch.load(CHECKPOINT_PATH, map_location="cpu")
fixed_state    = dict(fixed_model.state_dict())

LSTM_OLD_KEYS = {
    "bilstm.weight_ih_l0", "bilstm.weight_hh_l0",
    "bilstm.bias_ih_l0",   "bilstm.bias_hh_l0",
    "bilstm.weight_ih_l0_reverse", "bilstm.weight_hh_l0_reverse",
    "bilstm.bias_ih_l0_reverse",   "bilstm.bias_hh_l0_reverse",
}

# Copy all non-LSTM keys that match directly
matched = 0
for key in list(fixed_state.keys()):
    if key in original_state and fixed_state[key].shape == original_state[key].shape:
        fixed_state[key] = original_state[key]
        matched += 1

# Split nn.LSTM combined matrices → per-gate nn.Linear weights
def map_lstm_gates(original_state, wih_key, whh_key, bih_key, bhh_key,
                   target_prefix, fixed_state):
    H   = original_state[wih_key].shape[0] // 4
    wih = original_state[wih_key]
    whh = original_state[whh_key]
    bih = original_state[bih_key]
    bhh = original_state[bhh_key]
    for idx, gate in enumerate(["Wi", "Wf", "Wg", "Wo"]):
        w = torch.cat([wih[idx*H:(idx+1)*H], whh[idx*H:(idx+1)*H]], dim=1)
        b = bih[idx*H:(idx+1)*H] + bhh[idx*H:(idx+1)*H]
        fixed_state[f"{target_prefix}.{gate}.weight"] = w
        fixed_state[f"{target_prefix}.{gate}.bias"]   = b
    print(f"  ✅ Mapped {wih_key.split('.')[1]} → {target_prefix} (H={H})")

map_lstm_gates(original_state,
    "bilstm.weight_ih_l0", "bilstm.weight_hh_l0",
    "bilstm.bias_ih_l0",   "bilstm.bias_hh_l0",
    "bilstm.lstm_fwd", fixed_state)

map_lstm_gates(original_state,
    "bilstm.weight_ih_l0_reverse", "bilstm.weight_hh_l0_reverse",
    "bilstm.bias_ih_l0_reverse",   "bilstm.bias_hh_l0_reverse",
    "bilstm.lstm_bwd", fixed_state)

# Sanity check
model_keys  = set(fixed_model.state_dict().keys())
loaded_keys = set(fixed_state.keys())
missing     = model_keys  - loaded_keys
unexpected  = loaded_keys - model_keys
if missing or unexpected:
    if missing:    print(f"❌ Missing    : {missing}")
    if unexpected: print(f"❌ Unexpected : {unexpected}")
    raise RuntimeError("state_dict key mismatch — fix before continuing")

fixed_model.load_state_dict(fixed_state)
print(f"\n✅ Weights loaded: {matched} direct + 16 LSTM gate tensors mapped")

# Copy pre-trained weights into PatchedEmbeddings (fresh instance after swap)
emb = fixed_model.bert.embeddings
emb.word_embeddings.weight.data     = fixed_state["bert.embeddings.word_embeddings.weight"]
emb.position_embeddings.weight.data = fixed_state["bert.embeddings.position_embeddings.weight"]
emb.LayerNorm.weight.data           = fixed_state["bert.embeddings.LayerNorm.weight"]
emb.LayerNorm.bias.data             = fixed_state["bert.embeddings.LayerNorm.bias"]
print("✅ PatchedEmbeddings weights restored from checkpoint")




# ── Step 5: Export to ONNX ────────────────────────────────────────────────────
print("\n📦 Exporting to ONNX...")

dummy_ids   = torch.zeros(1, 128, dtype=torch.long)
dummy_mask  = torch.ones(1,  128, dtype=torch.long)
dummy_feats = torch.zeros(1,  6,  dtype=torch.float32)

with torch.no_grad():
    torch.onnx.export(
        fixed_model,
        (dummy_ids, dummy_mask, dummy_feats),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask", "extra_features"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch"},
            "attention_mask": {0: "batch"},
            "extra_features": {0: "batch"},
            "logits":         {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
    )
print(f"✅ ONNX saved: {ONNX_PATH}")


# ── Step 5b: Full static-shape ONNX patch ────────────────────────────────────
# onnxsim OOMs on this model. Instead we use onnxruntime to run a forward pass
# and observe ALL intermediate tensor shapes, then patch every dynamic op:
#   - Reshape nodes  with dynamic shape inputs  → hardcoded static shapes
#   - Expand nodes   with dynamic shape inputs  → hardcoded static shapes
#   - Unsqueeze nodes (opset 13 takes axes as input) → already constant
# This fully eliminates the 128→0 and 0,0 dimension crashes in TFLite.

import onnx as onnx_lib
from onnx import numpy_helper, helper, TensorProto
import numpy as np
import onnxruntime as ort
import os

ONNX_SIM_PATH = "model_patched.onnx"
print("\n📦 Patching ONNX — hardcoding all dynamic shape tensors...")

model_proto = onnx_lib.load(ONNX_PATH)
graph       = model_proto.graph
init_names  = {init.name for init in graph.initializer}

# ── Strategy: use onnxruntime to get ALL intermediate tensor shapes ───────────
# Then annotate the ONNX graph's value_info with static shapes so onnx2tf
# can propagate them through — this is more robust than patching individual nodes.

print("  Running onnxruntime to collect all intermediate shapes...")

# First expose ALL intermediate tensors as outputs
existing_output_names = {o.name for o in graph.output}
all_intermediate = [vi.name for vi in graph.value_info
                    if vi.name not in existing_output_names]

extra_outputs = []
for name in all_intermediate:
    extra_outputs.append(
        helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
    )
graph.output.extend(extra_outputs)
onnx_lib.save(model_proto, "model_debug_shapes.onnx")

try:
    sess = ort.InferenceSession("model_debug_shapes.onnx",
                                 providers=["CPUExecutionProvider"])
    dummy_input = {
        "input_ids":      np.zeros((1, 128), dtype=np.int64),
        "attention_mask": np.ones ((1, 128), dtype=np.int64),
        "extra_features": np.zeros((1,   6), dtype=np.float32),
    }
    results      = sess.run(None, dummy_input)
    output_names = [o.name for o in sess.get_outputs()]
    tensor_shapes = {name: results[i].shape
                     for i, name in enumerate(output_names)
                     if i < len(results)}
    print(f"  Collected shapes for {len(tensor_shapes)} intermediate tensors")
except Exception as e:
    print(f"  ⚠️  onnxruntime full-graph run failed: {e}")
    tensor_shapes = {}
finally:
    for eo in extra_outputs:
        graph.output.remove(eo)
    if os.path.exists("model_debug_shapes.onnx"):
        os.remove("model_debug_shapes.onnx")

# ── Patch dynamic shape inputs for Reshape/Expand/Tile nodes ─────────────────
# Also collect shape-building tensors (Concat of scalar dims → shape vector)
TARGET_OPS = {"Reshape", "Expand", "Tile"}
dynamic_shape_inputs = {}
for node in graph.node:
    if node.op_type in TARGET_OPS and len(node.input) >= 2:
        shape_input = node.input[1]
        if shape_input not in init_names:
            dynamic_shape_inputs[node.output[0]] = (node, shape_input)

# Run a second targeted pass just for shape tensors (int64 outputs)
shape_tensor_names = list({sname for _, sname in dynamic_shape_inputs.values()})
extra_outputs2 = []
existing2 = {o.name for o in graph.output}
for sname in shape_tensor_names:
    if sname not in existing2:
        extra_outputs2.append(
            helper.make_tensor_value_info(sname, TensorProto.INT64, None)
        )
graph.output.extend(extra_outputs2)
onnx_lib.save(model_proto, "model_debug_shapes2.onnx")

shape_values = {}
try:
    sess2 = ort.InferenceSession("model_debug_shapes2.onnx",
                                  providers=["CPUExecutionProvider"])
    results2      = sess2.run(None, dummy_input)
    output_names2 = [o.name for o in sess2.get_outputs()]
    for sname in shape_tensor_names:
        if sname in output_names2:
            idx = output_names2.index(sname)
            shape_values[sname] = results2[idx]
            print(f"  Shape tensor {sname} = {results2[idx].tolist()}")
except Exception as e:
    print(f"  ⚠️  Shape tensor pass failed: {e}")
finally:
    for eo in extra_outputs2:
        graph.output.remove(eo)
    if os.path.exists("model_debug_shapes2.onnx"):
        os.remove("model_debug_shapes2.onnx")

patched = 0
for out, (node, sname) in dynamic_shape_inputs.items():
    if sname in shape_values:
        arr        = shape_values[sname].astype(np.int64)
        const_name = f"_static_{sname}"
        if const_name not in init_names:
            graph.initializer.append(numpy_helper.from_array(arr, name=const_name))
            init_names.add(const_name)
        node.input[1] = const_name
        print(f"  ✅ [{node.op_type}] {sname} → {arr.tolist()}")
        patched += 1
    else:
        print(f"  ⚠️  Could not resolve shape for {sname}")

# ── Update value_info with observed static shapes ─────────────────────────────
# This tells onnx2tf the concrete shape of every intermediate tensor,
# so it can allocate static buffers instead of dynamic ones.
vi_map   = {vi.name: vi for vi in graph.value_info}
updated  = 0
for name, shape in tensor_shapes.items():
    if name in vi_map:
        vi = vi_map[name]
        tp = vi.type.tensor_type
        tp.shape.ClearField("dim")
        for dim_size in shape:
            dim = tp.shape.dim.add()
            dim.dim_value = int(dim_size)
        updated += 1

print(f"  Updated {updated} value_info shapes with static dimensions")

# ── Remove leftover Range nodes ───────────────────────────────────────────────
range_nodes = [n for n in graph.node if n.op_type == "Range"]
for rnode in range_nodes:
    print(f"  Removing Range node: {rnode.output[0]}")
    graph.node.remove(rnode)

onnx_lib.save(model_proto, ONNX_SIM_PATH)
print(f"\n  Patched {patched} shape inputs, updated {updated} value_info entries")
print(f"✅ Fully static ONNX saved: {ONNX_SIM_PATH}")


# ── Step 6+7: ONNX → TFLite directly via onnx2tf ─────────────────────────────
# Uses simplified ONNX (constant-folded) to avoid dynamic RESHAPE at runtime.
# -ois locks input shapes statically → fixes GATHER crash on Android
# NOTE: Do NOT use -b 1 here — it breaks LSTM matmul (384 vs 1024 dim error)
print("\n📦 Converting ONNX → TFLite (direct via onnx2tf -eatfp16, static shapes)...")

# Wipe output dir so no stale files from previous runs contaminate selection
if os.path.exists(TFLITE_OUT_DIR):
    shutil.rmtree(TFLITE_OUT_DIR)
os.makedirs(TFLITE_OUT_DIR)

ret = os.system(
    f"onnx2tf -i {ONNX_SIM_PATH} -o {TFLITE_OUT_DIR} "
    f"--non_verbose "
    f"-nuo "
    f"-eatfp16 "
    f"-coion "
    f"-ois input_ids:1,128 attention_mask:1,128 extra_features:1,6"
)
if ret != 0:
    raise RuntimeError(f"onnx2tf failed with exit code {ret}")

# onnx2tf writes one or more .tflite files into the output folder
tflite_files = glob.glob(os.path.join(TFLITE_OUT_DIR, "*.tflite"))
if not tflite_files:
    raise RuntimeError(
        f"No .tflite found in {TFLITE_OUT_DIR} after onnx2tf. "
        "Check the onnx2tf output above for errors."
    )

# Select the simplified float16 file specifically — avoids picking stale
# model_fixed_float16.tflite from a previous run of the original ONNX.
# e.g. ONNX_SIM_PATH=model_simplified.onnx → stem=model_simplified
sim_stem   = os.path.splitext(os.path.basename(ONNX_SIM_PATH))[0]
fp16_sim   = [f for f in tflite_files
              if sim_stem in os.path.basename(f) and "float16" in f.lower()]
fp32_sim   = [f for f in tflite_files
              if sim_stem in os.path.basename(f) and "float32" in f.lower()]
any_sim    = [f for f in tflite_files if sim_stem in os.path.basename(f)]
any_fp16   = [f for f in tflite_files if "float16" in f.lower()]

# Priority: simplified-fp16 > simplified-fp32 > simplified-any > any-fp16 > first
if fp16_sim:
    chosen = fp16_sim[0]
elif fp32_sim:
    chosen = fp32_sim[0]
elif any_sim:
    chosen = any_sim[0]
elif any_fp16:
    chosen = any_fp16[0]
else:
    chosen = tflite_files[0]

shutil.copy(chosen, OUTPUT_PATH)
size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"✅ TFLite saved: {OUTPUT_PATH} ({size_mb:.2f} MB)")
print(f"   (source: {chosen})")
if len(tflite_files) > 1:
    print(f"   Other variants: {[f for f in tflite_files if f != chosen]}")


# ── Step 8: Check Flex ops ────────────────────────────────────────────────────
print("\n🔍 Checking for Flex ops...")
with open(OUTPUT_PATH, "rb") as f:
    raw = f.read()
flex_ops = sorted(set(re.findall(b"Flex[A-Za-z]+", raw)))
if flex_ops:
    print(f"⚠️  Flex ops still present: {[op.decode() for op in flex_ops]}")
else:
    print("✅ Zero Flex ops — fully builtin compatible!")


# ── Step 9: Verify interpreter ────────────────────────────────────────────────
print("\n🔍 Verifying interpreter...")
try:
    interp = tf.lite.Interpreter(model_path=OUTPUT_PATH)

    # Must resize BEFORE allocate_tensors so TFLite can resolve all
    # internal dynamic reshape nodes with concrete sizes.
    # NOTE: correct method is resize_tensor_input (not resize_input_tensor)
    input_details = interp.get_input_details()
    interp.resize_tensor_input(input_details[0]['index'], [1, 128])  # input_ids
    interp.resize_tensor_input(input_details[1]['index'], [1, 128])  # attention_mask
    interp.resize_tensor_input(input_details[2]['index'], [1, 6])    # extra_features

    interp.allocate_tensors()  # allocate AFTER resize
    print("✅ Interpreter loaded OK")
    for t in interp.get_input_details():
        print(f"   Input : {t['name']} | {t['shape']} | {t['dtype']}")
    for t in interp.get_output_details():
        print(f"   Output: {t['name']} | {t['shape']} | {t['dtype']}")

    # ── Step 10: Smoke-test inference ─────────────────────────────────────────
    print("\n🔍 Running smoke-test inference...")
    output_details = interp.get_output_details()

    interp.set_tensor(input_details[0]['index'], np.zeros((1, 128), dtype=np.int64))   # input_ids
    interp.set_tensor(input_details[1]['index'], np.ones ((1, 128), dtype=np.int64))   # attention_mask
    interp.set_tensor(input_details[2]['index'], np.zeros((1,   6), dtype=np.float32)) # extra_features

    interp.invoke()
    logits = interp.get_tensor(output_details[0]['index'])
    print(f"✅ Smoke-test passed — Raw logits: {logits}")

    # Softmax → probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs      = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    print(f"   Fake prob: {probs[0][0]:.4f} | Real prob: {probs[0][1]:.4f}")

except Exception as e:
    print(f"❌ Verify/inference failed: {e}")
    raise


print("\n" + "=" * 50)
print("🎉 Export complete — model_fixed.tflite ready for Android")
print("=" * 50)
print("\nNext steps:")
print("  1. Copy model_fixed.tflite → assets/models/")
print("  2. Hot restart and check logcat:")
print("     - Token ID range: min >= 0, max <= 30521")
print("     - Raw logits printed")
print("     - No GATHER crash")
print("  3. Once confirmed → remove force-delete cache + token range debug print")
print("  4. flutter clean && gradlew clean && flutter run")