# check_ops.py
import tensorflow as tf
import numpy as np

MODEL_PATH = "scamshield_model.tflite"

print("=" * 50)
print("🔍 Checking TFLite model ops...")
print("=" * 50)

# ── 1. Load & inspect ──────────────────────────────
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    print("✅ Interpreter created successfully")
except Exception as e:
    print(f"❌ Failed to create interpreter: {e}")
    exit(1)

# ── 2. Print all tensor details ────────────────────
print("\n📊 Input tensors:")
interpreter.allocate_tensors()
for t in interpreter.get_input_details():
    print(f"  - {t['name']} | shape: {t['shape']} | dtype: {t['dtype']}")

print("\n📊 Output tensors:")
for t in interpreter.get_output_details():
    print(f"  - {t['name']} | shape: {t['shape']} | dtype: {t['dtype']}")

# ── 3. Check for Flex / unsupported ops ───────────
print("\n🔬 Scanning for Flex ops (unsupported builtins)...")
try:
    import re
    with open(MODEL_PATH, "rb") as f:
        raw = f.read()

    # Flex ops are embedded as strings like "FlexErf", "FlexGelu", etc.
    flex_ops = re.findall(b"Flex[A-Za-z]+", raw)
    flex_ops = list(set(flex_ops))

    if flex_ops:
        print(f"⚠️  Found {len(flex_ops)} Flex op(s) — SELECT_TF_OPS required:")
        for op in flex_ops:
            print(f"   ❌ {op.decode()}")
    else:
        print("✅ No Flex ops found — model uses only builtin ops")

except Exception as e:
    print(f"⚠️  Could not scan raw bytes: {e}")

# ── 4. Try dummy inference ─────────────────────────
print("\n🧪 Attempting dummy inference...")
try:
    inputs = interpreter.get_input_details()
    for inp in inputs:
        dummy = np.zeros(inp['shape'], dtype=inp['dtype'])
        interpreter.set_tensor(inp['index'], dummy)

    interpreter.invoke()
    print("✅ Dummy inference passed — model runs fine as-is")

except Exception as e:
    print(f"❌ Inference failed: {e}")
    print("👉 This confirms SELECT_TF_OPS or model rebuild is needed")

print("\n" + "=" * 50)
print("Done.")
print("=" * 50)