# fix_signature.py

import tensorflow as tf
import numpy as np
import os

print("Loading SavedModel...")
loaded = tf.saved_model.load("scamshield_tf_model")

# ────────────────────────────────────────────────────
# Find the correct callable in the SavedModel
# ────────────────────────────────────────────────────
print("Inspecting SavedModel...")

# Check what functions are available
print("Type:", type(loaded))

# onnx2tf saves the model with a specific function name
# Try common function names
fn = None
for attr_name in ["__call__", "serving_default", "call", "forward", "infer"]:
    if hasattr(loaded, attr_name):
        candidate = getattr(loaded, attr_name)
        print(f"Found: {attr_name} — type: {type(candidate)}")
        if callable(candidate):
            fn = candidate
            print(f"✅ Using function: {attr_name}")
            break

if fn is None:
    print("❌ No callable found, trying direct call...")
    fn = loaded

# ────────────────────────────────────────────────────
# Test the function with dummy inputs
# ────────────────────────────────────────────────────
print("\nTesting with dummy inputs...")

# FIXED: Changed tf.int32 to tf.int64 to match ONNX/PyTorch long tensors
dummy_ids  = tf.zeros([1, 128], dtype=tf.int64)
dummy_mask = tf.ones([1, 128],  dtype=tf.int64)
dummy_feat = tf.zeros([1, 6],   dtype=tf.float32)

try:
    # Try keyword arguments first
    result = fn(
        input_ids=dummy_ids,
        attention_mask=dummy_mask,
        extra_features=dummy_feat
    )
    print(f"✅ Keyword args work — output: {result}")
    use_kwargs = True

except Exception as e:
    print(f"Keyword args failed: {e}")
    try:
        # Try positional arguments
        result = fn(dummy_ids, dummy_mask, dummy_feat)
        print(f"✅ Positional args work — output: {result}")
        use_kwargs = False
    except Exception as e2:
        print(f"Positional args also failed: {e2}")
        # Try with a dict
        try:
            result = fn({
                "input_ids":      dummy_ids,
                "attention_mask": dummy_mask,
                "extra_features": dummy_feat,
            })
            print(f"✅ Dict input works — output: {result}")
            use_kwargs = "dict"
        except Exception as e3:
            print(f"Dict also failed: {e3}")
            raise Exception("Cannot find working call signature")

# ────────────────────────────────────────────────────
# Create serving function based on what works
# ────────────────────────────────────────────────────
print("\nCreating serving function...")

if use_kwargs == True:
    @tf.function(input_signature=[
        # FIXED: Changed tf.int32 to tf.int64
        tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="input_ids"),
        tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="attention_mask"),
        tf.TensorSpec(shape=[1, 6],   dtype=tf.float32, name="extra_features"),
    ])
    def serving_fn(input_ids, attention_mask, extra_features):
        return fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            extra_features=extra_features
        )

elif use_kwargs == False:
    @tf.function(input_signature=[
        # FIXED: Changed tf.int32 to tf.int64
        tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="input_ids"),
        tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="attention_mask"),
        tf.TensorSpec(shape=[1, 6],   dtype=tf.float32, name="extra_features"),
    ])
    def serving_fn(input_ids, attention_mask, extra_features):
        return fn(input_ids, attention_mask, extra_features)

else:
    @tf.function(input_signature=[
        # FIXED: Changed tf.int32 to tf.int64
        tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="input_ids"),
        tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="attention_mask"),
        tf.TensorSpec(shape=[1, 6],   dtype=tf.float32, name="extra_features"),
    ])
    def serving_fn(input_ids, attention_mask, extra_features):
        return fn({
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "extra_features": extra_features,
        })

print("✅ Serving function created\n")

# ────────────────────────────────────────────────────
# Convert directly from concrete function
# This avoids the SavedModel signature issue entirely
# ────────────────────────────────────────────────────
print("=" * 55)
print("Converting to TFLite from concrete function...")

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [serving_fn.get_concrete_function()],
    serving_fn
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
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
print(f"✅ TFLite saved: {tflite_path} — {size_mb:.2f} MB\n")

# ────────────────────────────────────────────────────
# Verify TFLite
# ────────────────────────────────────────────────────
print("=" * 55)
print("Verifying TFLite model...")

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput tensors:")
for inp in input_details:
    print(f"  [{inp['index']}] {inp['name']}")
    print(f"        shape = {inp['shape']}")
    print(f"        dtype = {inp['dtype']}")

print("\nOutput tensors:")
for out in output_details:
    print(f"  [{out['index']}] {out['name']}")
    print(f"        shape = {out['shape']}")
    print(f"        dtype = {out['dtype']}")

print("\nRunning test inference...")
for inp in input_details:
    interpreter.set_tensor(
        inp['index'],
        np.zeros(inp['shape'], dtype=inp['dtype'])
    )

interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"Output logits: {output}")
print("✅ TFLite inference successful!\n")

print("=" * 55)
print("🎉 CONVERSION COMPLETE!")
print(f"📱 File: {tflite_path} ({size_mb:.2f} MB)")
print("   Copy to:")
print("   scamshield_app/assets/models/scamshield_model.tflite")
print("=" * 55)