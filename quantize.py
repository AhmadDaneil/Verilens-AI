import tensorflow as tf
import numpy as np
import os

print("=" * 50)
print("Loading SavedModel...")

# Check if saved model exists
if not os.path.exists('scamshield_tf_model'):
    print("❌ scamshield_tf_model folder not found!")
    print("Run convert_to_tflite_fixed.py first")
    exit()

saved_model = tf.saved_model.load('scamshield_tf_model')
print("✅ SavedModel loaded")

# Check available signatures
sigs = list(saved_model.signatures.keys())
print(f"Available signatures: {sigs}")

# ✅ Build concrete function
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="input_ids"),
    tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="attention_mask"),
    tf.TensorSpec(shape=[1, 6],   dtype=tf.float32, name="extra_features"),
])
def serving_fn(input_ids, attention_mask, extra_features):
    return saved_model(input_ids, attention_mask, extra_features)

concrete_func = serving_fn.get_concrete_function()
print("✅ Concrete function created")

# ✅ Representative dataset
def representative_dataset():
    for _ in range(100):
        yield [
            np.zeros((1, 128), dtype=np.int64),
            np.ones((1, 128),  dtype=np.int64),
            np.zeros((1, 6),   dtype=np.float32),
        ]

print("=" * 50)
print("Converting with SELECT_TF_OPS + quantization...")

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# ✅ MUST include SELECT_TF_OPS to handle FlexErf
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,  # ← fixes FlexErf
]

converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()
print("✅ Conversion done")

output_path = 'scamshield_model_int8.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / 1024 / 1024
print(f"✅ Saved: {output_path} — {size_mb:.1f} MB")

# ✅ Verify
print("=" * 50)
print("Verifying...")
interpreter = tf.lite.Interpreter(model_path=output_path)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput tensors:")
for inp in input_details:
    print(f"  [{inp['index']}] {inp['name']}")
    print(f"           shape={inp['shape']} dtype={inp['dtype']}")

print("\nOutput tensors:")
for out in output_details:
    print(f"  [{out['index']}] {out['name']}")
    print(f"           shape={out['shape']} dtype={out['dtype']}")

print("\nRunning test inference...")
for inp in input_details:
    interpreter.set_tensor(
        inp['index'],
        np.zeros(inp['shape'], dtype=inp['dtype'])
    )

interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"✅ Output: {output}")
print("=" * 50)
print(f"Original : ~60.0 MB")
print(f"Quantized: {size_mb:.1f} MB")
print(f"Reduction: {(1 - size_mb / 60) * 100:.0f}%")
print("=" * 50)
print("📱 Next steps:")
print("  1. Copy scamshield_model_int8.tflite")
print("     → scamshield_app/assets/models/scamshield_model.tflite")
print("  2. flutter clean")
print("  3. flutter pub get")
print("  4. flutter run")