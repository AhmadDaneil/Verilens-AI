import tensorflow as tf
import numpy as np

print("Loading SavedModel...")
saved_model = tf.saved_model.load("tf_model")

@tf.function(input_signature=[
    tf.TensorSpec(shape=(1, 128), dtype=tf.int64,   name="inputs_0"),
    tf.TensorSpec(shape=(1, 128), dtype=tf.int64,   name="inputs_1"),
    tf.TensorSpec(shape=(1, 6),   dtype=tf.float32, name="inputs_2"),
])
def serving_fn(inputs_0, inputs_1, inputs_2):
    return saved_model(inputs_0, inputs_1, inputs_2)

concrete_func = serving_fn.get_concrete_function()
print("✅ Concrete function obtained!")

# =========================================================
# ✅ FIX — rep_data must yield a flat LIST of numpy arrays,
#    NOT a list-of-dicts (that caused the 'dict has no .shape' error)
# =========================================================
def rep_data():
    for _ in range(100):
        yield [
            np.zeros((1, 128), dtype=np.int64),    # inputs_0
            np.ones((1, 128),  dtype=np.int64),    # inputs_1
            np.zeros((1, 6),   dtype=np.float32),  # inputs_2
        ]

# =========================================================
# ATTEMPT 1 — Dynamic range quantization (float16)
# Targets ~67 MB (4x smaller than 270 MB)
# =========================================================
print("\n⚙️  Attempting float16 quantization...")
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func], saved_model
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]   # float16 weights
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False

try:
    tflite_f16 = converter.convert()
    with open("model_f16.tflite", "wb") as f:
        f.write(tflite_f16)
    print(f"🎉 Float16 model saved! ({len(tflite_f16) / 1024:.1f} KB  /  {len(tflite_f16) / 1024 / 1024:.1f} MB)")
except Exception as e:
    print(f"❌ Float16 failed: {e}")

# =========================================================
# ATTEMPT 2 — Full INT8 quantization with corrected rep_data
# Targets ~67 MB with faster inference
# =========================================================
print("\n⚙️  Attempting INT8 quantization...")
converter2 = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func], saved_model
)
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
converter2.representative_dataset = rep_data   # ✅ fixed format
converter2.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter2._experimental_lower_tensor_list_ops = False

try:
    tflite_int8 = converter2.convert()
    with open("model_int8.tflite", "wb") as f:
        f.write(tflite_int8)
    print(f"🎉 INT8 model saved! ({len(tflite_int8) / 1024:.1f} KB  /  {len(tflite_int8) / 1024 / 1024:.1f} MB)")
except Exception as e:
    print(f"❌ INT8 failed: {e}")