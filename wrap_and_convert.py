# wrap_and_convert.py
import tensorflow as tf
import os

SAVED_MODEL_PATH = r"C:\dev\python\project\Scamshield-AI Testing\scamshield_tf_model"
OUTPUT_F16_PATH  = "model_f16_flex.tflite"
OUTPUT_INT8_PATH = "model_int8_flex.tflite"

print("=" * 50)
print("📦 Loading model...")
print("=" * 50)

model = tf.saved_model.load(SAVED_MODEL_PATH)

# ── Step 1: Wrap __call__ as tf.function ──────────
print("\n🔧 Wrapping model as tf.function...")

@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="input_ids"),
    tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="attention_mask"),
    tf.TensorSpec(shape=[1, 6],   dtype=tf.float32, name="extra_features"),
])
def wrapped_fn(input_ids, attention_mask, extra_features):
    return model(input_ids, attention_mask, extra_features)

# ── Step 2: Get concrete function ─────────────────
print("🔍 Getting concrete function...")
try:
    cf = wrapped_fn.get_concrete_function()
    print("✅ Concrete function OK")
    print(f"   Inputs : {[i.name for i in cf.inputs]}")
    print(f"   Outputs: {[o.name for o in cf.outputs]}")
except Exception as e:
    print(f"❌ Failed: {e}")
    print("\n🔁 Trying to inspect model call signature...")
    import inspect
    try:
        sig = inspect.signature(model.__call__)
        print(f"   model.__call__ params: {list(sig.parameters.keys())}")
    except:
        pass

    # ── Fallback: try different arg order ─────────
    print("\n🔁 Trying keyword args via dict output...")
    try:
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="input_ids"),
            tf.TensorSpec(shape=[1, 128], dtype=tf.int64,   name="attention_mask"),
            tf.TensorSpec(shape=[1, 6],   dtype=tf.float32, name="extra_features"),
        ])
        def wrapped_fn2(input_ids, attention_mask, extra_features):
            out = model.__call__(input_ids, attention_mask, extra_features)
            return out

        cf = wrapped_fn2.get_concrete_function()
        wrapped_fn = wrapped_fn2
        print("✅ Fallback concrete function OK")
    except Exception as e2:
        print(f"❌ Fallback also failed: {e2}")
        exit(1)


# ── Helper ─────────────────────────────────────────
def make_converter(cf):
    converter = tf.lite.TFLiteConverter.from_concrete_functions([cf], model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.allow_custom_ops = True
    converter._experimental_lower_tensor_list_ops = False
    return converter


# ── F16 ───────────────────────────────────────────
print("\n📦 Converting → F16...")
try:
    conv = make_converter(cf)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]

    tflite_model = conv.convert()
    with open(OUTPUT_F16_PATH, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(OUTPUT_F16_PATH) / (1024 * 1024)
    print(f"✅ Saved: {OUTPUT_F16_PATH} ({size_mb:.2f} MB)")

except Exception as e:
    print(f"❌ F16 failed: {e}")


# ── INT8 ──────────────────────────────────────────
print("\n📦 Converting → INT8...")
try:
    conv = make_converter(cf)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = conv.convert()
    with open(OUTPUT_INT8_PATH, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(OUTPUT_INT8_PATH) / (1024 * 1024)
    print(f"✅ Saved: {OUTPUT_INT8_PATH} ({size_mb:.2f} MB)")

except Exception as e:
    print(f"❌ INT8 failed: {e}")


# ── Verify ────────────────────────────────────────
print("\n🔍 Verifying outputs...")
for path in [OUTPUT_F16_PATH, OUTPUT_INT8_PATH]:
    if not os.path.exists(path):
        print(f"⚠️  Missing: {path}")
        continue
    try:
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        print(f"✅ {path} — OK")
        for t in interp.get_input_details():
            print(f"   Input : {t['name']} | {t['shape']} | {t['dtype']}")
        for t in interp.get_output_details():
            print(f"   Output: {t['name']} | {t['shape']} | {t['dtype']}")
    except Exception as e:
        print(f"❌ {path} verify failed: {e}")

print("\n" + "=" * 50)
print("Done.")
print("=" * 50)