# rebuild_with_flex.py
import tensorflow as tf
import os

SAVED_MODEL_PATH = r"C:\dev\python\project\Scamshield-AI Testing\scamshield_tf_model"
OUTPUT_F16_PATH  = "model_f16_flex.tflite"
OUTPUT_INT8_PATH = "model_int8_flex.tflite"

print("=" * 50)
print("🔧 Rebuilding TFLite model with SELECT_TF_OPS...")
print("=" * 50)

def base_converter(saved_model_path: str) -> tf.lite.TFLiteConverter:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

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
    conv_f16 = base_converter(SAVED_MODEL_PATH)
    conv_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_f16.target_spec.supported_types = [tf.float16]

    model_f16 = conv_f16.convert()
    with open(OUTPUT_F16_PATH, "wb") as f:
        f.write(model_f16)

    size_mb = os.path.getsize(OUTPUT_F16_PATH) / (1024 * 1024)
    print(f"✅ Saved: {OUTPUT_F16_PATH}  ({size_mb:.2f} MB)")

except Exception as e:
    print(f"❌ F16 failed: {e}")


# ── INT8 ──────────────────────────────────────────
print("\n📦 Converting → INT8...")
try:
    conv_int8 = base_converter(SAVED_MODEL_PATH)
    conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]

    model_int8 = conv_int8.convert()
    with open(OUTPUT_INT8_PATH, "wb") as f:
        f.write(model_int8)

    size_mb = os.path.getsize(OUTPUT_INT8_PATH) / (1024 * 1024)
    print(f"✅ Saved: {OUTPUT_INT8_PATH}  ({size_mb:.2f} MB)")

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
        print(f"✅ {path} — loads OK")
        for t in interp.get_input_details():
            print(f"   Input : {t['name']} | {t['shape']} | {t['dtype']}")
        for t in interp.get_output_details():
            print(f"   Output: {t['name']} | {t['shape']} | {t['dtype']}")
    except Exception as e:
        print(f"❌ {path} failed: {e}")

print("\n" + "=" * 50)
print("Done. Use model_f16_flex.tflite in Flutter.")
print("=" * 50)