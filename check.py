import tensorflow as tf
import numpy as np

# Load and test the INT8 model
interpreter = tf.lite.Interpreter(
    model_path="model_int8.tflite",
    experimental_delegates=[tf.lite.experimental.load_delegate('tensorflowlite_flex.dll')]  # Windows
    # Use 'libtensorflowlite_flex.so' on Linux/Mac
)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== Input details ===")
for d in input_details:
    print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

print("\n=== Output details ===")
for d in output_details:
    print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

# Run a test inference
interpreter.set_tensor(input_details[0]['index'], np.zeros((1, 128), dtype=np.int64))
interpreter.set_tensor(input_details[1]['index'], np.ones((1, 128),  dtype=np.int64))
interpreter.set_tensor(input_details[2]['index'], np.zeros((1, 6),   dtype=np.float32))

interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
print(f"\n=== Test output ===")
print(f"Raw logits : {output}")
print(f"Prediction : {'SCAM' if np.argmax(output) == 1 else 'LEGIT'} (confidence: {tf.nn.softmax(output)[0][np.argmax(output)]:.2%})")