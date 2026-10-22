# check_tflite_inputs.py
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(
    model_path="scamshield_model.tflite"
)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input tensors:")
for inp in input_details:
    print(f"  [{inp['index']}] {inp['name']}")
    print(f"        shape = {inp['shape']}")
    print(f"        dtype = {inp['dtype']}")

print("\nOutput tensors:")
for out in output_details:
    print(f"  [{out['index']}] {out['name']}")
    print(f"        shape = {out['shape']}")
    print(f"        dtype = {out['dtype']}")