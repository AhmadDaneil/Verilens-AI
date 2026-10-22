# check_signatures.py
import tensorflow as tf

model = tf.saved_model.load("scamshield_tf_model")
print("Signatures:", list(model.signatures.keys()))
print("Attributes:", [x for x in dir(model) if not x.startswith("_")])