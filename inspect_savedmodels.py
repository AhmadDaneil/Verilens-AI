# inspect_savedmodels.py
import tensorflow as tf

paths = [
    r"C:\dev\python\project\Scamshield-AI Testing\scamshield_tf_model",
    r"C:\dev\python\project\Scamshield-AI Testing\tf_model",
]

for path in paths:
    print("=" * 50)
    print(f"📁 {path}")
    try:
        model = tf.saved_model.load(path)
        sigs = list(model.signatures.keys())
        print(f"  Signatures : {sigs}")
        print(f"  Callable   : {callable(model)}")
        print(f"  Has __call__: {hasattr(model, '__call__')}")

        # Try to find concrete functions
        if hasattr(model, '__call__'):
            cf = model.__call__.concrete_functions
            print(f"  Concrete fns: {len(cf)}")
            if cf:
                fn = cf[0]
                print(f"  Inputs:")
                for k, v in fn.structured_input_signature[1].items():
                    print(f"    - {k}: {v}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("=" * 50)