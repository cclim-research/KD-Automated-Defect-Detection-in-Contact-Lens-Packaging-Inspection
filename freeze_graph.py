import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

SAVED_MODEL_DIR = "saved_model"
FROZEN_OUT_FILE = "frozen_student_model.pb"

print("--> Loading low-level SavedModel structure...")
try:
    # 1. Load using core tf.saved_model instead of Keras
    imported = tf.saved_model.load(SAVED_MODEL_DIR)
    
    # 2. Extract the exact concrete serving function signature
    concrete_func = imported.signatures["serving_default"]
    
    print("--> Freezing graph architecture and weight variables together...")
    # 3. Convert variable tensors into embedded constants inside the graph structure
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_func.graph.as_graph_def()

    print(f"--> Writing standalone frozen graph to: {FROZEN_OUT_FILE}")
    # 4. Serialize and write the unified binary file out to the folder
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=".",
        name=FROZEN_OUT_FILE,
        as_text=False
    )

    print("\n" + "="*50)
    print("🎉 SUCCESS! Standalone legacy .pb file generated.")
    print(f"Target File: {FROZEN_OUT_FILE}")
    print("="*50)

except Exception as e:
    print(f"\n❌ ERROR during freezing pipeline: {e}")