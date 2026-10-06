import os
import shutil
import tensorflow as tf
from onnx_tf.backend import prepare
import onnx
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
# Input path of your exported ONNX model
ONNX_INPUT_PATH = r"C:\FYP Student_Boet Puai Ying 221151481\onnx_exports\student_tinyvit_opset14.onnx"

# Temporary directory needed by onnx-tf to save intermediate weights
TEMP_SAVED_MODEL_DIR = "temp_tf_saved_model"

# Ultimate single standalone binary file required for deployment
PB_OUTPUT_FILE = "frozen_student_tinyvit_model.pb"

def main():
    print("==================================================")
    print("🚀 STARTING: ONNX to Standalone Frozen .pb Pipeline")
    print("==================================================")

    # -----------------------------------------------------------------
    # STEP 1: Load ONNX Model and Parse Graph
    # -----------------------------------------------------------------
    if not os.path.exists(ONNX_INPUT_PATH):
        print(f"❌ ERROR: Cannot find input file at '{ONNX_INPUT_PATH}'")
        return

    print(f"--> Loading ONNX model from: {ONNX_INPUT_PATH} ...")
    onnx_model = onnx.load(ONNX_INPUT_PATH)
    
    # Optional check to make sure the ONNX file isn't corrupted before starting
    onnx.checker.check_model(onnx_model)
    print("✔ SUCCESS: ONNX structure verified as structurally sound.")

    # -----------------------------------------------------------------
    # STEP 2: Convert ONNX representation to intermediate TensorFlow format
    # -----------------------------------------------------------------
    print("--> Translating ONNX computational operators into TensorFlow graph...")
    tf_rep = prepare(onnx_model)

    print(f"--> Exporting intermediate SavedModel folder to: ./{TEMP_SAVED_MODEL_DIR}")
    # Clears old folder if you rerun the script to prevent appending bugs
    if os.path.exists(TEMP_SAVED_MODEL_DIR):
        shutil.rmtree(TEMP_SAVED_MODEL_DIR)
        
    tf_rep.export_graph(TEMP_SAVED_MODEL_DIR)
    print("✔ SUCCESS: Intermediate SavedModel assets generated.")

    # -----------------------------------------------------------------
    # STEP 3: Load SavedModel structure back via low-level APIs
    # -----------------------------------------------------------------
    print("--> Reading low-level SavedModel graph signature block...")
    imported = tf.saved_model.load(TEMP_SAVED_MODEL_DIR)
    concrete_func = imported.signatures["serving_default"]

    # -----------------------------------------------------------------
    # STEP 4: Weld variables into constants (Graph Freezing)
    # -----------------------------------------------------------------
    print("--> Freezing architecture graph and mathematical weight variables together...")
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_func.graph.as_graph_def()

    # -----------------------------------------------------------------
    # STEP 5: Serialize output out to a standalone binary file
    # -----------------------------------------------------------------
    print(f"--> Serializing unified architecture and weights to: {PB_OUTPUT_FILE}")
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=".",
        name=PB_OUTPUT_FILE,
        as_text=False  # Crucial: Forces binary encoding instead of plain text format
    )

    # -----------------------------------------------------------------
    # CLEANUP: Delete temporary intermediate directory asset
    # -----------------------------------------------------------------
    print("--> Cleaning up intermediate assets...")
    if os.path.exists(TEMP_SAVED_MODEL_DIR):
        shutil.rmtree(TEMP_SAVED_MODEL_DIR)

    print("\n" + "="*50)
    print("🎉 SUCCESS! Standalone legacy .pb file generated from ONNX.")
    print(f"Target Output: {os.path.abspath(PB_OUTPUT_FILE)}")
    print("="*50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL PIPELINE CRASH: {e}")