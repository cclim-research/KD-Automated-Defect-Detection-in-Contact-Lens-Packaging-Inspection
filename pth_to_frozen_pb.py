import os
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import onnx
from onnxsim import simplify
import tensorflow as tf
from onnx_tf.backend import prepare
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# =====================================================================
# PATH CONFIGURATIONS (Update these paths for MobileNetV3 vs TinyViT)
# =====================================================================
# 1. Your PyTorch distilled student trained checkpoint file
PTH_INPUT_PATH = r"C:\FYP Student_Boet Puai Ying 221151481\onnx_exports\student_bs64_lr0.001.pt"

# 2. Output targets
OUTPUT_SIMPLIFIED_ONNX = "student_tinyvit_static_opset12.onnx"
OUTPUT_FROZEN_PB = "frozen_student_tinyvit.pb"

# 3. Intermediate temporary directory required by the onnx-tf translator
TEMP_SAVED_MODEL_DIR = "temp_tf_saved_model_dir"

NUM_CLASSES = 7

# =====================================================================
# ARCHITECTURE DEFINITION (Must match your training script exactly)
# =====================================================================
class StudentWithFeat(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.fc1 = base.classifier[0]
        self.act1 = nn.Hardswish()
        self.drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(base.classifier[0].out_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.drop(self.act1(self.fc1(x)))
        out = self.fc2(feat)
        return out

def main():
    print("=====================================================================")
    print("🚀 PIPELINE INITIATED: Direct PyTorch (.pth) to Frozen Graph (.pb)")
    print("=====================================================================")

    # -----------------------------------------------------------------
    # STAGE 1: Reconstruct Architecture and Load Weights
    # -----------------------------------------------------------------
    if not os.path.exists(PTH_INPUT_PATH):
        print(f"❌ ERROR: Cannot locate PyTorch weights file at: {PTH_INPUT_PATH}")
        return

    print("--> Instantiating model structure and executing state_dict mapping...")
    model = StudentWithFeat(num_classes=NUM_CLASSES)
    state_dict = torch.load(PTH_INPUT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    
    # CRITICAL STEP FOR DEPLOYMENT: Lock weights and operational graph layers
    model.eval()
    print("✔ Success: PyTorch weights loaded securely in Evaluation mode.")

    # -----------------------------------------------------------------
    # STAGE 2: Trace and Export to Static Opset 11 ONNX
    # -----------------------------------------------------------------
    # Mandatory industrial constraint: Static tensor size instead of dynamic_axes
    dummy_input = torch.randn(1, 3, 224, 224)
    temp_raw_onnx = "temp_raw_unprocessed.onnx"

    print("--> Tracing graph via static 1x3x224x224 input tensor (OPSET 11)...")
    torch.onnx.export(
        model,
        dummy_input,
        temp_raw_onnx,
        export_params=True,
        opset_version=12,  # Lowered to 11 to prevent Unsqueeze version errors in onnx-tf
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    print("✔ Success: Primitive ONNX serialization complete.")

    # -----------------------------------------------------------------
    # STAGE 3: Optimize Graph via ONNX Simplifier
    # -----------------------------------------------------------------
    print("--> Optimizing model topology with onnx-simplifier...")
    raw_model_loaded = onnx.load(temp_raw_onnx)
    simplified_model, validation_check = simplify(raw_model_loaded)
    
    if not validation_check:
        print("❌ ERROR: ONNX optimization graph validation failed!")
        return
        
    onnx.save(simplified_model, OUTPUT_SIMPLIFIED_ONNX)
    
    # Remove the unoptimized temporary file from disk
    if os.path.exists(temp_raw_onnx):
        os.remove(temp_raw_unprocessed.onnx if 'temp_raw_unprocessed.onnx' in locals() else temp_raw_onnx)
        
    print(f"✔ Success: Clean static model saved at -> {OUTPUT_SIMPLIFIED_ONNX}")

    # -----------------------------------------------------------------
    # STAGE 4: Translate ONNX Representation to TensorFlow Graph
    # -----------------------------------------------------------------
    print("--> Translating graph operators into TensorFlow representation...")
    onnx_optimized_graph = onnx.load(OUTPUT_SIMPLIFIED_ONNX)
    tf_rep = prepare(onnx_optimized_graph)

    print(f"--> Exporting intermediate SavedModel folder asset...")
    if os.path.exists(TEMP_SAVED_MODEL_DIR):
        shutil.rmtree(TEMP_SAVED_MODEL_DIR)
    tf_rep.export_graph(TEMP_SAVED_MODEL_DIR)

    # -----------------------------------------------------------------
    # STAGE 5: Weld Variables to Constants (Graph Freezing)
    # -----------------------------------------------------------------
    print("--> Extracting signature paths and freezing tensor variables...")
    imported_tf_model = tf.saved_model.load(TEMP_SAVED_MODEL_DIR)
    concrete_func = imported_tf_model.signatures["serving_default"]
    
    # Inlines the weight constants directly into the graph network layout nodes
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_func.graph.as_graph_def()

    # -----------------------------------------------------------------
    # STAGE 6: Write Standalone Monolithic Binary File
    # -----------------------------------------------------------------
    print(f"--> Serializing final production asset to disk: {OUTPUT_FROZEN_PB} ...")
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=".",
        name=OUTPUT_FROZEN_PB,
        as_text=False  # Crucial: Save as binary stream to save footprint size
    )

    # Clean up intermediate junk folders to keep workspace tidy
    if os.path.exists(TEMP_SAVED_MODEL_DIR):
        shutil.rmtree(TEMP_SAVED_MODEL_DIR)

    print("\n" + "="*65)
    print("🎉 CONVERSIONS COMPLETED SUCCESSFULLY!")
    print(f"📦 Optimized ONNX (Backup): {os.path.abspath(OUTPUT_SIMPLIFIED_ONNX)}")
    print(f"🎯 Frozen PB Graph (ATS Tool Target): {os.path.abspath(OUTPUT_FROZEN_PB)}")
    print("="*65)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ PIPELINE HALTED DUE TO ENCOUNTERED EXCEPTION: {e}")