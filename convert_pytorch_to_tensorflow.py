import os

import torch
import torch.nn as nn

import torchvision
import torchvision.models as models

import onnx

# =========================================================
# PATH
# =========================================================

MODEL_PATH = r"C:\FYP Student_Boet Puai Ying 221151481\onnx_exports\student_bs32_lr0.001.pt"

ONNX_PATH = "student_model.onnx"

NUM_CLASSES = 7

# =========================================================
# STUDENT MODEL
# =========================================================

class StudentWithFeat(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()

        base = models.mobilenet_v3_small(weights=None)

        self.features = base.features

        self.avgpool = base.avgpool

        self.fc1 = base.classifier[0]

        self.act1 = nn.Hardswish()

        self.drop = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(
            base.classifier[0].out_features,
            num_classes
        )

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        feat = self.drop(
            self.act1(
                self.fc1(x)
            )
        )

        out = self.fc2(feat)

        return out

# =========================================================
# LOAD MODEL
# =========================================================

print("Building model...")

model = StudentWithFeat(NUM_CLASSES)

print("Loading weights...")

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(state_dict)

model.eval()

print("SUCCESS: weights loaded")

# =========================================================
# DUMMY INPUT
# =========================================================

dummy_input = torch.randn(
    1,
    3,
    224,
    224
)

# =========================================================
# EXPORT ONNX
# =========================================================

print("Exporting ONNX...")

torch.onnx.export(

    model,

    dummy_input,

    ONNX_PATH,

    export_params=True,

    opset_version=11,

    do_constant_folding=True,

    input_names=["input"],

    output_names=["output"],

    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("SUCCESS: ONNX exported")

# =========================================================
# VERIFY ONNX
# =========================================================

onnx_model = onnx.load(ONNX_PATH)

onnx.checker.check_model(onnx_model)

print("SUCCESS: ONNX verified")

print("\nDONE")