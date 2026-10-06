"""
================================================================================
  PyTorch → ONNX Export Script
  Contact Lens Defect Packaging Inspection — ATS SmartVision Configurator
================================================================================
  Author  : Industrial ML Engineer
  Purpose : Export a distilled student model (.pth / .pt) to a universal
            ONNX format for real-time DNN inferencing on the factory floor.
  Models  : MobileNetV3-Large | TinyViT-5M-224 | FastViT-SA12
  Classes : DOUBLE | INVERTED | NONE | OUTPOSITION | PASS | SEAL | TEAR
================================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import sys
import logging

# ── Third-Party ───────────────────────────────────────────────────────────────
import torch
import timm
import onnx
import onnxruntime as ort
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — LOGGING SETUP
#  Gives timestamped, colour-coded console feedback — useful for production
#  scripts running inside automated build pipelines.
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — GLOBAL CONFIGURATION
#  Change MODEL_CHOICE and CHECKPOINT_PATH below, then run the script.
#  Everything else is derived automatically.
# ══════════════════════════════════════════════════════════════════════════════

# ── Choose which student model to export ─────────────────────────────────────
#   Options: "mobilenetv3"  |  "tinyvit"  |  "fastvit"
MODEL_CHOICE = "tinyvit"

# ── Path to your trained .pth / .pt checkpoint ───────────────────────────────
CHECKPOINT_PATH = r"C:\FYP Student_Boet Puai Ying 221151481\150 epoch - RESULTS - TinyVit train (diff baseline and lr)\student_bs64_lr0.0001.pt"

# ── Where to save the exported ONNX file ─────────────────────────────────────
OUTPUT_DIR = "onnx_exports"

# ── Class labels (must match training order exactly) ─────────────────────────
CLASS_NAMES = ["DOUBLE", "INVERTED", "NONE", "OUTPOSITION", "PASS", "SEAL", "TEAR"]
NUM_CLASSES = len(CLASS_NAMES)          # 7

# ── Camera / input tensor dimensions ─────────────────────────────────────────
#   ATS SmartVision expects: Batch=1, Channels=3 (RGB), Height=224, Width=224
INPUT_HEIGHT = 224
INPUT_WIDTH  = 224
INPUT_CHANNELS = 3

# ── ONNX export settings ─────────────────────────────────────────────────────
OPSET_VERSION = 14          # Stable opset widely supported by industrial runtimes


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MODEL REGISTRY
#  Maps a short string key to the exact timm model identifier.
#  Adding a new architecture in the future is as simple as inserting one line.
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "mobilenetv3": "mobilenetv3_large_100",
    "tinyvit":     "tiny_vit_5m_224",  # Added underscore here
    "fastvit":     "fastvit_sa12",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_model(model_choice: str, num_classes: int) -> torch.nn.Module:
    """
    Instantiate a bare (randomly-weighted) model architecture from timm.

    Parameters
    ----------
    model_choice : str
        One of 'mobilenetv3', 'tinyvit', or 'fastvit'.
    num_classes : int
        Number of output logits (7 for this project).

    Returns
    -------
    torch.nn.Module
        An un-trained model skeleton ready to receive checkpoint weights.

    Notes
    -----
    pretrained=False is critical here: we do NOT want ImageNet weights loaded
    from the internet.  We will overwrite every weight from our own checkpoint
    in the next step.
    """
    if model_choice not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_choice}'. "
            f"Valid choices: {list(MODEL_REGISTRY.keys())}"
        )

    timm_name = MODEL_REGISTRY[model_choice]
    log.info(f"Building architecture  →  timm model: '{timm_name}'")

    model = timm.create_model(
        timm_name,
        pretrained=False,       # Do not download ImageNet weights
        num_classes=num_classes # Replace the classifier head with 7 outputs
    )

    log.info(
        f"Architecture built  →  "
        f"{sum(p.numel() for p in model.parameters()):,} total parameters"
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — CHECKPOINT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """
    Load trained weights from a .pth / .pt file into the model skeleton.

    Parameters
    ----------
    model          : The bare architecture built in build_model().
    checkpoint_path: Absolute or relative path to the checkpoint file.

    Returns
    -------
    torch.nn.Module
        The same model object, now carrying the trained factory weights.

    Design Decisions
    ----------------
    map_location=torch.device('cpu')
        Forces all tensors to land on the CPU regardless of where they were
        saved.  This is essential in industrial deployment scenarios where the
        export PC may not have an NVIDIA GPU, and it avoids CUDA-related errors
        when moving checkpoints between machines.

    weights_only=True  (PyTorch >= 2.0)
        Prevents arbitrary Python code execution during unpickling — a
        security best practice for production environments.  If your PyTorch
        version is older than 2.0, remove this argument.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'\n"
            "Check CHECKPOINT_PATH at the top of this script."
        )

    log.info(f"Loading checkpoint  →  '{checkpoint_path}'")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),   # CPU-safe loading (no GPU required)
        weights_only=True,                  # Security: blocks arbitrary code exec
    )

    # ── Handle different checkpoint formats ──────────────────────────────────
    # Training scripts sometimes wrap weights under a 'state_dict' or
    # 'model_state_dict' key (e.g., when saving optimizer state alongside
    # the model). The logic below handles all common conventions.
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            log.info("Detected key: 'state_dict'")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            log.info("Detected key: 'model_state_dict'")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            log.info("Detected key: 'model'")
        else:
            # Assume the checkpoint IS the state_dict directly
            state_dict = checkpoint
            log.info("Detected format: raw state_dict (no wrapper key)")
    else:
        state_dict = checkpoint

    # strict=True (default) raises an error if any weight is missing or
    # unexpected — guaranteeing a 100% weight match between checkpoint and
    # the architecture.  This is safer than silent partial loads.
    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    if missing:
        log.warning(f"Missing keys in checkpoint  →  {missing}")
    if unexpected:
        log.warning(f"Unexpected keys in checkpoint  →  {unexpected}")

    log.info("Checkpoint loaded successfully.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — ONNX EXPORTER
# ══════════════════════════════════════════════════════════════════════════════

def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 14,
) -> str:
    """
    Convert the PyTorch model to ONNX format and save it to disk.

    Parameters
    ----------
    model        : Trained PyTorch model (already in eval mode).
    output_path  : Where to write the .onnx file.
    input_shape  : Dummy tensor shape matching the industrial camera output.
    opset_version: ONNX operator set version (14 recommended for industry).

    Returns
    -------
    str : The final output path of the saved ONNX file.

    Key Export Arguments Explained (for FYP thesis)
    ------------------------------------------------
    model (torch.nn.Module)
        The PyTorch model to trace.  It must already be in .eval() mode so
        that inference-time behaviour (frozen BN, no Dropout) is captured.

    args (dummy_input)
        A representative input tensor.  PyTorch traces the computational
        graph by running this tensor through the model once.  The actual
        pixel values don't matter — only the shape does.

    f (output_path)
        Destination file path.  The ONNX graph is serialised as a
        Protocol Buffers binary here.

    export_params=True
        Embeds all trained weights directly inside the .onnx file, producing
        a single self-contained artefact.  Set False only if you want to
        ship weights separately (uncommon in industrial deployments).

    opset_version=14
        Selects which version of the ONNX operator specification to use.
        Higher versions support more modern ops (e.g., LayerNorm for ViTs).
        Opset 14 strikes the best balance between compatibility with
        industrial runtimes (TensorRT, OpenVINO, ATS SmartVision) and
        support for transformer-based architectures.

    do_constant_folding=True
        Pre-computes sub-expressions whose inputs are compile-time constants
        (e.g., fixed weight matrices in BatchNorm after folding).  This
        reduces graph node count and speeds up runtime inference — especially
        beneficial on embedded industrial PCs with limited CPU headroom.

    input_names=['input_image']
        Assigns a human-readable label to the input node.  ATS SmartVision's
        DNN Inferencing Tool may reference this name when binding camera
        frames to the network input buffer.

    output_names=['defect_classification']
        Labels the output node.  The inferencing tool reads logits from this
        node and maps them to CLASS_NAMES for the operator display panel.

    dynamic_axes
        By default, ONNX fixes every tensor dimension at export time.
        Specifying axis 0 (batch) as dynamic lets the runtime accept any
        batch size without graph recompilation.  This prevents the
        "fixed-shape mismatch" errors that industrial software often raises
        when it internally batches frames differently from the export batch.
    """

    # ── Step A: Lock the model in evaluation / inference mode ─────────────────
    # .eval() switches BatchNormalization layers from 'training mode'
    # (computing mean/variance from the current mini-batch) to 'inference mode'
    # (using the frozen running statistics learned during training).
    # It also disables Dropout, which randomly zeroes activations during
    # training but must be disabled for deterministic factory inference.
    model.eval()
    log.info("Model set to .eval() mode  →  BN frozen, Dropout disabled.")

    # ── Step B: Create a dummy input tensor ───────────────────────────────────
    # PyTorch's ONNX exporter works by *tracing*: it feeds a real tensor
    # through the model and records every operation into a static graph.
    # The values are irrelevant; only the dtype and shape matter.
    # Shape: (1 batch, 3 RGB channels, 224 height, 224 width)
    dummy_input = torch.zeros(input_shape, dtype=torch.float32)
    log.info(f"Dummy input tensor created  →  shape {tuple(dummy_input.shape)}, dtype {dummy_input.dtype}")

    # ── Step C: Define dynamic axes ───────────────────────────────────────────
    # Index 0 is the batch dimension for both input and output tensors.
    # Marking it as dynamic prevents hard-coded batch-size constraints.
    dynamic_axes = {
        "input_image":           {0: "batch_size"},
        "defect_classification": {0: "batch_size"},
    }

    # ── Step D: Ensure the output directory exists ────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # ── Step E: Run the ONNX export ───────────────────────────────────────────
    log.info(f"Exporting ONNX graph  →  opset {opset_version}  →  '{output_path}'")

    with torch.no_grad():       # Disable autograd: we are not training
        torch.onnx.export(
            model,                              # Trained PyTorch model (eval mode)
            dummy_input,                        # Representative input tensor for tracing
            output_path,                        # Destination .onnx file path
            export_params=True,                 # Embed trained weights in the file
            opset_version=opset_version,        # ONNX operator set version (14 = stable)
            do_constant_folding=True,           # Pre-fold constants → fewer nodes, faster inference
            input_names=["input_image"],        # Name for the input node (camera frame buffer)
            output_names=["defect_classification"],  # Name for the output node (logit vector)
            dynamic_axes=dynamic_axes,          # Allow variable batch sizes at runtime
            verbose=False,                      # Set True for node-by-node export log
        )

    log.info(f"ONNX export complete  →  '{output_path}'")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — ONNX VALIDATOR
#  Run two independent checks after export:
#    (a) onnx.checker  — validates the graph's structural integrity
#    (b) onnxruntime   — runs an actual forward pass to confirm numerical output
# ══════════════════════════════════════════════════════════════════════════════

def validate_onnx(onnx_path: str, input_shape: tuple = (1, 3, 224, 224)) -> bool:
    """
    Validate the exported ONNX model at two levels.

    Level 1 — Structural check (onnx.checker):
        Verifies that the graph conforms to the ONNX specification: all nodes
        have valid operator names, tensor shapes are consistent, and no required
        attributes are missing.  If this passes, any ONNX-compliant runtime
        (TensorRT, OpenVINO, ATS SmartVision) should be able to parse the file.

    Level 2 — Runtime check (onnxruntime):
        Loads the model into ONNX Runtime (a neutral C++ inference engine),
        runs a dummy forward pass, and reports the output tensor shape.
        This confirms the model is not just structurally valid but actually
        executable — catching edge cases like unsupported custom ops that
        pass structural checks but fail at runtime.

    Returns True if both checks pass, False otherwise.
    """
    log.info("─" * 60)
    log.info("VALIDATION — Level 1: ONNX structural check")

    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        log.info("✓  ONNX structural check PASSED")
    except onnx.checker.ValidationError as e:
        log.error(f"✗  ONNX structural check FAILED: {e}")
        return False

    log.info("VALIDATION — Level 2: ONNX Runtime forward-pass check")

    try:
        session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]  # CPU only — no GPU dependency
        )

        # Build a random float32 input (same shape as during export)
        dummy_np = np.random.rand(*input_shape).astype(np.float32)

        # Run inference through the ONNX Runtime session
        input_feed = {session.get_inputs()[0].name: dummy_np}
        outputs = session.run(None, input_feed)

        output_shape = outputs[0].shape
        log.info(f"✓  ONNX Runtime forward-pass PASSED  →  output shape: {output_shape}")

        # Sanity-check: last dimension should equal number of classes
        if output_shape[-1] != NUM_CLASSES:
            log.warning(
                f"  Output classes mismatch: got {output_shape[-1]}, "
                f"expected {NUM_CLASSES}.  Check NUM_CLASSES setting."
            )

    except Exception as e:
        log.error(f"✗  ONNX Runtime forward-pass FAILED: {e}")
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — SUMMARY REPORTER
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(onnx_path: str, model_choice: str) -> None:
    """Print a deployment-ready summary card for the engineer and FYP report."""

    file_size_mb = os.path.getsize(onnx_path) / (1024 ** 2)

    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           ONNX EXPORT SUMMARY — ATS SmartVision             ║
╠══════════════════════════════════════════════════════════════╣
║  Architecture  : {model_choice.upper():<43}║
║  Classes ({NUM_CLASSES})    : {", ".join(CLASS_NAMES[:4]):<43}║
║               : {", ".join(CLASS_NAMES[4:]):<43}║
║  Input shape   : (batch, 3, {INPUT_HEIGHT}, {INPUT_WIDTH})  — dynamic batch     ║
║  ONNX opset    : {OPSET_VERSION:<43}║
║  Output file   : {os.path.basename(onnx_path):<43}║
║  File size     : {file_size_mb:.2f} MB{" " * (39 - len(f"{file_size_mb:.2f} MB"))}║
╠══════════════════════════════════════════════════════════════╣
║  NEXT STEPS — Import into ATS SmartVision Configurator:     ║
║  1. Open the DNN Inferencing Tool                            ║
║  2. Load ONNX model  →  select '{os.path.basename(onnx_path)}'           ║
║  3. Map input node   →  'input_image'                        ║
║  4. Map output node  →  'defect_classification'              ║
║  5. Set class labels →  DOUBLE, INVERTED, NONE,              ║
║                         OUTPOSITION, PASS, SEAL, TEAR        ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(summary)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — MAIN PIPELINE
#  This is the entry point.  All configuration is in SECTION 2 above.
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("  PyTorch → ONNX Export Pipeline — Contact Lens Inspection")
    log.info("=" * 60)
    log.info(f"  Selected model  : {MODEL_CHOICE}")
    log.info(f"  Checkpoint      : {CHECKPOINT_PATH}")
    log.info(f"  Output dir      : {OUTPUT_DIR}")
    log.info(f"  Classes ({NUM_CLASSES})      : {CLASS_NAMES}")
    log.info("=" * 60)

    # ── Build output file path ────────────────────────────────────────────────
    onnx_filename = f"student_{MODEL_CHOICE}_opset{OPSET_VERSION}.onnx"
    onnx_output_path = os.path.join(OUTPUT_DIR, onnx_filename)

    # ── Step 1: Instantiate model architecture ────────────────────────────────
    model = build_model(MODEL_CHOICE, NUM_CLASSES)

    # ── Step 2: Load trained factory weights ─────────────────────────────────
    model = load_checkpoint(model, CHECKPOINT_PATH)

    # ── Step 3: Export to ONNX ───────────────────────────────────────────────
    onnx_path = export_to_onnx(
        model=model,
        output_path=onnx_output_path,
        input_shape=(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
        opset_version=OPSET_VERSION,
    )

    # ── Step 4: Validate the exported ONNX file ───────────────────────────────
    passed = validate_onnx(
        onnx_path=onnx_path,
        input_shape=(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
    )

    if not passed:
        log.error("Export validation FAILED.  Do not deploy this model.")
        sys.exit(1)

    # ── Step 5: Print deployment summary ─────────────────────────────────────
    print_summary(onnx_path, MODEL_CHOICE)

    log.info("Pipeline finished successfully.  Model is ready for ATS SmartVision.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — ENTRY POINT
#
#  HOW TO USE THIS SCRIPT
#  ──────────────────────
#  1. Install dependencies (once):
#       pip install torch torchvision timm onnx onnxruntime
#
#  2. Set your configuration in SECTION 2 (top of file):
#       MODEL_CHOICE    = "mobilenetv3"   ← or "tinyvit" or "fastvit"
#       CHECKPOINT_PATH = "best_student_model.pth"
#
#  3. Run from terminal:
#       python export_to_onnx.py
#
#  EXAMPLE — Export MobileNetV3:
#       MODEL_CHOICE = "mobilenetv3"
#       CHECKPOINT_PATH = "checkpoints/mobilenet_kd_epoch50.pth"
#       → output: onnx_exports/student_mobilenetv3_opset14.onnx
#
#  EXAMPLE — Export TinyViT:
#       MODEL_CHOICE = "tinyvit"
#       CHECKPOINT_PATH = "checkpoints/tinyvit_kd_best.pth"
#       → output: onnx_exports/student_tinyvit_opset14.onnx
#
#  EXAMPLE — Export FastViT:
#       MODEL_CHOICE = "fastvit"
#       CHECKPOINT_PATH = "checkpoints/fastvit_sa12_final.pth"
#       → output: onnx_exports/student_fastvit_opset14.onnx
#
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
