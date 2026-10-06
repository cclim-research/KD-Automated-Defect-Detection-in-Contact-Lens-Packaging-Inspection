import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

# --------------------------------------------------------
# 1. Configuration & Hyperparameters
# --------------------------------------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 7
MODEL_PATH = "saved_model"  # Points to your successful folder

# Update these paths to point to your real dataset directory!
# Change this line:
DATASET_DIR = r"C:\FYP Student_Boet Puai Ying 221151481\dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# --------------------------------------------------------
# 2. Data Pipelines (Loading & Preprocessing)
# --------------------------------------------------------
print("--> Loading image datasets...")

try:
    train_dataset = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_dataset = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
except Exception as e:
    print(f"\n❌ DATASET ERROR: Could not find your image directories. Check paths!\nDetails: {e}")
    exit()

# Normalize pixel inputs to [0, 1] as expected by the converted network
normalization_layer = keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Optimize backend pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# --------------------------------------------------------
# 3. Wrapping the SavedModel inside a Keras Model
# --------------------------------------------------------
print(f"--> Building Keras layer around SavedModel from '{MODEL_PATH}'...")

# 1. Define functional input layer matching the name ('input') discovered during verification
inputs = keras.Input(shape=(224, 224, 3), name="input")

# 2. Load the low-level graph as a Keras Layer wrapper
class SavedModelLayer(keras.layers.Layer):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self.imported = tf.saved_model.load(model_path)
        self.infer = self.imported.signatures["serving_default"]
    
    def call(self, inputs):
        # Dynamically pass inputs under the correct input key name
        outputs = self.infer(input=inputs)
        # Extract the dictionary value tensor output
        return list(outputs.values())[0]

# 3. Connect structural architecture
raw_features = SavedModelLayer(MODEL_PATH)(inputs)

# Create final trainable structural pipeline
model = keras.Model(inputs=inputs, outputs=raw_features)

# --------------------------------------------------------
# 4. Compilation & Fine-Tuning Setup
# --------------------------------------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Low LR keeps your distilled knowledge safe
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n"+"="*40)
print("🚀 READY FOR TRAINING!")
print("="*40)

# Optional: track and save optimal checkpoints
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, monitor="val_loss", restore_best_weights=True)
]

# Run the training process
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save the final trained weights
model.save_weights("final_student_tf_weights.h5")
print("\n🎉 Training complete! Weights successfully saved to 'final_student_tf_weights.h5'")