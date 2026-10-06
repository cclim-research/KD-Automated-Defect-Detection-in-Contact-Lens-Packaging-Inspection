import os
import numpy as np
import tensorflow as tf

# Define the exact path to your float32 TFLite file
TFLITE_MODEL_PATH = os.path.join("saved_model", "student_model_float32.tflite")

print(f"--> Initializing TFLite Interpreter for: {TFLITE_MODEL_PATH}")
try:
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n" + "="*40)
    print("✅ TFLITE MODEL LOADED SUCCESSFULLY!")
    print("="*40)
    print(f"Expected Input Shape:  {input_details[0]['shape']}")
    print(f"Expected Input Type:   {input_details[0]['dtype']}")
    print(f"Expected Output Shape: {output_details[0]['shape']}")
    print("="*40)

    # Generate a dummy image tensor (1 image, 224x224, 3 channels) matching expected float32 type
    print("\n--> Running an inference test with dummy image data...")
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

    # Set the tensor to point to our dummy image data
    interpreter.set_tensor(input_details[0]['index'], dummy_input)

    # Run the model execution
    interpreter.invoke()

    # Extract the resulting prediction matrix
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print("\n" + "🎉 TEST SUCCESSFUL!")
    print(f"Output Raw Layer Array:\n {output_data[0]}")
    print(f"Predicted Class Output Count: {output_data.shape[1]} classes successfully registered.")

except Exception as e:
    print(f"\n❌ ERROR testing TFLite model: {e}")