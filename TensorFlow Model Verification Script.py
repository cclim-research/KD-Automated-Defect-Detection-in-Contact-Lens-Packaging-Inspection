import tensorflow as tf
import numpy as np

MODEL_PATH = "saved_model"

try:
    print("--> Attempting to load the converted low-level SavedModel...")
    # Load as a generic signature-based model object
    imported = tf.saved_model.load(MODEL_PATH)
    print("SUCCESS: TensorFlow SavedModel structure read perfectly!\n")
    
    print("--> Locating model prediction signature...")
    # Get the default inference function mapping (usually called 'serving_default')
    infer = imported.signatures["serving_default"]
    
    # Let's peek at the actual input name the model expects
    input_name = list(infer.structured_input_signature[1].keys())[0]
    print(f"Model signature found! Expects input key name: '{input_name}'")
    
    print("\n--> Running a dummy data prediction test...")
    # Generate a random image batch (1 image, 224x224, 3 channels)
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # Feed input into the signature function using its exact keyword argument name
    tf_tensor = tf.convert_to_tensor(dummy_input)
    output_dict = infer(**{input_name: tf_tensor})
    
    # Get the key name of the final output array
    output_name = list(output_dict.keys())[0]
    final_output = output_dict[output_name]
    
    print("\n" + "="*40)
    print("🎉 VALIDATION COMPLETE: SUCCESS!")
    print("="*40)
    print("Model Output Shape produced:", final_output.shape)
    print("Sample raw output probabilities:\n", final_output.numpy()[0][:7]) 
    print("="*40)

except Exception as e:
    print(f"\n❌ ERROR during signature verification: {e}")