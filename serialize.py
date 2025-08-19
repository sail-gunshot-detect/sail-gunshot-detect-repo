import os
import argparse
import tensorflow as tf


def serialize_model(model_path, output_path):
    """Convert a trained Keras model to TensorFlow Lite format."""
    # Load the trained model
    print(f"Loading model from: {model_path}")
    loaded_model = tf.keras.models.load_model(model_path)
    
    # Wrap in a tf.function and trace it
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 249, 64, 1], dtype=tf.float32)])
    def wrapped_model(input_tensor):
        return loaded_model(input_tensor)

    # Convert using the concrete function
    concrete_func = wrapped_model.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_func], trackable_obj=loaded_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # Convert the model
    print("Converting model to TensorFlow Lite format...")
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    print(f"Saving TensorFlow Lite model to: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print("Model serialization completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Serialize trained model to TensorFlow Lite format')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained Keras model (.keras file)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for the TensorFlow Lite model (.tflite file)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Serialize the model
    serialize_model(args.model_path, args.output_path)


if __name__ == "__main__":
    main()
