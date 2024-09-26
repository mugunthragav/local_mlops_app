import os
import sys

models = {}

# Use a relative path for the models directory
models_directory = os.path.join(os.path.dirname(__file__))

if models_directory not in sys.path:
    sys.path.append(models_directory)

# Load models
for filename in os.listdir(models_directory):
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]  # Remove '.py'
        try:
            module = __import__(module_name)

            # Check all attributes of the module
            for item in dir(module):
                attr = getattr(module, item)

                # Check if the attribute is a class and its name ends with 'Model'
                if isinstance(attr, type) and item.endswith('Model'):
                    # Get the class name
                    class_name = attr.__name__

                    # Create an instance of the model class
                    model_instance = attr()  # Instantiate the model class

                    # Retrieve model class, parameters, test_size, and random_state
                    model = model_instance.get_model()  # This is now just the instance
                    parameters = model_instance.get_params()
                    test_size = model_instance.get_test_size()
                    random_state = model_instance.get_random_state()

                    # Store the model class reference instead of the instance
                    models[class_name] = {
                        "model": attr,  # Store the actual class reference
                        "parameters": parameters,  # Model parameters
                        "test_size": test_size,
                        "random_state": random_state
                    }
                    break  # Exit after loading the first valid class

        except Exception as e:
            print(f"Error loading model from {filename}: {str(e)}")

# This should only print once
if __name__ == "__main__":
    print("Final loaded models:")
    for model_name, model_info in models.items():
        print(
            f"{model_name}: {{'model': {model_info['model'].__name__}, 'parameters': {model_info['parameters']}, 'test_size': {model_info['test_size']}, 'random_state': {model_info['random_state']}}}")
