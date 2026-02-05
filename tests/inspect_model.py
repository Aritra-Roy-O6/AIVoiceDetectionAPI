import sys
import os
import logging

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)

# Set backend to TORCH provided torch is installed
os.environ["KERAS_BACKEND"] = "torch"

try:
    model_path = "model/model-1.keras"
    if not os.path.exists(model_path):
        print(f"File {model_path} DOES NOT EXIST.")
    else:
        print(f"File {model_path} exists. Size: {os.path.getsize(model_path)} bytes.")
        
        with open(model_path, "rb") as f:
            header = f.read(4)
        
        print(f"File Header: {header}")
        
        if header.startswith(b"PK"):
            print("Format: ZIP (Likely Keras 3 .keras)")
            is_zip = True
        elif header.startswith(b"\x89HDF"):
            print("Format: HDF5 (Likely Legacy .h5 saved as .keras)")
            is_zip = False
        else:
            print("Format: Unknown")
            is_zip = False

        import keras
        print(f"Keras version: {keras.__version__}")
        
        # Try loading
        if is_zip:
             print("Attempting load as Keras 3...")
             model = keras.models.load_model(model_path)
             model.summary()
        else:
             print("Format is HDF5. Creating temp .h5 file for loading...")
             import shutil
             temp_h5_path = "model/temp_model_inspection.h5"
             shutil.copyfile(model_path, temp_h5_path)
             
             try:
                print(f"Loading {temp_h5_path}...")
                model = keras.models.load_model(temp_h5_path)
                print("Model Summary:")
                model.summary()
                
                if hasattr(model, "input_shape"):
                    print(f"Input Shape: {model.input_shape}")
                    
             except Exception as e:
                print(f"Load failed: {e}")
             finally:
                if os.path.exists(temp_h5_path):
                    os.remove(temp_h5_path)
                
except Exception as e:
    print(f"Error: {e}")
