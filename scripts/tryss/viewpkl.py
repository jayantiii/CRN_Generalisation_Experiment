import pickle
import os
import json
from pprint import pprint

def view_pkl_structure(pkl_path):
    # Get absolute path
    abs_path = os.path.abspath(pkl_path)
    if not os.path.exists(abs_path):
        print(f"File not found: {abs_path}")
        return
    
    print(f"Loading: {abs_path}")
    
    # Load PKL with binary read
    with open(abs_path, 'rb') as f:
        try:
            data = pickle.load(f)
        except EOFError:
            print("Error: PKL file is corrupted or empty")
            return
        except Exception as e:
            print(f"Error loading PKL: {str(e)}")
            return
            
    # Validate data structure
    if not isinstance(data, (list, dict)):
        print(f"Unexpected data type: {type(data)}")
        return
        
    # Print info
    print("\n=== PKL Structure ===")
    if isinstance(data, list):
        print(f"List length: {len(data)}")
        if len(data) > 0:
            print("\nFirst item:")
            pprint(data[0], indent=2, width=80, depth=2)
    else:
        print("\nDict keys:")
        pprint(data.keys(), indent=2)

if __name__ == "__main__":
    pkl_path = "nuscenes_infos_val.pkl"
    view_pkl_structure(pkl_path)