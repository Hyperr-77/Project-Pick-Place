import os
import json

UI_config_path = "UI-database.json"
database_path = "database.json"

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    return config

def save_config(config, config_path):
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

def add_UI_entry(entry_name, eps, preview_image, object_contour):
    config = load_config(UI_config_path)
    config[entry_name] = {
        "eps": eps,
        "preview_image": preview_image,
        "object_contour": object_contour
    }
    save_config(config, UI_config_path)