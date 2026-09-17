import os
import yaml

def create_structure():
    """Crea la estructura de directorios siguiendo principios de separación de responsabilidades."""
    directories = [
        "data/raw",              # Videos crudos
        "data/processed",        # Clips procesados, JSONs de tracking
        "src/ingestion",         # Módulos para leer y decodificar video
        "src/tracking",          # Detección (YOLO) y Tracking (ByteTrack)
        "src/reid",              # Re-identificación (OCR, VLM, embeddings de color)
        "src/calibration",       # Homografía, JPR (Jump-Aware Position Rectification)
        "src/features",          # Extracción espacio-temporal, filtros (Kalman, Savitzky-Golay)
        "src/analytics",         # Motor táctico, integración con Gemini
        "src/utils",             # Utilidades geométricas (Shapely), visualización (Matplotlib)
        "tests",                 # Pruebas unitarias y de integración
        "notebooks",             # EDA y experimentación
        "config"                 # Configuraciones
    ]
    
    for d in directories:
        os.makedirs(d, exist_ok=True)
        # Crear __init__.py en los paquetes de Python
        if d.startswith("src") or d == "tests":
            with open(os.path.join(d, "__init__.py"), 'w') as f:
                pass
                
    print("✓ Estructura de directorios creada correctamente.")

def create_requirements():
    """Genera el archivo de dependencias base."""
    reqs = """# Core Data Science & Math
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Computer Vision & Deep Learning
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0

# Tracking & Filtering
filterpy>=1.4.5

# Geometry & Visualization
shapely>=2.0.0
matplotlib>=3.7.0

# Config & Validation
pydantic>=2.0.0
pyyaml>=6.0
"""
    with open("requirements.txt", 'w') as f:
        f.write(reqs)
    print("✓ requirements.txt generado.")

def create_config():
    """Genera la configuración base del proyecto."""
    config = {
        "project": {
            "name": "MVP-Basket-Games-Analysis",
            "version": "0.1.0"
        },
        "video": {
            "target_fps": 30,
            "processing_resolution": [1920, 1080]
        },
        "models": {
            "detection_weights": "yolo11n.pt",
            "tracker": "bytetrack.yaml",
            "reid_method": "color_histogram" # Opciones: ocr, vlm, color_histogram
        },
        "calibration": {
            "use_jpr": True
        },
        "analytics": {
            "smoothing_filter": "savitzky_golay",
            "max_target_players_per_possession": 2
        }
    }
    
    with open("config/config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print("✓ config/config.yaml generado.")

if __name__ == "__main__":
    print("Arrancando el bootstrap del proyecto...")
    create_structure()
    create_requirements()
    create_config()
    print("¡Arquitectura base lista! A codear con cabeza.")
