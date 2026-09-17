import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

class VideoTrackingExtractor:
    """
    Clase responsable de la FASE 1 (Mundo Inferencia).
    Acopla la lectura del video, inferencia de YOLO y seguimiento con ByteTrack.
    Su única responsabilidad es transformar píxeles en un dataset estructurado.
    """
    def __init__(self, model_weights: str = 'yolo11n.pt', tracker_config: str = 'bytetrack.yaml'):
        # Cargamos el modelo de Ultralytics (Descargará los pesos automáticamente si no existen)
        print(f"Cargando modelo YOLO: {model_weights}")
        self.model = YOLO(model_weights)
        self.tracker_config = tracker_config
        
    def process_video(self, video_path: str, output_csv_path: str):
        print(f"Iniciando procesamiento del video: {video_path}")
        
        # Clases COCO estándar: 0 (Persona), 32 (Pelota de deportes)
        # Usamos stream=True para no saturar la memoria RAM con todos los frames a la vez
        results = self.model.track(
            source=video_path,
            tracker=self.tracker_config,
            stream=True,
            persist=True,
            classes=[0, 32], 
            verbose=False
        )
        
        data = []
        for frame_idx, r in enumerate(results):
            # Si el frame no tiene detecciones o no pudo asignar IDs de tracking, salteamos
            if r.boxes is None or r.boxes.id is None:
                continue
            
            # Extraemos tensores y los pasamos a CPU -> Numpy
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            
            # Serializamos las detecciones del frame
            for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
                x1, y1, x2, y2 = box
                data.append({
                    'frame': frame_idx,
                    'track_id': int(track_id),
                    'class_id': int(class_id),
                    'x1': round(float(x1), 2),
                    'y1': round(float(y1), 2),
                    'x2': round(float(x2), 2),
                    'y2': round(float(y2), 2),
                    'confidence': round(float(conf), 3)
                })
                
            if frame_idx % 150 == 0:
                print(f"Procesando frame {frame_idx}...")
                
        # Exportamos el DataFrame a CSV
        df = pd.DataFrame(data)
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"¡Extracción completada! Datos guardados en: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ejecutar Inferencia CV")
    parser.add_argument("--video", type=str, required=True, help="Ruta al video crudo")
    parser.add_argument("--output", type=str, required=True, help="Ruta del CSV de salida")
    args = parser.parse_args()
    
    extractor = VideoTrackingExtractor()
    extractor.process_video(args.video, args.output)
