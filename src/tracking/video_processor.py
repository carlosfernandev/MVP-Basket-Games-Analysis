import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from src.tracking.camera_motion import CameraMotionTracker

class VideoTrackingExtractor:
    """
    FASE 1 (Inferencia): YOLO + ByteTrack + Camera Motion Compensation.
    Extrae detecciones y ajusta las coordenadas anulando el paneo de la cámara.
    """
    def __init__(self, model_weights: str = 'yolo11n.pt', tracker_config: str = 'bytetrack.yaml'):
        print(f"Cargando modelo YOLO: {model_weights}")
        self.model = YOLO(model_weights)
        self.tracker_config = tracker_config
        self.camera_tracker = CameraMotionTracker()
        
    def process_video(self, video_path: str, output_csv_path: str):
        print(f"Iniciando procesamiento con compensación de movimiento: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
            
        data = []
        frame_idx = 0
        
        # Procesamos frame a frame manualmente para poder inyectar el CameraMotionTracker
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. Anulamos el movimiento de la cámara respecto al Frame 0
            # M_cam transforma puntos de este frame a las coordenadas del frame original
            M_cam = self.camera_tracker.process_frame(frame)
            
            # 2. Inferencia de YOLO y ByteTrack
            results = self.model.track(
                source=frame,
                tracker=self.tracker_config,
                persist=True,
                classes=[0, 32], # Persona y pelota
                verbose=False
            )
            
            r = results[0]
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                
                for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
                    x1, y1, x2, y2 = box
                    
                    # 3. Aplicar compensación de cámara al Bounding Box
                    # Convertimos los puntos a formato homogéneo para multiplicarlos por M_cam
                    pts = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)
                    warped_pts = cv2.perspectiveTransform(pts, M_cam)
                    
                    wx1, wy1 = warped_pts[0][0]
                    wx2, wy2 = warped_pts[0][1]
                    
                    data.append({
                        'frame': frame_idx,
                        'track_id': int(track_id),
                        'class_id': int(class_id),
                        'x1': round(float(wx1), 2),
                        'y1': round(float(wy1), 2),
                        'x2': round(float(wx2), 2),
                        'y2': round(float(wy2), 2),
                        'confidence': round(float(conf), 3)
                    })
            
            if frame_idx % 30 == 0:
                print(f"Procesando frame {frame_idx}...")
                
            frame_idx += 1
            
        cap.release()
        
        df = pd.DataFrame(data)
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"¡Extracción y compensación completadas! Datos en: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Modelo YOLO (ej: yolo11m.pt para mejor detección)")
    args = parser.parse_args()
    
    extractor = VideoTrackingExtractor(model_weights=args.weights)
    extractor.process_video(args.video, args.output)
