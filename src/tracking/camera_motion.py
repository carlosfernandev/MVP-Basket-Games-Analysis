import cv2
import numpy as np

class CameraMotionTracker:
    """
    Rastrea el movimiento del fondo (la cancha) usando Flujo Óptico (Lucas-Kanade).
    Genera una matriz de transformación que 'anula' el movimiento de la cámara,
    proyectando cualquier frame al sistema de coordenadas del PRIMER FRAME.
    """
    def __init__(self):
        self.prev_gray = None
        # Matriz Identidad 3x3 (Sin movimiento inicial)
        self.accumulated_matrix = np.eye(3, dtype=np.float32)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return self.accumulated_matrix.copy()

        # 1. Buscar puntos fuertes en el frame anterior (líneas de la cancha, logos, gradas estáticas)
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        
        if prev_pts is not None and len(prev_pts) > 0:
            # 2. Rastrear adónde se movieron esos puntos en el frame actual
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None)
            
            # Filtrar solo los puntos que se rastrearon exitosamente
            valid = status.flatten() == 1
            p0 = prev_pts[valid]
            p1 = curr_pts[valid]
            
            # 3. Si tenemos suficientes puntos, calculamos cómo se movió la cámara
            if len(p0) >= 4:
                # Calculamos una transformación afín estricta (paneo y zoom).
                # Transformamos de p1 (actual) a p0 (anterior) para anular el movimiento.
                M, inliers = cv2.estimateAffinePartial2D(p1, p0, method=cv2.RANSAC)
                if M is not None:
                    # Convertir M (2x3) a Matriz 3x3 Homogénea
                    H_step = np.eye(3, dtype=np.float32)
                    H_step[0:2, :] = M
                    
                    # Acumulamos el movimiento multiplicando matrices
                    self.accumulated_matrix = self.accumulated_matrix @ H_step
                    
        self.prev_gray = gray
        return self.accumulated_matrix.copy()
