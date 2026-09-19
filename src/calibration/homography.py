import numpy as np
import cv2
import pandas as pd

class CourtCalibrator:
    """
    Mapea las coordenadas de la cámara (píxeles) a las coordenadas del plano de la cancha (metros).
    Aplica el principio de que los pies del jugador (Z=0) pueden proyectarse mediante Homografía 2D.
    """
    def __init__(self):
        self.H = None
        
    def calculate_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz H a partir de puntos conocidos.
        Ej: Puntos de intersección de la zona de tres puntos, tiros libres, esquinas.
        
        src_pts: Array Nx2 de coordenadas (x,y) en la imagen de video.
        dst_pts: Array Nx2 de coordenadas (x,y) en el plano 2D de la cancha real.
        """
        if len(src_pts) < 4:
            raise ValueError("Se necesitan al menos 4 puntos para calcular la homografía.")
            
        # RANSAC es clave acá porque los clicks manuales siempre tienen error humano
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.H = H
        return H
        
    def pixel_to_court(self, x_pixel: float, y_pixel: float) -> tuple[float, float]:
        """Proyecta un único punto de píxeles a metros."""
        if self.H is None:
            raise ValueError("Primero debés calcular la matriz llamando a calculate_homography()")
            
        pt = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, self.H)
        return float(dst[0][0][0]), float(dst[0][0][1])

    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Toma el dataset crudo y le agrega las columnas espaciales reales.
        Asumimos que el punto de contacto es el centro inferior del Bounding Box.
        """
        if self.H is None:
            raise ValueError("Matriz no calculada.")
            
        # Punto medio en X, y la base (y2) en Y (los pies)
        df = df.copy()
        df['x_base_px'] = (df['x1'] + df['x2']) / 2.0
        df['y_base_px'] = df['y2']
        
        # Vectorizamos la transformación para que sea súper rápida
        pts_px = np.stack((df['x_base_px'].values, df['y_base_px'].values), axis=-1)
        pts_px = pts_px.reshape(-1, 1, 2).astype(np.float32)
        
        pts_court = cv2.perspectiveTransform(pts_px, self.H)
        
        df['x_court_m'] = pts_court[:, 0, 0]
        df['y_court_m'] = pts_court[:, 0, 1]
        
        return df
