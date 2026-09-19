import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

class TrajectorySmoother:
    """
    Filtro de señales aplicado a trayectorias espaciales.
    YOLO produce pequeños 'temblores' en los bounding boxes frame a frame.
    Si no alisamos esto, la velocidad y aceleración derivadas van a ser puro ruido.
    """
    def __init__(self, window_length: int = 15, polyorder: int = 3):
        # La ventana debe ser impar. A 30 FPS, ventana de 15 = suavizado de 0.5 segundos.
        self.window_length = window_length if window_length % 2 != 0 else window_length + 1
        self.polyorder = polyorder
        
    def smooth_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica Savitzky-Golay por cada jugador (track_id) independientemente."""
        df_smoothed = df.copy()
        # Garantizar orden cronológico absoluto para no arruinar el filtro
        df_smoothed = df_smoothed.sort_values(by=['track_id', 'frame'])
        
        def apply_savgol(series: pd.Series) -> pd.Series:
            n = len(series)
            if n < self.window_length:
                return series
            return savgol_filter(series.values, window_length=self.window_length, polyorder=self.polyorder)
            
        df_smoothed['x_court_m_smooth'] = df_smoothed.groupby('track_id')['x_court_m'].transform(apply_savgol)
        df_smoothed['y_court_m_smooth'] = df_smoothed.groupby('track_id')['y_court_m'].transform(apply_savgol)
        
        return df_smoothed
