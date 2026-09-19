import pandas as pd
import numpy as np

class TacticalFeatureExtractor:
    """
    Calcula métricas físicas avanzadas a partir de las coordenadas suavizadas.
    Esto le ahorra trabajo mental a Gemini y evita alucinaciones.
    """
    def __init__(self, fps: int = 30):
        self.fps = fps
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=['track_id', 'frame']).copy()
        
        # 1. Calcular Velocidad (Metros por segundo)
        # Distancia euclidiana entre el punto actual y el anterior
        df['dx'] = df.groupby('track_id')['x_court_m_smooth'].diff()
        df['dy'] = df.groupby('track_id')['y_court_m_smooth'].diff()
        df['dist_m'] = np.sqrt(df['dx']**2 + df['dy']**2)
        
        # Velocidad = Distancia / Tiempo (1 frame = 1/fps segundos)
        df['speed_mps'] = df['dist_m'] / (1.0 / self.fps)
        
        # Llenar los NaN iniciales con 0
        df['speed_mps'] = df['speed_mps'].fillna(0)
        
        # 2. Dirección (Rumbo) en grados
        df['direction_deg'] = np.degrees(np.arctan2(df['dy'], df['dx']))
        df['direction_deg'] = df['direction_deg'].fillna(0)
        
        return df

    def to_gemini_json(self, df: pd.DataFrame, target_fps: int = 5) -> str:
        """
        Convierte el DataFrame en una lista JSON optimizada para el prompt del LLM.
        Aplica downsampling para no reventar la cuota de tokens gratuitos de la API.
        """
        step = max(1, self.fps // target_fps)
        
        # Filtramos solo las columnas tácticas
        tactical_cols = ['frame', 'track_id', 'class_id', 'x_court_m_smooth', 'y_court_m_smooth', 'speed_mps']
        df_tactical = df[tactical_cols].dropna()
        
        # Downsampling: Nos quedamos solo con los frames que necesitamos (Ej: 5 FPS en vez de 30)
        df_tactical = df_tactical[df_tactical['frame'] % step == 0]
        
        # Formatear a 2 decimales para ahorrar miles de tokens en el string final
        df_tactical = df_tactical.round(2)
        
        return df_tactical.to_json(orient='records')
