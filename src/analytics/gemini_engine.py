import os
import re
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. Esquema hiper-especializado en Pasar y Cortar
class PassAndCutAnalysis(BaseModel):
    is_pass_executed: bool = Field(description="¿Se ejecutó un pase desde el eje central hacia un jugador a 45 grados (ala)?")
    is_cut_executed: bool = Field(description="¿El jugador que dio el pase realizó un corte hacia el centro/aro?")
    cut_speed_classification: str = Field(description="Clasificación de la velocidad de corte: 'Rapido' (si alcanzó más de 4.0 m/s), 'Lento' o 'No hubo corte'")
    cut_destination: str = Field(description="Decisión post-corte, ¿hacia dónde fue?: 'Pintura/Aro', 'Esquina Derecha', 'Esquina Izquierda' o 'No aplica'")
    confidence: float = Field(description="Nivel de confianza general de la IA entre 0.0 y 1.0")
    key_players_ids: list[int] = Field(description="IDs numéricos (track_id) del jugador pasador/cortador y del receptor")
    technical_explanation: str = Field(description="Explicación detallada de las coordenadas: en qué posición (X,Y) estaba al pasar, a qué velocidad cortó y dónde frenó.")

class GeminiTacticalEngine:
    """
    Motor Táctico Especializado. Restringido 100% a la acción de "Pasar y Cortar"
    para maximizar la precisión evaluando decisiones complejas.
    """
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key)
        
        available_models = list(self.client.models.list())
        flash_models = [
            m.name for m in available_models 
            if re.fullmatch(r'models/gemini-\d+\.\d+-flash', m.name)
        ]
        
        if flash_models:
            self.model_name = flash_models[-1].replace('models/', '') 
        else:
            self.model_name = available_models[0].name.replace('models/', '')
            
    def analyze_play(self, tactical_json_data: str) -> PassAndCutAnalysis:
        # Prompt Especializado con conocimiento de dominio de Baloncesto
        system_instruction = """
        Eres un Entrenador Especialista en Desarrollo de Jugadores de Baloncesto.
        Tu ÚNICA tarea es analizar matrices cinemáticas espaciales para detectar y evaluar EXCLUSIVAMENTE UNA ACCIÓN COMPLEJA: "PASAR Y CORTAR" (Give and Go).
        DEBES IGNORAR cualquier otra jugada. Tu objetivo es medir la ejecución técnica y la toma de decisiones del jugador.
        
        Zonas Geométricas de la Cancha a tener en cuenta:
        1. Eje Central (Top of the key): Zona superior central de la línea de 3 puntos.
        2. 45 grados (Alas): Zonas laterales alrededor de la línea de 3 puntos.
        3. Pintura (Zona restringida): Zona muy cercana al aro.
        4. Esquinas (Corners): Intersección de la línea de fondo y la línea de 3 puntos (extremos de la cancha).
        
        Algoritmo Mental para evaluar la Acción "Pasar y Cortar":
        1. IDENTIFICACIÓN: Diferencia qué objeto es qué. 'class_id' = 0 siempre indica que es una Persona/Jugador. 'class_id' = 32 siempre indica que es el Balón (es una etiqueta universal de YOLO, NO un ID de tracking). Cada jugador tiene un 'track_id' único que lo identifica.
        2. PASE: Busca el momento donde el Balón (class_id=32) viaja rápidamente desde un Jugador A (situado en el eje central) hacia un Jugador B (a 45 grados).
        3. CORTE: Inmediatamente después de soltar el balón, evalúa si el Jugador A cambia de dirección y avanza hacia el aro.
        4. VELOCIDAD DE CORTE: Observa la columna 'speed_mps' del Jugador A durante ese corte. Considera "Rapido" si supera los 4.0 m/s, o "Lento" si es un trote inferior.
        5. TOMA DE DECISIÓN: Mapea la trayectoria final (coordenadas finales X, Y) del Jugador A. ¿El corte terminó metiéndose de lleno a la Pintura, o el jugador giró hacia alguna Esquina para abrir la cancha?
        
        Devuelve tu evaluación estrictamente en el formato JSON solicitado. No inventes datos que no estén respaldados por el CSV.
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=tactical_json_data,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=PassAndCutAnalysis,
                temperature=0.0 # Determinismo absoluto: 0 creatividad, 100% matemática
            ),
        )
        
        return PassAndCutAnalysis.model_validate_json(response.text)
