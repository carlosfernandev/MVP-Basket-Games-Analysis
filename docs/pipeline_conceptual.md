# Tema: Evaluación del impacto de la Rectificación Sensible al Salto (JPR) en la extracción de trayectorias espacio-temporales para el reconocimiento de patrones tácticos de interacción en baloncesto

## Un poco de contexto

## 1. Proceso Base (5 Pasos)
1. **Detección (YOLO):** Identificar dónde están los jugadores y el balón en cada frame de la imagen.
2. **Seguimiento (ByteTrack):** Asignar un ID único a cada detección para trazar su recorrido continuo a lo largo del tiempo.
3. **Estabilización de Cámara (Optical Flow):** Calcular y anular el movimiento del camarógrafo (paneo/zoom) para operar sobre un fondo estático.
4. **Mapeo Geométrico (Homografía 2D):** Traducir las coordenadas de la pantalla (píxeles) a una vista satelital o pizarra de la cancha (metros).
5. **Motor Táctico (LLM/Gemini):** Interpretar las trayectorias espaciales puras para identificar jugadas de básquet.

---

## 2. Features y su Rol en el Pipeline

### A. Rectificación Sensible al Salto (JPR - Jump-Aware Position Rectification)
* **El Problema:** La homografía matemática asume que los pies siempre tocan el piso ($Z=0$). Cuando un jugador salta, sus píxeles suben, y el modelo proyecta erróneamente que el jugador se desplazó hacia atrás en la cancha.
* **La Solución:** Un filtro heurístico que detecta desplazamientos verticales bruscos sin traslación horizontal en los píxeles, "congelando" la coordenada proyectada en el piso $(x, y)$ hasta que el jugador vuelva a caer de su salto.

### B. Extracción de Trayectorias Espaciotemporales
* **El Problema:** El output crudo del modelo de visión tiene "ruido" y los *bounding boxes* tiemblan frame a frame. Calcular la velocidad con esos datos permite obtener picos físicos imposibles.
* **La Solución:** Aplicación de filtros de procesamiento de señales (ej. Savitzky-Golay o Filtro de Kalman) para suavizar las rutas. Esto permite derivar métricas cinemáticas precisas y reales (coordenadas $(x,y)$, velocidad $v$, y aceleración $a$ en el instante $t$).

### C. Reconocimiento de Patrones Tácticos
* **El Problema:** Darle un video en formato MP4 a una IA visual para analizar táctica profunda es computacionalmente lento y propenso a alucinaciones.
* **La Solución:** Convertir la matriz de trayectorias limpias (las "flechas" en la pizarra táctica) en un esquema JSON estructurado. Al darle a la IA (Gemini) las posiciones relativas y velocidades puras en texto/JSON, esta puede identificar interacciones complejas (ej. *Pick & Roll*, cortinas, *Drive & Kick*) aplicando su conocimiento teórico sobre geometría perfecta.
