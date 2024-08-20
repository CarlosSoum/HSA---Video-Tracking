# HSA---Video-Tracking

Sistema de seguimiento de objetos en video basado en el algoritmo ZNCC (Zero Mean Normalized Cross-Correlation) potenciado por el algoritmo meta-heurístico HSA (Honey-Bee Search Algorithm).

Las pruebas de este sistema se han realizado sobre el dataset ALOV300++ y evaluadas mediante la métrica F1-Score. 

De momento, las categorías de dicho dataset dónde el sistema obtuvo puntajes más altos de F1-Score son: Oclusión (0.21), Larga Duración (0.25), Transparencia (0.33) y Cobertura de Superficie (0.35).

Así mismo, las categorías dónde el sistema ha sido más rápido capaz de analizar más de 2 frames por segundo (FPS) han sido: Especularidad (2.1 FPS), Oclusión (2.2 FPS), Cobertura de Superficie (2.7 FPS), Transparencia (2.9), Objetos desordenados (3 FPS), Zoom (3.1 FPS), Confusión (3.1 FPS), Movimientos Suaves (3.8 FPS), Luz (3.9 FPS), Movimientos Coherentes (4.4 FPS) y Larga Duración (5.6 FPS).

Las categorías con mejor desempeño en ambos atributos combinados (F1-Score y FPS) son: Oclusión, Cobertura de Superficie, Transparencia y Larga Duración.

Las pruebas se realizaron en los siguientes equipos de cómputo:

DELL XPS 8700:
- CPU: Intel Core i7-4790
  - Reloj: 3.60GHz
  - Hilos: 8
  - Núcleos: 4
  - 64 bits
- GPU: AMD Radeon R9 270
  - Núcleos: 1280
  - Reloj: 900MHz

Modelo de laptop desconocido:
- CPU: AMD FX-6100
  - Reloj: 3.3GHz
  - Hilos: 6
  - Núcleos: 6
  - 64 bits
- GPU: NVDIA GeForce GTX-680
  - Núcleos: 1536
  - Reloj: 1006MHz

Lenovo ThinkStation P300
- CPU: Intel Xeon v3.
  - Reloj: 3.3GHz
  - Hilos: 8
  - Núcleos: 4
  - 64 bits
- GPU: NVIDIA GK107GL
  - Núcleos: 192
  - Reloj: 876MHz
