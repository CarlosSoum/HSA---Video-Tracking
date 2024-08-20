# HSA---Video-Tracking

Sistema de seguimiento de objetos en video basado en el algoritmo ZNCC (Zero Mean Normalized Cross-Correlation) potenciado por el algoritmo meta-herustico HSA (Honey-Bee Search Algorithm).

La pruebas de este sistema se han realizado sobre el dataset ALOV300++ y evaluadas mediante la metrica F1-Score. 

De momento, las categorias de dicho dataset dónde el sistema obtuvo puntajes más altos de F1-Score son: Oclusión (0.21), Larga Duración (0.25), Transparencia (0.33) y Covertura de Superficie (0.35).

Asi mismo, las categorias dónde el sistema ha sido mas rapido capaz de analizar mas de 2 frames por segundo (FPS) han sido: Especularidad (2.1 FPS), Oclusión (2.2 FPS), Covertura de Superficie (2.7 FPS), Transparencia (2.9), Objetos desordenados (3 FPS), Zoom (3.1 FPS), Confusión (3.1 FPS), Movimientos Suaves (3.8 FPS), Luz (3.9 FPS), Movimientos Coherentes (4.4 FPS) y Larga Duración (5.6 FPS).

Las categorias con mejor desempeño en ambos atributos combinados (F1-Score y FPS) son: Oclusión, Covertura de Superficie, Transparencia y Larga Duración.

La pruebas se realizaron en los siguientes equipos de computo:

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
