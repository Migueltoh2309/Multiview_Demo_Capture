# Papers de referencia — confusión izq/derecha y fusión multivista robusta

Lecturas seleccionadas a raíz de la primera prueba real de MediaPipe Pose +
triangulación, donde se observó cobertura de landmarks inconsistente entre tomas
y una posible confusión izquierda/derecha de MediaPipe en vistas laterales de
~50°. Los PDFs no se incluyen en el repositorio: cada entrada enlaza a arXiv.

---

## 1. Bultmann & Behnke (2021) — feedback semántico para pose 3D en tiempo real

**Cita (IEEE):** [1] S. Bultmann and S. Behnke, "Real-Time Multi-View 3D
Human Pose Estimation using Semantic Feedback to Smart Edge Sensors,"
*arXiv:2106.14729*, 2021. [Online]. Available: https://arxiv.org/abs/2106.14729

```bibtex
@misc{bultmann2021realtime,
  title         = {Real-Time Multi-View 3D Human Pose Estimation using Semantic Feedback to Smart Edge Sensors},
  author        = {Bultmann, Simon and Behnke, Sven},
  year          = {2021},
  eprint        = {2106.14729},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2106.14729}
}
```

- **Por qué:** el más directo al problema observado. Detección 2D
  local por cámara + fusión central por triangulación, con un mecanismo de
  *feedback* que reproyecta la pose 3D de vuelta a cada sensor para mejorar
  su propia detección 2D. Aborda explícitamente casos de oclusión e
  inversión izquierda/derecha como el fallo típico que este feedback corrige.

---

## 2. Bragagnolo, Terreran, Allegro & Ghidoni (2024) — fusión consciente de oclusión

**Cita (IEEE):** [2] L. Bragagnolo, M. Terreran, D. Allegro, and S. Ghidoni,
"Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation,"
*arXiv:2408.15810*, 2024. [Online]. Available: https://arxiv.org/abs/2408.15810

```bibtex
@misc{bragagnolo2024multiview,
  title         = {Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation},
  author        = {Bragagnolo, Laura and Terreran, Matteo and Allegro, Davide and Ghidoni, Stefano},
  year          = {2024},
  eprint        = {2408.15810},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2408.15810}
}
```

- **Por qué:** ya citado en `README.md` como justificación del ángulo de
  cámaras (±50° vs. el margen ±60° de OpenCap). En vez de triangular
  keypoints 2D ruidosos, fusiona esqueletos 3D monoculares por cámara
  optimizando con restricciones de simetría de longitud de extremidad —
  alternativa directa a nuestro filtro actual (umbral duro de RMS de
  reproyección en `common/triangulation.py`). Escenario de aplicación
  (colaboración humano-robot) es el más cercano al nuestro de los 5.

---

## 3. Cotton, Cimorelli, Shah, Anarwala, Uhlrich & Karakostas (2023) — trayectorias robustas

**Cita (IEEE):** [3] R. J. Cotton, A. Cimorelli, K. Shah, S. Anarwala,
S. Uhlrich, and T. Karakostas, "Improved Trajectory Reconstruction for
Markerless Pose Estimation," *arXiv:2303.02413*, 2023. [Online]. Available:
https://arxiv.org/abs/2303.02413

```bibtex
@misc{cotton2023improved,
  title         = {Improved Trajectory Reconstruction for Markerless Pose Estimation},
  author        = {Cotton, R. James and Cimorelli, Anthony and Shah, Kunal and Anarwala, Shawana and Uhlrich, Scott and Karakostas, Tasos},
  year          = {2023},
  eprint        = {2303.02413},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2303.02413}
}
```

- **Por qué:** compara detectores 2D y métodos de reconstrucción de
  trayectorias; reportan ruido de solo 8 mm en ancho de paso reconstruyendo
  con una función implícita en vez de triangular frame a frame e
  independiente (que es justo lo que hace hoy `09_pose_triangulate.py`, sin
  ninguna señal temporal entre frames). Coautor Scott Uhlrich es el mismo
  autor principal del paper de OpenCap ya citado en el README.

---

## 4. Davoodnia, Ghorbani, Carbonneau, Messier & Etemad (2024) — UPose3D

**Cita (IEEE):** [4] V. Davoodnia, S. Ghorbani, M.-A. Carbonneau,
A. Messier, and A. Etemad, "UPose3D: Uncertainty-Aware 3D Human Pose
Estimation with Cross-View and Temporal Cues," *arXiv:2404.14634*, 2024.
[Online]. Available: https://arxiv.org/abs/2404.14634

```bibtex
@misc{davoodnia2024upose3d,
  title         = {UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues},
  author        = {Davoodnia, Vandad and Ghorbani, Saeed and Carbonneau, Marc-Andr\'e and Messier, Alexandre and Etemad, Ali},
  year          = {2024},
  eprint        = {2404.14634},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2404.14634}
}
```

- **Por qué:** un "pose compiler" refina las detecciones 2D combinando
  información temporal Y entre cámaras, usando incertidumbre explícita para
  ser robusto a outliers — no requiere anotación 3D de entrenamiento.
  Referencia directa si la solución termina siendo agregar una señal
  temporal (opción (c) de la pregunta de abajo).

---

## 5. Nogueira, Oliveira & Teixeira (2024) — survey

**Cita (IEEE):** [5] A. F. R. Nogueira, H. P. Oliveira, and L. F. Teixeira,
"Markerless Multi-view 3D Human Pose Estimation: a survey,"
*arXiv:2407.03817*, 2024. [Online]. Available: https://arxiv.org/abs/2407.03817

```bibtex
@misc{nogueira2024markerless,
  title         = {Markerless Multi-view 3D Human Pose Estimation: a survey},
  author        = {Nogueira, Ana Filipa Rodrigues and Oliveira, H\'elder P. and Teixeira, Lu\'is F.},
  year          = {2024},
  eprint        = {2407.03817},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2407.03817}
}
```

- **Por qué:** mapa general del área — clasifica métodos en geométricos
  puramente supervisados vs. los que incorporan consistencia temporal,
  profundidad o extracción de features 3D. Punto de partida para tener
  vocabulario correcto y no reinventar taxonomía en el manuscrito.

---

## Pregunta abierta

¿La corrección al problema observado (cobertura de landmarks inconsistente
entre lados del cuerpo y entre tomas) debería ir en:

- **(a)** los parámetros actuales (`max_reproj_px`, `min_cameras`),
- **(b)** el criterio de rechazo — pesos por consistencia geométrica en vez
  de un umbral duro de RMS (papers 2 y 3),
- **(c)** agregar una señal temporal entre frames (papers 3 y 4), o
- **(d)** tratar la ambigüedad izquierda/derecha en la fuente 2D, antes de
  triangular (paper 1)?
