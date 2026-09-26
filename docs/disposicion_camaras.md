# Disposición física de las cámaras

Abanico de 3 cámaras (no una fila, y sin llegar a 90°) apuntando todas al mismo punto
sobre la faja —condición para poder triangular—, con la cámara frontal a 0° de azimut y las dos
laterales a ≈50°. El ángulo lateral de 50° (no 90°) se eligió porque la autooclusión y la
ambigüedad izquierda/derecha degradan la detección 2D de MediaPipe Pose antes de que la
triangulación entre en juego; 50° queda dentro del margen de ±60° respecto al frente del sujeto
que usa OpenCap para la misma arquitectura (cámaras RGB + pose 2D + triangulación), con colchón de
seguridad.

Las tres cámaras van a la **misma altura** (1.45 m, por encima del borde de la mesa) — no hace
falta variar la altura entre ellas: lo que evita que los rayos triangulen mal es que no queden casi
paralelos, y eso ya lo da la diferencia de distancia horizontal al punto de interés (cámara frontal
más lejos, laterales más cerca), que por sí sola produce inclinaciones distintas (≈25° la frontal,
≈28° las laterales) sin tocar la altura. Una sola altura de montaje para las tres simplifica el
armado del rig.

Referencias usadas para justificar el ángulo:

- Uhlrich, S. D., Falisse, A., Kidziński, Ł., et al. (2023). *OpenCap: Human movement dynamics
  from smartphone videos*. PLOS Computational Biology, 19(10), e1011462.
  <https://doi.org/10.1371/journal.pcbi.1011462> — origen de la restricción de ±60° respecto al
  frente del sujeto en la colocación de cámaras para no degradar la red de pose 2D.
- Bragagnolo, L., Terreran, M., Allegro, D., & Ghidoni, S. (2024). *Multi-view Pose Fusion for
  Occlusion-Aware 3D Human Pose Estimation*. arXiv:2408.15810. <https://arxiv.org/abs/2408.15810>
  — la autooclusión como fuente principal de ambigüedad en pose 3D multivista, y por qué el
  multivista solo la resuelve si cada cámara sigue dando un keypoint 2D confiable.
- Literatura de fotogrametría/calibración estéreo sobre ángulo de convergencia: el error de
  triangulación crece cuando los rayos de dos cámaras se acercan al paralelismo (0°/180°), y un
  ángulo menor a 90° suele triangular mejor que uno exactamente perpendicular — la ganancia de
  ángulo lateral tiene rendimientos decrecientes frente a la pérdida de fiabilidad del detector 2D
  en vistas muy laterales. Ver p. ej. *On the accuracy of stereo-plotting of convergent aerial
  photographs* y *Uncertainty, Baseline, and Noise Analysis for L1 Error-Based Multi-View
  Triangulation* (<https://www.researchgate.net/publication/261511328>).
