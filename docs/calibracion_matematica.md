# Matemática de la calibración

Lo mínimo para entender qué calcula `06_calibrate_intrinsics.py` (intrínseca)
y `07_calibrate_extrinsics.py` (extrínseca).

### 1. Modelo pinhole y la matriz intrínseca `K`

Un punto 3D en el frame de la cámara se proyecta a píxel así:

\[
\lambda
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
=
K
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
\qquad
K =
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\]

`fx, fy` (distancia focal en píxeles) y `cx, cy` (centro óptico) son
exactamente los 4 números que imprime el script para cada cámara.

### 2. Distorsión de lente

La lente real no es un pinhole ideal. El modelo (Brown-Conrady, el que usa
OpenCV) corrige la posición normalizada `(x, y)` antes de aplicar `K`:

\[
x_d = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1 xy + p_2(r^2 + 2x^2)
\]
\[
y_d = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1(r^2 + 2y^2) + 2p_2 xy
\]

donde \( r^2 = x^2+y^2 \) (distancia al centro óptico). `k1,k2,k3` son
distorsión **radial**, `p1,p2` **tangencial**. Estos 5 números son el vector
`distortion` que también imprime el script.

**Por qué importa la cobertura de bordes (`edge_point_ratio`):** los términos
radiales escalan con \(r^2, r^4, r^6\) — cerca del centro (\(r\approx0\)) su
efecto es casi cero, así que un dataset de tomas centradas no tiene señal
para estimar `k1,k2,k3`: el ajuste converge igual (RMS bajo) pero con una `D`
que no representa la lente real. Por eso el script exige tomas en el 20% exterior del encuadre.

### 3. Error de reproyección (la métrica `rms_reprojection_px`)

Para cada esquina detectada \( \mathbf{u}_i \) y el punto 3D correspondiente
del tablero \( \mathbf{X}_i \) (conocido, porque el tablero es un patrón
impreso de dimensiones fijas), se reproyecta con la calibración estimada:

\[
\hat{\mathbf{u}}_i = \pi\big(K, D, R, t, \mathbf{X}_i\big)
\]

y se compara contra lo observado:

\[
e_i = \lVert \mathbf{u}_i - \hat{\mathbf{u}}_i \rVert
\qquad
\text{RMS} = \sqrt{\frac{1}{N}\sum_i e_i^2}
\]

Ese es el número que el script exige `< ~0.5-0.7 px` para aceptar la
calibración (checkpoint 3).

### 4. Por qué el tablero debe ser rígido y plano

El método de calibración (Zhang, 2000 — el que implementa
`cv2.calibrateCamera`) asume que **todos los puntos del tablero, en cada
toma, están exactamente sobre un plano** (\(Z=0\) en el frame del tablero).
Si la hoja se curva, cada esquina tiene un pequeño desplazamiento fuera de
ese plano que el modelo no puede distinguir de error de lente o de pose —
se cuela como sesgo en `K` y `D`. Medido en este proyecto: papel
suelto → RMS 0.78 px con solo 2/23 tomas limpias; mismo tablero, montado
rígido → RMS 0.51 px con 38/40 limpias (cam2).

### 5. La optimización en sí

`cv2.calibrateCamera` primero obtiene una solución cerrada por cada vista
(una homografía tablero→imagen, porque el tablero es plano), y después
refina **todo junto** (`K`, `D`, y la pose `R_i, t_i` de cada vista) con
mínimos cuadrados no lineales (Levenberg-Marquardt), minimizando la suma de
\(e_i^2\) de la sección 3 sobre todas las vistas y todas las esquinas a la
vez.

### 6. Extrínseca: transformaciones rígidas SE(3)

La extrínseca no estima un solo número por cámara — estima **dónde está cada
cámara respecto a las otras**, como una transformación rígida:

\[
T =
\begin{bmatrix} R & t \\ \mathbf{0}^\top & 1 \end{bmatrix}
\in SE(3),
\qquad R \in SO(3)\ (\text{ortogonal}, \det R = 1),\ \ t \in \mathbb{R}^3
\]

Convención `T_a_b` (la que usa `common/transforms.py`): transforma
coordenadas del frame `b` al frame `a`, \( \mathbf{X}_a = T_{a\_b}\, \mathbf{X}_b \).

**Cómo se ata una cámara a otra sin medir nada con una regla:** en cada toma
donde una cámara ve el tablero, `solvePnP` da \( T_{\text{cam\_board}} \)
(pose del tablero respecto a esa cámara). Si dos cámaras ven el **mismo**
tablero en el **mismo instante** (la toma compartida — por eso hace falta la
sesión con las 3 conectadas a la vez):

\[
T_{A\_B} = T_{A\_board} \cdot T_{board\_B} = T_{A\_board} \cdot T_{B\_board}^{-1}
\]

**El origen `{W}`** lo define la toma marcada como ancla: \( T_{W\_board} \)
para esa toma (por convención, normalmente la identidad si el tablero se
apoya alineado con `{W}`). Cualquier cámara que haya visto esa toma queda en
`{W}` por \( T_{W\_C} = T_{W\_board} \cdot T_{C\_board}^{-1} \).

**Encadenamiento:** una cámara que *nunca* vio la toma ancla igual puede
quedar referida a `{W}` si comparte *alguna otra* toma con una cámara que sí
está atada — componiendo transformaciones: \( T_{W\_D} = T_{W\_C}\cdot T_{C\_D} \).
Esto es un problema de conectividad de grafo (cámaras = nodos, tomas
compartidas = aristas), y es exactamente lo que ya cubre el test de
`common/transforms.py` (`tests/test_extrinsics.py`). También
es la razón matemática de por qué la calibración intrínseca se pudo hacer
cámara por cámara pero la extrínseca **no**: sin una toma compartida no hay
arista en el grafo, sin importar cuán buena sea la intrínseca de cada una.
