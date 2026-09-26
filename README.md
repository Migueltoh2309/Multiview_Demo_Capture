# Multiview Demo Capture

Sistema de **captura markerless de demostraciones humanas con 3 cámaras RGB**
para aprendizaje por demostración (LfD). Reconstruye en 3D el movimiento de
ambos brazos de una persona (hombros, codos, muñecas) mientras realiza una
tarea de picking, para luego transferirlo a un robot humanoide **Unitree H1-2**.

Forma parte de mi tesis de maestría en UTEC: *picking bimanual de mandarinas
sobre faja transportadora con un humanoide de base fija*. El control y la
simulación del robot están en [H1_2_MiTo](https://github.com/Migueltoh2309/H1_2_MiTo).

## Cómo funciona

```text
 3 cámaras RGB (abanico 0° / ±50°)
        │  captura sincronizada (timestamps monótonos)
        ▼
 Calibración ChArUco ── intrínseca (K, D) por cámara
        │               extrínseca (pose de cada cámara en el frame {W})
        ▼
 MediaPipe Pose ─────── keypoints 2D por cámara
        ▼
 Triangulación N-vistas (DLT + refinamiento, rechazo por error de reproyección)
        ▼
 Trayectorias 3D de ambos brazos en {W}  ──►  dataset de demostraciones
```

- **Sin marcadores:** solo cámaras RGB comerciales (Logitech BRIO) y pose 2D con MediaPipe.
- **Multivista genérico:** el número de cámaras sale de `config/cameras.yaml`;
  toda la geometría funciona con N vistas y tolera oclusiones (basta con 2 cámaras por punto).
- **Calibración con checkpoints:** cada etapa tiene un criterio numérico de
  aceptación y el script correspondiente devuelve error si no se cumple.

Incluye además un subsistema de **visión RGB-D** (Intel RealSense D435i) que
detecta la fruta, estima su centro 3D ajustando una esfera a la nube de puntos y
lo expresa en el mismo frame `{W}` que las demostraciones.

## Resultados

Sobre el rig real de 3 cámaras, con un tablero ChArUco A3:

| Métrica | Resultado |
|---|---|
| Error de reproyección intrínseco (RMS) | 0.67 – 0.70 px por cámara |
| Error medio de distancia entre esquinas | 1.13 mm |
| Distancia conocida de 176 mm reconstruida | 1.08 % de error |

La cadena de captura se verificó a 30 FPS en 1280×720 sin frames perdidos (prueba
con una cámara). En escenas sintéticas con verdad de terreno (`pytest`), la
localización de la fruta con RGB-D tiene un error de ~3 mm entre 0.4 y 1.2 m.

## Estructura

```text
├── config/                  # Entrada, editada a mano: cámaras, ChArUco, frame {W}
├── calib/                   # Salida de calibración: camN.yaml (K, D, T_world_cam)
├── src/tesis_picking/
│   ├── common/              # Capa geométrica compartida: SE(3), modelo de cámara,
│   │                        #   triangulación, timebase, carga de config
│   ├── demostraciones_humanas/
│   │   ├── capture/         # Captura multicámara sincronizada y grabación
│   │   ├── calibration/     # ChArUco, intrínseca, extrínseca, validación metrológica
│   │   └── pose/            # MediaPipe Pose + reconstrucción 3D del esqueleto
│   ├── vision/              # RealSense: detección, localización 3D, cambio de frame
│   ├── control/             # (pendiente)
│   └── aprendizaje/         # (pendiente)
├── scripts/                 # CLIs numerados en el orden del flujo de trabajo
├── results/metricas/        # Reportes JSON de calibración y validación
├── docs/                    # Documentación técnica y tableros ChArUco para imprimir
└── tests/                   # Tests sintéticos de la capa geométrica y de visión
```

Los videos y datasets (`data/`) no se versionan.

## Instalación

Probado en Ubuntu 22.04 con Python 3.10.

```bash
git clone https://github.com/Migueltoh2309/Multiview_Demo_Capture.git
cd Multiview_Demo_Capture
python3 -m venv .venv
source env.sh          # activa el venv y limpia PYTHONPATH (evita conflictos con ROS)
pip install -e ".[dev]"
pytest -q
```

## Uso

Flujo de captura y calibración, en orden:

| # | Script | Qué hace |
|---|---|---|
| 1 | `01_probe_cameras.py` | Detecta las cámaras y verifica resolución y FPS |
| 2 | `02_preview_camera.py` | Vista previa para encuadrar físicamente cada cámara |
| 3 | `03_record_session.py --name <sesion>` | Graba una sesión sincronizada |
| 4 | `04_make_charuco.py` | Genera el tablero ChArUco para imprimir |
| 5 | `05_capture_charuco.py --set <set>` | Captura tomas del tablero |
| 6 | `06_calibrate_intrinsics.py --set <set>` | Calibración intrínseca (K, D) |
| 7 | `07_calibrate_extrinsics.py --set <set>` | Calibración extrínseca (pose en `{W}`) |
| 8 | `08_validate_geometry.py --set <set>` | Validación metrológica en mm |
| 9 | `09_pose_triangulate.py` | MediaPipe + triangulación en vivo, graba trayectorias 3D |

Visión RGB-D (`scripts/vision/`): `10_check_realsense.py` → `12_ajustar_color.py`
→ `13_detectar_y_localizar.py` (también funciona con `--sintetico`, sin
hardware) → `14_extrinsecos_camara.py` → `15_exactitud_profundidad.py`.

## Documentación

- [`docs/calibracion_matematica.md`](docs/calibracion_matematica.md): modelo pinhole, distorsión, error de reproyección y SE(3)
- [`docs/disposicion_camaras.md`](docs/disposicion_camaras.md): por qué 3 cámaras en abanico a ±50° y a la misma altura
- [`docs/papers/`](docs/papers): referencias sobre fusión multivista robusta
- [`MIGRACION.md`](MIGRACION.md): cómo mover el proyecto a otra máquina

## Estado

- **Captura y calibración:** implementadas y validadas con hardware real.
- **Pose 3D (MediaPipe + triangulación):** primeras pruebas reales hechas; se
  está trabajando en la robustez frente a la confusión izquierda/derecha.
- **Visión RGB-D:** implementada y validada en simulación.
- **Control y aprendizaje:** pendientes.

## Autor

Miguel Olortegui — Maestría, UTEC
