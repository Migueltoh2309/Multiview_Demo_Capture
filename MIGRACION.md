# Migrar el proyecto a otra máquina

## Qué se copia y qué NO

**Copiar todo el directorio `~/Maestria_tesis`, excepto `.venv/`.**

El `.venv/` lleva rutas absolutas grabadas dentro (la instalación editable del
paquete apunta a `/home/utec/Maestria_tesis/src`) y binarios compilados para
este CPU. Copiarlo produce fallos difíciles de leer. Se reconstruye en 2 minutos.

Todo lo demás es portable: `paths.ROOT` se calcula a partir de `__file__`, así
que el proyecto funciona desde cualquier ruta, con cualquier usuario.

## Pasos

```bash
# --- en la laptop vieja ---
cd ~
tar --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
    --exclude='*.egg-info' -czf maestria_tesis.tar.gz Maestria_tesis/

# --- en la laptop nueva ---
tar -xzf maestria_tesis.tar.gz
cd ~/Maestria_tesis
python3 -m venv .venv
source env.sh                 # hace `unset PYTHONPATH` y activa el venv
pip install -e ".[dev]"
pytest -q                     # deben pasar 23 tests
```

Si `pytest` pasa, la migración está completa: la capa geométrica, la de visión y
toda la cadena de localización quedan verificadas en la máquina nueva.

## Revisar después de migrar

### 1. `env.sh` y ROS

`env.sh` hace `unset PYTHONPATH` porque en esta máquina `/opt/ros/humble/...`
se antepone al `site-packages` del venv y filtra un numpy antiguo que rompe
pandas. **Si la laptop nueva no tiene ROS, la línea es inofensiva; déjala.**
Si tiene ROS y en el futuro se necesita, se quita entonces — no antes.

### 2. Puertos USB 3 (lo primero que hay que hacer)

El mapeo de conectores es **específico de cada equipo**. Identificar cuál es el
USB 3 en la máquina nueva:

```bash
for p in /sys/bus/usb/devices/usb*/*-0:1.0/*-port*; do
  [ -L "$p/peer" ] && echo "$(basename $p) -> $(basename $(readlink -f $p/peer))"
done
```

Los puertos que aparecen **emparejados** (`peer`) son los USB 3. Los que no
tienen pareja son USB 2 por hardware y ningún cable los cambia.

Luego, con la RealSense conectada:

```bash
bash scripts/vision/09_usb_realsense.sh
```

Debe decir `USB 3 OK` y un puerto del tipo `4-1`, no `3-N`.

### 3. `config/cameras.yaml` (solo para demostraciones humanas)

Los `/dev/videoN` cambian de una máquina a otra. Redescubrirlos con:

```bash
python scripts/01_probe_cameras.py --scan
```

y preferir rutas `/dev/v4l/by-id/...`, que son estables ante reinicios.

No afecta al subsistema de visión: la RealSense se localiza por serial, no por
ruta de dispositivo.

### 4. Calibraciones existentes en `calib/`

- **Intrínsecos de la RealSense**: son de fábrica y viajan con la cámara, no con
  el ordenador. Siguen siendo válidos.
- **Extrínsecos (`T_world_cam`)**: describen dónde está la cámara. Se invalidan
  en cuanto se mueva el montaje, que es lo que va a pasar. Rehacer con
  `scripts/vision/14_extrinsecos_camara.py` una vez montada en su sitio.

Ahora mismo `calib/` está vacío, así que no hay nada que perder.

## Aprovechar la GPU en la máquina nueva

Nada de lo que hay hoy usa GPU y no hace falta tocarlo. Importa a partir de la
etapa **V9 (YOLO)** del roadmap de visión: ahí sí conviene instalar PyTorch con
CUDA. Cuando llegue el momento, comprobar primero:

```bash
nvidia-smi                                  # driver y versión de CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

El resto del pipeline —detección por color, localización 3D, calibración— corre
en CPU de sobra a 30 FPS.

## Estado al momento de migrar (2026-08-26)

| | estado |
|---|---|
| Demostraciones humanas, etapas 1–7 | código completo, 13 tests |
| Visión, etapas V0–V5 | código completo, 10 tests |
| Validación en hardware | pendiente: la RealSense nunca llegó a transmitir |
| `calib/` | vacío |
| `data/` | vacío |

Lo pendiente y por qué está en `docs/roadmap_vision.md` y en
`docs/bitacora/2026-08-26_realsense_d435i.md`.
