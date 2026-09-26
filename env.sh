#!/usr/bin/env bash
# Activa el entorno del proyecto.  Uso:  source env.sh
#
# PYTHONPATH se limpia a proposito: /opt/ros/humble/... se antepone al
# site-packages del venv y filtra paquetes de ROS (incluido un numpy antiguo)
# dentro del entorno. Este proyecto NO usa ROS todavia.
_here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
unset PYTHONPATH
source "${_here}/.venv/bin/activate"
export TESIS_ROOT="${_here}"
echo "entorno tesis activo: $(python --version) en ${_here}"
