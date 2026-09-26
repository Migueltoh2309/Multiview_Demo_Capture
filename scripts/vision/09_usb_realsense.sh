#!/usr/bin/env bash
# Estado del enlace USB de la RealSense D435i.
#
# En un laptop, cada puerto fisico esta cableado a DOS controladoras: la USB 2
# y la SuperSpeed. Al conectar algo, el enlace negocia: si el dispositivo Y EL
# CABLE soportan SuperSpeed, el aparato aparece en el bus de 5000/10000 Mbps;
# si no, cae al bus de 480 Mbps. La D435i en el bus de 480 no transmite.
#
# Uso:
#   bash scripts/vision/09_usb_realsense.sh        una lectura
#   bash scripts/vision/09_usb_realsense.sh -w     en bucle, para ir probando
#                                                  cables y puertos en vivo
DEV=/sys/bus/usb/devices

buses() {
  echo "--- controladoras del equipo ---"
  local h base spd n
  for h in "$DEV"/usb*; do
    base=$(basename "$h")
    spd=$(cat "$h/speed" 2>/dev/null) || continue
    # Dispositivos reales de este bus: rutas <bus>-<puerto> sin ':' (las que
    # llevan ':' son interfaces, no aparatos).
    n=$(ls -d "$DEV"/"${base#usb}"-[0-9]* 2>/dev/null | grep -vc ':')
    if [ "$spd" -ge 5000 ] 2>/dev/null; then
      printf '  %-6s \033[36m%6s Mbps  SuperSpeed\033[0m  %2s dispositivos\n' \
             "$base" "$spd" "$n"
    else
      printf '  %-6s %6s Mbps  USB 2      %2s dispositivos\n' "$base" "$spd" "$n"
    fi
  done
}

realsense() {
  local d spd ver encontrada=0
  for d in "$DEV"/*/; do
    [ -f "$d/idVendor" ] || continue
    [ "$(cat "$d/idVendor")" = "8086" ] || continue
    [ "$(cat "$d/idProduct" 2>/dev/null)" = "0b3a" ] || continue
    encontrada=1
    spd=$(cat "$d/speed"); ver=$(tr -d ' ' < "$d/version")
    printf '%s  ' "$(date +%H:%M:%S)"
    if [ "$spd" -ge 5000 ] 2>/dev/null; then
      printf '\033[32mUSB 3 OK\033[0m  %5s Mbps (USB %s)  puerto %s\n' \
             "$spd" "$ver" "$(basename "$d")"
      printf '            siguiente: python scripts/vision/10_check_realsense.py\n'
    else
      printf '\033[31mUSB 2\033[0m     %5s Mbps (USB %s)  puerto %s\n' \
             "$spd" "$ver" "$(basename "$d")"
      printf '            probar el cable USB 3 de fabrica y un puerto azul/SS\n'
    fi
  done
  [ "$encontrada" -eq 0 ] && printf '%s  \033[33mno se detecta la RealSense\033[0m\n' "$(date +%H:%M:%S)"
  return 0
}

buses
echo "--- enlace de la RealSense ---"
if [ "$1" = "-w" ]; then
  echo "(bucle; Ctrl-C para salir. Cambiar cable o puerto y observar)"
  while true; do realsense; sleep 2; done
else
  realsense
fi
