"""Etapas 8-13: MediaPipe Pose por camara y triangulacion del skeleton 3D.

Implementado 2026-09-23, tras superar el checkpoint 6 (distancia conocida
reconstruida, 2026-09-22) -- condicion que se habia puesto para no confundir
error de pose con error de calibracion. La extrinseca del rig de practica
sigue sin pasar el checkpoint 4 limpio (ver docs/bitacora/); por eso
`skeleton.triangulate_skeleton` usa un `max_reproj_px` mas permisivo que el
default de `common.triangulation`, documentado ahi mismo.

Landmarks del alcance inicial (seccion 19 del roadmap):
    11 LEFT_SHOULDER   13 LEFT_ELBOW    15 LEFT_WRIST
    12 RIGHT_SHOULDER  14 RIGHT_ELBOW   16 RIGHT_WRIST
    23 LEFT_HIP        24 RIGHT_HIP     (para el frame corporal)

Ver `detector.PoseDetector` (etapa 8) y `skeleton.triangulate_skeleton`
(etapas 9-13).
"""
