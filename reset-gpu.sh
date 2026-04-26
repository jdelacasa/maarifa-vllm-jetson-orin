#!/bin/bash
# reset-gpu.sh — Resetea el estado GPU del Jetson Orin sin reboot completo.
#
# Ejecutar desde el HOST (no desde dentro del container) cuando vLLM/ComfyUI
# falla al reiniciar con: NVML_SUCCESS == r INTERNAL ASSERT FAILED
#
# Causa: un SIGKILL previo deja estado corrupto en el driver nvhost del kernel.
# Este script fuerza la liberacion de ese estado.
#
# Uso:
#   sudo ./reset-gpu.sh
#   docker compose --profile agents up -d    # ya puede reiniciar OK

set -euo pipefail

RAILGATE="/sys/devices/platform/bus@0/17000000.gpu/railgate_enable"
COMPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[reset-gpu] Parando contenedores GPU..."
docker compose -f "${COMPOSE_DIR}/docker-compose.yml" --profile agents down 2>/dev/null || true
# Otros contenedores con GPU (ComfyUI, etc.)
docker stop $(docker ps -q --filter "label=com.nvidia.volumes.needed") 2>/dev/null || true

echo "[reset-gpu] Esperando que los procesos liberen /dev/nvhost-gpu..."
sleep 3

# Verificar que no queden procesos usando la GPU
GPU_PROCS=$(lsof /dev/nvhost-gpu /dev/nvhost-as-gpu 2>/dev/null | awk 'NR>1 {print $2}' | sort -u || true)
if [ -n "$GPU_PROCS" ]; then
    echo "[reset-gpu] ADVERTENCIA: procesos aun con GPU abierta: $GPU_PROCS"
    echo "[reset-gpu] Forzando kill..."
    kill -9 $GPU_PROCS 2>/dev/null || true
    sleep 2
fi

# Limpiar memoria de sistema
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || \
    echo "[reset-gpu] drop_caches: sin permisos (no critico)"

# Resetear GPU via railgate: apagar y encender la GPU
# Esto fuerza que el driver nvhost limpie todo el estado del contexto anterior.
if [ -f "$RAILGATE" ]; then
    echo "[reset-gpu] Railgate OFF (power-gate GPU)..."
    echo 1 > "$RAILGATE"
    sleep 2
    echo "[reset-gpu] Railgate ON (power-up GPU)..."
    echo 1 > "$RAILGATE"
    sleep 2
    echo "[reset-gpu] Railgate estado: $(cat $RAILGATE)"
else
    echo "[reset-gpu] ADVERTENCIA: railgate no encontrado en $RAILGATE"
fi

# Verificar que la GPU responde
echo "[reset-gpu] Verificando GPU..."
if docker run --rm --runtime=nvidia \
    jdelacasa/vllm-jetson:0.19.1-torch2.10.0-cu126-l4tr36.4-sm87 \
    python3 -c "import torch; print(f'GPU OK: {torch.cuda.get_device_name(0)}, {torch.cuda.mem_get_info()[0]/1e9:.1f} GB libres')" \
    2>/dev/null; then
    echo "[reset-gpu] GPU lista."
else
    echo "[reset-gpu] GPU aun no responde. Puede requerirse reboot."
    exit 1
fi

echo "[reset-gpu] Listo. Ahora puedes ejecutar:"
echo "  docker compose --profile agents up -d"
