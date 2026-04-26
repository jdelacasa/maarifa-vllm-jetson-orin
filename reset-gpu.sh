#!/bin/bash
# reset-gpu.sh — Resetea el estado GPU del Jetson Orin sin reboot completo.
#
# Ejecutar desde el HOST (no desde dentro del container) cuando vLLM/ComfyUI
# falla al reiniciar con: NVML_SUCCESS == r INTERNAL ASSERT FAILED
#
# Uso:
#   sudo ./reset-gpu.sh
#   docker compose up -d    # ya puede reiniciar OK

set -euo pipefail

RAILGATE="/sys/devices/platform/bus@0/17000000.gpu/railgate_enable"
COMPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="jdelacasa/vllm-jetson:0.19.1-torch2.10.0-cu126-l4tr36.4-sm87"

echo "[reset-gpu] Parando contenedores GPU..."
docker compose -f "${COMPOSE_DIR}/docker-compose.yml" down 2>/dev/null || true
docker stop $(docker ps -q --filter "ancestor=dustynv/comfyui") 2>/dev/null || true

echo "[reset-gpu] Esperando que los procesos liberen /dev/nvhost-gpu..."
sleep 3

GPU_PROCS=$(lsof /dev/nvhost-gpu /dev/nvhost-as-gpu 2>/dev/null | awk 'NR>1 {print $2}' | sort -u || true)
if [ -n "$GPU_PROCS" ]; then
    echo "[reset-gpu] ADVERTENCIA: procesos aun con GPU abierta: $GPU_PROCS — forzando kill..."
    kill -9 $GPU_PROCS 2>/dev/null || true
    sleep 2
fi

echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

# Ciclo railgate: habilitar (GPU puede apagarse) → esperar → deshabilitar (GPU siempre ON).
# Esto fuerza que el driver nvhost limpie el estado del contexto anterior.
# El estado final es 0 (railgate desactivado), que es el estado correcto para produccion.
if [ -f "$RAILGATE" ]; then
    echo "[reset-gpu] Railgate: habilitando para forzar power-cycle..."
    echo 1 > "$RAILGATE"
    sleep 3
    echo "[reset-gpu] Railgate: deshabilitando (GPU siempre ON)..."
    echo 0 > "$RAILGATE"
    sleep 2
    echo "[reset-gpu] Railgate estado final: $(cat $RAILGATE)  (0=desactivado, correcto)"
else
    echo "[reset-gpu] ADVERTENCIA: railgate no encontrado en $RAILGATE"
fi

echo "[reset-gpu] Verificando GPU..."
# --entrypoint python3 necesario para no pasar por 'vllm serve' del ENTRYPOINT
if docker run --rm --runtime=nvidia --entrypoint python3 "$IMAGE" \
    -c "import torch; free,total=torch.cuda.mem_get_info(0); print(f'GPU OK: {torch.cuda.get_device_name(0)}, {free/1e9:.1f} GB libres / {total/1e9:.1f} GB total')"; then
    echo "[reset-gpu] GPU lista."
else
    echo "[reset-gpu] GPU aun no responde. Puede requerirse reboot."
    exit 1
fi

echo ""
echo "[reset-gpu] Listo. Ahora puedes ejecutar:"
echo "  docker compose up -d"
