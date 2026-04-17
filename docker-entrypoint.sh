#!/bin/bash
set -e

# ── LD_PRELOAD: libcuda.so.1 del driver Jetson ───────────────────────────────
# vllm._C.abi3.so llama cuPointerGetAttribute (CUDA Driver API) pero libcuda.so.1
# no es dependencia directa del .so → no se carga automáticamente al hacer dlopen.
# Con LD_PRELOAD se carga primero y sus símbolos quedan disponibles globalmente.
LIBCUDA_NVIDIA=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1
if [ -f "$LIBCUDA_NVIDIA" ]; then
    export LD_PRELOAD="${LIBCUDA_NVIDIA}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# ── LD_LIBRARY_PATH ──────────────────────────────────────────────────────────
# 1. Libs de torch (libtorch.so, libtorch_cuda.so, etc.) — primero para que
#    el torch de Jetson (sm_87, CUDA) tenga prioridad sobre el CUDA compat.
TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')" 2>/dev/null || echo "")
if [ -n "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

# 2. Libs CUDA del sistema
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH:-}"

# 3. Libs del paquete nvidia.cu12 (pip) si está disponible
NVIDIA_CU12_LIB=$(python3 -c "import nvidia.cu12; print(nvidia.cu12.__path__[0])" 2>/dev/null || echo "")
if [ -n "$NVIDIA_CU12_LIB" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_CU12_LIB}/lib:${LD_LIBRARY_PATH}"
fi

# ── pip install opcional antes de arrancar ────────────────────────────────────
# Añade paquetes extra sin reconstruir la imagen.
# Uso en docker-compose.yml:
#   environment:
#     - PIP_INSTALL=paquete1 paquete2==1.2.3
if [ -n "${PIP_INSTALL:-}" ]; then
    echo "[entrypoint] pip install ${PIP_INSTALL}"
    pip install --quiet ${PIP_INSTALL}
fi

# Si el primer argumento empieza por '-' o es '--help', lanzar vllm serve
if [ "${1#-}" != "$1" ] || [ "$1" = "--help" ]; then
    exec vllm serve "$@"
fi

# Si se pasa un modelo directamente, lanzar serve con él
exec vllm serve "$@"
