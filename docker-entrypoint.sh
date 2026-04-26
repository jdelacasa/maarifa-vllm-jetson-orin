#!/bin/bash
set -e

# ── LD_LIBRARY_PATH ──────────────────────────────────────────────────────────
TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')" 2>/dev/null || echo "")
if [ -n "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH:-}"
NVIDIA_CU12_LIB=$(python3 -c "import nvidia.cu12; print(nvidia.cu12.__path__[0])" 2>/dev/null || echo "")
if [ -n "$NVIDIA_CU12_LIB" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_CU12_LIB}/lib:${LD_LIBRARY_PATH}"
fi

# ── pip install opcional ──────────────────────────────────────────────────────
if [ -n "${PIP_INSTALL:-}" ]; then
    echo "[entrypoint] pip install ${PIP_INSTALL}"
    pip install --quiet ${PIP_INSTALL}
fi

# ── Parche torchvision._meta_registrations — vaciar para evitar register_fake ─
# torch 2.10 lanza RuntimeError en _dispatch_has_kernel_for_dispatch_key cuando
# el operador aún no está registrado (se registra después, al cargar _torchvision.so).
# _meta_registrations.py solo añade implementaciones "fake/meta" para torch.compile
# con tensores meta — no necesarias para inferencia con CUDA real.
# Vaciarlo es seguro: las ops reales siguen disponibles vía _torchvision.so.
# Este parche va DESPUÉS de pip install para sobrescribir el fichero recién instalado.
python3 - <<'PYEOF'
import os
path = '/opt/venv/lib/python3.10/site-packages/torchvision/_meta_registrations.py'
if not os.path.exists(path):
    print('[entrypoint] _meta_registrations.py: torchvision no instalado, skip')
    exit()
with open(path) as f:
    first = f.read(32)
if first.startswith('# noop'):
    print('[entrypoint] _meta_registrations.py: ya no-op')
else:
    with open(path, 'w') as f:
        f.write('# noop: register_fake disabled; ops load via _torchvision.so\n')
    pyc_dir = os.path.join(os.path.dirname(path), '__pycache__')
    for fn in os.listdir(pyc_dir) if os.path.isdir(pyc_dir) else []:
        if '_meta_registrations' in fn:
            os.remove(os.path.join(pyc_dir, fn))
    print('[entrypoint] _meta_registrations.py: convertido a no-op')
PYEOF

# ── Parche torch standalone_compile.py — fix UnboundLocalError _sc_args0 ──────
# Bug en torch._inductor: _sc_args0 se usa antes de asignarse en ciertos
# grafos MoE. El parche inicializa a [] si no fue asignado, lo que equivale
# a "no hay argumentos en este fragmento" — comportamiento correcto.
python3 - <<'PYEOF'
import sys
path = '/opt/venv/lib/python3.10/site-packages/torch/_inductor/standalone_compile.py'
marker = 'for node in (_sc_args0 if isinstance(_sc_args0, (list, tuple)) else [_sc_args0]):'
already = 'vllm-jetson patch _sc_args0'
try:
    with open(path) as f:
        lines = f.readlines()
    if any(already in l for l in lines):
        print('[entrypoint] standalone_compile.py: ya parchado')
        sys.exit(0)
    patched = False
    for i, line in enumerate(lines):
        if marker in line:
            sp = ' ' * (len(line) - len(line.lstrip()))
            patch = [
                f'{sp}try:  # {already}\n',
                f'{sp}    _sc_args0\n',
                f'{sp}except (UnboundLocalError, NameError):\n',
                f'{sp}    _sc_args0 = []\n',
            ]
            lines[i:i] = patch
            patched = True
            break
    if patched:
        with open(path, 'w') as f:
            f.writelines(lines)
        import glob, os
        for pyc in glob.glob(path.replace('standalone_compile.py',
                '__pycache__/standalone_compile*.pyc')):
            os.remove(pyc)
        print('[entrypoint] standalone_compile.py: parchado _sc_args0')
    else:
        print('[entrypoint] standalone_compile.py: marcador no encontrado, skip')
except Exception as e:
    print(f'[entrypoint] standalone_compile.py: error al parchear: {e}')
PYEOF

# ── CUDA pre-flight: detecta estado corrupto antes de cargar el modelo ────────
# En Jetson (GPU integrada), un SIGKILL previo puede dejar estado NVML corrupto
# en el driver nvhost del kernel. El fallo se manifiesta como NVML_SUCCESS==r en
# CUDACachingAllocator.cpp durante la allocación del KV cache (~15 min después).
# Este test lo detecta en segundos forzando la misma llamada cudaMemGetInfo.
python3 - <<'PYEOF'
import sys
import torch

print('[entrypoint] CUDA pre-flight...', flush=True)

if not torch.cuda.is_available():
    print('[entrypoint] FATAL: CUDA no disponible', flush=True)
    sys.exit(1)

try:
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info(0)
    print(f'[entrypoint] GPU: {torch.cuda.get_device_name(0)} | '
          f'{free/1024**3:.1f} GB libres / {total/1024**3:.1f} GB total', flush=True)

    # 512 MiB fuerza el mismo codigo path (cudaMemGetInfo en CUDACachingAllocator)
    # que fallara durante la allocacion del KV cache si la GPU esta en mal estado.
    probe = torch.zeros(512 * 1024 * 1024 // 2, dtype=torch.float16, device='cuda')
    del probe
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('[entrypoint] CUDA pre-flight OK', flush=True)

except RuntimeError as e:
    msg = str(e)
    if 'NVML' in msg or 'INTERNAL ASSERT' in msg or 'CUDA' in msg:
        print(f'[entrypoint] FATAL: GPU estado corrupto (run anterior termino con SIGKILL): {msg}',
              flush=True)
        print('[entrypoint] Fix rapido:  sudo reboot', flush=True)
        print('[entrypoint] Fix sin reboot: ver leeme.txt o ejecutar reset-gpu.sh en el host',
              flush=True)
        sys.exit(1)
    raise
PYEOF

# ── Arrancar vLLM ─────────────────────────────────────────────────────────────
exec vllm serve "$@"
