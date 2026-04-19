# ============================================================
# vLLM para Jetson AGX Orin — JetPack 6.2 (L4T R36.4.x)
# CUDA 12.9 · sm_87 · PyTorch 2.8.0 (Jetson AI Lab)
#
# Base: dustynv/vllm:r36.4-cu129-24.04
#   → CUDA 12.9, PyTorch 2.8.0 sm_87, triton 3.4.0, flashinfer 0.2.8,
#     xgrammar 0.1.21, bitsandbytes 0.47.0, flash-attn 2.8.0.post2
#   → Desinstalamos la versión vieja de vLLM y compilamos 0.19.0 desde fuente
#
# Parches críticos para sm_87 (JetPack 6.2.1, driver CUDA 12.6):
#   1. CMakeLists.txt — añadir 8.7+PTX a MARLIN_ARCHS y FA2_ARCHS
#      Sin esto: nvcc 12.9 genera PTX ISA 8.5 que driver 12.6 no puede JIT-compilar
#   2. Restaurar torch 2.8.0 tras build (vLLM instala 2.10.0 CPU-only de PyPI)
#   3. backends.py — autograd_cache_normalize_inputs no existe en torch 2.8.0
#   4. standalone_compile.py — args[0] es un Node, no lista (torch 2.8.0)
#
# Compilación: ~2-3 h (CPU-bound, nvcc sm_87)
# ============================================================

ARG BASE_IMAGE=dustynv/vllm:r36.4-cu129-24.04
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_BRANCH=v0.19.0
# FA commit que pina vLLM 0.19.0 (de cmake/external_projects/vllm_flash_attn.cmake)
ARG FA_COMMIT=29210221863736a08f71a866459e368ad1ac4a95

# ── Stage auxiliar: preservar torch 2.8.0 antes de que vLLM lo sobreescriba ──
FROM ${BASE_IMAGE} AS jetson-base

# ── Stage principal ───────────────────────────────────────────
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG VLLM_REPO
ARG VLLM_BRANCH
ARG FA_COMMIT

# PyPI estándar; jetson-ai-lab como extra para paquetes arm64 específicos
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu129

# ── Herramientas de compilación ───────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    gcc \
    g++ \
  && rm -rf /var/lib/apt/lists/*

# ── Desinstalar la versión vieja de vLLM del base image ──────
RUN pip uninstall vllm -y || true

# ── Clonar vLLM 0.19.0 ───────────────────────────────────────
WORKDIR /build
RUN git clone --depth 1 --branch ${VLLM_BRANCH} ${VLLM_REPO} vllm

# ── Pre-clonar flash-attn al commit exacto que pina vLLM 0.19.0 ──
# VLLM_FLASH_ATTN_SRC_DIR le indica a FetchContent usar este directorio
# en vez de descargarlo, permitiéndonos aplicar el parche sm_87 primero.
RUN git clone https://github.com/vllm-project/flash-attention.git /build/flash-attn && \
    git -C /build/flash-attn checkout ${FA_COMMIT} && \
    git -C /build/flash-attn submodule update --init --depth 1 csrc/cutlass

# ── Parche 1: CMakeLists.txt — añadir sm_87 a todas las arch lists ──
#
# vLLM 0.19.0 ya incluye 8.7 en CUDA_SUPPORTED_ARCHS (línea 97), pero
# MARLIN_ARCHS y MOE siguen usando "8.0+PTX" como limite superior.
# La FA del commit pinado sigue sin 8.7 en CUDA_SUPPORTED_ARCHS.
#
# Sin estos parches: sm_87 cae al fallback 8.0+PTX → PTX generado con
# nvcc 12.9 (ISA 8.5) → driver CUDA 12.6 no puede JIT-compilarlo (max ISA 8.4).
RUN \
    # vLLM: Marlin GPTQ kernels
    sed -i \
        's|cuda_archs_loose_intersection(MARLIN_ARCHS "8\.0+PTX"|cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX;8.7+PTX"|g' \
        /build/vllm/CMakeLists.txt && \
    sed -i \
        's|cuda_archs_loose_intersection(MARLIN_BF16_ARCHS "8\.0+PTX;9\.0+PTX"|cuda_archs_loose_intersection(MARLIN_BF16_ARCHS "8.0+PTX;8.7+PTX;9.0+PTX"|g' \
        /build/vllm/CMakeLists.txt && \
    sed -i \
        's|cuda_archs_loose_intersection(MARLIN_OTHER_ARCHS "7\.5;8\.0+PTX"|cuda_archs_loose_intersection(MARLIN_OTHER_ARCHS "7.5;8.0+PTX;8.7+PTX"|g' \
        /build/vllm/CMakeLists.txt && \
    # vLLM: Marlin MOE kernels
    sed -i \
        's|cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8\.0+PTX"|cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX;8.7+PTX"|g' \
        /build/vllm/CMakeLists.txt && \
    sed -i \
        's|cuda_archs_loose_intersection(MARLIN_MOE_OTHER_ARCHS "7\.5;8\.0+PTX"|cuda_archs_loose_intersection(MARLIN_MOE_OTHER_ARCHS "7.5;8.0+PTX;8.7+PTX"|g' \
        /build/vllm/CMakeLists.txt && \
    # flash-attn: CUDA_SUPPORTED_ARCHS no incluye 8.7 en el commit pinado
    sed -i \
        's|set(CUDA_SUPPORTED_ARCHS "8\.0;8\.6;8\.9;9\.0")|set(CUDA_SUPPORTED_ARCHS "8.0;8.6;8.7;8.9;9.0")|g' \
        /build/flash-attn/CMakeLists.txt && \
    sed -i \
        's|cuda_archs_loose_intersection(FA2_ARCHS "8\.0+PTX"|cuda_archs_loose_intersection(FA2_ARCHS "8.0+PTX;8.7+PTX"|g' \
        /build/flash-attn/CMakeLists.txt && \
    # Verificar
    echo "=== vLLM MARLIN patches ===" && \
    grep -n "8\.7" /build/vllm/CMakeLists.txt | grep -E "MARLIN|FA2" && \
    echo "=== flash-attn patches ===" && \
    grep -n "8\.7" /build/flash-attn/CMakeLists.txt

# ── Desactivar _C_stable_libtorch (requiere torch >= 2.10.0) ──
# Este target usa TORCH_BOX y torch/headeronly/ — APIs exclusivas de torch 2.10.
# Jetson Orin no tiene torch 2.10 disponible (el máximo es 2.8.0 en dustynv).
# El bloque se identifica por su comentario único y se reemplaza if(...) → if(FALSE).
RUN python3 - <<'PYEOF'
# CMakeLists.txt: desactivar el bloque _C_stable_libtorch
path = '/build/vllm/CMakeLists.txt'
with open(path) as f:
    content = f.read()
old = ('# add OR VLLM_GPU_LANG STREQUAL "HIP" here once\n'
       '# https://github.com/vllm-project/vllm/issues/35163 is resolved\n'
       'if(VLLM_GPU_LANG STREQUAL "CUDA")')
new = ('# add OR VLLM_GPU_LANG STREQUAL "HIP" here once\n'
       '# https://github.com/vllm-project/vllm/issues/35163 is resolved\n'
       'if(FALSE) # [jetson] _C_stable_libtorch disabled: requires torch >= 2.10')
if old in content:
    with open(path, 'w') as f:
        f.write(content.replace(old, new, 1))
    print('[OK] CMakeLists: _C_stable_libtorch desactivado')
else:
    raise RuntimeError('CMakeLists pattern not found')

# setup.py: eliminar ext_module y package_data de _C_stable_libtorch
path2 = '/build/vllm/setup.py'
with open(path2) as f:
    content2 = f.read()

# Quitar el bloque if completo (if vacío causa IndentationError)
old2 = ('    if _is_cuda():\n'
        '        ext_modules.append(CMakeExtension(name="vllm._C_stable_libtorch"))\n')
if old2 in content2:
    content2 = content2.replace(old2, '')
    print('[OK] setup.py: bloque _C_stable_libtorch eliminado')
else:
    raise RuntimeError('setup.py ext_module pattern not found')

# Quitar la entrada en package_data (el .so que ya no se genera)
old3 = '        "vllm/_C_stable_libtorch.abi3.so",\n'
if old3 in content2:
    content2 = content2.replace(old3, '')
    print('[OK] setup.py: package_data _C_stable_libtorch.so eliminado')
else:
    print('[WARN] setup.py: package_data pattern not found (puede no ser problema)')

with open(path2, 'w') as f:
    f.write(content2)
PYEOF

# ── Filtrar requirements/cuda.txt para Jetson ─────────────────
# vLLM 0.19.0 setup.py lee cuda.txt en VLLM_TARGET_DEVICE=cuda.
# Eliminamos paquetes que:
#   - No tienen wheel aarch64 o son CPU-only en PyPI (torch 2.10.0)
#   - Son solo para FA4/sm_90+ (nvidia-cutlass-dsl, quack-kernels)
#   - Conflictan con la versión Jetson ya instalada (flashinfer-cubin)
# torch/torchaudio/torchvision se restauran del jetson-base al final.
# flashinfer: se intenta actualizar a 0.6.x tras el build (ver paso dedicado).
RUN python3 - <<'PYEOF'
import re

path = '/build/vllm/requirements/cuda.txt'
with open(path) as f:
    lines = f.readlines()

skip_patterns = [
    r'^torch==',
    r'^torchaudio==',
    r'^torchvision==',
    r'^flashinfer-cubin',   # cubins precompilados solo para x86 sm_90
    r'^nvidia-cutlass-dsl', # FA4 CuteDSL — solo sm_90+
    r'^quack-kernels',      # FA4 — solo sm_90+
]

new_lines = []
for line in lines:
    stripped = line.strip()
    if any(re.match(p, stripped) for p in skip_patterns):
        new_lines.append(f'# [jetson-skip] {line}')
        print(f'  skipped: {stripped}')
    else:
        new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
print('[OK] requirements/cuda.txt filtered')
PYEOF

WORKDIR /build/vllm

ENV TORCH_CUDA_ARCH_LIST="8.7"
ENV MAX_JOBS=9
ENV VLLM_TARGET_DEVICE=cuda
ENV CUDA_HOME=/usr/local/cuda
ENV VLLM_NO_USAGE_STATS=1
# Usar el flash-attn pre-clonado y parcheado en vez de descargarlo
ENV VLLM_FLASH_ATTN_SRC_DIR=/build/flash-attn

# ── Instalar dependencias comunes ─────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements/common.txt

# ── Build deps de pyproject.toml (excepto torch, que ya tenemos) ──
# --no-build-isolation no instala [build-system].requires — hay que hacerlo manualmente.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "cmake>=3.26.1" \
        "ninja" \
        "packaging>=24.2" \
        "setuptools>=77.0.3,<81.0.0" \
        "setuptools-scm>=8.0" \
        "wheel" \
        "jinja2"

# ── Stubs de torch/headeronly/util para torch 2.8.0 ──────────
# vLLM 0.19.0 usa torch/headeronly/util/Float8_*.h (añadidos en torch 2.10.0).
# torch 2.8.0 tiene los mismos tipos en c10/util/ — creamos stubs que redirigen.
RUN TORCH_INC=/opt/venv/lib/python3.12/site-packages/torch/include && \
    mkdir -p ${TORCH_INC}/torch/headeronly/util && \
    for TYPE in Float8_e4m3fn Float8_e4m3fnuz Float8_e5m2 Float8_e5m2fnuz; do \
        printf '#pragma once\n#include <c10/util/%s.h>\n' "${TYPE}" \
            > ${TORCH_INC}/torch/headeronly/util/${TYPE}.h; \
    done && \
    ls ${TORCH_INC}/torch/headeronly/util/

# ── Compilar e instalar vLLM 0.19.0 (~2-3 h) ─────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation .

# ── Parche 2: Restaurar torch 2.8.0 (CUDA, sm_87) de Jetson ──
# pip install . ha sobreescrito torch con 2.10.0 CPU-only de PyPI (aarch64).
# Copiamos torch 2.8.0 compilado con CUDA desde el stage jetson-base.
RUN --mount=from=jetson-base,source=/opt/venv/lib/python3.12/site-packages,target=/jetson-pkgs \
    rm -rf \
        /opt/venv/lib/python3.12/site-packages/torch \
        /opt/venv/lib/python3.12/site-packages/torch-*.dist-info \
        /opt/venv/lib/python3.12/site-packages/torchgen \
        /opt/venv/lib/python3.12/site-packages/functorch \
        /opt/venv/lib/python3.12/site-packages/torchvision \
        /opt/venv/lib/python3.12/site-packages/torchvision-*.dist-info \
        /opt/venv/lib/python3.12/site-packages/torchaudio \
        /opt/venv/lib/python3.12/site-packages/torchaudio-*.dist-info && \
    cp -rp /jetson-pkgs/torch       /opt/venv/lib/python3.12/site-packages/torch && \
    cp -rp /jetson-pkgs/torchgen    /opt/venv/lib/python3.12/site-packages/torchgen && \
    cp -rp /jetson-pkgs/functorch   /opt/venv/lib/python3.12/site-packages/functorch && \
    cp -rp /jetson-pkgs/torchvision /opt/venv/lib/python3.12/site-packages/torchvision && \
    for d in /jetson-pkgs/torch-*.dist-info /jetson-pkgs/torchvision-*.dist-info; do \
        cp -rp "$d" /opt/venv/lib/python3.12/site-packages/; \
    done

# ── Actualizar flashinfer a 0.6.x (requerido por vLLM 0.19.0) ──
# La base tiene 0.2.8; vLLM 0.19.0 necesita 0.6.6.
# Si no hay wheel aarch64 disponible (probable), flashinfer cae en modo JIT:
# los kernels se compilan al primer uso (~30 s extra en primera petición).
# La variable VLLM_USE_FLASHINFER_SAMPLER=0 desactiva el sampler si hay crash.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "flashinfer-python==0.6.6" --no-deps \
        --index-url https://pypi.org/simple \
        --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu129 \
    && echo "[OK] flashinfer 0.6.6 instalado" \
    || echo "[WARN] flashinfer 0.6.6 no disponible para aarch64 — se usará JIT mode (compilación al primer uso)"

# ── Stub _C_stable_libtorch (requiere torch >= 2.10, no disponible en Jetson) ──
# cuda.py importa este módulo solo para trigger de op registration (side effect).
# Las ops que registra son W8A8 fp8/int8 — no usadas en GPTQ Marlin / sm_87.
# Un módulo Python vacío satisface el import sin necesitar compilar el .so.
RUN python3 - <<'PYEOF'
path = '/opt/venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.py'
with open(path, 'w') as f:
    f.write('# stub: W8A8 fp8/int8 ops not available on torch 2.8 / Jetson sm_87\n')
print('[OK] stub _C_stable_libtorch creado')
PYEOF

# ── Parches 3+4: compatibilidad torch 2.8.0 ↔ vLLM 0.19.x ────
RUN python3 - <<'PYEOF'
# Parche 3: backends.py — autograd_cache_normalize_inputs (añadida en torch 2.10.0)
# Sin este parche: KeyError si no se usa --enforce-eager
path1 = '/opt/venv/lib/python3.12/site-packages/vllm/compilation/backends.py'
old1 = '                torch._functorch.config.patch(autograd_cache_normalize_inputs=True),'
new1 = '                (__import__("contextlib").nullcontext() if not hasattr(torch._functorch.config, "autograd_cache_normalize_inputs") else torch._functorch.config.patch(autograd_cache_normalize_inputs=True)),'
with open(path1) as f:
    content = f.read()
if old1 in content:
    with open(path1, 'w') as f:
        f.write(content.replace(old1, new1))
    print(f'[OK] backends.py patched')
elif new1 in content:
    print(f'[SKIP] backends.py already patched')
else:
    print(f'[WARN] backends.py: pattern not found — check if torch._functorch.config API changed')

# Parche 4a: torch.accelerator — añadir empty_cache/memory_stats faltantes (añadidas en torch 2.10)
# vLLM 0.19.0 usa estas APIs en ~20 archivos; parchear torch mismo cubre todos a la vez.
path_acc = '/opt/venv/lib/python3.12/site-packages/torch/accelerator/__init__.py'
with open(path_acc) as f:
    acc = f.read()
shim = '''
# [jetson-compat] torch 2.8.0 compat: add APIs introduced in torch 2.10.0
import sys as _sys, torch.cuda as _cuda
_m = _sys.modules[__name__]
_missing = {
    'empty_cache': _cuda.empty_cache,
    'memory_stats': _cuda.memory_stats,
    'memory_reserved': _cuda.memory_reserved,
    'max_memory_allocated': _cuda.max_memory_allocated,
    'reset_peak_memory_stats': _cuda.reset_peak_memory_stats,
}
for _k, _v in _missing.items():
    if not hasattr(_m, _k):
        setattr(_m, _k, _v)
del _m, _sys, _cuda, _missing, _k, _v
'''
if '[jetson-compat]' not in acc:
    with open(path_acc, 'a') as f:
        f.write(shim)
    print('[OK] torch.accelerator: empty_cache + memory_stats shims añadidos')
else:
    print('[SKIP] torch.accelerator: shims ya presentes')

# Parche 4b-0: gdn_linear_attn.py — dynamo genera guarda para in_proj_qkv que no existe
# cuando create_in_proj_qkvz=True (nuestro caso sin LoRA). Dynamo no puede guardar
# hasattr() — reemplazamos por flag booleano que sí puede guardar.
path_gdn = '/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py'
with open(path_gdn) as f:
    gdn = f.read()
orig_gdn = gdn

# Añadir flag _use_lora_path justo después del if/else de create_in_proj_qkvz
old_flag = ('        if create_in_proj_qkvz:\n'
            '            self.in_proj_qkvz = self.create_qkvz_proj(\n')
new_flag = ('        self._use_lora_path = not create_in_proj_qkvz\n'
            '        if create_in_proj_qkvz:\n'
            '            self.in_proj_qkvz = self.create_qkvz_proj(\n')

# Reemplazar hasattr check en forward
old_hasattr = '        if hasattr(self, "in_proj_qkv"):\n'
new_hasattr = '        if self._use_lora_path:\n'

if old_flag in gdn and old_hasattr in gdn:
    gdn = gdn.replace(old_flag, new_flag, 1)
    gdn = gdn.replace(old_hasattr, new_hasattr, 1)
    with open(path_gdn, 'w') as f:
        f.write(gdn)
    print('[OK] gdn_linear_attn.py: _use_lora_path flag añadido (dynamo guard fix)')
elif '_use_lora_path' in gdn:
    print('[SKIP] gdn_linear_attn.py: ya parcheado')
else:
    raise RuntimeError('gdn_linear_attn.py: patterns not found')

# Parche 4b: matcher_utils.py + rms_quant_fusion.py — per_token_group_fp8_quant
# no compilado para sm_87 (requiere sm_89+). Añadimos hasattr guard igual que scaled_fp4_quant.
old_fp8 = ('if current_platform.is_cuda():\n'
           '    QUANT_OPS[kFp8Dynamic128Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501\n'
           '    QUANT_OPS[kFp8Dynamic64Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501\n')
new_fp8 = ('if current_platform.is_cuda() and hasattr(torch.ops._C, "per_token_group_fp8_quant"):\n'
           '    QUANT_OPS[kFp8Dynamic128Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501\n'
           '    QUANT_OPS[kFp8Dynamic64Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501\n')
fusion_dir = '/opt/venv/lib/python3.12/site-packages/vllm/compilation/passes/fusion/'
for fname in ['matcher_utils.py', 'rms_quant_fusion.py']:
    p = fusion_dir + fname
    with open(p) as f:
        c = f.read()
    if old_fp8 in c:
        with open(p, 'w') as f:
            f.write(c.replace(old_fp8, new_fp8))
        print(f'[OK] {fname}: fp8_quant hasattr guard añadido')
    elif 'hasattr(torch.ops._C, "per_token_group_fp8_quant")' in c:
        print(f'[SKIP] {fname}: ya parcheado')
    else:
        raise RuntimeError(f'{fname}: pattern not found')

# Parche 4c: standalone_compile.py — args[0] es un único Node, no lista
# Sin este parche: TypeError durante profile_run con CUDA graphs
import os
path2 = '/opt/venv/lib/python3.12/site-packages/torch/_inductor/standalone_compile.py'
if os.path.exists(path2):
    old2 = '        for node in last_node.args[0]:'
    new2 = '        _sc_args0 = last_node.args[0]\n        for node in (_sc_args0 if isinstance(_sc_args0, (list, tuple)) else [_sc_args0]):'
    with open(path2) as f:
        content = f.read()
    if old2 in content:
        with open(path2, 'w') as f:
            f.write(content.replace(old2, new2))
        print(f'[OK] standalone_compile.py patched')
    elif '_sc_args0' in content:
        print(f'[SKIP] standalone_compile.py already patched')
    else:
        print(f'[WARN] standalone_compile.py: pattern not found — may be fixed in newer torch build')
else:
    print(f'[WARN] standalone_compile.py not found — torch may have moved this file')
PYEOF

# ── Invalidar bytecode cacheado de los módulos parcheados ────
# Python puede usar __pycache__/*.pyc en subprocesos (EngineCore) y saltarse
# los .py ya parcheados. Borramos todos para forzar recompilación limpia.
RUN find /opt/venv/lib/python3.12/site-packages/vllm -name '*.pyc' -delete && \
    find /opt/venv/lib/python3.12/site-packages/torch/_inductor \
         -name 'standalone_compile*.pyc' -delete 2>/dev/null || true

# ── Limpiar código fuente y artefactos ────────────────────────
RUN rm -rf /build /tmp/pip-* /root/.cache/pip

# ── Variables de entorno de runtime ──────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
# Pre-cargar el libcuda real de Tegra con RTLD_GLOBAL al arrancar Python.
# El NVIDIA container runtime inyecta /usr/local/cuda/compat/libcuda.so.1
# que es un stub sin cuPointerGetAttribute → ImportError en _C.abi3.so.
# sitecustomize.py se ejecuta antes de cualquier import de usuario.
RUN python3 - <<'PYEOF'
# Parchear cuda.py: pre-cargar el libcuda real de Tegra con RTLD_GLOBAL
# justo antes de "import vllm._C".
# El NVIDIA container runtime inyecta /usr/local/cuda/compat/libcuda.so.1 en
# LD_LIBRARY_PATH, pero ese stub no tiene cuPointerGetAttribute.
path = '/opt/venv/lib/python3.12/site-packages/vllm/platforms/cuda.py'
old = '# import custom ops, trigger op registration\nimport vllm._C  # noqa'
new = (
    '# Pre-load Tegra libcuda before vllm._C to expose cuPointerGetAttribute.\n'
    '# The compat stub injected by the NVIDIA runtime lacks this symbol.\n'
    'import ctypes as _ctypes, os as _os\n'
    'for _p in [\'/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1\',\n'
    '           \'/usr/lib/aarch64-linux-gnu/libcuda.so.1\']:\n'
    '    if _os.path.exists(_p):\n'
    '        try:\n'
    '            _ctypes.CDLL(_p, _ctypes.RTLD_GLOBAL)\n'
    '        except OSError:\n'
    '            pass\n'
    '        break\n'
    '# import custom ops, trigger op registration\n'
    'import vllm._C  # noqa'
)
with open(path) as f:
    content = f.read()
if old in content:
    with open(path, 'w') as f:
        f.write(content.replace(old, new, 1))
    print('[OK] cuda.py: preload libcuda patch aplicado')
else:
    raise RuntimeError('cuda.py: pattern not found')
PYEOF

ENV VLLM_NO_USAGE_STATS=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=0

ENV HF_HOME=/data/huggingface
ENV TRANSFORMERS_CACHE=/data/huggingface
ENV XDG_CACHE_HOME=/data/vllm-cache

WORKDIR /app

VOLUME ["/data/huggingface", "/data/vllm-cache"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--help"]
