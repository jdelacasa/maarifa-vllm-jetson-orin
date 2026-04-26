# ============================================================
# vLLM para Jetson AGX Orin — JetPack 6.2 (L4T R36.4.x)
# CUDA 12.6 · sm_87 · PyTorch 2.10.0
#
# Base: pytorch:2.10-r36.4.tegra-aarch64-cu126-22.04
#   → CUDA 12.6, PyTorch 2.10.0 sm_87, Ubuntu 22.04, Python 3.10
#
# Con torch 2.10.0 ya disponible en la base, no son necesarios:
#   - Restaurar torch tras el build (pip instala 2.10.0 igual)
#   - Stubs torch/headeronly/util (existen en 2.10)
#   - Deshabilitar _C_stable_libtorch (requiere torch >= 2.10 ✓)
#   - Parche autograd_cache_normalize_inputs (existe en 2.10 ✓)
#
# Parches que siguen siendo necesarios para sm_87:
#   1. CMakeLists.txt — añadir 8.7+PTX a MARLIN_ARCHS, FA2_ARCHS, etc.
#      Garantiza kernels nativos sm_87 en Marlin GPTQ + flash-attn
#   2. cuda.py — pre-cargar libcuda real de Tegra (stub compat no tiene
#      cuPointerGetAttribute)
#   3. fp8_quant hasattr guard — sm_87 no compila per_token_group_fp8_quant
#      (requiere sm_89+); se necesita guard para no crashear al importar
#   4. gdn_linear_attn dynamo guard — hasattr() no puede ser guardado por
#      torch.compile; reemplazar por flag booleano
#
# Compilación: ~2-3 h (CPU-bound, nvcc sm_87)
# ============================================================

ARG BASE_IMAGE=pytorch:2.10-r36.4.tegra-aarch64-cu126-22.04
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_BRANCH=v0.19.1
# FA commit pinado por vLLM 0.19.x (de cmake/external_projects/vllm_flash_attn.cmake)
# Si el build falla con "GIT_TAG mismatch", actualizar este valor.
ARG FA_COMMIT=29210221863736a08f71a866459e368ad1ac4a95

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG VLLM_REPO
ARG VLLM_BRANCH
ARG FA_COMMIT

ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126

# ── Herramientas de compilación ───────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    gcc \
    g++ \
  && rm -rf /var/lib/apt/lists/*

# ── Clonar vLLM 0.19.1 ───────────────────────────────────────
WORKDIR /build
RUN git clone --depth 1 --branch ${VLLM_BRANCH} ${VLLM_REPO} vllm

# ── Pre-clonar flash-attn al commit exacto que pina vLLM ─────
# VLLM_FLASH_ATTN_SRC_DIR indica a FetchContent usar este directorio
# en vez de descargarlo, permitiéndonos aplicar el parche sm_87 primero.
RUN git clone https://github.com/vllm-project/flash-attention.git /build/flash-attn && \
    git -C /build/flash-attn checkout ${FA_COMMIT} && \
    git -C /build/flash-attn submodule update --init --depth 1 csrc/cutlass

# ── Parche 1: CMakeLists.txt — añadir sm_87 a todas las arch lists ──
#
# vLLM 0.19.x ya incluye 8.7 en CUDA_SUPPORTED_ARCHS, pero MARLIN_ARCHS
# y MOE usan "8.0+PTX" como límite superior, lo que hace que sm_87 compile
# kernels Marlin solo en PTX en vez de SASS nativo (menor rendimiento).
# La FA del commit pinado tampoco incluye 8.7 en CUDA_SUPPORTED_ARCHS.
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
    echo "=== vLLM MARLIN patches ===" && \
    grep -n "8\.7" /build/vllm/CMakeLists.txt | grep -E "MARLIN|FA2" && \
    echo "=== flash-attn patches ===" && \
    grep -n "8\.7" /build/flash-attn/CMakeLists.txt

# ── Filtrar requirements/cuda.txt para Jetson ─────────────────
# Evitar que pip reinstale/degrade torch, torchaudio, torchvision
# (ya tenemos 2.10.0 CUDA en la base), y excluir paquetes sm_90+ only.
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
ENV VLLM_FLASH_ATTN_SRC_DIR=/build/flash-attn

# ── Instalar dependencias comunes ─────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements/common.txt

# ── Build deps de pyproject.toml ──────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "cmake>=3.26.1" \
        "ninja" \
        "packaging>=24.2" \
        "setuptools>=77.0.3,<81.0.0" \
        "setuptools-scm>=8.0" \
        "wheel" \
        "jinja2"

# ── Compilar e instalar vLLM 0.19.1 (~2-3 h) ─────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation .

# ── Instalar torchvision ──────────────────────────────────────
# La base pytorch:2.10 no incluye torchvision. vLLM la necesita en el
# import chain de qwen3_5.py → qwen3_vl.py → transformers → torchvision.
# --no-deps evita que pip reinstale torch desde PyPI.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torchvision --no-deps \
        --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# ── Actualizar flashinfer (requerido por vLLM 0.19.x) ────────
# Si no hay wheel aarch64, flashinfer cae en modo JIT:
# los kernels se compilan al primer uso (~30 s extra en primera petición).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "flashinfer-python==0.6.6" --no-deps \
        --index-url https://pypi.org/simple \
        --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    && echo "[OK] flashinfer 0.6.6 instalado" \
    || echo "[WARN] flashinfer 0.6.6 no disponible para aarch64 — se usará JIT mode"

# ── Parche torch fake_impl.py — torchvision 0.26 + torch 2.10 ──
# torch 2.10.0 cambió _dispatch_has_kernel_for_dispatch_key: ahora lanza
# RuntimeError si el op no existe en lugar de retornar False. torchvision
# no lo contempla → crash al importar antes de que _torchvision.so cargue.
RUN python3 - <<'PYEOF'
path = '/opt/venv/lib/python3.10/site-packages/torch/_library/fake_impl.py'
old = '        if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, "Meta"):'
new = ('        try:\n'
       '            _hm = torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, "Meta")\n'
       '        except RuntimeError:\n'
       '            _hm = False\n'
       '        if _hm:')
with open(path) as f:
    c = f.read()
if old in c:
    with open(path, 'w') as f:
        f.write(c.replace(old, new))
    print('[OK] fake_impl.py patched')
elif '_hm' in c:
    print('[SKIP] fake_impl.py: ya parcheado')
else:
    raise RuntimeError('fake_impl.py: pattern not found')
PYEOF

# ── Parches de runtime: fp8, dynamo, libcuda ─────────────────
RUN python3 - <<'PYEOF'
import os

# Parche 2a: fp8_quant hasattr guard — sm_87 no compila per_token_group_fp8_quant
old_fp8 = ('if current_platform.is_cuda():\n'
           '    QUANT_OPS[kFp8Dynamic128Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501\n'
           '    QUANT_OPS[kFp8Dynamic64Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501\n')
new_fp8 = ('if current_platform.is_cuda() and hasattr(torch.ops._C, "per_token_group_fp8_quant"):\n'
           '    QUANT_OPS[kFp8Dynamic128Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501\n'
           '    QUANT_OPS[kFp8Dynamic64Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501\n')
fusion_dir = '/opt/venv/lib/python3.10/site-packages/vllm/compilation/passes/fusion/'
for fname in ['matcher_utils.py', 'rms_quant_fusion.py']:
    p = fusion_dir + fname
    if not os.path.exists(p):
        print(f'[SKIP] {fname}: not found')
        continue
    with open(p) as f:
        c = f.read()
    if old_fp8 in c:
        with open(p, 'w') as f:
            f.write(c.replace(old_fp8, new_fp8))
        print(f'[OK] {fname}: fp8_quant hasattr guard añadido')
    elif 'hasattr(torch.ops._C, "per_token_group_fp8_quant")' in c:
        print(f'[SKIP] {fname}: ya parcheado')
    else:
        print(f'[WARN] {fname}: pattern not found — puede estar ya corregido')

# Parche 2b: gdn_linear_attn dynamo guard
# hasattr() en forward no puede ser guardado por torch.compile; usar flag booleano
path_gdn = '/opt/venv/lib/python3.10/site-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py'
if os.path.exists(path_gdn):
    with open(path_gdn) as f:
        gdn = f.read()
    old_flag = ('        if create_in_proj_qkvz:\n'
                '            self.in_proj_qkvz = self.create_qkvz_proj(\n')
    new_flag = ('        self._use_lora_path = not create_in_proj_qkvz\n'
                '        if create_in_proj_qkvz:\n'
                '            self.in_proj_qkvz = self.create_qkvz_proj(\n')
    old_hasattr = '        if hasattr(self, "in_proj_qkv"):\n'
    new_hasattr = '        if self._use_lora_path:\n'
    if old_flag in gdn and old_hasattr in gdn:
        gdn = gdn.replace(old_flag, new_flag, 1).replace(old_hasattr, new_hasattr, 1)
        with open(path_gdn, 'w') as f:
            f.write(gdn)
        print('[OK] gdn_linear_attn.py: _use_lora_path flag añadido')
    elif '_use_lora_path' in gdn:
        print('[SKIP] gdn_linear_attn.py: ya parcheado')
    else:
        print('[WARN] gdn_linear_attn.py: patterns not found — puede estar corregido en 0.19.1')
else:
    print('[SKIP] gdn_linear_attn.py: not found')

# Parche 2c: standalone_compile.py — args[0] puede ser Node suelto, no lista
path_sc = '/opt/venv/lib/python3.10/site-packages/torch/_inductor/standalone_compile.py'
if os.path.exists(path_sc):
    old_sc = '        for node in last_node.args[0]:'
    new_sc = ('        _sc_args0 = last_node.args[0]\n'
              '        for node in (_sc_args0 if isinstance(_sc_args0, (list, tuple)) else [_sc_args0]):')
    with open(path_sc) as f:
        content = f.read()
    if old_sc in content:
        with open(path_sc, 'w') as f:
            f.write(content.replace(old_sc, new_sc))
        print('[OK] standalone_compile.py patched')
    elif '_sc_args0' in content:
        print('[SKIP] standalone_compile.py: ya parcheado')
    else:
        print('[INFO] standalone_compile.py: pattern not found — puede estar corregido en torch 2.10')
else:
    print('[SKIP] standalone_compile.py: not found')
PYEOF

# ── Parche 3: cuda.py — pre-cargar libcuda real de Tegra ─────
# El container runtime inyecta /usr/local/cuda/compat/libcuda.so.1
# (stub sin cuPointerGetAttribute), que hace fallar la carga de vllm._C.
RUN python3 - <<'PYEOF'
path = '/opt/venv/lib/python3.10/site-packages/vllm/platforms/cuda.py'
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
elif 'cuPointerGetAttribute' in content:
    print('[SKIP] cuda.py: ya parcheado')
else:
    raise RuntimeError('cuda.py: pattern not found — revisar si vLLM 0.19.1 cambió cuda.py')
PYEOF

# ── Invalidar bytecode cacheado de módulos parcheados ────────
RUN find /opt/venv/lib/python3.10/site-packages/vllm -name '*.pyc' -delete && \
    find /opt/venv/lib/python3.10/site-packages/torch/_inductor \
         -name 'standalone_compile*.pyc' -delete 2>/dev/null || true

# ── Limpiar código fuente y artefactos de build ───────────────
RUN rm -rf /build /tmp/pip-* /root/.cache/pip

# ── Variables de entorno de runtime ──────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
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
