# ============================================================
# vLLM para Jetson AGX Orin — JetPack 6.2 (L4T R36.4.x)
# CUDA 12.9 · sm_87 · PyTorch 2.8.0 (Jetson AI Lab, compilado en imagen)
#
# Base: dustynv/comfyui:r36.4-cu129-24.04
#   → CUDA 12.9 instalado → sin problemas de libcuda / cuMemcpyBatchAsync
#   → PyTorch 2.8.0 con CUDA sm_87 precompilado
#   → No requiere autenticación NGC
#
# Compilación: ~150 min (CPU-bound, nvcc con sm_87)
#
# Uso:
#   docker build -t vllm-jetson:latest .
#   docker compose up
# ============================================================

ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_BRANCH=v0.17.0
ARG FA_COMMIT=140c00c0241bb60cc6e44e7c1be9998d4b20d8d2

# ── Stage auxiliar: preservar torch de Jetson antes de que vLLM lo reemplace ──
# vLLM v0.17.0 instala torch>=2.10.0 desde PyPI durante "pip install .",
# que en aarch64 es CPU-only (sin libtorch_cuda.so). Copiamos torch 2.8.0
# (CUDA, sm_87, compilado en la imagen dustynv) desde aquí al final del build.
FROM dustynv/comfyui:r36.4-cu129-24.04 AS jetson-base

# ── Stage principal ───────────────────────────────────────────
FROM dustynv/comfyui:r36.4-cu129-24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG VLLM_REPO
ARG VLLM_BRANCH
ARG FA_COMMIT

# Sobrescribir el índice pip de dustynv (pypi.jetson-ai-lab.dev no resuelve DNS)
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

# ── PyTorch para JetPack 6 (sm_87 nativo) ────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision triton \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# ── Clonar vLLM v0.17.0 ──────────────────────────────────────
WORKDIR /build
RUN git clone --depth 1 --branch ${VLLM_BRANCH} ${VLLM_REPO} vllm

# ── Pre-clonar flash-attn para parchearlo ─────────────────────
# vLLM usa FetchContent para descargar flash-attn durante el build.
# Lo pre-clonamos aquí para poder parchearlo antes de la compilación.
# Inicializamos solo el submodule csrc/cutlass (necesario para compilar FA2).
# csrc/composable_kernel es ROCm-only, no lo necesitamos.
RUN git clone https://github.com/vllm-project/flash-attention.git /build/flash-attn && \
    git -C /build/flash-attn checkout ${FA_COMMIT} && \
    git -C /build/flash-attn submodule update --init --depth 1 csrc/cutlass

# ── Parchear CMakeLists.txt: añadir sm_87 a las listas de arch ──
# Problema: vLLM y flash-attn usan "8.0+PTX" como arch fallback para sm_87+.
# Esto compila los kernels para sm_80 SASS + compute_80 PTX.
# En Jetson JetPack 6.2.1, el driver (CUDA 12.6) no puede JIT-compilar
# ese PTX porque fue generado con ISA 8.5 (nvcc 12.9) > ISA 8.4 (CUDA 12.6 max).
# Fix: añadir "8.7+PTX" para generar sm_87 SASS nativo → sin PTX JIT en Jetson.
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
    # vLLM: Marlin MOE kernels (gptq_marlin_moe_repack, etc.)
    sed -i \
        's|cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8\.0+PTX"|cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX;8.7+PTX"|g' \
        /build/vllm/CMakeLists.txt && \
    sed -i \
        's|cuda_archs_loose_intersection(MARLIN_MOE_OTHER_ARCHS "7\.5;8\.0+PTX"|cuda_archs_loose_intersection(MARLIN_MOE_OTHER_ARCHS "7.5;8.0+PTX;8.7+PTX"|g' \
        /build/vllm/CMakeLists.txt && \
    # flash-attn v2: añadir 8.7 a CUDA_SUPPORTED_ARCHS (lista hardcodeada: "8.0;8.6;8.9;9.0" sin 8.7)
    # Sin este parche CUDA_ARCHS queda en "8.6" y FA2_ARCHS sigue siendo "8.0+PTX" → sm_80 PTX fallback
    # Con este parche CUDA_ARCHS="8.7" y FA2_ARCHS="8.7+PTX" → sm_87 SASS nativo, sin JIT
    sed -i \
        's|set(CUDA_SUPPORTED_ARCHS "8\.0;8\.6;8\.9;9\.0")|set(CUDA_SUPPORTED_ARCHS "8.0;8.6;8.7;8.9;9.0")|g' \
        /build/flash-attn/CMakeLists.txt && \
    sed -i \
        's|cuda_archs_loose_intersection(FA2_ARCHS "8\.0+PTX"|cuda_archs_loose_intersection(FA2_ARCHS "8.0+PTX;8.7+PTX"|g' \
        /build/flash-attn/CMakeLists.txt && \
    # Verificar todos los parches
    grep -n "MARLIN_ARCHS\|FA2_ARCHS\|MARLIN_BF16\|MARLIN_OTHER\|MARLIN_MOE" /build/vllm/CMakeLists.txt | grep -E "8\.7" && \
    grep -n "FA2_ARCHS\|CUDA_SUPPORTED_ARCHS" /build/flash-attn/CMakeLists.txt | grep -E "8\.7"

WORKDIR /build/vllm

# ── Variables de compilación: sm_87 = Marlin tensor cores Orin ──
ENV TORCH_CUDA_ARCH_LIST="8.7"
ENV MAX_JOBS=9
ENV VLLM_TARGET_DEVICE=cuda
ENV CUDA_HOME=/usr/local/cuda
ENV VLLM_NO_USAGE_STATS=1
# Usar el flash-attn pre-clonado y parcheado
ENV VLLM_FLASH_ATTN_SRC_DIR=/build/flash-attn

# ── Instalar dependencias y compilar (~150 min) ───────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements/common.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation .

# ── Restaurar torch de Jetson con CUDA ───────────────────────
# pip install . instala torch 2.10.0 de PyPI (CPU-only en aarch64).
# Restauramos torch 2.8.0 (CUDA, sm_87) copiándolo desde el stage jetson-base.
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

# ── Parches de compatibilidad torch 2.8.0 ↔ vLLM 0.17.x ─────
# 1. vllm/compilation/backends.py: torch._functorch.config.patch() falla si
#    'autograd_cache_normalize_inputs' no existe (añadida en torch 2.10.0).
#    Fix: usar nullcontext si el atributo no está presente.
# 2. torch/_inductor/standalone_compile.py: espera que last_node.args[0] sea
#    una lista iterable de Tensors, pero con el grafo FX de vLLM 0.17.x es un
#    único Node. Fix: envolver en lista si no es iterable.
RUN python3 - <<'PYEOF'
import re

# --- Patch 1: backends.py ---
path1 = '/opt/venv/lib/python3.12/site-packages/vllm/compilation/backends.py'
old1 = '                torch._functorch.config.patch(autograd_cache_normalize_inputs=True),'
new1 = '                (__import__("contextlib").nullcontext() if not hasattr(torch._functorch.config, "autograd_cache_normalize_inputs") else torch._functorch.config.patch(autograd_cache_normalize_inputs=True)),'
with open(path1) as f:
    content = f.read()
if old1 in content:
    with open(path1, 'w') as f:
        f.write(content.replace(old1, new1))
    print(f'[OK] {path1}')
elif new1 in content:
    print(f'[SKIP] {path1} already patched')
else:
    raise RuntimeError(f'Pattern not found in {path1}')

# --- Patch 2: standalone_compile.py ---
path2 = '/opt/venv/lib/python3.12/site-packages/torch/_inductor/standalone_compile.py'
old2 = '        for node in last_node.args[0]:'
new2 = '        _sc_args0 = last_node.args[0]\n        for node in (_sc_args0 if isinstance(_sc_args0, (list, tuple)) else [_sc_args0]):'
with open(path2) as f:
    content = f.read()
if old2 in content:
    with open(path2, 'w') as f:
        f.write(content.replace(old2, new2))
    print(f'[OK] {path2}')
elif '_sc_args0' in content:
    print(f'[SKIP] {path2} already patched')
else:
    raise RuntimeError(f'Pattern not found in {path2}')
PYEOF

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
