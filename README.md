# maarifa-vllm-jetson-orin

vLLM compilado para Jetson AGX Orin 64GB (JetPack 6.2.1, sm_87).

## Imágenes disponibles

| Versión | Imagen | Estado |
|---|---|---|
| **v0.20.2** | `jdelacasa/vllm-jetson:0.20.2-torch2.10.0-cu126-l4tr36.4-sm87` | ✅ activa |
| v0.19.1 | `jdelacasa/vllm-jetson:0.19.1-torch2.10.0-cu126-l4tr36.4-sm87` | ✅ funcional |
| v0.19.0 | `jdelacasa/vllm-jetson:0.19.0-torch2.8.0-cu129-l4tr36.4-sm87` | archivada |

**Stack activo (v0.20.2):**
- JetPack 6.2.1 · CUDA 12.6 · driver 540.4.0 · sm_87
- PyTorch 2.10.0 · vLLM 0.20.2 · flash-attn `f5bc33fc`
- Ubuntu 22.04 · Python 3.10

---

## Uso rápido

```bash
docker compose up -d
```

Variables de entorno en `.env`:
```env
HF_TOKEN=hf_...
HF_CACHE_DIR=/ruta/a/cache
```

> El arranque tarda **5–8 min** (CUDA graphs desde caché) o **15–20 min** la primera vez.

---

## Build — versión actual (v0.20.2)

```bash
docker build \
  --build-arg BASE_IMAGE=jdelacasa/vllm-jetson:0.19.1-torch2.10.0-cu126-l4tr36.4-sm87 \
  --build-arg VLLM_BRANCH=v0.20.2 \
  -t jdelacasa/vllm-jetson:0.20.2-torch2.10.0-cu126-l4tr36.4-sm87 \
  .
```

Tiempo: ~2–2.5 h (CPU-bound, nvcc sm_87).

> ⚠️ La imagen base `pytorch:2.10-r36.4.tegra-aarch64-cu126-22.04` compilada localmente
> ya no está disponible. Se usa `v0.19.1` como base. Para reconstruirla desde cero:
> ```bash
> CUDA_VERSION=12.6 LSB_RELEASE=22.04 jetson-containers build pytorch:2.10
> ```

---

## Build — guía para futuras versiones de vLLM

Al actualizar a una nueva versión de vLLM hay **4 cosas que verificar y posiblemente actualizar** en el Dockerfile. El resto de parches suele ser estable.

### 1. FA_COMMIT — commit de flash-attn

Cada versión de vLLM pinta un commit concreto de flash-attn en su `CMakeLists.txt`. Si no coincide el build falla con `GIT_TAG mismatch`.

**Cómo encontrar el commit correcto:**
```bash
# Sustituye X.Y.Z por la versión objetivo
curl -s https://raw.githubusercontent.com/vllm-project/vllm/vX.Y.Z/CMakeLists.txt \
  | grep -A2 "flash-attention"
```

Busca la línea `GIT_TAG` dentro del bloque `FetchContent_Declare(flash-attn ...)`.

**Historial conocido:**
| vLLM | FA_COMMIT |
|---|---|
| v0.19.x | `29210221863736a08f71a866459e368ad1ac4a95` |
| v0.20.2 | `f5bc33cfc02c744d24a2e9d50e6db656de40611c` |

Actualizar en el Dockerfile:
```dockerfile
ARG FA_COMMIT=<nuevo_commit>
```

---

### 2. Versión de flashinfer

vLLM requiere una versión específica de flashinfer. Si la versión cambia, la instalación falla o da warnings.

**Cómo encontrar la versión correcta:**
```bash
curl -s https://raw.githubusercontent.com/vllm-project/vllm/vX.Y.Z/requirements/cuda.txt \
  | grep flashinfer
```

**Historial conocido:**
| vLLM | flashinfer |
|---|---|
| v0.19.x | `0.6.6` |
| v0.20.2 | `0.6.8.post1` |

Actualizar en el Dockerfile (la instalación usa `|| true` para no fallar si no hay wheel aarch64):
```dockerfile
pip install flashinfer-python==<version> ... || echo "[WARN] flashinfer no disponible"
```

---

### 3. Parche fp8 — patterns de matcher_utils.py y rms_quant_fusion.py

Este parche añade un guard `hasattr` para que el código fp8 no explote en sm_87 (que no tiene FP8 hardware). El pattern exacto puede cambiar entre versiones si los autores añaden o quitan comentarios.

**Cómo verificar el pattern correcto antes de hacer el build:**
```bash
# Clona la rama objetivo y mira el archivo
git clone --depth 1 --branch vX.Y.Z https://github.com/vllm-project/vllm /tmp/vllm-check
grep -n "per_token_group_fp8_quant" /tmp/vllm-check/vllm/model_executor/layers/quantization/utils/quant_utils/matcher_utils.py
```

**Historial:**
- v0.19.x: las líneas tenían `# noqa: E501` al final
- v0.20.2: **también tienen `# noqa: E501`** (un WebFetch truncado indujo a error en la sesión anterior — siempre verificar con el repo real)

Si el build imprime `[WARN] fp8 pattern not found`, el old_pattern del Dockerfile no coincide con el archivo instalado. Busca las líneas actuales en el repo y actualiza el `old_fp8` en el Dockerfile.

---

### 4. Parche fake_impl.py — compatibilidad PyTorch 2.10

Este parche envuelve una llamada a `_dispatch_has_kernel_for_dispatch_key` en un try/except para que no explote con PyTorch 2.10. Usa regex para detectar la indentación real del archivo.

**Cuándo es necesario:** siempre que la imagen base siga usando PyTorch 2.10. Si en el futuro se actualiza a PyTorch 2.11+, este parche puede no ser necesario (vLLM 0.20.x pide 2.11 pero funciona con 2.10 omitiendo el requisito en `requirements/cuda.txt`).

**Señal de que el parche falló:** error en runtime al arrancar el servidor:
```
RuntimeError: Tried to call torch._C._dispatch_has_kernel_for_dispatch_key...
```

El parche detecta automáticamente si ya está aplicado (`'_hm' in c`) y hace SKIP — es seguro re-aplicarlo.

---

### Checklist completo para una nueva versión

```
[ ] 1. Actualizar ARG VLLM_BRANCH=vX.Y.Z en el Dockerfile
[ ] 2. Buscar FA_COMMIT correcto en CMakeLists.txt de la nueva versión
[ ] 3. Buscar versión de flashinfer en requirements/cuda.txt
[ ] 4. Clonar la rama y verificar el pattern del parche fp8 en:
        vllm/model_executor/layers/quantization/utils/quant_utils/matcher_utils.py
        vllm/model_executor/layers/quantization/utils/quant_utils/rms_quant_fusion.py
[ ] 5. Verificar si gdn_linear_attn.py sigue teniendo hasattr(self, "in_proj_qkv")
[ ] 6. Build con la imagen de la versión anterior como BASE_IMAGE
[ ] 7. Arrancar el servidor y verificar que todos los parches dicen [OK] en los logs
[ ] 8. Comprobar que "Application startup complete" aparece sin errores
```

---

## Parches aplicados (estables entre versiones)

Estos 5 parches son necesarios para sm_87 y no han cambiado entre v0.19.x y v0.20.2:

### Parche 1 — CMakeLists.txt: sm_87 en MARLIN y FA2_ARCHS
Añade `8.7+PTX` a `MARLIN_ARCHS`, `MARLIN_MOE_ARCHS` y `FA2_ARCHS` para compilar kernels Marlin GPTQ y flash-attn como SASS nativo sm_87 (sin este parche se compila PTX genérico y es ~20% más lento).

### Parche 2 — cuda.py: preload libcuda Tegra
Pre-carga `/usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1` con `RTLD_GLOBAL` antes de `import vllm._C`. El stub que inyecta el runtime de NVIDIA carece de `cuPointerGetAttribute`, necesario para los kernels de atención.

### Parche 3 — fp8 hasattr guard
`matcher_utils.py` y `rms_quant_fusion.py`: añade `hasattr(torch.ops._C, 'per_token_group_fp8_quant')` antes de registrar los ops fp8. Sin este guard explota en sm_87 que no compila esos ops (requieren sm_89+).

### Parche 4 — gdn_linear_attn dynamo guard
Reemplaza `hasattr(self, "in_proj_qkv")` en el forward de la linear attention por un flag booleano fijo. `torch.compile` no puede guardar un guard dinámico sobre `hasattr` de atributos de módulo y lanza `torch._dynamo.exc.InternalTorchDynamoError`.

### Parche 5 — fake_impl.py: compatibilidad PyTorch 2.10
Envuelve `torch._C._dispatch_has_kernel_for_dispatch_key(...)` en try/except para que no explote durante el registro de ops con PyTorch 2.10. Usa regex para detectar la indentación real (evita el bug de substring-match que corrompía el bloque en versiones anteriores del parche).

---

## Configuraciones según caso de uso

Flags comunes a todas las configuraciones:
```
--dtype half                     # float16, más estable que bfloat16 en sm_87
--tool-call-parser qwen3_coder
--reasoning-parser qwen3
--enable-auto-tool-choice
--no-enable-log-requests
# SIN --enforce-eager  (CUDA graphs activos)
# SIN --quantization   (vLLM detecta gptq_marlin automáticamente)
```

### Caso 1: agentes / 1–4 usuarios interactivos (configuración actual)

```yaml
command: >
  palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
  --host 0.0.0.0 --port 8001
  --dtype half
  --gpu-memory-utilization 0.77
  --max-model-len 128000
  --max-num-batched-tokens 64000
  --max-num-seqs 8
  --enable-prefix-caching
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
  --limit-mm-per-prompt '{"image":4}'
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --no-enable-log-requests
environment:
  VLLM_MARLIN_USE_ATOMIC_ADD=1
  VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=0
```

Rendimiento medido (benchmark 2026-05-11):

| usuarios | contexto | TTFT | TPS/req | throughput |
|---|---|---|---|---|
| 1 | ~500 tok  | 0.45s | 32 tok/s | 31 tok/s |
| 4 | ~500 tok  | 1.2s  | 26 tok/s | 93 tok/s |
| 4 | ~8k tok   | 4.7s  | 21 tok/s | 64 tok/s |
| 8 | ~8k tok   | 9.6s  | 15 tok/s | 82 tok/s |
| 4 | ~32k tok  | 5.2s  | 14 tok/s | 45 tok/s |
| 8 | ~32k tok  | 10s   |  9 tok/s | 56 tok/s |

MTP acceptance rate observado: 65–95% (media ~2.5 tokens aceptados de 3 posibles).

### Caso 2: máximo throughput, muchos usuarios

Sin MTP (penaliza con batches grandes), más `--max-num-seqs`:

```yaml
command: >
  palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
  --host 0.0.0.0 --port 8001
  --dtype half
  --gpu-memory-utilization 0.77
  --max-model-len 32000
  --max-num-batched-tokens 32000
  --max-num-seqs 16
  --enable-prefix-caching
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --no-enable-log-requests
```

### Caso 3: contexto muy largo, usuario único

```yaml
command: >
  palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
  --host 0.0.0.0 --port 8001
  --dtype half
  --gpu-memory-utilization 0.77
  --max-model-len 100000
  --max-num-batched-tokens 64000
  --max-num-seqs 1
  --enable-prefix-caching
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --no-enable-log-requests
```

Rendimiento con 64k tokens de contexto: TTFT ~27s (frío) / ~2s (prefix cache hit), ~20 tok/s de decode.

### Resumen de trade-offs

| | Caso 1 (actual) | Caso 2 (throughput) | Caso 3 (contexto largo) |
|---|---|---|---|
| MTP | ✅ sí | ❌ no | ❌ no |
| Prefix caching | ✅ sí | ✅ sí | ✅ sí |
| max-num-seqs | 8 | 16 | 1 |
| max-model-len | 128K | 32K | 100K |
| Latencia/req | Baja | Media | Alta (prefill) |
| Throughput total | Alto | Muy alto | Bajo |

---

## Bug: GPU estado corrupto tras reinicio de container

### Síntoma

Primera vez funciona. Al reiniciar el container, vLLM falla ~15 minutos después del arranque:
```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:1154
```

### Causa raíz

`stop_grace_period` por defecto de Docker es 10s → SIGKILL → PyTorch no destruye el contexto CUDA → el driver `nvhost` retiene estado corrupto. El GPU **railgating** (power-gate automático de Jetson) agrava el problema.

### Fixes aplicados en este repo

**1. `stop_grace_period: 120s` en `docker-compose.yml`** ← fix principal

**2. CUDA pre-flight en `docker-entrypoint.sh`:** allocación de prueba antes de cargar el modelo — falla rápido en segundos si la GPU está en mal estado, en vez de tras 15 min de carga.

**3. Fix del railgate (ejecutar una vez en el host):**
```bash
sudo cp nvidia-gpu-no-railgate.service /etc/systemd/system/
sudo systemctl enable --now nvidia-gpu-no-railgate.service
```

**4. Recuperación sin reboot:**
```bash
sudo ./reset-gpu.sh
docker compose up -d
```

| Archivo | Propósito |
|---|---|
| `docker-compose.yml` | `stop_grace_period: 120s` |
| `docker-entrypoint.sh` | CUDA pre-flight |
| `reset-gpu.sh` | Reset GPU sin reboot |
| `nvidia-gpu-no-railgate.service` | Deshabilita railgate al boot |

---

## Prueba rápida del modelo

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4",
       "messages": [{"role": "user", "content": "Hola, ¿en qué puedo ayudarte?"}]}'
```

## Benchmark

Ver resultados completos en [`benchmark/resultados.md`](benchmark/resultados.md).

```bash
# Test rápido
python3 benchmark/bench.py --concurrency 1 2 4 --ctx tiny small --max-tokens 200 --repeats 1

# Test completo
python3 benchmark/bench.py --concurrency 1 2 4 6 8 --max-tokens 300 --repeats 2

# Con contextos de 32k y 64k
python3 benchmark/bench.py --include-xlarge --concurrency 1 2 4 6 8
```
