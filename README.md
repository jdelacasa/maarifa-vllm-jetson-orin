# maarifa-vllm-jetson-orin

vLLM 0.19.1 compilado para Jetson AGX Orin 64GB (JetPack 6.2.1, sm_87).

## Imagen

```
jdelacasa/vllm-jetson:0.19.1-torch2.10.0-cu126-l4tr36.4-sm87
sha256:4a0975efcb0446e6b439d136804beeeb79c1509d5209dae4f4f6a65b48668a2a
```

**Stack:**
- JetPack 6.2.1 · CUDA 12.6 · driver 540.4.0 · sm_87
- PyTorch 2.10.0 (compilado con jetson-containers)
- vLLM 0.19.1 + flash-attn `29210221863736`
- Ubuntu 22.04 · Python 3.10

## Uso rápido

```bash
docker compose  up -d
```

Variables de entorno en `.env`:

```env
HF_TOKEN=hf_...
HF_CACHE_DIR=/ruta/a/cache
```

## Build

```bash
docker build \
  --build-arg BASE_IMAGE=pytorch:2.10-r36.4.tegra-aarch64-cu126-22.04 \
  --build-arg VLLM_BRANCH=v0.19.1 \
  -t jdelacasa/vllm-jetson:0.19.1-torch2.10.0-cu126-l4tr36.4-sm87 \
  -t vllm-jetson:latest \
  .
```

Tiempo: ~2.2 h (CPU-bound, nvcc sm_87).

La base `pytorch:2.10-r36.4.tegra-aarch64-cu126-22.04` se compila con:

```bash
CUDA_VERSION=12.6 LSB_RELEASE=22.04 jetson-containers build pytorch:2.10
```

## Parches aplicados

Con PyTorch 2.10.0 ya disponible en la base, solo son necesarios 4 parches (vs 11 en la build anterior con torch 2.8.0):

1. **CMakeLists.txt** — añade `8.7+PTX` a MARLIN_ARCHS, MARLIN_MOE_ARCHS y FA2_ARCHS para compilar kernels Marlin GPTQ + flash-attn como SASS nativo sm_87.
2. **cuda.py** — pre-carga la `libcuda.so.1` real de Tegra con `RTLD_GLOBAL` antes de `import vllm._C`. El stub que inyecta el runtime de NVIDIA carece de `cuPointerGetAttribute`.
3. **fp8_quant hasattr guard** — `matcher_utils.py` y `rms_quant_fusion.py`: `per_token_group_fp8_quant` no se compila para sm_87 (requiere sm_89+).
4. **gdn_linear_attn dynamo guard** — reemplaza `hasattr(self, "in_proj_qkv")` por un flag booleano que torch.compile sí puede guardar.

## Configuraciones según caso de uso

Flags comunes a todas las configuraciones (no cambiar):
```
--dtype half                     # float16, más estable que bfloat16 en sm_87
--tool-call-parser qwen3_coder   # parser correcto para este modelo
--reasoning-parser qwen3         # maneja bloques <think>...</think>
--enable-auto-tool-choice
--no-enable-log-requests
```

Sin `--enforce-eager` (CUDA graphs activos). Sin `--quantization` (vLLM detecta gptq_marlin automáticamente).

---

### Caso 1: agentes / uso interactivo (1–2 usuarios)

Prioridad: **mínima latencia por request**. MTP acelera la generación ~1.3–1.4x cuando hay pocas requests simultáneas y la acceptance rate es alta (70–95% observado).

```yaml
command: >
  palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
  --host 0.0.0.0 --port 8001
  --dtype half
  --gpu-memory-utilization 0.70
  --max-model-len 64000
  --max-num-batched-tokens 64000
  --max-num-seqs 4
  --enable-prefix-caching
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --no-enable-log-requests
```

Rendimiento observado en Jetson AGX Orin 64GB:
- Generación: **32–35 t/s** (1 request) / **47–48 t/s** total (2 requests)
- Prefill: ~350–1000 t/s
- MTP acceptance rate: 70–95%
- Memoria: ~50 GB

> ⚠️ Prefix caching con capas Mamba es experimental en vLLM 0.19. Si aparecen
> respuestas incoherentes en contextos largos, quitar `--enable-prefix-caching`.

---

### Caso 2: muchos usuarios concurrentes (5+ simultáneos)

Prioridad: **máximo throughput total**. Sin MTP — el batch size inflado por los
draft tokens penaliza cuando hay muchas secuencias activas. Más `--max-num-seqs`
para saturar bien la GPU con trabajo real.

```yaml
command: >
  palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
  --host 0.0.0.0 --port 8001
  --dtype half
  --gpu-memory-utilization 0.75
  --max-model-len 32000
  --max-num-batched-tokens 32000
  --max-num-seqs 16
  --enable-prefix-caching
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --no-enable-log-requests
```

Diferencias clave vs Caso 1:
- Sin `--speculative-config` — MTP no ayuda con muchos usuarios concurrentes
- `--max-num-seqs 16` — más requests en paralelo, mayor throughput total
- `--max-model-len 32000` — contexto reducido para que quepan más KV slots
- `--gpu-memory-utilization 0.75` — más presupuesto KV para acomodar 16 seqs

---

### Caso 3: contexto muy largo, un usuario

Prioridad: **máximo contexto posible**. Sin MTP (no ayuda en prefill) y sin
prefix caching experimental para máxima estabilidad.

```yaml
command: >
  palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
  --host 0.0.0.0 --port 8001
  --dtype half
  --gpu-memory-utilization 0.70
  --max-model-len 75000
  --max-num-batched-tokens 75000
  --max-num-seqs 1
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --no-enable-log-requests
```

> ⚠️ El pool KV con MTP activo es ~65–70K tokens. Sin MTP puede llegar a ~77K.
> Con `--max-model-len 75000` verificar en los logs de arranque que:
> `GPU KV cache size: XXXXX tokens` sea ≥ 75000. Si no, reducir a 65000.
>
> Prefill de 75K tokens ≈ 3–4 minutos a ~350 t/s. Planificar en consecuencia.

---

### Resumen de trade-offs

| | Caso 1 (agentes) | Caso 2 (concurrencia) | Caso 3 (contexto largo) |
|---|---|---|---|
| MTP | ✅ sí | ❌ no | ❌ no |
| Prefix caching | ✅ sí | ✅ sí | ❌ opcional |
| max-num-seqs | 4 | 16 | 1 |
| max-model-len | 64K | 32K | 75K |
| Latencia/req | Baja | Media-alta | Alta (prefill) |
| Throughput total | Medio | Alto | Bajo |
| Memoria | ~50 GB | ~48 GB | ~46 GB |

---

## Bug: GPU estado corrupto tras reinicio de container

### Síntoma

La primera vez que arranca todo funciona. Al reiniciar el container (con `docker compose restart` o `docker compose down && up`), vLLM falla al cabo de ~15 minutos con:

```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at
"/opt/pytorch/c10/cuda/CUDACachingAllocator.cpp":1154
```

El mismo problema afecta a otros containers con GPU (ComfyUI, etc.). Un reboot del sistema lo arregla.

### Causa raíz

El `stop_grace_period` por defecto de Docker es **10 segundos**. Si vLLM no termina en ese tiempo, Docker envía **SIGKILL**. PyTorch no tiene oportunidad de destruir el contexto CUDA limpiamente.

En la Jetson Orin con GPU integrada (nvgpu), el driver `nvhost` del kernel retiene estado de esa sesión CUDA. Cuando el siguiente proceso intenta hacer la allocación grande del KV cache y el `CUDACachingAllocator` llama a `cudaMemGetInfo` vía NVML, encuentra estado inconsistente → assert fatal.

Evidencia en los logs: la run fallida reporta **15.02 GiB de KV cache disponibles** (vs 5.89 GiB en la run que funcionó). El profiling pasa, el modelo carga, pero `torch.zeros()` sobre los 15 GiB peta porque esa llamada NVML encuentra el estado corrupto de la run anterior.

El GPU **railgating** (power-gate automático, activado por defecto en Jetson) agrava el problema: el ciclo de power-off/on entre reinicios puede dejar el estado del driver en una situación inconsistente.

### Fixes aplicados en este repo

**1. `stop_grace_period: 120s` en `docker-compose.yml`** ← fix principal

Da 2 minutos a vLLM para cerrarse limpiamente con SIGTERM antes de que Docker lo mate. vLLM maneja SIGTERM correctamente y destruye el contexto CUDA.

**2. CUDA pre-flight en `docker-entrypoint.sh`**

Antes de cargar el modelo, el entrypoint intenta una allocación de 512 MiB que fuerza el mismo código path (`CUDACachingAllocator` → `cudaMemGetInfo`) que fallaría durante el KV cache. Si la GPU está en mal estado, el error aparece **en segundos** en vez de tras 15 minutos de carga del modelo:

```
[entrypoint] FATAL: GPU estado corrupto (run anterior termino con SIGKILL): ...
[entrypoint] Fix rapido:  sudo reboot
[entrypoint] Fix sin reboot: ver leeme.txt o ejecutar reset-gpu.sh en el host
```

### Fix del railgate (ejecutar una vez en el host)

Desactivar el GPU railgate mantiene la GPU siempre encendida, eliminando el ciclo power-off/on entre reinicios de container:

```bash
# Aplicar ahora mismo (hasta el próximo reboot):
sudo sh -c 'echo 0 > /sys/devices/platform/bus@0/17000000.gpu/railgate_enable'

# Hacerlo permanente (persiste tras reboot):
sudo cp nvidia-gpu-no-railgate.service /etc/systemd/system/
sudo systemctl enable --now nvidia-gpu-no-railgate.service
```

El consumo adicional es mínimo en un sistema dedicado a inferencia.

### Recuperación sin reboot (si ya estás en el estado malo)

```bash
sudo ./reset-gpu.sh
docker compose --profile agents up -d
```

`reset-gpu.sh` para los containers, espera a que liberen `/dev/nvhost-gpu`, limpia la caché del sistema y verifica que la GPU responde antes de salir.

### Resumen de archivos relevantes

| Archivo | Propósito |
|---|---|
| `docker-compose.yml` | `stop_grace_period: 120s` — previene el SIGKILL |
| `docker-entrypoint.sh` | CUDA pre-flight — falla rápido si GPU en mal estado |
| `reset-gpu.sh` | Reset GPU en el host sin reboot |
| `nvidia-gpu-no-railgate.service` | Servicio systemd para deshabilitar railgate al boot |


### Prueba el modelo

```bash
curl http://localhost:8001/v1/chat/completions  -H "Content-Type: application/json"   -d '{"model": "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4", "messages": [{"role": "user", "content": "Hola!, puedes darme un listado de los discos de michael jackson? quiero una tabla en formato markdown con el nombre y la fecga , respondeme de manera breve"}]}'
```

** Importante levantar el modelo puede llevar 5 a 10 minutos, sea paciente
