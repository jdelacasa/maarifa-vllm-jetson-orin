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

## Configuración validada (docker-compose)

```
--dtype half
--gpu-memory-utilization 0.70
--max-model-len 44000
--max-num-batched-tokens 44000
--max-num-seqs 4
--no-enable-log-requests
```

Sin `--enforce-eager` (CUDA graphs activos). Sin `--quantization` (vLLM detecta gptq_marlin automáticamente).

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
