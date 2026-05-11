# Benchmark — vLLM en Jetson AGX Orin 64GB

**Hardware:** Jetson AGX Orin 64GB · GPU integrada nvgpu sm_87 · LPDDR5 64 GB unificada  
**Stack:** vLLM 0.20.2 · PyTorch 2.10.0 · CUDA 12.6 · JetPack 6.2.1  
**Modelo:** palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 (MoE, 3B parámetros activos por forward pass)  
**Última actualización:** 2026-05-11

---

## Configuración activa (v0.20.2)

```yaml
image: jdelacasa/vllm-jetson:0.20.2-torch2.10.0-cu126-l4tr36.4-sm87
command:
  --dtype half
  --gpu-memory-utilization 0.77
  --max-model-len 128000
  --max-num-batched-tokens 64000
  --max-num-seqs 8
  --enable-prefix-caching
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
  --limit-mm-per-prompt '{"image":4}'
environment:
  VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=0
  VLLM_MARLIN_USE_ATOMIC_ADD=1
KV cache pool: 411,151 tokens (9.55 GiB)
```

---

## Benchmark completo v0.20.2 — throughput por contexto y concurrencia

**Metodología:**
- `max_tokens=300` por request, 2 repeticiones por bloque, con thinking activado (reasoning-parser qwen3)
- TTFT = tiempo hasta el primer token generado (incluye prefill + inicio de reasoning)
- TPS/req = tokens/s de decode por usuario individual (excluye prefill)
- Throughput = tokens/s totales del sistema (suma de todos los usuarios)
- Script: `benchmark/bench.py`

### Tabla completa

| ctx | tok entrada | conc | TTFT_med | TTFT_p90 | TPS/req | throughput |
|---|---|---|---|---|---|---|
| tiny   | ~50   | 1 | 0.42s |  0.43s | 34.2 |  32.7 tok/s |
| tiny   | ~50   | 2 | 0.75s |  0.79s | 28.4 |  51.0 tok/s |
| tiny   | ~50   | 4 | 0.81s |  0.82s | 27.3 |  99.1 tok/s |
| small  | ~500  | 1 | 0.45s |  0.45s | 32.4 |  30.9 tok/s |
| small  | ~500  | 2 | 0.79s |  0.84s | 26.9 |  49.3 tok/s |
| small  | ~500  | 4 | 1.23s |  1.23s | 26.1 |  93.4 tok/s |
| medium | ~2k   | 1 | 1.06s |  1.06s | 30.3 |  27.3 tok/s |
| medium | ~2k   | 2 | 2.96s |  3.96s | 26.2 |  41.0 tok/s |
| medium | ~2k   | 4 | 3.56s |  3.71s | 24.7 |  74.4 tok/s |
| medium | ~2k   | 6 | 8.26s | 12.68s | 20.6 |  74.6 tok/s |
| medium | ~2k   | 8 | 6.72s | 25.55s | 18.9 |  73.7 tok/s |
| large  | ~8k   | 1 | 2.45s |  3.54s | 28.5 |  23.1 tok/s |
| large  | ~8k   | 2 | 2.51s |  2.62s | 24.3 |  38.8 tok/s |
| large  | ~8k   | 4 | 4.74s |  4.76s | 21.5 |  63.6 tok/s |
| large  | ~8k   | 6 | 8.83s | 10.53s | 17.1 |  67.6 tok/s |
| large  | ~8k   | 8 | 9.64s |  9.71s | 15.5 |  81.7 tok/s |
| xlarge | ~32k  | 1 | 8.24s | 15.04s | 24.8 |  14.7 tok/s |
| xlarge | ~32k  | 2 | 2.71s |  2.81s | 18.8 |  32.0 tok/s |
| xlarge | ~32k  | 4 | 5.17s |  5.18s | 14.2 |  45.2 tok/s |
| xlarge | ~32k  | 6 |14.15s | 20.68s | 10.9 |  42.8 tok/s |
| xlarge | ~32k  | 8 |10.11s | 10.14s |  9.2 |  55.8 tok/s |
| 64k    | ~64k  | 1 |14.66s | 27.10s | 20.3 |  10.2 tok/s |
| 64k    | ~64k  | 2 | 4.26s |  4.43s | 14.4 |  23.9 tok/s |
| 64k    | ~64k  | 4 | 8.38s |  8.77s |  9.5 |  29.8 tok/s |
| 64k    | ~64k  | 6 |11.32s | 12.52s |  7.1 |  33.2 tok/s |
| 64k    | ~64k  | 8 |16.50s | 16.82s |  5.6 |  33.8 tok/s |

### Throughput del sistema por contexto (tok/s totales)

| ctx \ conc | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| tiny (~50)   | 32.7 | 51.0 | **99.1** | — | — |
| small (~500) | 30.9 | 49.3 | **93.4** | — | — |
| medium (~2k) | 27.3 | 41.0 | 74.4 | 74.6 | 73.7 |
| large (~8k)  | 23.1 | 38.8 | 63.6 | 67.6 | **81.7** |
| xlarge (~32k)| 14.7 | 32.0 | 45.2 | 42.8 | **55.8** |
| 64k          | 10.2 | 23.9 | 29.8 | 33.2 | **33.8** |

---

## Análisis y conclusiones

### Saturación de throughput según contexto

El throughput del sistema satura en distintos puntos según el tamaño del contexto:

- **Contexto corto (≤2k):** satura en 4 usuarios (~74-99 tok/s). Pasar a 6-8 no gana nada, solo aumenta TTFT.
- **Contexto largo (≥8k):** sorprendentemente, 8 usuarios da *más* throughput que 6. El batch más grande amortiza mejor los kernels de atención: con large/8 se obtienen 81.7 tok/s vs 63.6 tok/s con 4 usuarios.
- **64k de contexto:** techo absoluto ~34 tok/s independientemente de cuántos usuarios. El cuello de botella es el ancho de banda de la LPDDR5: cada decode step lee el KV cache completo de 10 capas × N tokens para todos los usuarios simultáneos.

### Punto óptimo para uso interactivo

| escenario | TTFT | throughput | veredicto |
|---|---|---|---|
| 4 usuarios, contexto ≤2k  | ~1-4s  | ~74 tok/s | excelente |
| 4 usuarios, contexto ~8k  | ~5s    | ~64 tok/s | muy bueno |
| 4-6 usuarios, contexto ~32k | ~5-14s | ~43-55 tok/s | bueno |
| 8 usuarios, contexto ~64k | ~17s   | ~34 tok/s | límite de lo tolerable |

El sweet spot real es **4-6 usuarios con contextos de hasta 32k**: ~45 tok/s de throughput con TTFTs por debajo de 15s.

### Efecto del prefix cache

El prefix cache tiene impacto dramático cuando múltiples requests comparten el mismo contexto largo:

| contexto | TTFT frío | TTFT cacheado | mejora |
|---|---|---|---|
| ~32k (1 usuario) | 15.0s | 1.5s | **10x** |
| ~64k (1 usuario) | 27.1s | 2.2s | **12x** |

En producción, los system prompts largos compartidos se pagan solo una vez. Todos los usuarios siguientes con el mismo prefijo tienen latencia casi equivalente a un contexto corto.

### Degradación de TPS/req con contexto creciente

| ctx | TPS/req (conc=1) | TPS/req (conc=4) | TPS/req (conc=8) |
|---|---|---|---|
| tiny  | 34.2 | 27.3 | — |
| small | 32.4 | 26.1 | — |
| medium| 30.3 | 24.7 | 18.9 |
| large | 28.5 | 21.5 | 15.5 |
| xlarge| 24.8 | 14.2 |  9.2 |
| 64k   | 20.3 |  9.5 |  5.6 |

La caída se debe a que las 10 capas de full-attention escalan linealmente con la longitud del KV cache (O(N) memoria leída por token generado). Las 30 capas de linear-attention (GDN) del modelo no tienen este problema, lo que hace que la degradación sea menos severa que en un transformer puro.

---

## Thinking vs No-think

El modo de razonamiento (chain-of-thought) no penaliza el throughput:

| ctx | conc | throughput CON thinking | throughput SIN thinking | diferencia |
|---|---|---|---|---|
| tiny  | 4 | 99.1 tok/s | 96.2 tok/s | −3% |
| small | 4 | 93.4 tok/s | 90.7 tok/s | −3% |
| medium| 4 | 74.4 tok/s | 73.2 tok/s | −2% |

La GPU genera tokens de razonamiento y de respuesta a la misma velocidad. Desactivar thinking (`/no_think`) reduce el total de tokens generados pero no libera capacidad GPU. Solo merece la pena para casos donde la latencia total de respuesta es crítica y la calidad puede sacrificarse.

---

## Historial — configuraciones evaluadas en v0.19.x (2026-04-26)

| ID | Cuantización | MTP | Prefix cache | max-model-len |
|---|---|---|---|---|
| A | GPTQ Int4 gptq_marlin | No  | No  | 44K |
| B | GPTQ Int4 gptq_marlin | Sí (2 tok) | No  | 44K |
| C | GPTQ Int4 gptq_marlin | Sí (2 tok) | Sí  | 64K |
| D | AWQ Int4              | Sí (2 tok) | Sí  | 64K |

Resultados destacados:
- **Config B vs A:** MTP da +1.3–1.4x en un usuario, acceptance rate 65–95%
- **Config C:** prefix cache hit rate 23.2% con mismo system prompt
- **GPTQ vs AWQ:** GPTQ ~10–15% más rápido en sm_87 gracias a kernels Marlin nativos

---

## Notas de infraestructura

- **GPU railgate:** `echo 0 > /sys/devices/platform/bus@0/17000000.gpu/railgate_enable` — previene corrupción de estado CUDA entre reinicios.
- **stop_grace_period: 120s** — crítico para que PyTorch libere el contexto CUDA limpiamente antes del SIGKILL.
- **gpu-memory-utilization 0.77** — valor máximo que pasa el check de arranque. Con 0.80 falla (`Free memory < desired`).
- **Tiempo de arranque con CUDA graphs:** 15–20 min en frío. Con caché en `/data/vllm-cache`: ~5–8 min.
- **--limit-mm-per-prompt:** requiere formato JSON (`'{"image":4}'`), no `image=4`.
