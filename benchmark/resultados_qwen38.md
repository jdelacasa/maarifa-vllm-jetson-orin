# Benchmark — Qwen3.8-27B-AWQ-INT4 en Jetson AGX Orin 64GB

**Hardware:** Jetson AGX Orin 64GB · GPU integrada nvgpu sm_87 · LPDDR5 64 GB unificada
**Stack:** vLLM 0.23.0 (imagen genérica `vllm/vllm-openai:latest`, sin parches sm_87) · JetPack **R39.2** (L4T), CUDA 13.2, driver 595.78
**Modelo:** `cyankiwi/Qwen3.8-27B-AWQ-INT4` (dense, 27B parámetros activos por forward pass — no MoE)
**Fecha:** 2026-08-16

> **Cambio de entorno respecto al benchmark anterior:** entre el benchmark de Qwen3.6-35B-A3B
> y este, el Jetson fue actualizado de JetPack 6.2.1 (L4T R36.4, CUDA 12.6) a **JetPack R39.2
> (CUDA 13.2)**. Ninguna imagen custom del repo (parches sm_87: Marlin, libcuda Tegra, fp8 guard,
> gdn dynamo guard) sirve ya — la imagen genérica `vllm/vllm-openai:latest` (vLLM 0.23.0) funciona
> sin parchear en el nuevo JetPack.

---

## Configuración usada

```yaml
image: vllm/vllm-openai:latest
command:
  --dtype half
  --gpu-memory-utilization 0.70      # bajado de 0.77: otros contenedores (comfyui, etc.)
                                      # ya usan ~16GB de la memoria unificada
  --max-model-len 80000              # bajado de 128000: KV cache no daba para más
                                      # con la memoria libre real del host
  --max-num-batched-tokens 64000
  --max-num-seqs 4
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --limit-mm-per-prompt '{"image":4}'
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --no-enable-log-requests
  # SIN --enable-prefix-caching: descartado por el bug de corrupción de KV
  # reportado para esta arquitectura GDN (ver models/qwen38-27b.yml)
environment:
  VLLM_MARLIN_USE_ATOMIC_ADD=1
KV cache pool: 93,571 tokens
Tiempo de arranque: ~9 min (incluye 230s de compilación)
```

### ⚠️ Comparación NO apples-to-apples con el benchmark anterior

- **Prefix caching estaba ACTIVADO** en el benchmark de Qwen3.6 y **DESACTIVADO** acá (riesgo de
  corrupción de KV documentado para GDN). Esto castiga fuerte el TTFT en contextos largos con
  prompts repetidos entre reps — el benchmark viejo se beneficiaba de hasta 10-12x de mejora por
  cache hit en esos casos (ver `resultados.md`, sección "Efecto del prefix cache").
- `max-model-len` (80K vs 128K) y `gpu-memory-utilization` (0.70 vs 0.77) más bajos por memoria
  GPU ya ocupada por otros contenedores en el host.
- Stack de software completamente distinto (vLLM 0.23.0 sin parches vs vLLM 0.20.2 con 5 parches
  sm_87 a medida).

Con esas salvedades, sigue sirviendo para comparar el **tipo de arquitectura**: MoE 3B-activos
(Qwen3.6-35B-A3B) vs dense 27B-activos (Qwen3.8-27B).

---

## Tabla completa

| ctx | tok entrada | conc | TTFT_med | TTFT_p90 | TPS/req | throughput | éxito |
|---|---|---|---|---|---|---|---|
| tiny   | ~50   | 1 |   0.51s |   0.51s | 17.8 | 16.0 tok/s | 100% |
| tiny   | ~50   | 2 |   2.08s |   3.46s | 17.9 | 22.1 tok/s | 100% |
| tiny   | ~50   | 4 |   7.42s |  13.77s | 17.3 | 24.0 tok/s | 100% |
| tiny   | ~50   | 6 |   1.08s |   6.05s | 15.6 | 40.8 tok/s | 100% |
| tiny   | ~50   | 8 |   2.98s |   7.50s | 15.2 | 49.8 tok/s | 100% |
| small  | ~500  | 1 |   1.22s |   1.23s | 16.9 | 15.8 tok/s | 100% |
| small  | ~500  | 2 |   2.12s |   2.12s | 16.4 | 28.1 tok/s | 100% |
| small  | ~500  | 4 |   3.99s |   4.80s | 14.8 | 47.8 tok/s | 100% |
| small  | ~500  | 6 |   4.02s |  24.09s | 15.1 | 41.5 tok/s | 100% |
| small  | ~500  | 8 |  13.18s |  28.31s | 14.1 | 47.9 tok/s | 100% |
| medium | ~2k   | 1 |   3.57s |   3.64s | 13.1 | 11.3 tok/s | 100% |
| medium | ~2k   | 2 |   6.62s |   7.07s | 13.2 | 19.6 tok/s | 100% |
| medium | ~2k   | 4 |  14.00s |  14.04s | 11.5 | 29.5 tok/s | 100% |
| medium | ~2k   | 6 |  13.96s |  44.84s | 10.6 | 25.6 tok/s | 100% |
| medium | ~2k   | 8 |  26.36s |  54.34s |  9.6 | 29.4 tok/s | 100% |
| large  | ~8k   | 1 |  14.01s |  14.01s | 13.5 |  8.3 tok/s | 100% |
| large  | ~8k   | 2 |  28.26s |  28.77s | 11.8 | 11.1 tok/s | 100% |
| large  | ~8k   | 4 |  57.96s |  58.10s | 10.6 | 13.7 tok/s | 100% |
| large  | ~8k   | 6 |  58.04s | 112.65s |  7.6 | 13.2 tok/s | 100% |
| large  | ~8k   | 8 |  77.41s | 141.58s |  6.6 | 14.1 tok/s | 100% |
| xlarge | ~32k  | 1 |  62.88s |  62.96s | 11.3 |  3.4 tok/s | 100% |
| xlarge | ~32k  | 2 | 123.23s | 126.48s |  9.7 |  3.9 tok/s | 100% |
| xlarge | ~32k  | 4 | 170.82s | 284.92s |  8.5 |  3.8 tok/s | 100% |
| xlarge | ~32k  | 6 | 253.10s | 442.80s |  3.3 |  3.8 tok/s | 100% |
| xlarge | ~32k  | 8 | 253.31s | 444.23s |  3.2 |  3.0 tok/s | **75%** (2 timeouts) |
| 64k    | ~64k  | — | — | — | — | — | **no corrido** (cortado antes de empezar, ver nota) |

**64k (xxlarge):** no se corrió — el bloque `xlarge/concurrencia=8` ya tardó ~10min por
repetición con TTFT p90 >7min y 2 timeouts de 8, señal clara de que 64k iba a ser peor y a
tomar otra hora+. Se cortó el sweep ahí por decisión explícita.

---

## Comparación de throughput del sistema (tok/s totales) — MoE vs dense

| ctx \ conc | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| tiny   — Qwen3.6 (MoE) | 32.7 | 51.0 | 99.1 | — | — |
| tiny   — Qwen3.8 (dense) | 16.0 | 22.1 | 24.0 | 40.8 | 49.8 |
| small  — Qwen3.6 (MoE) | 30.9 | 49.3 | 93.4 | — | — |
| small  — Qwen3.8 (dense) | 15.8 | 28.1 | 47.8 | 41.5 | 47.9 |
| medium — Qwen3.6 (MoE) | 27.3 | 41.0 | 74.4 | 74.6 | 73.7 |
| medium — Qwen3.8 (dense) | 11.3 | 19.6 | 29.5 | 25.6 | 29.4 |
| large  — Qwen3.6 (MoE) | 23.1 | 38.8 | 63.6 | 67.6 | 81.7 |
| large  — Qwen3.8 (dense) |  8.3 | 11.1 | 13.7 | 13.2 | 14.1 |
| xlarge — Qwen3.6 (MoE) | 14.7 | 32.0 | 45.2 | 42.8 | 55.8 |
| xlarge — Qwen3.8 (dense) |  3.4 |  3.9 |  3.8 |  3.8 |  3.0 |

### Ratio Qwen3.6/Qwen3.8 (cuánto más lento es el dense, a igual conc)

| ctx | conc=1 | conc=4 | conc=8 |
|---|---|---|---|
| tiny   | 2.0x | 4.1x | 2.0x |
| small  | 2.0x | 2.0x | 1.9x |
| medium | 2.4x | 2.5x | 2.5x |
| large  | 2.8x | 4.6x | 5.8x |
| xlarge | 4.3x | 11.9x | **18.6x** |

---

## Análisis

- **Contexto corto (tiny/small), 1 usuario:** el dense ya arranca ~2x más lento que el MoE. Es
  el piso esperable por el ratio de parámetros activos (27B vs 3B), algo amortiguado por MTP.
- **La brecha se dispara con el contexto.** En `xlarge` (~32k) el dense es hasta **18.6x más
  lento** que el MoE a 8 usuarios. Dos factores se suman ahí, no solo el cómputo dense:
  1. **Sin prefix caching** (ver caveat arriba) — cada rep repite el prefill completo del prompt,
     cosa que el benchmark del MoE no pagaba.
  2. Las 16 capas de full-attention (de 64) siguen escalando con el contexto igual que un
     transformer clásico; a diferencia del MoE viejo (10 de 40 capas full-attention aprox.,
     proporción distinta), acá cada forward pass además mueve pesos densos de 27B.
- **Falla de confiabilidad:** a partir de `large/6` empiezan a aparecer TTFT p90 muy por encima
  de la mediana (colas largas), y en `xlarge/8` un 25% de requests dieron timeout — el modelo
  MoE nunca mostró fallos en el mismo rango de la tabla vieja.
- **Contextos cortos con concurrencia alta (tiny/small a 6-8) son el único terreno donde el dense
  se acerca al MoE** (~1.9-2x, no 5-18x) — ahí MTP + batch grande compensan mejor.

## Conclusión práctica

Para el patrón de uso de este repo (agentes, contexto medio/largo, varios usuarios) **Qwen3.6-35B-A3B
sigue siendo la mejor opción en este hardware**. Qwen3.8-27B-AWQ-INT4 solo tiene sentido si:
- se puede reactivar prefix caching de forma segura (parche/fix para el bug de KV en GDN), y
- el caso de uso es de contexto corto y alta concurrencia (chat corto, no agentes con contexto
  largo).

Con contexto largo (≥8k) el dense se vuelve significativamente más lento y menos confiable
(timeouts) que el MoE actual.

---

## Efecto de `reasoning_effort` (xhigh / medium / low)

Qwen3.8 expone niveles de esfuerzo de razonamiento vía `chat_template_kwargs:
{"reasoning_effort": "xhigh"|"medium"|"low"}` en el request (no es un flag de servidor, es
por-request). `xhigh` es el default si no se manda nada — es lo que corrió el sweep de arriba.

Subset rápido: concurrencia 1 y 4, ctx tiny/medium/large, max-tokens 300, **repeats=1** para
medium y low (vs repeats=2 para el xhigh de referencia — algo más de ruido en estos dos).

| ctx | conc | modo | TTFT_med | TPS/req | throughput |
|---|---|---|---|---|---|
| tiny   | 1 | xhigh  | 0.51s  | 17.8 | 16.0 tok/s |
| tiny   | 1 | medium | 0.42s  | 20.4 | 18.2 tok/s |
| tiny   | 1 | low    | 0.51s  | 19.2 | 16.8 tok/s |
| tiny   | 4 | xhigh  | ~1.07s (rep warm) | 17.3 | 24.0 tok/s |
| tiny   | 4 | medium | 0.697s | 16.9 | 51.8 tok/s |
| tiny   | 4 | low    | 0.975s | 17.4 | 43.3 tok/s |
| medium | 1 | xhigh  | 3.57s  | 13.1 | 11.3 tok/s |
| medium | 1 | medium | 3.60s  | 17.9 | 14.8 tok/s |
| medium | 1 | low    | 3.64s  | 15.4 | 13.0 tok/s |
| medium | 4 | xhigh  | 14.00s | 11.5 | 29.5 tok/s |
| medium | 4 | medium | 13.60s | 15.5 | 35.8 tok/s |
| medium | 4 | low    | 13.93s | 14.8 | 32.8 tok/s |
| large  | 1 | xhigh  | 14.01s | 13.5 |  8.3 tok/s |
| large  | 1 | medium | 14.03s | 15.8 |  9.1 tok/s |
| large  | 1 | low    | 13.90s | 16.9 |  9.5 tok/s |
| large  | 4 | xhigh  | 57.96s | 10.6 | 13.7 tok/s |
| large  | 4 | medium | 56.84s | 13.1 | 14.7 tok/s |
| large  | 4 | low    | 57.45s | 12.2 | 14.6 tok/s |

**Lecturas:**

- **TTFT no cambia entre modos** (diferencias <1s, dentro del ruido). Tiene sentido: TTFT mide
  hasta el primer token, y el reasoning_effort solo afecta *qué tan largo/profundo* piensa el
  modelo después de arrancar — no toca el prefill.
- **TPS/req sube ~15-25% con medium/low vs xhigh** en la mayoría de los bloques (ej. large/conc1:
  13.5 → 15.8-16.9 tok/s). Hipótesis: con instrucciones de razonar menos, el texto generado es
  más corto/predecible, lo que sube la tasa de aceptación de MTP (no medida directamente acá,
  este script no expone acceptance rate). No es una mejora enorme — no esperes que `low` resuelva
  el problema de fondo del contexto largo visto arriba, ese es un cuello de botella de prefill/
  memoria, no de reasoning_effort.
- **En `medium`/`large` con max-tokens=300, la respuesta se corta antes de terminar de razonar**
  (`output_tokens` pega el techo de 300 en casi todos los casos) independientemente del modo —
  con este límite de tokens no se llega a ver el ahorro real de tokens totales que da `low` vs
  `xhigh` (`low` debería necesitar menos tokens de razonamiento para llegar a la respuesta final,
  pero acá se corta antes de que eso se note). Para medir eso habría que correr con `max-tokens`
  más alto y comparar tokens totales hasta `finish_reason=stop`, no tok/s.
