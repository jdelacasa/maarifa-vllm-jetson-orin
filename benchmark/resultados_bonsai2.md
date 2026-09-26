# Benchmark — Ternary Bonsai 2 27B en Jetson AGX Orin 64GB (llama.cpp)

**Hardware:** Jetson AGX Orin 64GB · sm_87 · LPDDR5 64 GB unificada · JetPack R39 (CUDA 13.2, driver 595.78)
**Stack:** fork de PrismML de llama.cpp (`prism-b10743-adfffbe`) compilado para sm_87 en contenedor (`bonsai/Dockerfile`, `models/bonsai2-27b.yml`)
**Modelo:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7,2 GB) + `mmproj-Q8_0` (0,6 GB). Base: Qwen3.8-27B, ternario (−1/0/+1)
**Fecha:** 2026-09-26

> El llama.cpp oficial NO sirve para estos GGUF (carga `Q2_0` y genera basura): hace falta el fork.
> vLLM no es un backend oficial de Bonsai 2. Existe un port de la comunidad
> (`fraserprice/bonsai-vllm`, solo Blackwell probado, sin visión) que NO se ha probado aquí.

## Configuración

```
-m PQ2_0 --mmproj Q8_0 -ngl 99 -fa on --jinja
-c 262144 -np 1        # contexto completo, un slot (los benchmarks de concurrencia usaron -c 65536 -np 4)
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05
reasoning_effort: medium
```

Notas de build: hay que enlazar contra el stub de libcuda (`-Wl,-rpath-link`) y borrar
`/usr/local/cuda/compat*` de la imagen runtime, porque el parser de `nvidia-cdi-hook` peta con esas libs.

## 1. Throughput vs Qwen3.8-27B AWQ (vLLM + MTP)

`benchmark/bench.py`, `--reasoning-effort medium`, 300 tokens de salida, `-np 4`.
Datos de Qwen: `bench_results_qwen38_medium.json` (solo tiene algunos bloques).

| Caso | Bonsai 2 | Qwen3.8 |
|---|---|---|
| tiny, 1 usuario | 13,6 tok/s, TTFT 0,45 s | 20,4 tok/s, TTFT 0,42 s |
| medium (~2k), 1 usuario | 13,4 tok/s, TTFT 3,4 s | 17,9 tok/s, TTFT 3,6 s |
| tiny, 4 usuarios | 6,9 tok/s/req, 22,7 tok/s total | 16,9 tok/s/req, 51,8 tok/s total |
| medium, 4 usuarios | 6,8 tok/s/req, 23,5 tok/s total, TTFT 4,5 s | 15,5 tok/s/req, 35,8 tok/s total, TTFT 13,6 s |

Bonsai es ~25–35 % más lento en decode con 1 usuario y mucho peor con concurrencia. Solo gana en TTFT
con medium y 4 usuarios. Todas las peticiones terminaron (100 %).

## 2. Contexto largo (`bonsai_longctx.py`, aguja a mitad, `-c 262144 -np 1`)

| Prompt | Prefill | Decode | ¿Recupera la aguja? |
|---|---|---|---|
| 12,9k tokens | 50 s (259 tok/s) | 12,6 tok/s | sí |
| 51,9k | 223 s (233 tok/s) | 10,7 tok/s | sí |
| 104k | 545 s (192 tok/s) | 8,8 tok/s | sí |
| 188,6k | 1.250 s (151 tok/s) | 6,9 tok/s | sí |

## 3. Memoria

Contenedor con `-c 262144`: **~25 GiB** (pesos 7,2 + mmproj 0,6 + KV + buffers), estable de 13k a 188k tokens.
RAM del sistema con todo cargado: 37 GB de 62 GB usados, **~25,7 GB disponibles**.
Qwen3.8 en vLLM reservaba el 70 % (~43 GB) para solo 80k de contexto.

## 4. Cache de prefijo en conversación (`bonsai_conv_cache.py`)

Documento de ~8,3k tokens + 6 turnos encadenados.

Sin thinking (`max_tokens` 120):

| Turno | Con cache: procesa / reutiliza | Prefill con cache | Prefill sin cache |
|---|---|---|---|
| 1 | 8.346 / 0 | 32,1 s | 32,0 s |
| 2–6 | 22–30 / ~8,4–8,5k | 0,5–0,6 s | 32,5–33,4 s |
| **Total** | | **34,8 s** | **197,2 s** (−82 %) |

Con thinking (`reasoning_effort: medium`, `max_tokens` 2500):

| Turno | A: por defecto, cliente reenvía solo `content` | B: `--reasoning-preserve` + reenvía `reasoning_content` |
|---|---|---|
| 2–6 | 31–92 tokens procesados, 0,6–0,9 s | 20–29 tokens procesados, 0,5–0,6 s |

- La cache por defecto ya funciona con thinking: el template quita el razonamiento de turnos viejos y aun así se reutiliza el documento.
- `--reasoning-preserve` ahorra décimas de segundo y hace crecer el contexto (8,8k → 11,3k desde el turno 4). No compensa.
- El turno 1 de A no es válido: reutilizó 8.103 tokens de la prueba anterior (mismo documento en el slot).
- El coste real es el thinking: un turno trivial (contar palabras) agotó 2.500 tokens de razonamiento (195 s). Poner `max_tokens` generoso y valorar `--reasoning-budget`.

## Conclusiones

- Bonsai 2 ocupa ~25 GiB con 262k de contexto y recupera información en todo el rango probado.
- Decode 13,5 tok/s (1 usuario), 6,9 tok/s con 188k de contexto. Más lento que Qwen3.8+MTP, sin batching competitivo.
- Prefill ~150–260 tok/s: un primer prompt largo cuesta minutos, pero con prompt cache las preguntas siguientes sobre el mismo prefijo cuestan ~1 s.
- Pendiente: calidad real (resumen diario contra Qwen3.8) y, opcionalmente, probar el port vLLM y `PTQ1_0`.
