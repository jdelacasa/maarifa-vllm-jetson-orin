#!/usr/bin/env python3
"""
Benchmark de throughput y latencia para vLLM en Jetson AGX Orin.

Mide TTFT, tokens/s por request y throughput total del sistema
según usuarios concurrentes y tamaño de contexto de entrada.

Uso rápido:
    python3 bench.py
    python3 bench.py --concurrency 1 2 4 --ctx tiny small medium
    python3 bench.py --include-xlarge --max-tokens 500 --repeats 3
"""

import asyncio
import aiohttp
import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8001"
MODEL = "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"

# Texto de relleno en castellano para generar contextos largos (~4 chars/token)
_FILLER = (
    "La inteligencia artificial ha transformado múltiples industrias en la última década. "
    "Los modelos de lenguaje de gran escala permiten realizar tareas complejas de razonamiento, "
    "generación de texto y análisis semántico con una precisión sin precedentes. "
    "El hardware especializado como las GPUs y TPUs ha sido fundamental para hacer viable "
    "el entrenamiento y la inferencia de estos sistemas a gran escala. "
    "En el ámbito edge, plataformas como Jetson AGX Orin permiten ejecutar modelos MoE "
    "de decenas de miles de millones de parámetros aprovechando la memoria unificada. "
    "La arquitectura Mixture-of-Experts activa solo una fracción de los pesos en cada paso, "
    "reduciendo drásticamente el coste computacional sin sacrificar capacidad del modelo. "
)

# (tokens_entrada_aprox, pregunta)
CONTEXT_SIZES: dict[str, tuple[int, str]] = {
    "tiny":   (50,    "¿Cuál es la capital de Francia? Responde en una frase."),
    "small":  (500,   "Resume los puntos clave del texto en tres párrafos concisos."),
    "medium": (2_000, "Analiza en detalle el impacto del hardware edge en la IA moderna."),
    "large":  (8_000, "Realiza un análisis crítico exhaustivo: pros, contras y recomendaciones."),
    "xlarge": (32_000,"Proporciona un resumen ejecutivo detallado y un análisis técnico profundo."),
    "xxlarge": (64_000,"Sintetiza los puntos más relevantes del documento y elabora conclusiones accionables."),
}

DEFAULT_CONCURRENCY = [1, 2, 4, 6, 8]
DEFAULT_MAX_TOKENS  = 300
DEFAULT_REPEATS     = 2

# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ReqResult:
    ok: bool
    ttft: float      # s hasta primer token
    total: float     # s totales (incluyendo prefill)
    out_tokens: int  # tokens de salida (de usage chunk)
    tps: float       # tokens/s de decode (sin prefill)
    error: str = ""


@dataclass
class BlockResult:
    concurrency: int
    ctx_name: str
    ctx_tokens: int
    wall_time: float
    reqs: list[ReqResult] = field(default_factory=list)

    @property
    def ok(self) -> list[ReqResult]:
        return [r for r in self.reqs if r.ok]

    @property
    def success_rate(self) -> float:
        return len(self.ok) / len(self.reqs) if self.reqs else 0.0

    @property
    def median_ttft(self) -> float:
        v = [r.ttft for r in self.ok]
        return statistics.median(v) if v else 0.0

    @property
    def p90_ttft(self) -> float:
        v = sorted(r.ttft for r in self.ok)
        return v[int(len(v) * 0.9)] if v else 0.0

    @property
    def median_tps(self) -> float:
        v = [r.tps for r in self.ok]
        return statistics.median(v) if v else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.out_tokens for r in self.ok)

    @property
    def throughput(self) -> float:
        """Tokens/s totales del sistema (suma de todos los usuarios)."""
        return self.total_tokens / self.wall_time if self.wall_time > 0 else 0.0

# ─── Generador de prompts ─────────────────────────────────────────────────────

def build_prompt(approx_tokens: int, question: str, no_think: bool) -> str:
    chars_needed = max(0, approx_tokens * 4 - len(question) - 60)
    filler = (_FILLER * (chars_needed // len(_FILLER) + 1))[:chars_needed]
    base = f"Texto de referencia:\n{filler}\n\nPregunta: {question}" if filler else question
    return base

def system_msg(no_think: bool) -> Optional[dict]:
    if no_think:
        return {"role": "system", "content": "/no_think Eres un asistente útil y conciso."}
    return None

# ─── Request individual ───────────────────────────────────────────────────────

async def do_request(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int,
    no_think: bool,
) -> ReqResult:
    messages = []
    sys = system_msg(no_think)
    if sys:
        messages.append(sys)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    out_tokens = 0

    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return ReqResult(False, 0.0, time.perf_counter() - t0, 0, 0.0,
                                 f"HTTP {resp.status}: {body[:300]}")

            async for raw in resp.content:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Chunk final con usage (stream_options)
                if "usage" in chunk:
                    out_tokens = chunk["usage"].get("completion_tokens", out_tokens)
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content") or ""

                # El modelo usa reasoning-parser: los tokens salen en delta.reasoning
                # (pensamiento) y luego en delta.content (respuesta final).
                # TTFT se mide desde el primer token de cualquier tipo.
                reasoning = (choices[0].get("delta") or {}).get("reasoning") or ""
                any_token = content or reasoning

                if any_token and ttft is None:
                    ttft = time.perf_counter() - t0

    except asyncio.TimeoutError:
        return ReqResult(False, 0.0, time.perf_counter() - t0, 0, 0.0, "timeout")
    except Exception as exc:
        return ReqResult(False, 0.0, time.perf_counter() - t0, 0, 0.0, str(exc)[:200])

    total = time.perf_counter() - t0

    if ttft is None:
        return ReqResult(False, 0.0, total, out_tokens, 0.0, "sin tokens de salida")

    decode_time = total - ttft
    tps = out_tokens / decode_time if decode_time > 0.05 and out_tokens > 0 else 0.0
    return ReqResult(True, ttft, total, out_tokens, tps)

# ─── Bloque de N requests simultáneos ────────────────────────────────────────

async def run_block(
    concurrency: int,
    prompt: str,
    max_tokens: int,
    no_think: bool,
) -> tuple[list[ReqResult], float]:
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.perf_counter()
        tasks = [do_request(session, prompt, max_tokens, no_think) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    return list(results), wall

# ─── Warmup ───────────────────────────────────────────────────────────────────

async def warmup():
    print("Calentando servidor con una petición pequeña...", flush=True)
    reqs, _ = await run_block(1, "Responde solo 'ok'", 5, no_think=True)
    if reqs[0].ok:
        print(f"  warmup OK ({reqs[0].total:.1f}s)\n")
    else:
        print(f"  warmup FAILED: {reqs[0].error}\n")

# ─── Runner principal ─────────────────────────────────────────────────────────

async def run_bench(
    concurrency_levels: list[int],
    ctx_filter: Optional[list[str]],
    max_tokens: int,
    repeats: int,
    skip_xlarge: bool,
    no_think: bool,
) -> list[BlockResult]:

    await warmup()

    all_blocks: list[BlockResult] = []

    ctx_items = [
        (name, tokens, q)
        for name, (tokens, q) in CONTEXT_SIZES.items()
        if (not ctx_filter or name in ctx_filter) and not (skip_xlarge and name in ("xlarge", "xxlarge"))
    ]

    for ctx_name, ctx_tokens, question in ctx_items:
        prompt = build_prompt(ctx_tokens, question, no_think)
        prompt_chars = len(prompt)

        for n in concurrency_levels:
            print(f"▶ ctx={ctx_name} (~{ctx_tokens} tok entrada, {prompt_chars} chars) | "
                  f"concurrencia={n} | max_output={max_tokens}", flush=True)

            block = BlockResult(
                concurrency=n,
                ctx_name=ctx_name,
                ctx_tokens=ctx_tokens,
                wall_time=0.0,
            )

            total_wall = 0.0
            for rep in range(repeats):
                reqs, wall = await run_block(n, prompt, max_tokens, no_think)
                block.reqs.extend(reqs)
                total_wall += wall

                ok_count = sum(1 for r in reqs if r.ok)
                total_toks = sum(r.out_tokens for r in reqs if r.ok)
                tput = total_toks / wall if wall > 0 else 0.0
                ttfts = [r.ttft for r in reqs if r.ok]
                med_ttft = statistics.median(ttfts) if ttfts else 0.0
                print(f"  rep {rep+1}/{repeats}: wall={wall:.1f}s | ok={ok_count}/{n} | "
                      f"throughput={tput:.1f} tok/s | TTFT_med={med_ttft:.2f}s", flush=True)

            block.wall_time = total_wall
            all_blocks.append(block)
            _print_block_summary(block)
            print()

    return all_blocks


def _print_block_summary(b: BlockResult):
    ok = b.ok
    if not ok:
        print("  ✗ Todas las peticiones fallaron:")
        for r in b.reqs:
            if not r.ok:
                print(f"    {r.error}")
        return
    print(f"  TTFT mediana / p90 : {b.median_ttft:.2f}s / {b.p90_ttft:.2f}s")
    print(f"  TPS/req mediana    : {b.median_tps:.1f} tok/s")
    print(f"  Throughput sistema : {b.throughput:.1f} tok/s  (todos los usuarios)")
    print(f"  Tokens totales out : {b.total_tokens}")
    print(f"  Éxito              : {b.success_rate*100:.0f}%  ({len(ok)}/{len(b.reqs)})")

# ─── Tabla resumen ────────────────────────────────────────────────────────────

def print_table(blocks: list[BlockResult]):
    print("\n" + "═" * 84)
    print("TABLA RESUMEN — vLLM Jetson AGX Orin")
    print("═" * 84)
    print(f"{'ctx':<8} {'conc':>5} {'TTFT_med':>10} {'TTFT_p90':>10} "
          f"{'TPS/req':>9} {'throughput':>12} {'éxito':>7}")
    print("─" * 84)
    for b in blocks:
        print(
            f"{b.ctx_name:<8} {b.concurrency:>5} "
            f"{b.median_ttft:>10.2f} {b.p90_ttft:>10.2f} "
            f"{b.median_tps:>9.1f} {b.throughput:>12.1f} "
            f"{b.success_rate*100:>6.0f}%"
        )
    print("═" * 84)
    print("  TTFT      = tiempo hasta el primer token (s)")
    print("  TPS/req   = tokens/s de decode por request individual")
    print("  throughput = tokens/s totales del sistema (suma todos los usuarios)")


def save_json(blocks: list[BlockResult], path: str):
    data = []
    for b in blocks:
        data.append({
            "ctx": b.ctx_name,
            "ctx_tokens_approx": b.ctx_tokens,
            "concurrency": b.concurrency,
            "wall_time_total": round(b.wall_time, 2),
            "median_ttft_s": round(b.median_ttft, 3),
            "p90_ttft_s": round(b.p90_ttft, 3),
            "median_tps": round(b.median_tps, 1),
            "throughput_tok_s": round(b.throughput, 1),
            "success_rate": round(b.success_rate, 3),
            "total_output_tokens": b.total_tokens,
            "requests": [
                {
                    "ok": r.ok,
                    "ttft_s": round(r.ttft, 3),
                    "total_s": round(r.total, 2),
                    "output_tokens": r.out_tokens,
                    "tps": round(r.tps, 1),
                    "error": r.error,
                }
                for r in b.reqs
            ],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados → {path}")

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    global BASE_URL
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM Jetson AGX Orin — tokens/s por concurrencia y contexto",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--concurrency", nargs="+", type=int,
                        default=DEFAULT_CONCURRENCY,
                        metavar="N",
                        help="Niveles de concurrencia a probar")
    parser.add_argument("--ctx", nargs="+",
                        choices=list(CONTEXT_SIZES.keys()),
                        metavar="NAME",
                        help="Contextos a probar (por defecto todos excepto xlarge)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Tokens máximos de salida por request")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="Repeticiones por bloque (concurrencia × contexto)")
    parser.add_argument("--include-xlarge", action="store_true",
                        help="Incluir contexto xlarge (~32k tokens entrada) — muy lento")
    parser.add_argument("--no-think", action="store_true",
                        help="Deshabilitar razonamiento (chain-of-thought) durante el benchmark")
    parser.add_argument("--url", default=BASE_URL,
                        help="URL base del servidor vLLM")
    parser.add_argument("--output", default="benchmark/bench_results.json",
                        help="Fichero JSON donde guardar los resultados")
    args = parser.parse_args()

    BASE_URL = args.url

    print("═" * 84)
    print("vLLM Benchmark — Jetson AGX Orin 64GB")
    print(f"  URL          : {BASE_URL}")
    print(f"  Modelo       : {MODEL}")
    print(f"  Concurrencias: {args.concurrency}")
    print(f"  Contextos    : {args.ctx or [k for k in CONTEXT_SIZES if k != 'xlarge']}")
    print(f"  Max tokens   : {args.max_tokens}")
    print(f"  Repeticiones : {args.repeats}")
    print(f"  No-think     : {args.no_think}")
    print("═" * 84 + "\n")

    blocks = asyncio.run(run_bench(
        concurrency_levels=args.concurrency,
        ctx_filter=args.ctx,
        max_tokens=args.max_tokens,
        repeats=args.repeats,
        skip_xlarge=not args.include_xlarge,
        no_think=args.no_think,
    ))

    print_table(blocks)
    save_json(blocks, args.output)


if __name__ == "__main__":
    main()
