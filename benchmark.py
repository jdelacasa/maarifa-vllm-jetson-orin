#!/usr/bin/env python3
"""
Benchmark vLLM: múltiples usuarios y tamaños de contexto.
Mide TTFT, TPS/req y throughput total para cada escenario.

Uso:
    python3 benchmark.py
    python3 benchmark.py --url http://localhost:8001  # otro host
    python3 benchmark.py --thinking                   # activar razonamiento (más lento)
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import aiohttp

BASE_URL = "http://localhost:8001"
MODEL = "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
MAX_OUTPUT_TOKENS = 400

SCENARIOS = [
    (1,  500),
    (4,  500),
    (4,  8_000),
    (8,  8_000),
    (4,  32_000),
    (8,  32_000),
]

FILLER = (
    "La inteligencia artificial es una rama de la informática que crea sistemas "
    "capaces de realizar tareas que requieren inteligencia humana, como el aprendizaje "
    "automático, el procesamiento del lenguaje natural y la visión por computador. "
    "Los modelos de lenguaje grande han transformado radicalmente la forma en que "
    "interactuamos con la tecnología moderna. Estos sistemas aprenden patrones "
    "estadísticos a partir de enormes cantidades de texto y son capaces de generar "
    "respuestas coherentes y útiles en una amplia variedad de tareas cotidianas. "
)


def build_prompt(target_tokens: int) -> str:
    chars = target_tokens * 3  # ~3 chars/token en español
    body = (FILLER * (chars // len(FILLER) + 2))[:chars]
    return f"{body}\n\nResume los puntos más importantes del texto anterior con todo el detalle posible:"


@dataclass
class ReqResult:
    ttft: float
    output_tokens: int
    total_time: float

    @property
    def tps(self) -> float:
        gen_time = self.total_time - self.ttft
        return self.output_tokens / gen_time if gen_time > 0 else 0.0


async def do_request(
    session: aiohttp.ClientSession, url: str, prompt: str, thinking: bool
) -> ReqResult:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Qwen3: desactivar thinking por defecto para comparar con benchmark anterior
        "chat_template_kwargs": {"enable_thinking": thinking},
    }

    t0 = time.perf_counter()
    ttft: float | None = None
    output_tokens = 0

    async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                if chunk.get("usage"):
                    output_tokens = chunk["usage"].get("completion_tokens", output_tokens)
                    continue
                delta = chunk["choices"][0]["delta"]
                # Primer token puede ser reasoning_content o content
                first = delta.get("content") or delta.get("reasoning_content")
                if first and ttft is None:
                    ttft = time.perf_counter() - t0
            except Exception:
                pass

    total = time.perf_counter() - t0
    return ReqResult(ttft=ttft or total, output_tokens=output_tokens, total_time=total)


async def run_scenario(url: str, users: int, ctx_tokens: int, thinking: bool) -> dict:
    prompt = build_prompt(ctx_tokens)
    connector = aiohttp.TCPConnector(limit=users + 4)
    timeout = aiohttp.ClientTimeout(total=900)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # warmup: una request suelta antes de medir
        try:
            await do_request(session, url, "Di hola.", thinking)
        except Exception:
            pass

        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[do_request(session, url, prompt, thinking) for _ in range(users)]
        )
        wall = time.perf_counter() - t0

    ttfts = [r.ttft for r in results]
    tps_list = [r.tps for r in results]
    total_tokens = sum(r.output_tokens for r in results)

    return {
        "users": users,
        "context": ctx_tokens,
        "ttft_mean": statistics.mean(ttfts),
        "ttft_p95": sorted(ttfts)[int(len(ttfts) * 0.95)] if len(ttfts) > 1 else ttfts[0],
        "tps_req": statistics.mean(tps_list),
        "throughput": total_tokens / wall,
        "total_tokens": total_tokens,
    }


async def main(url: str, thinking: bool):
    mode = "con thinking" if thinking else "sin thinking"
    header = f"{'usuarios':>8}  {'contexto':>10}  {'TTFT':>8}  {'TTFT p95':>9}  {'TPS/req':>9}  {'throughput':>12}  {'tokens out':>11}"
    sep = "─" * len(header)

    print(f"\nBenchmark → {url}  [{mode}]")
    print(f"Modelo: {MODEL}")
    print(f"Output máx: {MAX_OUTPUT_TOKENS} tokens por request\n")
    print(header)
    print(sep)

    for users, ctx in SCENARIOS:
        ctx_label = f"~{ctx // 1000}k" if ctx >= 1000 else f"~{ctx}"
        print(f"  {users:>6}  {ctx_label:>10}  {'…':>8}  {'…':>9}  {'…':>9}  {'…':>12}  {'…':>11}", end="\r", flush=True)
        try:
            r = await run_scenario(url, users, ctx, thinking)
            print(
                f"  {r['users']:>6}  {ctx_label:>10}"
                f"  {r['ttft_mean']:>7.2f}s"
                f"  {r['ttft_p95']:>8.2f}s"
                f"  {r['tps_req']:>8.1f}/s"
                f"  {r['throughput']:>10.1f}/s"
                f"  {r['total_tokens']:>11}"
            )
        except Exception as e:
            print(f"  {users:>6}  {ctx_label:>10}  ERROR: {e}")

    print(sep)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BASE_URL, help="Base URL del servidor vLLM")
    parser.add_argument("--thinking", action="store_true", default=False,
                        help="Activar razonamiento Qwen3 (desactivado por defecto para comparar con benchmark anterior)")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.thinking))
