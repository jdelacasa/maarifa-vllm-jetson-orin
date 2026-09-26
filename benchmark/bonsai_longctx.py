import json, sys, time, random, urllib.request
random.seed(1)
words = "red de datos modelo cuantización memoria unificada tensor kernel latencia caché contexto atención capa peso escala bloque flujo ruta nodo señal borde".split()
def filler(n_chars):
    out=[];n=0;i=0
    while n<n_chars:
        s=f"Registro {i}: "+" ".join(random.choice(words) for _ in range(random.randint(12,25)))+f" (ref {random.randint(1000,99999)}). "
        out.append(s);n+=len(s);i+=1
    return "".join(out)
NEEDLE="NOTA IMPORTANTE: el código secreto del proyecto Bonsai es ZORRO-7431-AZUL. "
for target in [int(x) for x in sys.argv[1:]]:
    chars=int(target*3.3)
    a=filler(chars//2); b=filler(chars//2)
    prompt=a+"\n"+NEEDLE+"\n"+b+"\n\nPregunta: ¿cuál es el código secreto del proyecto Bonsai? Responde solo con el código."
    body=json.dumps({"model":"x","messages":[{"role":"user","content":prompt}],"max_tokens":3000,"reasoning_effort":"medium"}).encode()
    t=time.time()
    r=json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8001/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=3000))
    tm=r["timings"];m=r["choices"][0]["message"]
    print(f"target={target} prompt_tokens={r['usage']['prompt_tokens']} prefill={tm['prompt_ms']/1000:.1f}s ({tm['prompt_per_second']:.0f} tok/s) decode={tm['predicted_per_second']:.1f} tok/s out={tm['predicted_n']} total={time.time()-t:.0f}s found={'ZORRO-7431-AZUL' in (m['content'] or '')} ans={(m['content'] or '').strip()[:60]!r}",flush=True)
