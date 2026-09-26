import json, sys, time, random, urllib.request
random.seed(7)
words = "red de datos modelo cuantización memoria unificada tensor kernel latencia caché contexto atención capa peso escala bloque flujo ruta nodo señal borde".split()
def doc(n_chars):
    out=[];n=0;i=0
    while n<n_chars:
        s=f"Sección {i}: "+" ".join(random.choice(words) for _ in range(random.randint(12,25)))+f" (ref {random.randint(1000,99999)}). "
        out.append(s);n+=len(s);i+=1
    return "".join(out)
DOC=doc(int(sys.argv[1]))
QS=["Resume en una frase de qué va el documento.","¿Cuántas secciones aproximadamente tiene? Responde en una frase.","Dame las tres palabras que más aparecen.","¿Qué referencia (ref) aparece en la sección 5?","Dame un título corto para el documento.","Despídete en una frase."]
def call(msgs, cache):
    body={"model":"x","messages":msgs,"max_tokens":2500,"cache_prompt":cache,"reasoning_effort":"medium"}
    r=json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8001/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=3000))
    return r
SEND_R = len(sys.argv)>2 and sys.argv[2]=="send_reasoning"
for cache in (True,):
    print(f"\n=== cache_prompt={cache} ===",flush=True)
    msgs=[{"role":"system","content":"Eres un asistente conciso."},{"role":"user","content":"Documento:\n"+DOC+"\n\n"+QS[0]}]
    tot=0
    for i,q in enumerate(QS):
        if i>0: msgs.append({"role":"user","content":q})
        t=time.time();r=call(msgs,cache);tm=r["timings"];m=r["choices"][0]["message"];a=m["content"] or "";rc=m.get("reasoning_content") or ""
        tot+=tm["prompt_ms"]
        print(f"turno {i+1}: prompt_tokens={r['usage']['prompt_tokens']:>6} reutilizados={tm['cache_n']:>6} salida={tm['predicted_n']:>5} procesados={tm['prompt_n']:>6} prefill={tm['prompt_ms']/1000:6.1f}s total={time.time()-t:5.1f}s",flush=True)
        am={"role":"assistant","content":a}
        if SEND_R and rc: am["reasoning_content"]=rc
        msgs.append(am)
    print(f"prefill total: {tot/1000:.1f}s")
