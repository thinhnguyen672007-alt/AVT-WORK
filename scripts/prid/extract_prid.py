import json, os, struct, subprocess, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

WORK = os.path.expanduser("~/workspace/prid-work")
SHARE = "https://cloud.tugraz.at/index.php/s/KtC8rK9io6HxNc2/download"
URL = subprocess.run(["curl","-sIL","-A","Mozilla/5.0","-o","/dev/null","-w","%{url_effective}",SHARE],
                     capture_output=True, text=True).stdout.strip()
sel = json.load(open(f"{WORK}/selected.json"))
groups = {}
for e in sel:
    groups.setdefault("/".join(e["name"].split("/")[:3]), []).append(e)
todo = []
for es in groups.values():
    es = sorted(es, key=lambda x: x["name"])
    if len(es) > 24:
        step = len(es) / 24
        es = [es[int(i*step)] for i in range(24)]
    todo.extend(es)

def get_rng(a, b):
    r = subprocess.run(["curl","-sL","-A","Mozilla/5.0","-r",f"{a}-{b}",URL], capture_output=True)
    return r.stdout

def fetch(e):
    dest = os.path.join(WORK, "data", e["name"])
    if os.path.exists(dest) and os.path.getsize(dest) == e["usize"]:
        return "skip"
    hdr = get_rng(e["lho"], e["lho"] + 29)
    if hdr[:4] != b"PK\x03\x04":
        return f"fail-hdr {e['name']} {hdr[:20]!r}"
    nlen, elen = struct.unpack("<HH", hdr[26:30])
    off = e["lho"] + 30 + nlen + elen
    raw = get_rng(off, off + e["csize"] - 1)
    if len(raw) != e["csize"]:
        return f"fail-len {e['name']} {len(raw)} vs {e['csize']}"
    data = zlib.decompress(raw, -15) if e["method"] == 8 else raw
    if len(data) != e["usize"]:
        return f"fail-usz {e['name']}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, "wb").write(data)
    return "ok"

ok = skip = 0; fails = []
with ThreadPoolExecutor(12) as ex:
    futs = {ex.submit(fetch, e): e for e in todo}
    for f in as_completed(futs):
        try:
            r = f.result()
        except Exception as ex_:
            r = f"fail-exc {futs[f]['name']} {ex_}"
        if r == "ok": ok += 1
        elif r == "skip": skip += 1
        else: fails.append(r)
print(f"ok={ok} skip={skip} fails={len(fails)}")
for x in fails[:10]: print(x)
