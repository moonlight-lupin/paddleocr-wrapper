import base64, json, urllib.request

b64 = base64.b64encode(open("bench_images/receipt.png", "rb").read()).decode()
req = urllib.request.Request("http://127.0.0.1:8651/ocr",
    data=json.dumps({"image_b64": b64, "mime": "image/png"}).encode(),
    headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=180).read())
print("tier1 GPU:", d["device"], d["elapsed_s"], "s |", len(d["lines"]), "lines")
h = json.loads(urllib.request.urlopen("http://127.0.0.1:8651/health", timeout=10).read())
print("health:", json.dumps({k: h[k] for k in ("status", "device", "engine_chain", "rapid_worker", "fallback_active")}, ensure_ascii=False, indent=1))