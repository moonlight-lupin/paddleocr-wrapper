# Opus SHIP-condition smoke: threadpool handlers live — GPU tier-1 + /health responsive
# while a long tier-3 request runs.
import base64, json, threading, time, urllib.request

def status_of(port):
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10).read())

def ocr(port, name):
    b64 = base64.b64encode(open(name, "rb").read()).decode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/ocr",
        data=json.dumps({"image_b64": b64, "mime": "image/png"}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())

# --- 8651: auto mode, GPU tier-1 predict called from a worker thread
r = ocr(8651, "bench_images/en_prose.png")
print("auto tier-1 threadpool :", r["device"], r["elapsed_s"], "s |", len(r["lines"]), "lines")

# --- 8650: forced tier-2, gated -> tier-3 lazy init on worker thread; /health
# polled concurrently to prove the event loop stays responsive
res = {}
def t3():
    res["r"] = ocr(8650, "hard_images/en_prose__noise.png")
th = threading.Thread(target=t3)
th.start()
time.sleep(4.5)  # inside the gated tier-2/paddle run
healths = []
for _ in range(3):
    t0 = time.time()
    status_of(8650)
    healths.append(round(time.time() - t0, 3))
    time.sleep(1.5)
th.join()
r = res["r"]
print("health latency during tier-3 (s):", healths)
print("tier-3 threadpool result:", r["device"], r["elapsed_s"], "s |", len(r["lines"]), "lines")
h = status_of(8650)
print("health:", h["status"], "| spawned:", h["rapid_worker"]["spawned"], "| gated:", h["rapid_worker"]["gated"])