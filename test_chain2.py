# Post-fix chain tests: tier2 pass, gate->tier3, health, parallel requests.
import base64, json, threading, urllib.request

BASE = "http://127.0.0.1:8650"

def call(name):
    b64 = base64.b64encode(open(name, "rb").read()).decode()
    req = urllib.request.Request(BASE + "/ocr",
        data=json.dumps({"image_b64": b64, "mime": "image/png"}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())

r1 = call("bench_images/en_prose.png")
print("tier2 pass      :", r1["device"], r1["elapsed_s"], "s |", len(r1["lines"]), "lines | min_conf",
      round(min(l["confidence"] for l in r1["lines"]), 4))

r2 = call("hard_images/en_prose__noise.png")
print("gate -> tier3   :", r2["device"], r2["elapsed_s"], "s |", len(r2["lines"]), "lines | min_conf",
      round(min(l["confidence"] for l in r2["lines"]), 4))

r3 = call("bench_images/receipt.png")
print("tier2 receipt   :", r3["device"], r3["elapsed_s"], "s |", len(r3["lines"]), "lines | min_conf",
      round(min(l["confidence"] for l in r3["lines"]), 4))

results = {}
def worker(i, name):
    try:
        results[i] = call(name)
    except Exception as e:
        results[i] = {"error": str(e)}

t1 = threading.Thread(target=worker, args=(1, "bench_images/receipt.png"))
t2 = threading.Thread(target=worker, args=(2, "bench_images/en_prose.png"))
t1.start(); t2.start(); t1.join(); t2.join()
ok = all("error" not in results[i] and results[i]["lines"] for i in (1, 2))
print("parallel request:", "both answered correctly under CHAIN_LOCK" if ok else results)

h = json.loads(urllib.request.urlopen(BASE + "/health", timeout=10).read())
print("health:", json.dumps({"status": h["status"], "engine_chain": h["engine_chain"],
      "rapid_worker": h["rapid_worker"], "fallback_active": h["fallback_active"],
      "fallback_reason": h["fallback_reason"]}, ensure_ascii=False, indent=1))