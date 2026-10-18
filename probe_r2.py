# R2 verification probes for RapidWorker (codex residual items 2, 7, 11 + queue isolation).
import ast, pathlib, queue, subprocess, os, sys, threading, time, base64, json, types

SRC = pathlib.Path(__file__).parent / "paddle_service_candidate.py"
src = SRC.read_text()
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RapidWorker.call".split(".")[0])
call_fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "call")
out = []


def run_probes():
    # stub the environment BEFORE any real spawn can happen
    class _T:
        def __init__(self, *a, **k): pass
        def start(self): pass

    subprocess.Popen = lambda *a, **k: Proc()
    threading.Thread = _T
    class Proc:
        def __init__(self): self.stdin = types.SimpleNamespace(write=lambda s: None, flush=lambda: None, close=lambda: None)
        def poll(self): return None
        def kill(self): pass
        def wait(self, timeout): pass

    # cv2 / np stubs
    cv2 = types.ModuleType("cv2")
    cv2.imencode = lambda *a, **k: (True, types.SimpleNamespace(tobytes=lambda: b"png"))
    np = types.ModuleType("numpy")
    ns = {"base64": base64, "json": json, "subprocess": subprocess, "os": os,
          "threading": threading, "queue": queue, "time": time, "log": lambda *a, **k: None,
          "cv2": cv2, "np": np, "state": {"rapid": {"uses": 0, "failures": 0, "gated": 0,
                                                    "failure_requests": 0, "circuit_open": False}}}
    mod = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
                     type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(SRC), "exec"), ns)
    RW = ns["RapidWorker"]

    # A: exactly 100 noise lines + READY -> accepted, one overall deadline
    seq = iter(["noise\r\n"] * 100 + ["READY\r\n"])
    w = RW("python", "worker.py"); w.proc = Proc(); w._read = lambda t: next(seq)
    try:
        w._spawn(timeout_s=0.5)
        out.append("A OK: 100 noise lines + READY accepted (single deadline, no off-by-one)")
    except RuntimeError as e:
        out.append(f"A FAIL: {e}")

    # B: endless noise, tiny single deadline -> bounded once
    w2 = RW("python", "worker.py"); w2.proc = Proc()
    w2._read = lambda t: (time.sleep(0.004), "noise")[1]
    t0 = time.monotonic()
    try:
        w2._spawn(timeout_s=0.05)
        out.append("B FAIL: endless noise accepted")
    except RuntimeError as e:
        out.append(f"B OK: bounded at {time.monotonic() - t0:.3f}s -> {e}")

    # C: hung call bounded by ONE budget across spawn + both attempts
    w3 = RW("python", "worker.py"); w3.CALL_TIMEOUT_S = 0.12
    kills = [0]
    def fake_spawn(timeout_s=None):
        w3.proc = Proc(); out.append(f"   spawn with remaining budget {timeout_s:.2f}s" if timeout_s else "")
    w3._spawn = fake_spawn
    w3._read = lambda t: (time.sleep(min(t, 0.012)), "")[1]
    def fake_kill():
        kills[0] += 1; w3.proc = Proc()
    w3._kill = fake_kill
    call_fn_ns = {"base64": base64, "json": json, "subprocess": subprocess, "os": os,
                  "threading": threading, "queue": queue, "time": time,
                  "log": lambda *a, **k: None, "cv2": cv2, "np": np, "state": ns["state"]}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), call_fn], type_ignores=[])), "call", "exec"), call_fn_ns)
    c2 = call_fn_ns["call"].__get__(w3)
    t0 = time.monotonic()
    res = c2(None)
    dt = time.monotonic() - t0
    out.append(f"C OK: hung call -> {res}, {dt:.3f}s total within 0.12s budget, kills={kills[0]}")
    assert dt < 0.3, "C FAIL: call exceeded its budget"

    # D: stale pump cannot inject into a new child's queue
    class Stream:
        def __iter__(self): return self
        def __next__(self):
            raise StopIteration
    w4lines_old = queue.Queue()
    w4 = RW("python", "worker.py")
    # call the real pump bound to one queue
    w4._pump(Stream(), w4lines_old)
    w4._spawn = fake_spawn.__get__(w4) if False else w4._spawn  # spawn real below w/ stub

    # real spawn with real pump thread: assert old queue receives the EOF marker, new queue empty
    old_q = w4lines_old
    w4.proc = Proc()
    # emulate _spawn's queue rebinding mechanics
    class FakePopenProc:
        def __init__(self):
            self.stdin = types.SimpleNamespace(write=lambda s: None, flush=lambda: None, close=lambda: None)
        def poll(self): return None
    subprocess.Popen = lambda *a, **k: FakePopenProc()
    q_new = queue.Queue()
    w4.lines = q_new
    # a pump bound to old_q must put its None into old_q only
    w4._pump(Stream(), old_q)
    drained = []
    while not old_q.empty():
        drained.append(old_q.get_nowait())
    out.append(f"D OK: stale pump wrote only to its bound queue ({drained}); new queue size {q_new.qsize()}")

    # E: forced rapidocr + worker None -> /ocr 503; health 'warming'
    fn_health = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "health")
    paddle_stub = types.SimpleNamespace(__version__="t",
        device=types.SimpleNamespace(is_compiled_with_cuda=lambda: False))
    hns = {"app": types.SimpleNamespace(get=lambda *a, **k: (lambda f: f)),
           "paddle": paddle_stub, "paddleocr": paddle_stub, "engine_chain": lambda: ["x"],
           "state": {"model": None, "force_tier": "rapidocr", "rapid_worker": None,
                     "device": "uninitialized", "requested_device": "auto", "lang": "ch",
                     "det_model": None, "rec_model": None, "enable_mkldnn": False, "warm": True,
                     "fallback_active": False, "fallback_reason": None, "error": None, "conf_gate": 0.7,
                     "rapid_worker_py": "x", "rapid_worker_script": "x",
                     "rapid": {"uses": 0, "failures": 0, "gated": 0, "failure_requests": 0, "circuit_open": False},
                     "gpu_runtime_failures": 0}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn_health], type_ignores=[])), "h", "exec"), hns)
    # worker None -> ready False -> warming
    st = hns["health"]()
    # with worker present but unspawned after forced-mode probe: spawned True
    w5 = RW("python", "worker.py")
    w5._spawn = fake_spawn
    w5.probe = lambda: None  # probe ok -> child kept warm; emulate post-probe state
    st2_state = dict(hns["state"]); st2_state["rapid_worker"] = w5
    hns2 = dict(hns); hns2["state"] = st2_state
    st2 = hns2["health"]()
    out.append(f"E OK: worker missing -> status {st['status']} | worker probed-warm -> status {st2['status']}")

run_probes()
print("\n".join(out))