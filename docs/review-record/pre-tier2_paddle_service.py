"""PaddleOCR service — FastAPI wrapper for the Windows PC.

Runs the paddleocr 3.x defaults (PP-OCRv6 det+rec). One script, one port:
GPU build -> serves fast OCR with automatic CPU fallback.

Devices:
  --device auto  GPU first (build + driver + real prewarm predict), CPU fallback
  --device cpu   force CPU
  --device gpu   force GPU (gpu:0)
  --device gpu:N pin a specific CUDA device
Models:
  --det-model/--rec-model override packaged defaults (auto-downloaded; e.g. a newer release)
GUI:
  GET /       in-browser UI: upload + box preview + live logs + API cheatsheet
  GET /docs   FastAPI interactive Swagger UI
  GET /logs   recent log lines (ring buffer)
"""
import argparse
import base64
import binascii
import datetime as dt
import logging
import time
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import paddle
import paddleocr
from paddleocr import PaddleOCR

app = FastAPI(title="paddle-ocr-service")
state = {
    "model": None,
    "device": "uninitialized",
    "requested_device": "auto",
    "lang": "ch",
    "det_model": None,
    "rec_model": None,
    "enable_mkldnn": False,
    "fallback_active": False,
    "fallback_reason": None,
    "first_error": None,
    "warm": False,
    "error": None,
}

# --- ring log -------------------------------------------------------------
LOGS = deque(maxlen=300)


def log(msg: str, console: bool = True):
    line = f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    LOGS.append(line)
    if console:
        print(line, flush=True)


class _RingHandler(logging.Handler):
    """Capture uvicorn/paddle log records into the ring buffer (no console echo)."""

    def emit(self, record):
        try:
            log(f"[{record.levelname}] {record.getMessage()}", console=False)
        except Exception:
            pass


for _name in ("uvicorn", "uvicorn.access", "uvicorn.error", "paddleocr", "paddlex", "paddleocr.paddleocr"):
    logging.getLogger(_name).addHandler(_RingHandler())


class OcrJson(BaseModel):
    image_b64: str
    mime: str | None = None


def resolve_device(requested: str) -> str:
    """Map CLI arg to a paddle device string."""
    if requested == "cpu":
        return "cpu"
    compiled = paddle.device.is_compiled_with_cuda()
    count = 0
    if compiled:
        try:
            count = paddle.device.cuda.device_count()
        except Exception:
            count = 0
    if requested == "auto":
        return "gpu:0" if compiled and count > 0 else "cpu"
    if requested == "gpu":
        if not compiled:
            raise SystemExit("GPU requested but this paddle build is CPU-only (paddlepaddle, not paddlepaddle-gpu). Reinstall with a GPU wheel.")
        return "gpu:0"
    if requested.startswith("gpu:"):
        if not compiled:
            raise SystemExit(f"GPU requested ({requested}) but this paddle build is CPU-only.")
        return requested
    raise SystemExit(f"Unknown device {requested!r} (use auto|cpu|gpu|gpu:N)")


def get_model():
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail={"error": "model not initialized", "detail": state["error"]})
    return model


def result_fields(res) -> dict:
    """Pull text/score/box fields from a paddleocr 3.x OCRResult, defensively."""
    texts = scores = polys = None
    try:
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        polys = res["rec_polys"]
    except Exception:
        try:
            j = res.json["res"]
            texts, scores, polys = j["rec_texts"], j["rec_scores"], j["rec_polys"]
        except Exception:
            pass
    if texts is None:
        return {"lines": [], "text": ""}
    lines = []
    for i, t in enumerate(texts):
        score = float(scores[i]) if scores is not None and i < len(scores) else None
        poly = polys[i] if polys is not None and i < len(polys) else None
        box = [[float(pt[0]), float(pt[1])] for pt in poly] if poly is not None else None
        lines.append({"index": i, "text": str(t), "confidence": score, "box": box})
    return {"lines": lines, "text": "\n".join(str(t) for t in texts)}


def _run_one(image_bytes: bytes, filename: str | None) -> dict:
    """Shared handler body for both /ocr and /ocr/upload."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"cannot decode image (filename={filename!r})")
    t0 = time.time()
    try:
        results = ensure_predict(img)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "paddle predict failed", "detail": str(exc)})
    elapsed = time.time() - t0
    fields = result_fields(results[0]) if results else {"lines": [], "text": ""}
    log(f"ocr ok {filename or 'base64'} {elapsed:.2f}s {len(fields['lines'])} lines device={state['device']}", console=False)
    return {
        "text": fields["text"],
        "lines": fields["lines"],
        "elapsed_s": round(elapsed, 3),
        "device": state["device"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if state["model"] is not None else "warming",
        "service": "paddle-ocr-service",
        "device": state["device"],
        "device_requested": state["requested_device"],
        "lang": state["lang"],
        "det_model": state["det_model"],
        "rec_model": state["rec_model"],
        "enable_mkldnn": state["enable_mkldnn"],
        "model_initialized": state["model"] is not None,
        "warm": state["warm"],
        "fallback_active": state["fallback_active"],
        "fallback_reason": state["fallback_reason"],
        "init_error": state["error"],
        "paddle_version": paddle.__version__,
        "paddleocr_version": paddleocr.__version__,
        "paddle_compiled_with_cuda": paddle.device.is_compiled_with_cuda(),
    }


@app.post("/ocr")
async def ocr_json(body: OcrJson):
    get_model()
    img_bytes = decode_image_bytes(body.image_b64, body.mime)
    return _run_one(img_bytes, None)


def decode_image_bytes(raw_b64: str, mime: str | None) -> bytes:
    try:
        raw = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64")
    return raw


@app.post("/ocr/upload")
async def ocr_upload(image: UploadFile = File(...)):
    get_model()
    raw = await image.read()
    return _run_one(raw, image.filename)

# --- GUI + logs -----------------------------------------------------------
UI_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PaddleOCR service</title>
<style>
:root{--bg:#faf9f7;--panel:#fff;--ink:#1f1d1a;--muted:#6b675f;--line:#e6e2da;
--accent:#8a6d3b;--accent-soft:#f4efe6;--warn:#a1541c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:36px 20px 64px}
h1{font-size:20px;font-weight:600;margin:0 0 2px}
.sub{color:var(--muted);font-size:13px;margin-bottom:22px}
.sub a{color:var(--accent)}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:22px}
.chip{font-size:12px;padding:4px 12px;border:1px solid var(--line);
border-radius:999px;background:var(--panel)}
.chip b{font-weight:600}
.chip.warn b{color:var(--warn)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:20px 22px;margin-bottom:18px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.9px;
color:var(--muted);font-weight:600;margin:0 0 12px}
.drop{border:1.5px dashed var(--line);border-radius:10px;padding:36px 16px;
text-align:center;color:var(--muted);cursor:pointer;
transition:border-color .5s cubic-bezier(.25,.46,.45,.94),background .5s cubic-bezier(.25,.46,.45,.94)}
.drop:hover,.drop.on{border-color:var(--accent);background:var(--accent-soft);color:var(--ink)}
.btn{display:inline-block;padding:6px 14px;border:1px solid var(--ink);
border-radius:8px;background:var(--ink);color:#fff;font-size:13px;cursor:pointer}
.btn.ghost{background:transparent;color:var(--ink)}
.result{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:16px}
@media (max-width:760px){.result{grid-template-columns:1fr}}
canvas{width:100%;border:1px solid var(--line);border-radius:8px}
pre{margin:0;font:12.5px/1.5 ui-monospace,Consolas,monospace;
white-space:pre-wrap;word-break:break-word;max-height:360px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;width:180px}
code{font:12.5px ui-monospace,Consolas,monospace;background:var(--accent-soft);
padding:1px 6px;border-radius:5px}
details summary{cursor:pointer;color:var(--muted);font-size:13px;margin-top:12px}
.rowline{padding:5px 6px;border-bottom:1px solid var(--line);font-size:13.5px;cursor:pointer;
transition:background .35s cubic-bezier(.25,.46,.45,.94),border-color .35s cubic-bezier(.25,.46,.45,.94)}
.rowline:hover{background:#f0e8d8}
.rowline.sel{background:#f0e8d8;border-left:2px solid var(--accent);padding-left:4px;box-shadow:inset 2px 0 0 var(--accent)}
.rowline b{color:var(--accent);font-weight:600;margin-right:8px}
.rowline .conf{color:var(--muted);font-size:12px;float:right}
.rowline.low .conf{color:var(--warn)}
.statusbar{min-height:20px;color:var(--muted);font-size:13px;margin-top:10px}
</style></head><body><div class="wrap">
<h1>PaddleOCR service</h1>
<div class="sub">PP-OCR det+rec · Windows PC · LAN only · interactive API docs at
<a href="/docs">/docs</a> (Swagger) and <a href="/redoc">/redoc</a></div>
<div class="chips" id="chips">loading…</div>

<div class="card"><h2>Try it</h2>
<div class="drop" id="drop">Drop an image here, or click to browse<br>
<small>PNG / JPG / WEBP</small>
<input type="file" id="file" accept="image/*" style="display:none"></div>
<div class="statusbar" id="status"></div>
<div class="result hidden" id="result">
<div><canvas id="canvas"></canvas></div>
<div><h2 id="linesTitle">Lines</h2><div id="lines"></div></div>
</div>
<details id="jsonwrap" style="display:none"><summary>Raw JSON response</summary>
<pre id="json"></pre></details>
</div>

<div class="card"><h2>API cheatsheet</h2>
<table>
<tr><th>GET /health</th><td>JSON: status, device, versions, fallback_active, fallback_reason</td></tr>
<tr><th>POST /ocr</th><td>JSON body <code>{"image_b64": "...", "mime": "image/png"}</code> — same result shape, base64 in</td></tr>
<tr><th>POST /ocr/upload</th><td>multipart field <code>image</code> — result: <code>{text, lines:[{index,text,confidence,box}], elapsed_s, device}</code>; <code>box</code> is the 4-point polygon in source-image pixels</td></tr>
<tr><th>GET /docs</th><td>Interactive Swagger — run any endpoint from the browser</td></tr>
<tr><th>GET /logs</th><td>recent log lines (ring buffer, 300)</td></tr>
</table></div>

<div class="card"><h2>Logs</h2>
<div style="display:flex;gap:14px;align-items:center;margin-bottom:10px">
<label style="font-size:13px;color:var(--muted)">
<input type="checkbox" id="livecheck" checked> live (2s)</label>
<button class="btn ghost" id="clearview">clear view</button>
</div>
<pre id="logs" style="max-height:260px">—</pre>
</div>
</div>

<script>
const $=id=>document.getElementById(id);
const DROP_HTML=$('drop').innerHTML;

async function health(){try{
  const d=await (await fetch('/health')).json();
  const chips=[['status',d.status],['device',d.device+(d.fallback_active?' · fallback':'')],
    ['models',(d.det_model||d.rec_model)?`${d.det_model||'default det'} + ${d.rec_model||'default rec'}`:'package default']];
  chips.push(['versions',`paddle ${d.paddle_version} · paddleocr ${d.paddleocr_version} · cuda ${d.paddle_compiled_with_cuda}`]);
  if(d.fallback_reason)chips.push(['reason',d.fallback_reason]);
  $('chips').innerHTML=chips.map(([k,v])=>
    `<span class="chip ${d.fallback_active&&k==='device'?'warn':''}">${k}: <b>${v}</b></span>`).join('');
}catch(e){$('chips').innerHTML='<span class="chip warn">service unreachable</span>';}}
health();setInterval(health,5000);

const drop=$('drop'),file=$('file');
function bindDrop(){
  drop.innerHTML=DROP_HTML;
  drop.onclick=()=>document.getElementById('file').click();
  const nf=document.getElementById('file');
  nf.onchange=()=>{if(nf.files[0])run(nf.files[0]);};
}
bindDrop();
/* drag listeners ride on the drop element itself — attach once */
['dragover','dragenter'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('on');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('on');}));
drop.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f)run(f);});

async function run(f){
  drop.innerHTML='<span style="opacity:.7">OCR running…</span>';
  $('status').textContent=f.name+' ('+(f.size/1024).toFixed(0)+' KB)';
  const fd=new FormData();fd.append('image',f);
  try{
    const r=await fetch('/ocr/upload',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok){const msg=d.detail&&(d.detail.error||JSON.stringify(d.detail))||r.status;
      $('status').innerHTML='<span style="color:var(--warn)">Failed: '+msg+'</span>';
      $('result').classList.add('hidden');return;}
    $('status').textContent=`${d.lines.length} line${d.lines.length===1?'':'s'} · `+(d.elapsed_s*1000).toFixed(0)+' ms · device '+d.device;
    render(f,d);
  }catch(err){$('status').innerHTML='<span style="color:var(--warn)">Upload failed: '+err+'</span>';}
  finally{bindDrop();}
}

function render(f,d){
  $('jsonwrap').style.display='';
  $('json').textContent=JSON.stringify(d,null,2);
  window.__res=d;window.__sel=null;
  const img=new Image();
  img.onload=()=>{window.__img=img;window.__scale=Math.min(1,760/img.naturalWidth);
    $('canvas').title='click a box to jump to its line';
    drawBoxes(null);};
  img.src=URL.createObjectURL(f);
  /* line rows */
  $('linesTitle').textContent='Lines ('+d.lines.length+')';
  $('lines').innerHTML=d.lines.map((ln,i)=>
    `<div class="rowline ${ln.confidence!=null&&ln.confidence<.9?'low':''}" data-i="${ln.index}">
     <b>${ln.index??i}</b>${(ln.confidence!=null?`<span class="conf">${(ln.confidence*100).toFixed(1)}%</span>`:'')}
     ${ln.text.replace(/</g,'&lt;')}</div>`).join('')
    ||'<div style="color:var(--muted)">No text found.</div>';
  /* copy full text */
  window.__text=d.text;
  $('result').classList.remove('hidden');
}

/* box <-> line interactivity */
function drawBoxes(hi){
  const d=window.__res,img=window.__img;
  if(!d||!img||!window.__scale)return;
  const c=$('canvas'),s=window.__scale;
  c.width=img.naturalWidth*s;c.height=img.naturalHeight*s;
  const ctx=c.getContext('2d');
  ctx.drawImage(img,0,0,c.width,c.height);
  ctx.font=`${Math.max(10,11*s)}px ui-monospace,Consolas,monospace`;
  d.lines.forEach(ln=>{
    if(!ln.box)return;
    const active=(hi!=null&&ln.index===hi)||(window.__sel!=null&&ln.index===window.__sel);
    const p=ln.box.map(([x,y])=>[x*s,y*s]);
    ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);
    for(let i=1;i<4;i++)ctx.lineTo(p[i][0],p[i][1]);
    ctx.closePath();
    if(active){ctx.fillStyle='rgba(138,109,59,.34)';ctx.fill();}
    ctx.lineWidth=Math.max(1,(active?3.2:1.3)*s);
    ctx.strokeStyle=active?'rgba(166,124,46,1)':'rgba(138,109,59,.30)';
    ctx.stroke();
    if(active){ctx.fillStyle='rgba(166,124,46,1)';
      ctx.fillText('#'+ln.index,p[0][0],Math.max(11*s,p[0][1]-4));}
  });
}
function pointBoxDist(px,py,box){
  let inside=false,minD=1e9;
  for(let i=0;i<4;i++){
    const [x1,y1]=box[i],[x2,y2]=box[(i+1)%4];
    const dx=x2-x1,dy=y2-y1;
    const t=Math.max(0,Math.min(1,((px-x1)*dx+(py-y1)*dy)/((dx*dx+dy*dy)||1e-9)));
    const dd=Math.hypot(px-(x1+t*dx),py-(y1+t*dy));
    if(dd<minD)minD=dd;
    if(((y1>py)!==(y2>py))&&px<(x2-x1)*(py-y1)/((y2-y1)||1e-9)+x1)inside=!inside;
  }
  return inside?0:minD;
}
const LINES_=$('lines');
LINES_.addEventListener('mouseover',e=>{const row=e.target.closest('.rowline');if(row)drawBoxes(+row.dataset.i);});
LINES_.addEventListener('mouseout',e=>{if(!e.relatedTarget||!e.relatedTarget.closest('.rowline'))drawBoxes(null);});
LINES_.addEventListener('click',e=>{
  const row=e.target.closest('.rowline');if(!row)return;
  const i=+row.dataset.i;
  window.__sel=(window.__sel===i)?null:i;
  LINES_.querySelectorAll('.rowline').forEach(r=>r.classList.toggle('sel',+r.dataset.i===window.__sel));
  drawBoxes(null);
});
$('canvas').addEventListener('click',e=>{
  const c=$('canvas');
  if(!window.__res||!window.__scale)return;
  const rc=c.getBoundingClientRect();
  const px=(e.clientX-rc.left)*(c.width/rc.width);
  const py=(e.clientY-rc.top)*(c.height/rc.height);
  let best=null,bd=1e9;
  window.__res.lines.forEach(ln=>{if(!ln.box)return;
    const dd=pointBoxDist(px,py,ln.box);if(dd<bd){bd=dd;best=ln.index;}});
  if(best==null||bd>30)return;
  const row=LINES_.querySelector(`.rowline[data-i="${best}"]`);
  if(row){window.__sel=(window.__sel===best)?null:best;
    LINES_.querySelectorAll('.rowline').forEach(r=>r.classList.toggle('sel',+r.dataset.i===window.__sel));
    drawBoxes(null);row.scrollIntoView({block:'nearest',behavior:'smooth'});}
});

/* logs */
let logsTimer=null;
async function refreshLogs(){
  try{const d=await (await fetch('/logs')).json();
      $('logs').textContent=d.lines.join('\\n')||'—';
      $('logs').scrollTop=$('logs').scrollHeight;}catch(e){}
}
function setLive(on){if(logsTimer){clearInterval(logsTimer);logsTimer=null;}if(on)logsTimer=setInterval(refreshLogs,2000);}
$('livecheck').onchange=e=>setLive(e.target.checked);
$('clearview').onclick=()=>{$('logs').textContent='—';};
setLive(true);refreshLogs();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML


@app.get("/logs")
def recent_logs():
    return {"count": len(LOGS), "lines": list(LOGS)}


def warm_predict(model) -> bool:
    """Run a tiny predict to prove the device actually works."""
    img = np.full((64, 200, 3), 255, dtype=np.uint8)
    cv2.putText(img, "warmup", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    try:
        model.predict(img)
        return True
    except Exception as exc:
        log(f"prewarm predict failed: {exc}")
        return False


def init_model(dev):
    kwargs = dict(
        lang=state["lang"],
        device=dev,
        enable_mkldnn=state["enable_mkldnn"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    # Optional explicit model overrides. When set, paddleocr auto-downloads
    # the named model on first start. Note: explicit names make lang ignored.
    if state.get("det_model"):
        kwargs["text_detection_model_name"] = state["det_model"]
    if state.get("rec_model"):
        kwargs["text_recognition_model_name"] = state["rec_model"]
    return PaddleOCR(**kwargs)


def ensure_predict(img):
    """Predict, falling back to CPU once if a GPU run dies (init-time or runtime)."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail={"error": "model not initialized", "detail": state["error"]})
    try:
        return state["model"].predict(img)
    except Exception as exc:
        gpu_broken = state["device"] != "cpu" and state["requested_device"] == "auto"
        if not gpu_broken:
            raise HTTPException(status_code=500, detail={"error": "paddle predict failed", "detail": str(exc)}) from exc
        log(f"GPU predict failed ({exc}); falling back to CPU")
        try:
            model = init_model("cpu")
            if not warm_predict(model):
                raise RuntimeError("CPU prewarm failed too")
            state["model"] = model
            state["device"] = "cpu"
            state["fallback_active"] = True
            state["fallback_reason"] = f"GPU runtime failure: {exc}"
            return model.predict(img)
        except Exception as exc2:
            raise HTTPException(
                status_code=500,
                detail={"error": "predict failed on GPU and CPU fallback", "detail": str(exc2)},
            ) from exc2


@app.on_event("startup")
def startup():
    req = state["requested_device"]
    if req == "auto":
        candidates = ["cpu"]
        if paddle.device.is_compiled_with_cuda():
            try:
                if paddle.device.cuda.device_count() > 0:
                    candidates = ["gpu:0", "cpu"]
            except Exception:
                pass
    else:
        candidates = [resolve_device(req)]

    for i, dev in enumerate(candidates):
        try:
            model = init_model(dev)
            if not warm_predict(model):
                raise RuntimeError("prewarm predict failed")
            state["model"] = model
            state["device"] = dev
            state["warm"] = True
            state["fallback_active"] = i > 0
            state["fallback_reason"] = (
                f"GPU on this host not usable ({state['first_error']}), using {dev}" if i > 0 else None
            )
            log(f"initialized device={dev} lang={state['lang']} det={state['det_model'] or 'default'} "
                f"rec={state['rec_model'] or 'default'} mkldnn={state['enable_mkldnn']} "
                f"fallback_active={state['fallback_active']}")
            return
        except SystemExit:
            raise
        except Exception as exc:
            state["error"] = f"{dev}: {exc}"
            if i == 0:
                state["first_error"] = str(exc)
            log(f"init on {dev} failed: {exc}")
    log("all device candidates failed")


@app.on_event("shutdown")
def shutdown():
    log("service stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8649)
    parser.add_argument("--device", default="auto",
                        help="auto = GPU first, automatic CPU fallback (init-time and runtime) | cpu | gpu | gpu:N")
    parser.add_argument("--lang", default="ch", help="ch (zh+en) | en | ja | ...")
    parser.add_argument("--det-model", default=None,
                        help="text_detection_model_name override (e.g. PP-OCRv6_server_det or a newer release). "
                             "Downloaded automatically on first start; empty = package default.")
    parser.add_argument("--rec-model", default=None,
                        help="text_recognition_model_name override (e.g. PP-OCRv6_server_rec). "
                             "Downloaded automatically on first start; empty = package default.")
    parser.add_argument("--mkldnn", action="store_true",
                        help="enable oneDNN/MKLDNN on CPU. Default off: paddle 3.3.1 CPU + "
                             "MKLDNN raises '(Unimplemented) ConvertPirAttribute2RuntimeAttribute "
                             "not support' inside predict.")
    args = parser.parse_args()
    state["requested_device"] = args.device
    state["lang"] = args.lang
    state["det_model"] = args.det_model
    state["rec_model"] = args.rec_model
    state["enable_mkldnn"] = bool(args.mkldnn)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")