"""FastAPI backend for the CarLaneI showcase.

Wraps the real perception pipeline (src/run_pipeline_fast.run_pipeline). A video is
uploaded or a bundled sample is picked; the pipeline runs on the GPU and produces
an annotated MP4 + a REAL telemetry JSON (no fabricated numbers). The frontend
polls job status, then plays the video and renders the telemetry.

Endpoints:
  GET  /api/health            -> models present, gpu status
  GET  /api/samples           -> bundled demo clips
  POST /api/process           -> start a job (multipart upload OR {"sample": name})
  GET  /api/job/{id}          -> {status, progress, error}
  GET  /api/result/{id}       -> telemetry summary + frames
  GET  /media/{id}.mp4        -> the annotated video (range-served by StaticFiles)

GPU is single, so jobs run one-at-a-time on a worker thread.

Run:  uvicorn web.backend:app --host 127.0.0.1 --port 8000   (from project root)
"""

import os
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")   # silence FFmpeg NAL spam (set before cv2)

import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import MODELS_DIR, PROJECT_ROOT

WEB_OUT = PROJECT_ROOT / "output" / "web"
UPLOADS = WEB_OUT / "uploads"
for d in (WEB_OUT, UPLOADS):
    d.mkdir(parents=True, exist_ok=True)

# Bundled sample clips the user can run with one click. Ordered so the clips that
# best exercise the timeline + signal toasts (varied R/Y/G) come first. `hint`
# is shown in the UI so the demo lands on a clip that visibly shows the features.
SAMPLES = {
    # India-first: the showcase footage is Indian roads (matches the IDD lane
    # model + Indian sign detector).
    "kolkata":    {"path": PROJECT_ROOT / "downloads" / "kolkata_720p_seg.mp4",
                   "hint": "Kolkata city - dense mixed traffic, signals, signs"},
    "mumbai":     {"path": PROJECT_ROOT / "downloads" / "mumbai_720p_seg.mp4",
                   "hint": "South Mumbai - busy metro roads, lane weaving"},
    "india_night":{"path": PROJECT_ROOT / "downloads" / "india_night_720p_seg.mp4",
                   "hint": "NH-44 highway at night - headlights, low light"},
    # Full-length source videos for LONG live validation (streamed, never saved).
    "kolkata_full":     {"path": PROJECT_ROOT / "vidssave.com 4K Drive in Kolkata _ East India's Tier-1 City 720P.mp4",
                         "hint": "Kolkata FULL ~45 min - long live validation", "full": True},
    "mumbai_full":      {"path": PROJECT_ROOT / "vidssave.com 4K Drive to South Mumbai, India (via Mahim) 720P.mp4",
                         "hint": "Mumbai FULL ~29 min - long live validation", "full": True},
    "india_night_full": {"path": PROJECT_ROOT / "india_night_full_fixed.mp4",
                         "hint": "NH-44 Night FULL ~3.5 min (re-encoded clean)", "full": True},
    "nyc":        {"path": PROJECT_ROOT / "data" / "nyc_clip.mp4",
                   "hint": "dense signals - red/green, timeline + alerts"},
    "munich":     {"path": PROJECT_ROOT / "downloads" / "frankfurt_720p_5min.mp4",
                   "hint": "European urban, many state changes"},
    "urban_dense":{"path": PROJECT_ROOT / "BDDA" / "test" / "camera_videos" / "1003.mp4",
                   "hint": "bumper-to-bumper, red lights, occlusion"},
    "highway":    {"path": PROJECT_ROOT / "BDDA" / "test" / "camera_videos" / "100.mp4",
                   "hint": "open road, clear ego lane"},
    "green_flow": {"path": PROJECT_ROOT / "BDDA" / "test" / "camera_videos" / "1013.mp4",
                   "hint": "steady green - mostly lane + vehicles"},
}

app = FastAPI(title="CarLaneI Showcase API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
# Serve annotated videos with HTTP range support (needed for <video> seeking).
app.mount("/media", StaticFiles(directory=str(WEB_OUT)), name="media")

# --- job registry + single-GPU worker ---------------------------------------
_jobs = {}                       # id -> {status, progress, error, video, telemetry}
_frames = {}                     # id -> latest annotated JPEG bytes (live preview)
_cancel = set()                  # job ids that have been asked to stop
_active_long = {"id": None}      # the currently-running long-live job, if any
_lock = threading.Lock()
_gpu_lock = threading.Lock()     # serialize GPU access


def _cancel_active_long():
    """Ask any in-flight long-live job to stop, so a 45-min run never blocks the
    single GPU forever. Called before starting a new job."""
    with _lock:
        aid = _active_long["id"]
    if aid:
        with _lock:
            _cancel.add(aid)


def _set(job_id, **kw):
    with _lock:
        _jobs.setdefault(job_id, {}).update(kw)


def _find_ffmpeg():
    """Locate an ffmpeg binary (PATH or common Windows choco path)."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for c in (r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",):
        if Path(c).exists():
            return c
    return None


def _to_h264(mp4_path):
    """Transcode an OpenCV-written mp4 (mp4v/MPEG-4 Part 2) to H.264 + faststart
    so it plays in a browser <video>. No-op if ffmpeg is missing (falls back to
    the original file — which may not play in all browsers)."""
    import subprocess
    ff = _find_ffmpeg()
    if not ff:
        return False
    src = Path(mp4_path)
    tmp = src.with_name(src.stem + "_h264.mp4")
    cmd = [ff, "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-preset", "veryfast", "-movflags", "+faststart", "-an", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        tmp.replace(src)          # atomic-ish swap to the served filename
        return True
    except Exception as e:
        print(f"  [h264 transcode failed, serving original: {e}]")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def _run_job(job_id, input_path, seconds, mode="egoseg", lane_model="auto"):
    """Worker: run the real pipeline, capped to `seconds` for responsiveness.

    mode: "egoseg" (learned lane model) or "corridor" (heuristic) — for the
    before/after comparison in the UI.
    lane_model: "auto" | "ego_seg" (BDD/US) | "ego_seg_idd" (Indian) — which
    trained lane weights to use (only relevant in egoseg mode).

    Always captures the latest annotated frame as a JPEG so the /api/stream
    endpoint can serve a live MJPEG preview while the render is in progress.
    """
    import cv2
    import run_pipeline_fast as rp

    def _capture_frame(bgr):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _frames[job_id] = buf.tobytes()

    def _should_stop():
        with _lock:
            return job_id in _cancel

    out_video = WEB_OUT / f"{job_id}.mp4"
    out_json = WEB_OUT / f"{job_id}.json"

    # Cap the clip to `seconds` so a demo returns promptly (write a trimmed copy).
    src = input_path
    if seconds:
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(fps * seconds)
        w = int(cap.get(3)); h = int(cap.get(4))
        trimmed = WEB_OUT / f"{job_id}_src.mp4"
        vw = cv2.VideoWriter(str(trimmed), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
        i = 0
        while i < n:
            ret, fr = cap.read()
            if not ret:
                break
            vw.write(fr); i += 1
        cap.release(); vw.release()
        if i > 0:
            src = trimmed

    try:
        with _lock:
            _active_long["id"] = job_id      # track so Stop can cancel any run
        _set(job_id, status="processing", progress=1)
        with _gpu_lock:
            rp.run_pipeline(str(src), str(out_video), crop_center=False,
                            telemetry_out=str(out_json),
                            progress_cb=lambda p: _set(job_id, progress=p),
                            force_corridor=(mode == "corridor"),
                            lane_model=lane_model,
                            frame_cb=_capture_frame,
                            stop_cb=_should_stop)
        # If the user stopped mid-run, don't present a half-written result.
        if _should_stop():
            _set(job_id, status="stopped", progress=100, video=None)
        else:
            # Transcode to browser-playable H.264 (OpenCV writes mp4v, which many
            # browsers won't play in <video>). Done under no GPU lock — CPU encode.
            _set(job_id, progress=99)
            _to_h264(out_video)
            _set(job_id, status="done", progress=100,
                 video=f"/media/{job_id}.mp4", telemetry=str(out_json))
    except Exception as e:
        import traceback
        traceback.print_exc()
        _set(job_id, status="error", error=str(e))
    finally:
        with _lock:
            _cancel.discard(job_id)
            if _active_long["id"] == job_id:
                _active_long["id"] = None
        # free the live-preview buffer once the stream has drained
        def _drop():
            import time
            time.sleep(5)
            with _lock:
                _frames.pop(job_id, None)
        threading.Thread(target=_drop, daemon=True).start()


def _run_live_long(job_id, input_path, mode="egoseg", lane_model="auto"):
    """Long live-validation worker: stream the FULL video's annotated frames.

    No file write, no telemetry JSON, no length cap, fast cadence. The job stays
    'processing' (which keeps the MJPEG stream open) until the video ends. This
    is the browser equivalent of run.bat's live window — watch a 45-min clip
    without saving anything.
    """
    import cv2
    import run_pipeline_fast as rp

    def _capture_frame(bgr):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _frames[job_id] = buf.tobytes()

    def _should_stop():
        with _lock:
            return job_id in _cancel

    try:
        with _lock:
            _active_long["id"] = job_id
        _set(job_id, status="processing", progress=1)
        with _gpu_lock:
            rp.run_pipeline(str(input_path), None, crop_center=False,
                            telemetry_out=None, force_corridor=(mode == "corridor"),
                            lane_model=lane_model, frame_cb=_capture_frame,
                            fast=True, stop_cb=_should_stop)
        _set(job_id, status="done", progress=100, video=None)
    except Exception as e:
        import traceback
        traceback.print_exc()
        _set(job_id, status="error", error=str(e))
    finally:
        with _lock:
            _cancel.discard(job_id)
            if _active_long["id"] == job_id:
                _active_long["id"] = None
        def _drop():
            import time
            time.sleep(5)
            with _lock:
                _frames.pop(job_id, None)
        threading.Thread(target=_drop, daemon=True).start()


# --- endpoints ---------------------------------------------------------------
@app.get("/api/health")
def health():
    import shutil
    def _has(stem):
        return (MODELS_DIR / f"{stem}.engine").exists() or (MODELS_DIR / f"{stem}.pt").exists()
    models = {
        "ego_seg": _has("ego_seg"),                    # BDD lane
        "ego_seg_idd": _has("ego_seg_idd"),            # Indian lane
        "signs_merged_detector": _has("signs_merged_detector"),
        "german_sign_detector": _has("german_sign_detector"),
        "light_state": _has("light_state"),            # trained light detector
        "yolop_384": _has("yolop_384"),
        "yolov8n": _has("yolov8n"),
    }
    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        pass
    return {"ok": True, "gpu": gpu, "models": models}


@app.get("/api/samples")
def samples():
    return [{"id": k, "name": k.replace("_", " ").title(),
             "hint": v.get("hint", ""), "available": v["path"].exists(),
             "full": v.get("full", False)}
            for k, v in SAMPLES.items()]


@app.post("/api/process")
async def process(sample: str = Form(None), seconds: int = Form(12),
                  mode: str = Form("egoseg"), lane_model: str = Form("auto"),
                  live_long: int = Form(0),
                  file: UploadFile = File(None)):
    # Resolve the input: uploaded file OR a named sample.
    if file is not None:
        dest = UPLOADS / f"{uuid.uuid4().hex}_{Path(file.filename).name}"
        with open(dest, "wb") as f:
            f.write(await file.read())
        input_path = dest
    elif sample and sample in SAMPLES and SAMPLES[sample]["path"].exists():
        input_path = SAMPLES[sample]["path"]
    else:
        raise HTTPException(400, "Provide an uploaded 'file' or a valid 'sample'.")

    mode = "corridor" if mode == "corridor" else "egoseg"
    if lane_model not in ("auto", "ego_seg", "ego_seg_idd"):
        lane_model = "auto"
    # A new run should stop any long-live job still holding the GPU (else the
    # single GPU is blocked for the whole 45-min clip).
    _cancel_active_long()

    job_id = uuid.uuid4().hex[:12]
    _set(job_id, status="queued", progress=0, error=None)
    if live_long:
        # Long live validation: stream the FULL video's annotated frames only.
        # No file write, no telemetry, no seconds cap, fast cadence.
        threading.Thread(target=_run_live_long,
                         args=(job_id, input_path, mode, lane_model),
                         daemon=True).start()
    else:
        threading.Thread(target=_run_job,
                         args=(job_id, input_path, seconds, mode, lane_model),
                         daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/stop/{job_id}")
def stop_job(job_id: str):
    """Ask any running job (render, live preview, or long-live) to stop. The
    pipeline checks this every frame and breaks promptly. Idempotent."""
    with _lock:
        _cancel.add(job_id)
    return {"stopping": job_id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    with _lock:
        j = _jobs.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    return {k: j.get(k) for k in ("status", "progress", "error", "video")}


@app.get("/api/stream/{job_id}")
def stream(job_id: str):
    """Live MJPEG preview of the annotated frames while the job renders.

    Serves a multipart/x-mixed-replace stream an <img> can consume directly.
    Ends when the job leaves the processing state.
    """
    import time
    from fastapi.responses import StreamingResponse

    def gen():
        boundary = b"--frame"
        last = None
        # stream until the job finishes (or errors), then one final frame
        while True:
            with _lock:
                j = _jobs.get(job_id)
                buf = _frames.get(job_id)
            if j is None:
                break
            if buf is not None and buf is not last:
                last = buf
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
                       + f"Content-Length: {len(buf)}\r\n\r\n".encode()
                       + buf + b"\r\n")
            if j.get("status") in ("done", "error"):
                break
            time.sleep(0.04)   # ~25 fps cap on the preview push

    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/result/{job_id}")
def result(job_id: str):
    import json
    with _lock:
        j = _jobs.get(job_id)
    if not j or j.get("status") != "done":
        raise HTTPException(404, "result not ready")
    data = json.load(open(j["telemetry"]))
    return JSONResponse({"video": j["video"], "summary": data["summary"],
                         "frames": data["frames"]})


# Serve the frontend (index.html) at the root.
FRONTEND = Path(__file__).resolve().parent / "static"
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
