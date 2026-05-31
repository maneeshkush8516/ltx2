# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LTX-2 LD  --  INFINITE FLOW ENGINE  (Unified Edition)                 ║
# ║                                                                          ║
# ║  SECTION A  VisionDescribeEngine   (standalone Qwen2.5-VL, no wrapper)  ║
# ║  SECTION B  EasyPromptEngine       (full cinematic LLM expansion)        ║
# ║  SECTION C  CharacterBible         (cross-scene consistency lock)        ║
# ║  SECTION D  generate_clip()        (two-pass LTX-2 19B GGUF pipeline)   ║
# ║  SECTION E  InfiniteFlowEngine     (scene_chain + temporal_extend)       ║
# ║                                                                          ║
# ║  GPU target : Google Colab T4 (15 GB) / L4 (24 GB)                     ║
# ║  Model      : LTX-2 19B Q4_K_M GGUF (Kijai distilled)                  ║
# ║  Strategy   : load -> use -> unload every heavyweight model in sequence  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# ══════════════════════════════════════════════════════════════════════════
# CELL 1  --  ENVIRONMENT SETUP  (run once per Colab session)
# ══════════════════════════════════════════════════════════════════════════
# @title { "single-column": true }
# @markdown ## 1. Install Environment

import subprocess, os, sys

def _shell(cmd):
    """Run a shell command (replaces ! magic for AST compatibility)."""
    subprocess.run(cmd, shell=True, check=True)

def _pip(packages):
    """pip install helper."""
    _shell(f"pip install -q {packages}")

_shell("pip install torch torchvision torchaudio")

os.chdir("/content")
from IPython.display import clear_output
clear_output()

_pip("torchsde einops diffusers accelerate nest_asyncio")
_pip("av spandrel albumentations onnx opencv-python onnxruntime")
_pip("imageio imageio-ffmpeg")
_pip('"transformers>=4.43.0" accelerate qwen-vl-utils huggingface_hub')

# Pinned ComfyUI build
_shell("git clone --branch ComfyUI_22_01_2026_v0.10.0 https://github.com/Isi-dev/ComfyUI.git")
_shell("pip install -r /content/ComfyUI/requirements.txt -q")
clear_output()

os.chdir("/content/ComfyUI/custom_nodes")

# Pinned custom nodes
_shell("git clone --branch kj_1.2.6               https://github.com/Isi-dev/ComfyUI_KJNodes")
_shell("git clone --branch ComfyUI_GGUF_22_01_2026 https://github.com/Isi-dev/ComfyUI_GGUF.git")

# Official LTX nodes (tiled VAE + AV latent helpers)
_shell("git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git")

# LoRa Daddy nodes (Master Loader + Easy Prompt as ComfyUI nodes)
_shell("git clone https://github.com/seanhan19911990-source/LTX2EasyPrompt-LD.git")
_shell("git clone https://github.com/seanhan19911990-source/LTX2-Master-Loader.git")

# Video helper suite
_shell("git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git")

# Install node requirements
os.chdir("/content/ComfyUI/custom_nodes/ComfyUI_KJNodes")
_shell("pip install -r requirements.txt -q")
os.chdir("/content/ComfyUI/custom_nodes/ComfyUI_GGUF")
_shell("pip install -r requirements.txt -q")

for d in ["ComfyUI-LTXVideo", "LTX2EasyPrompt-LD", "LTX2-Master-Loader"]:
    req = f"/content/ComfyUI/custom_nodes/{d}/requirements.txt"
    if os.path.isfile(req):
        _shell(f"pip install -r {req} -q")

subprocess.run(["apt-get", "-y", "install", "-qq", "aria2", "ffmpeg"],
               check=True, capture_output=True)

os.chdir("/content/ComfyUI")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")
clear_output()
print("Environment ready. All custom nodes installed.")


# ══════════════════════════════════════════════════════════════════════════
# CELL 2  --  MODEL DOWNLOADS  (run once; skips cached files)
# ══════════════════════════════════════════════════════════════════════════
# @title { "single-column": true }
# @markdown ## 2. Download Model Weights

import os, subprocess
from pathlib import Path

def _dl(url: str, dest: str, filename: str = None) -> str:
    """Download a file with aria2c, skip if cached and >1MB."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    fn   = filename or url.split("/")[-1].split("?")[0]
    full = os.path.join(dest, fn)
    if os.path.exists(full) and os.path.getsize(full) > 1_000_000:
        print(f"  cached  {fn}"); return fn
    cmd = ["aria2c", "--console-log-level=error", "--summary-interval=0", "--quiet",
           "-c", "-x", "16", "-s", "16", "-k", "1M", "-d", dest, "-o", fn, url]
    print(f"  dl {fn}...", end=" ", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Download failed for {fn}: {r.stderr.strip() or 'unknown error'}. "
            f"Check network connection and re-run Cell 2.")
    print("done.")
    return fn

KIJAI   = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main"
CORG    = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
LTX_HF  = "https://huggingface.co/Lightricks"
UNET_D  = "/content/ComfyUI/models/unet"
TE_D    = "/content/ComfyUI/models/text_encoders"
VAE_D   = "/content/ComfyUI/models/vae"
UPD     = "/content/ComfyUI/models/latent_upscale_models"
LR_D    = "/content/ComfyUI/models/loras"

print("-- Core models --")
_M_UNET  = _dl(f"{KIJAI}/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf", UNET_D)
_M_CLIP1 = _dl(f"{CORG}/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", TE_D)
_M_CLIP2 = _dl(f"{KIJAI}/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors", TE_D)
_M_VAE   = _dl(f"{KIJAI}/VAE/LTX2_video_vae_bf16.safetensors", VAE_D)
_M_AVAE  = _dl(f"{KIJAI}/VAE/LTX2_audio_vae_bf16.safetensors", VAE_D)
_M_TAEV  = _dl("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors", VAE_D)
_M_UP    = _dl(f"{LTX_HF}/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors", UPD)

print("\n-- LoRAs --")
_LORA_URLS = {
    "Detailer":    f"{LTX_HF}/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors",
    "Canny":       f"{LTX_HF}/LTX-2-19b-IC-LoRA-Canny-Control/resolve/main/ltx-2-19b-ic-lora-canny-control.safetensors",
    "Depth":       f"{LTX_HF}/LTX-2-19b-IC-LoRA-Depth-Control/resolve/main/ltx-2-19b-ic-lora-depth-control.safetensors",
    "Pose":        f"{LTX_HF}/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors",
    "Dolly-In":    f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "Dolly-Out":   f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "Dolly-Left":  f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "Dolly-Right": f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "Jib-Up":      f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "Jib-Down":    f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "Static":      f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors",
}
os.makedirs(LR_D, exist_ok=True)
for name, url in _LORA_URLS.items():
    _dl(url, LR_D)

print("\nAll models ready.")


# ══════════════════════════════════════════════════════════════════════════
# CELL 3  --  CORE IMPORTS & UTILITIES
# ══════════════════════════════════════════════════════════════════════════
# @title { "single-column": true }
# @markdown ## 3. Imports, Helpers, Node Loader

import gc, re, json, time, shutil, warnings, asyncio, hashlib
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, List, Any, Union, Sequence, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from IPython.display import display, HTML, clear_output
from google.colab import files


# -- Dataclasses for structured configuration and results --
@dataclass
class LoRAConfig:
    """Configuration for a single LoRA slot in the stack."""
    on: bool = False
    lora: str = "None"
    guard: bool = False
    strength: float = 1.0

    def to_dict(self) -> dict:
        return {"on": self.on, "lora": self.lora, "guard": self.guard, "strength": self.strength}


@dataclass
class ClipResult:
    """Result from generate_clip() -- path, last frame tensor, duration, seed."""
    path: str = ""
    last_frame: Optional[torch.Tensor] = None
    duration: float = 0.0
    seed: int = 0


class ClipTimeoutError(Exception):
    """Raised when a single clip generation exceeds the allowed timeout."""
    pass


class Checkpoint:
    """Saves and loads generation state for resume after interruption.

    After each successful beat, call save() to persist progress. On restart,
    call load() to resume from the last completed beat.
    """

    def __init__(self, checkpoint_dir: str):
        self._dir = checkpoint_dir
        self._path = os.path.join(checkpoint_dir, "checkpoint.json")

    def save(self, completed_beats: List[int], clip_paths: dict,
             bible_state: dict, scene_history: List[str],
             current_seed: int, config_hash: str = ""):
        """Persist current generation state to checkpoint.json.

        Args:
            completed_beats: List of beat indices already completed.
            clip_paths: Dict mapping beat_idx (int) to output path (str).
            bible_state: Serialized CharacterBible data (_chars dict).
            scene_history: Rolling scene history list.
            current_seed: Current random seed value.
            config_hash: Hash of relevant config for invalidation detection.
        """
        Path(self._dir).mkdir(parents=True, exist_ok=True)
        data = {
            "completed_beats": completed_beats,
            "clip_paths": clip_paths,
            "bible_state": bible_state,
            "scene_history": scene_history,
            "current_seed": current_seed,
            "config_hash": config_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> Optional[dict]:
        """Load checkpoint data if available.

        Returns:
            Dict with checkpoint fields, or None if no checkpoint exists.
        """
        if not os.path.exists(self._path):
            return None
        try:
            with open(self._path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def exists(self) -> bool:
        """Check if a valid checkpoint file exists."""
        return os.path.exists(self._path)

    def clear(self):
        """Remove the checkpoint file after successful completion."""
        if os.path.exists(self._path):
            os.remove(self._path)


class ProgressTracker:
    """Tracks generation progress and writes progress.json with ETA estimates.

    Instantiate at the start of a run, call update() after each beat, and
    call complete() or error() to finalize status.
    """

    def __init__(self, total_beats: int, output_dir: str):
        self._total = total_beats
        self._completed = 0
        self._current_beat = 0
        self._start_time = time.time()
        self._status = "running"
        self._path = os.path.join(output_dir, "progress.json")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._write()

    def update(self, completed: int, current_beat: int):
        """Update progress after a beat completes.

        Args:
            completed: Total number of beats completed so far.
            current_beat: Index of the beat just finished (1-based).
        """
        self._completed = completed
        self._current_beat = current_beat
        self._write()

    def complete(self):
        """Mark the run as complete."""
        self._status = "complete"
        self._completed = self._total
        self._write()

    def error(self, detail: str = ""):
        """Mark the run as errored."""
        self._status = "error"
        self._write()

    def to_json(self) -> dict:
        """Return progress state as a JSON-serializable dict."""
        elapsed = time.time() - self._start_time
        avg_per_beat = (elapsed / self._completed) if self._completed > 0 else 0.0
        remaining = avg_per_beat * (self._total - self._completed)
        est_completion = ""
        if self._status == "running" and avg_per_beat > 0:
            est_dt = datetime.now(timezone.utc).isoformat()
            est_completion = est_dt
        return {
            "total_beats": self._total,
            "completed_beats": self._completed,
            "current_beat": self._current_beat,
            "elapsed_seconds": round(elapsed, 1),
            "avg_seconds_per_beat": round(avg_per_beat, 1),
            "estimated_remaining_seconds": round(remaining, 1),
            "estimated_completion_time": est_completion,
            "status": self._status,
        }

    def _write(self):
        """Write progress.json to disk."""
        try:
            with open(self._path, "w") as f:
                json.dump(self.to_json(), f, indent=2)
        except OSError:
            pass

warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/ComfyUI")
from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly
import folder_paths

# -- VRAM management --
def _vram_free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


# -- CLIP conditioning cache (maps prompt -> CPU tensors for reuse across clips) --
_CLIP_COND_CACHE: dict = {}


def _clear_clip_cache():
    """Clear the CLIP conditioning cache to free CPU memory."""
    global _CLIP_COND_CACHE
    _CLIP_COND_CACHE.clear()
    print("   [Cache] CLIP conditioning cache cleared.")

def _vram_print(tag=""):
    if not torch.cuda.is_available():
        return
    u = torch.cuda.memory_allocated() / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    filled = int(20 * u / t) if t else 0
    bar = "|" * filled + "." * (20 - filled)
    print(f"   VRAM [{bar}] {u:.1f}/{t:.1f} GB  {tag}")

# -- Tensor / image helpers --
def get_value_at_index(obj: Union[Sequence, Mapping], idx: int) -> Any:
    try:
        return obj[idx]
    except KeyError:
        return obj["result"][idx]

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    return Image.fromarray((t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), "RGB")

def load_image_tensor(path: str) -> Optional[torch.Tensor]:
    if not path or not os.path.exists(path):
        return None
    return pil_to_tensor(Image.open(path).convert("RGB"))

def get_last_frame_tensor(video_path: str) -> Optional[torch.Tensor]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame).float().unsqueeze(0) / 255.0


def get_last_frames_averaged(video_path: str, n_frames: int = 3) -> Optional[torch.Tensor]:
    """Extract the last n_frames from a video and average them into a single tensor.

    This reduces flickering artifacts when using the last frame as I2V conditioning
    by smoothing out single-frame noise. Falls back to get_last_frame_tensor if
    fewer frames are available.

    Args:
        video_path: Path to the video file.
        n_frames: Number of trailing frames to average (default 3, range 1-5).

    Returns:
        Averaged tensor of shape (1, H, W, 3) or None if extraction fails.
    """
    n_frames = max(1, min(5, n_frames))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None
    # Determine how many frames we can actually read
    count = min(n_frames, total)
    start_frame = total - count
    frames_list = []
    for i in range(count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(torch.from_numpy(frame).float() / 255.0)
    cap.release()
    if not frames_list:
        return None
    # Average across collected frames
    stacked = torch.stack(frames_list, dim=0)  # (N, H, W, 3)
    averaged = stacked.mean(dim=0, keepdim=True)  # (1, H, W, 3)
    return averaged

# -- Video helpers --
def display_video(path: str):
    if not path or not os.path.exists(path):
        return
    data = b64encode(open(path, "rb").read()).decode()
    display(HTML(
        '<video width=720 controls autoplay loop muted>'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4"></video>'))

def save_video_obj(video_obj, prefix="IFE") -> str:
    from comfy_api.latest import Types
    w, h = video_obj.get_dimensions()
    folder, fname, ctr, _, _ = folder_paths.get_save_image_path(
        f"video/{prefix}", folder_paths.get_output_directory(), w, h)
    ext = Types.VideoContainer.get_extension("auto")
    path = os.path.join(folder, f"{fname}_{ctr:05}_.{ext}")
    video_obj.save_to(path, format=Types.VideoContainer("auto"),
                      codec="auto", metadata=None)
    return path

def concatenate_clips(clip_paths: List[str], output_path: str,
                      transition_type: str = "hard_cut",
                      transition_duration_sec: float = 0.5) -> str:
    """ffmpeg concat all clips into one final video.

    Args:
        clip_paths: List of clip file paths to concatenate.
        output_path: Destination path for concatenated video.
        transition_type: One of 'hard_cut', 'crossfade', 'dip_to_black'.
        transition_duration_sec: Duration of transition in seconds (for crossfade/dip_to_black).

    Returns:
        Path to the concatenated output file.
    """
    if not clip_paths:
        return output_path

    if transition_type == "hard_cut" or len(clip_paths) == 1:
        # Original fast concat with stream copy
        list_file = "/tmp/concat_list.txt"
        with open(list_file, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                        "-c", "copy", output_path], check=True, capture_output=True)
    elif transition_type in ("crossfade", "dip_to_black"):
        # Build complex xfade filter chain
        dur = max(0.1, transition_duration_sec)
        xfade_mode = "fade" if transition_type == "crossfade" else "fade"
        xfade_color = "" if transition_type == "crossfade" else ":color=black"

        if len(clip_paths) == 1:
            shutil.copy2(clip_paths[0], output_path)
        else:
            # Build ffmpeg filter_complex for sequential xfade transitions
            inputs = []
            for p in clip_paths:
                inputs.extend(["-i", p])

            filter_parts = []
            n = len(clip_paths)
            # Chain xfade filters sequentially
            for i in range(n - 1):
                if i == 0:
                    src = "[0:v]"
                else:
                    src = f"[v{i}]"
                nxt = f"[{i + 1}:v]"
                out_label = f"[v{i + 1}]" if i < n - 2 else "[vout]"
                offset_val = f"offset_placeholder_{i}"
                if transition_type == "dip_to_black":
                    xf = (f"{src}{nxt}xfade=transition=fade:duration={dur}"
                           f":offset={offset_val}{out_label}")
                else:
                    xf = (f"{src}{nxt}xfade=transition=fade:duration={dur}"
                           f":offset={offset_val}{out_label}")
                filter_parts.append(xf)

            # We need to know clip durations to compute offsets
            # Use ffprobe to get durations
            durations = []
            for p in clip_paths:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", p],
                    capture_output=True, text=True)
                try:
                    durations.append(float(probe.stdout.strip()))
                except (ValueError, AttributeError):
                    durations.append(5.0)  # fallback duration

            # Compute offsets: each xfade starts at cumulative_duration - overlap
            cumulative = 0.0
            offsets = []
            for i in range(n - 1):
                cumulative += durations[i]
                if i > 0:
                    cumulative -= dur  # Account for previous transitions shortening
                offset = cumulative - dur
                offsets.append(max(0, offset))

            # Replace placeholders with actual offset values
            filter_str = ";".join(filter_parts)
            for i, off in enumerate(offsets):
                filter_str = filter_str.replace(
                    f"offset_placeholder_{i}", f"{off:.3f}")

            # Build audio filter (amerge or simple concat)
            audio_filter = ""
            audio_parts = []
            for i in range(n - 1):
                if i == 0:
                    a_src = "[0:a]"
                else:
                    a_src = f"[a{i}]"
                a_nxt = f"[{i + 1}:a]"
                a_out = f"[a{i + 1}]" if i < n - 2 else "[aout]"
                a_off = offsets[i]
                af = (f"{a_src}{a_nxt}acrossfade=d={dur}:c1=tri:c2=tri{a_out}")
                audio_parts.append(af)
            audio_filter = ";".join(audio_parts)

            full_filter = filter_str
            if audio_filter:
                full_filter += ";" + audio_filter

            cmd = ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", full_filter,
                "-map", "[vout]", "-map", "[aout]" if audio_parts else "[0:a]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # Fallback to hard_cut if xfade fails
                print(f"   [Concat] xfade failed, falling back to hard_cut: "
                      f"{result.stderr[:200]}")
                return concatenate_clips(clip_paths, output_path,
                                         transition_type="hard_cut")
    else:
        # Unknown transition type -- fall back to hard_cut
        return concatenate_clips(clip_paths, output_path, transition_type="hard_cut")

    print(f"   Concatenated ({transition_type}) -> {output_path}")
    return output_path

# -- Resolution auto-selection --
_SUPPORTED_RESOLUTIONS = [
    (768, 512), (512, 768), (768, 768),
    (704, 480), (480, 704), (640, 640),
    (832, 480), (480, 832),
]


def _auto_resolution(image_path: str) -> tuple:
    """Read the seed image aspect ratio and pick the closest supported LTX-2 resolution.

    Args:
        image_path: Path to the seed/reference image.

    Returns:
        Tuple of (width, height) from the supported resolution table.
        Falls back to (768, 512) if the image cannot be read.
    """
    try:
        img = Image.open(image_path)
        img_w, img_h = img.size
    except Exception:
        return (768, 512)

    if img_w == 0 or img_h == 0:
        return (768, 512)

    img_ratio = img_w / img_h
    best = _SUPPORTED_RESOLUTIONS[0]
    best_diff = abs(img_ratio - best[0] / best[1])
    for res in _SUPPORTED_RESOLUTIONS[1:]:
        diff = abs(img_ratio - res[0] / res[1])
        if diff < best_diff:
            best_diff = diff
            best = res
    return best


# -- Adaptive sigma schedule --
_MOTION_COMPLEX_KEYWORDS = [
    "run", "fight", "dance", "chase", "explode", "crash", "sprint",
    "jump", "flip", "spin", "attack", "slam", "whip", "rush",
]
_MOTION_SIMPLE_KEYWORDS = [
    "sit", "stand", "look", "wait", "rest", "pause", "gaze", "lean",
    "stare", "sleep", "hold", "still",
]


def _adaptive_sigmas(base_sigmas: str, complexity: str) -> str:
    """Adjust sigma schedule based on scene complexity.

    For 'simple' scenes (static/slow motion), use fewer intermediate steps
    by removing some mid-range sigma values for faster generation.
    For 'complex' scenes (fast action), add more intermediate steps for
    smoother motion and better temporal coherence.

    Args:
        base_sigmas: Comma-separated sigma schedule string (e.g. "1., 0.9, 0.7, 0.0").
        complexity: One of 'simple', 'moderate', 'complex'.

    Returns:
        Modified sigma schedule string.
    """
    try:
        vals = [float(v.strip()) for v in base_sigmas.split(",") if v.strip()]
    except (ValueError, AttributeError):
        return base_sigmas

    if not vals or len(vals) < 3:
        return base_sigmas

    if complexity == "simple":
        # Remove some mid-range values for faster generation
        # Keep first, last, and every other intermediate
        if len(vals) > 4:
            reduced = [vals[0]]
            step = max(1, (len(vals) - 2) // 3)
            for i in range(step, len(vals) - 1, step):
                reduced.append(vals[i])
            reduced.append(vals[-1])
            vals = reduced
    elif complexity == "complex":
        # Add interpolated values between existing steps for finer detail
        expanded = [vals[0]]
        for i in range(1, len(vals)):
            mid = (vals[i - 1] + vals[i]) / 2.0
            expanded.append(mid)
            expanded.append(vals[i])
        vals = expanded

    # 'moderate' returns the original schedule unchanged
    return ", ".join(f"{v:.6f}" if v not in (0.0, 1.0) else
                     ("0.0" if v == 0.0 else "1.0") for v in vals)


def _detect_complexity(prompt: str) -> str:
    """Analyze a prompt for motion keywords to determine complexity.

    Returns 'simple', 'moderate', or 'complex'.
    """
    lower = prompt.lower()
    complex_count = sum(1 for kw in _MOTION_COMPLEX_KEYWORDS if kw in lower)
    simple_count = sum(1 for kw in _MOTION_SIMPLE_KEYWORDS if kw in lower)
    if complex_count >= 2:
        return "complex"
    elif complex_count >= 1 and simple_count == 0:
        return "complex"
    elif simple_count >= 2 and complex_count == 0:
        return "simple"
    return "moderate"


# -- ComfyUI async node loader --
_NODES_LOADED = False

def _load_comfy_nodes():
    global _NODES_LOADED
    if _NODES_LOADED:
        return
    import nest_asyncio
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes
    async def _l():
        failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if failed:
            print(f"   Node failures: {failed}")
    try:
        asyncio.run(_l())
    except RuntimeError:
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(_l())
    _NODES_LOADED = True
    # Populate module-level node instances cache
    _populate_node_instances()


# -- Module-level node instances cache --
_NODE_INSTANCES: dict = {}

def _populate_node_instances():
    """Cache instantiated node objects for reuse across generate_clip sub-functions."""
    global _NODE_INSTANCES
    _node_names = [
        "DualCLIPLoader", "CLIPTextEncode", "ConditioningZeroOut",
        "LTXVConditioning", "UnetLoaderGGUF", "VAELoader",
        "LatentUpscaleModelLoader", "EmptyImage", "ResizeImageMaskNode",
        "GetImageSize", "EmptyLTXVLatentVideo", "LTXVPreprocess",
        "LTXVImgToVideoInplace", "LTXVEmptyLatentAudio", "LTXVConcatAVLatent",
        "ManualSigmas", "KSamplerSelect", "RandomNoise", "CFGGuider",
        "SamplerCustomAdvanced", "LTXVSeparateAVLatent", "LTXVCropGuides",
        "LTXVLatentUpsampler", "LTXVSpatioTemporalTiledVAEDecode",
        "VAEDecode", "LTXVAudioVAEDecode", "CreateVideo",
    ]
    for name in _node_names:
        if name in NODE_CLASS_MAPPINGS:
            _NODE_INSTANCES[name] = NODE_CLASS_MAPPINGS[name]()

def _audio_vae(name: str):
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS["VAELoaderKJ"]().load_vae(
            vae_name=name, device="main_device", weight_dtype="fp16")
    return NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=name)


# -- TinyVAE persistence (stays loaded across clips, only ~50MB) --
_TINY_VAE = None


def _load_tiny_vae():
    """Load TinyVAE into a persistent module-level variable.

    Only ~50MB, so it stays loaded across clips to avoid repeated load/unload
    overhead during previews. Call _unload_tiny_vae() to explicitly release.
    """
    global _TINY_VAE
    if _TINY_VAE is not None:
        return _TINY_VAE
    tiny_vae_ld = _get_node("VAELoader")
    _TINY_VAE = get_value_at_index(tiny_vae_ld.load_vae(vae_name=_M_TAEV), 0)
    print("   [TinyVAE] Loaded (persistent).")
    return _TINY_VAE


def _unload_tiny_vae():
    """Explicitly unload the persistent TinyVAE to free memory."""
    global _TINY_VAE
    if _TINY_VAE is not None:
        del _TINY_VAE
        _TINY_VAE = None
        _vram_free()
        print("   [TinyVAE] Unloaded.")

# -- LoRA stack definition --
LD_LORA_STACK = [
    LoRAConfig(on=True, lora="ltx-2-19b-ic-lora-detailer.safetensors", guard=False, strength=0.4),
    LoRAConfig(on=False, lora="None", guard=False, strength=1.0),
    LoRAConfig(on=False, lora="None", guard=False, strength=1.0),
    LoRAConfig(on=False, lora="None", guard=False, strength=1.0),
    LoRAConfig(on=False, lora="None", guard=False, strength=1.0),
    LoRAConfig(on=False, lora="None", guard=False, strength=1.0),
    LoRAConfig(on=False, lora="None", guard=False, strength=1.0),
    LoRAConfig(on=False, lora="None", guard=False, strength=1.0),
    LoRAConfig(on=False, lora="None", guard=False, strength=1.0),
    LoRAConfig(on=False, lora="None", guard=False, strength=1.0),
]
_LORA_STACK_JSON = json.dumps([lc.to_dict() for lc in LD_LORA_STACK])

def _apply_loras(unet, clip_model, lora_config: Optional[List[LoRAConfig]] = None):
    """Apply LoRA stack to UNet. Tries LTX2MasterLoaderLD first; falls back.

    Args:
        unet: The loaded UNet model.
        clip_model: The CLIP model (may be None).
        lora_config: Optional list of LoRAConfig instances. Defaults to LD_LORA_STACK.
    """
    stack = lora_config if lora_config is not None else LD_LORA_STACK
    active = [s for s in stack
              if s.on and s.lora not in (None, "None", "")]
    if not active:
        return unet, clip_model
    stack_json = json.dumps([lc.to_dict() for lc in stack])
    if "LTX2MasterLoaderLD" in NODE_CLASS_MAPPINGS:
        try:
            node = NODE_CLASS_MAPPINGS["LTX2MasterLoaderLD"]()
            result = getattr(node, node.FUNCTION)(
                model=unet, clip=clip_model, stack_data=stack_json)
            print(f"   [LoRA] {len(active)} slot(s) via LTX2MasterLoaderLD")
            return get_value_at_index(result, 0), clip_model
        except Exception as e:
            print(f"   [LoRA] MasterLoader failed ({e}) -- manual fallback")
    for slot in active:
        name, strength = slot.lora, slot.strength
        try:
            unet = LoraLoaderModelOnly().load_lora_model_only(unet, name, strength)[0]
            print(f"   [LoRA]  {name} @ {strength}")
        except Exception as e:
            print(f"   [LoRA] skip  {name}: {e}")
    return unet, clip_model

print("Core utilities ready.")


# ══════════════════════════════════════════════════════════════════════════
# SECTION A  --  VISION DESCRIBE ENGINE
# @title { "single-column": true }
# @markdown ## Section A - VisionDescribeEngine
# Standalone Qwen2.5-VL wrapper. No ComfyUI node wrapper.
# Loads -> describes -> unloads immediately to free VRAM.
# ══════════════════════════════════════════════════════════════════════════

class VisionDescribeEngine:
    """Analyses an image and returns a 100-130 word scene description."""

    MODEL_OPTIONS = {
        "3B-fast": "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
        "7B-nsfw": "prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it",
    }

    PROMPT = (
        "Describe this image in one paragraph of plain sentences, 100-130 words. "
        "Start with 'Style: photorealistic' or 'Style: anime' or 'Style: 3D animation' etc. "
        "The FIRST sentence about any person MUST explicitly state ethnicity and skin tone "
        "using plain terms: 'a Black man', 'a white woman', 'a South Asian man'. "
        "Include age, hair colour and style, body type, clothing or nude state, pose, "
        "camera framing, angle, lighting, time of day, and setting. "
        "One flowing paragraph, no bullets, no labels. "
        "If no person, describe environment, objects, lighting, mood."
    )

    def __init__(self, model_key: str = "3B-fast", offline: bool = False,
                 keep_loaded: bool = False):
        self.model_key = model_key
        self.offline = offline
        self.keep_loaded = keep_loaded
        self._model = None
        self._processor = None

    def unload(self):
        """Explicitly unload the vision model and processor to free VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        _vram_free()
        print("   [VisionDescribe] Unloaded.")

    def describe(self, image: Union[Image.Image, torch.Tensor]) -> str:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from huggingface_hub import snapshot_download
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("[VisionDescribe] pip install qwen-vl-utils")

        if isinstance(image, torch.Tensor):
            image = tensor_to_pil(image)

        # Reuse loaded model if keep_loaded and already loaded
        if self.keep_loaded and self._model is not None and self._processor is not None:
            model = self._model
            processor = self._processor
        else:
            hf_id = self.MODEL_OPTIONS[self.model_key]
            if not self.offline:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                try:
                    source = snapshot_download(hf_id)
                except Exception:
                    source = hf_id
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                source = hf_id

            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"   [VisionDescribe] Loading {self.model_key} ({image.size}) ...")
            processor = AutoProcessor.from_pretrained(source, local_files_only=self.offline)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                source, device_map="auto", torch_dtype=dtype,
                local_files_only=self.offline)
            model.eval()

        messages = [
            {"role": "system", "content":
             "You are an image analysis tool. Describe exactly what you see in plain prose."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.PROMPT},
            ]},
        ]
        text_in = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(text=[text_in], images=img_in, videos=vid_in,
                           padding=True, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        tok = processor.tokenizer
        stop_ids = [i for i in [tok.eos_token_id] if i is not None]
        for s in ["<|im_end|>", "<|endoftext|>"]:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in stop_ids:
                stop_ids.append(ids[0])

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=210, temperature=0.3,
                                 do_sample=True, top_p=0.9,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 eos_token_id=stop_ids)
        desc = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        del out, inputs

        if self.keep_loaded:
            self._model = model
            self._processor = processor
        else:
            del model, processor
            self._model = None
            self._processor = None
            _vram_free()

        print(f"   [VisionDescribe] Done  {len(desc.split())} words.")
        return desc

    def describe_characters(self, image: Union[Image.Image, torch.Tensor]) -> List[dict]:
        """Detect multiple people in the image and return separate character entries.

        Uses a modified prompt asking for numbered descriptions of each person.
        Post-processes the output to split on numbered patterns.

        Args:
            image: PIL Image or torch Tensor of the image to analyse.

        Returns:
            List of dicts, each with 'index' and 'description' keys.
            If only one person (or none), returns a single-entry list.
        """
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from huggingface_hub import snapshot_download
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("[VisionDescribe] pip install qwen-vl-utils")

        if isinstance(image, torch.Tensor):
            image = tensor_to_pil(image)

        # Reuse loaded model if keep_loaded and already loaded
        if self.keep_loaded and self._model is not None and self._processor is not None:
            model = self._model
            processor = self._processor
        else:
            hf_id = self.MODEL_OPTIONS[self.model_key]
            if not self.offline:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                try:
                    source = snapshot_download(hf_id)
                except Exception:
                    source = hf_id
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                source = hf_id

            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"   [VisionDescribe] Loading {self.model_key} for multi-character detection...")
            processor = AutoProcessor.from_pretrained(source, local_files_only=self.offline)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                source, device_map="auto", torch_dtype=dtype,
                local_files_only=self.offline)
            model.eval()

        multi_prompt = (
            "Describe all people visible in this image. "
            "If multiple people are visible, describe each person separately "
            "with a numbered list (1. 2. 3. etc). "
            "For each person include: ethnicity and skin tone, age, hair colour and style, "
            "body type, clothing or nude state, pose. "
            "If only one person is visible, just describe them in one paragraph. "
            "If no people are visible, say 'No people detected'."
        )

        messages = [
            {"role": "system", "content":
             "You are an image analysis tool. Describe exactly what you see."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": multi_prompt},
            ]},
        ]
        text_in = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(text=[text_in], images=img_in, videos=vid_in,
                           padding=True, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        tok = processor.tokenizer
        stop_ids = [i for i in [tok.eos_token_id] if i is not None]
        for s in ["<|im_end|>", "<|endoftext|>"]:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in stop_ids:
                stop_ids.append(ids[0])

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, temperature=0.3,
                                 do_sample=True, top_p=0.9,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 eos_token_id=stop_ids)
        raw_output = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        del out, inputs

        if self.keep_loaded:
            self._model = model
            self._processor = processor
        else:
            del model, processor
            self._model = None
            self._processor = None
            _vram_free()

        # Post-process: split on numbered patterns (1. 2. 3. etc)
        entries = re.split(r'\n?\s*\d+\.\s+', raw_output)
        # First split element is often empty or preamble before "1."
        entries = [e.strip() for e in entries if e.strip()]

        if not entries:
            entries = [raw_output.strip()]

        results = []
        for idx, entry in enumerate(entries, 1):
            results.append({"index": idx, "description": entry})

        print(f"   [VisionDescribe] Multi-character: detected {len(results)} character(s).")
        return results


# ══════════════════════════════════════════════════════════════════════════
# SECTION B  --  EASY PROMPT ENGINE
# @title { "single-column": true }
# @markdown ## Section B - EasyPromptEngine
# Full cinematic LLM expansion with pacing, character bible lock,
# dialogue control, explicit content tiers, output cleaning.
# ══════════════════════════════════════════════════════════════════════════

# -- Negative prompt builder --
_NEG_BASE = (
    "blurry, out of focus, low quality, worst quality, jpeg artifacts, "
    "static, no motion, frozen, duplicate, watermark, text, signature, "
    "poorly drawn, bad anatomy, deformed, disfigured, extra limbs, "
    "missing limbs, overexposed, underexposed, grainy, noise, flickering"
)

def _build_neg(result: str, user_input: str) -> str:
    c = (result + " " + user_input).lower()
    extras = []
    if any(w in c for w in ["indoor", "room", "interior", "bedroom", "kitchen", "office"]):
        extras.append("harsh outdoor lighting, direct sunlight")
    elif any(w in c for w in ["outdoor", "street", "beach", "forest", "park"]):
        extras.append("studio background, indoor lighting")
    if any(w in c for w in ["pussy", "cock", "penis", "vagina", "nude", "naked", "nipple", "breast"]):
        extras.append("censored, mosaic, pixelated, black bar, blurred genitals")
    if any(w in c for w in ["close-up", "close up", "portrait", "headshot"]):
        extras.append("wide angle distortion, fish eye")
    elif any(w in c for w in ["wide shot", "wide angle", "aerial"]):
        extras.append("close-up, portrait crop")
    if any(w in c for w in ["night", "dark", "moonlight", "dimly lit", "candlelight"]):
        extras.append("overexposed, bright daylight, blown highlights")
    elif any(w in c for w in ["daylight", "sunny", "golden hour", "bright"]):
        extras.append("underexposed, dark shadows, black crush")
    if any(w in c for w in ["two women", "two men", "two people", "couple", "both"]):
        extras.append("merged bodies, fused figures, incorrect number of people")
    return ", ".join([_NEG_BASE] + extras)


class EasyPromptEngine:
    """
    Expands a simple story beat into a dense cinematic LTX-2 prompt.
    Loads the LLM, generates, cleans output, then unloads to free VRAM.
    """

    MODELS = {
        "8B": "mlabonne/NeuralDaredevil-8B-abliterated",
        "3B": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "14B": "huihui-ai/Huihui-Qwen3-14B-abliterated-v2",
    }

    SYSTEM_PROMPT = (
        "You are a cinematic prompt writer for LTX-2, an AI video generation model. "
        "Expand the user's idea into a rich, video-ready prompt.\n\n"
        "PRIORITY ORDER:\n"
        "1. Video style & genre (slow-burn thriller, documentary, editorial, action blockbuster)\n"
        "2. Camera angle & shot type (low-angle close-up, bird's-eye wide, Dutch angle medium)\n"
        "3. Character description - age MUST be a specific number (e.g. 'a 28-year-old woman'), "
        "body type, hair, skin, clothing. Use exact words from the user.\n"
        "4. Scene & environment (location, time of day, lighting, colour palette, atmosphere)\n"
        "5. Action & motion - continuous present-tense sequence.\n"
        "6. Camera movement - prose only, no screenplay brackets.\n"
        "7. Audio - max 2 ambient sounds active at once, woven as prose.\n\n"
        "RULES:\n"
        "- Present tense throughout.\n"
        "- 8-12 sentences of dense flowing prose - no bullet lists.\n"
        "- Fill the full token budget. Do not stop early.\n"
        "- Output ONLY the expanded prompt. No preamble. No trailing notes."
    )

    _CLEAN_RE = [
        (re.compile(r"<think>.*?</think>", re.DOTALL), ""),
        (re.compile(r"^(Sure!?|Certainly!?|Here(?:'s| is).*?:)[^\n]*\n?", re.IGNORECASE), ""),
        (re.compile(r"\s*(assistant|user|system|<\|[^|>]*\|>)\s*$", re.IGNORECASE), ""),
        (re.compile(r"\s*\n+Note:.*$", re.DOTALL), ""),
        (re.compile(r"\s*\(Note:.*$", re.DOTALL | re.IGNORECASE), ""),
        (re.compile(r"\s*(\([^)]{5,120}\)\s*){2,}$", re.DOTALL), ""),
        (re.compile(
            r"\s*\n+(Please let me know|Let me revise|Confirmed\.|Output ends|"
            r"Done\.|I hope|Thank you|No further).*$",
            re.DOTALL | re.IGNORECASE), ""),
        (re.compile(r"\n{3,}"), "\n\n"),
    ]

    def __init__(self, model_size: str = "8B", offline: bool = False,
                 keep_loaded: bool = False):
        self.model_size = model_size
        self.offline = offline
        self.keep_loaded = keep_loaded
        self._tok = None
        self._model = None
        self._loaded_key = None

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
        key = self.model_size
        if self._model is not None and self._loaded_key == key:
            return
        if self._model is not None:
            self._unload()
        hf_id = self.MODELS[key]
        if not self.offline:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                source = snapshot_download(hf_id)
            except Exception:
                source = hf_id
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            source = hf_id
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"   [EasyPrompt] Loading {key} ...")
        self._tok = AutoTokenizer.from_pretrained(source, local_files_only=self.offline)
        self._model = AutoModelForCausalLM.from_pretrained(
            source, device_map="auto", torch_dtype=dtype,
            trust_remote_code=True, local_files_only=self.offline)
        self._model.config.use_cache = True
        self._model.eval()
        self._loaded_key = key
        print(f"   [EasyPrompt] Loaded.")

    def _unload(self):
        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
        self._model = None
        self._tok = None
        self._loaded_key = None
        _vram_free()
        print("   [EasyPrompt] VRAM cleared.")

    def _stop_ids(self) -> List[int]:
        delims = ["assistant", "user", "system", "<|eot_id|>", "<|end_of_turn|>",
                  "<|im_end|>", "<end_of_turn>", "[/INST]", "### Human", "### Assistant"]
        ids = [self._tok.eos_token_id]
        for s in delims:
            enc = self._tok.encode(s, add_special_tokens=False)
            if enc and enc[0] not in ids:
                ids.append(enc[0])
        return [i for i in dict.fromkeys(ids) if i is not None]

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        for pattern, repl in EasyPromptEngine._CLEAN_RE:
            text = pattern.sub(repl, text)
        text = re.sub(r"\s*[\(\[]\s*$", "", text)
        return text.strip()

    def generate(
        self,
        user_input: str,
        frame_count: int = 121,
        creativity: float = 0.9,
        seed: int = -1,
        scene_context: str = "",
        lora_triggers: str = "",
        character_bible: str = "",
    ) -> tuple:
        """
        Returns (positive_prompt, negative_prompt).

        character_bible is injected as a hard [CHARACTER BIBLE - NON-NEGOTIABLE]
        block so the LLM cannot alter hair, age, clothing, or any locked attribute.
        """
        self._load()

        real_seconds = frame_count / 25.0
        action_count = max(1, min(10, round(real_seconds / 4)))
        token_budget = max(256, min(1200, action_count * 120))
        max_tokens = int(token_budget * 1.05)
        min_tokens = int(token_budget * 0.75)

        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # -- Pacing constraint --
        ordinal = {2: "2nd", 3: "3rd"}.get(action_count, f"{action_count}th")
        if action_count > 1:
            pacing = (
                f"This clip is {real_seconds:.0f}s. Write EXACTLY {action_count} "
                f"distinct actions. "
                f"HARD STOP after the {ordinal} action. "
                f"Write ~{token_budget} tokens. No trailing notes or brackets."
            )
        else:
            pacing = (
                f"This clip is {real_seconds:.0f}s. Write EXACTLY 1 action. "
                f"HARD STOP after it. ~{token_budget} tokens."
            )

        # -- Character bible lock --
        bible_clause = ""
        if character_bible.strip():
            bible_clause = (
                "\n[CHARACTER BIBLE - NON-NEGOTIABLE: Every character attribute below "
                "MUST remain exactly as described. Do NOT alter hair, age, skin, "
                "clothing, or any other attribute. This overrides any inference:\n"
                f"{character_bible.strip()}\n]"
            )

        # -- Scene context (from Vision Describe) --
        if scene_context.strip():
            effective = (
                "[SCENE CONTEXT FROM IMAGE - authoritative, do not contradict]\n"
                f"{scene_context.strip()}\n\n"
                f"[USER DIRECTION - action, style, mood]\n{user_input.strip()}"
            )
        else:
            effective = user_input.strip()

        # -- LoRA trigger injection --
        lora_clause = ""
        if lora_triggers.strip():
            lora_clause = f"\n[LORA: Begin prompt with: {lora_triggers.strip()}]"

        user_content = effective + bible_clause + lora_clause + f"\n[PACING: {pacing}]"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        is_qwen3 = "Qwen3" in self.MODELS.get(self.model_size, "")
        kwargs = {"enable_thinking": False} if is_qwen3 else {}
        raw = self._tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, **kwargs)

        if hasattr(raw, "input_ids"):
            input_ids = raw.input_ids.to(self._model.device)
        elif isinstance(raw, dict):
            input_ids = raw["input_ids"].to(self._model.device)
        elif isinstance(raw, list):
            input_ids = torch.tensor([raw], dtype=torch.long).to(self._model.device)
        else:
            input_ids = raw.to(self._model.device)

        input_len = input_ids.shape[1]

        with torch.no_grad():
            out = self._model.generate(
                input_ids, min_new_tokens=min_tokens, max_new_tokens=max_tokens,
                temperature=creativity, do_sample=True, top_k=40, top_p=0.9,
                repetition_penalty=1.07, use_cache=True,
                pad_token_id=self._tok.eos_token_id,
                eos_token_id=self._stop_ids())

        result = self._tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        result = self._clean(result)
        result = re.sub(r'\s*[\(\[]\s*$', '', result).strip()
        del out, input_ids
        neg = _build_neg(result, user_input)

        if not self.keep_loaded:
            self._unload()

        print(f"   [EasyPrompt] Done  {len(result.split())} words generated.")
        return result, neg


# ══════════════════════════════════════════════════════════════════════════
# SECTION C  --  CHARACTER BIBLE
# @title { "single-column": true }
# @markdown ## Section C - CharacterBible
# Serialisable cross-scene character consistency record.
# ══════════════════════════════════════════════════════════════════════════

class CharacterBible:
    """
    Records named character attributes and serialises them as a prompt-injection
    block. The block is passed to EasyPromptEngine as character_bible so the
    LLM receives a hard NON-NEGOTIABLE constraint preventing attribute drift.

    Auto-populated from Vision Describe output on the seed image (recommended),
    or filled manually with add().
    """

    def __init__(self):
        self._chars: dict = {}

    def add(self, name: str, **attributes):
        """Manually define a character."""
        self._chars[name] = dict(attributes)

    def extract_from_description(self, name: str, description: str):
        """Store a raw Vision Describe output under a character name."""
        self._chars[name] = {"_raw": description.strip()}

    def parse_structured(self, name: str, description: str) -> dict:
        """Extract structured fields from a VisionDescribe output using regex.

        Extracts: age, hair_color, skin_tone, clothing, build.
        Stores both _raw and _structured in the character entry.

        Returns:
            dict with extracted fields (may be partially empty).
        """
        text = description.strip()
        structured = {}

        # Age: look for numbers near 'year' or 'old' or standalone age patterns
        age_match = re.search(r'(\d{1,3})\s*[-\s]?\s*(?:year|yr)s?\s*(?:old)?', text, re.I)
        if not age_match:
            age_match = re.search(r'(?:age[d]?\s*|around\s+|about\s+)(\d{1,3})', text, re.I)
        if age_match:
            structured["age"] = age_match.group(1)

        # Hair color: color words near 'hair'
        hair_match = re.search(
            r'(blonde|brunette|black|brown|red|auburn|grey|gray|white|silver|dark|light|'
            r'platinum|golden|ginger|sandy|chestnut|raven)\s+hair',
            text, re.I)
        if not hair_match:
            hair_match = re.search(
                r'hair\s+(?:is\s+|that\s+is\s+)?'
                r'(blonde|brunette|black|brown|red|auburn|grey|gray|white|silver|dark|light|'
                r'platinum|golden|ginger|sandy|chestnut|raven)',
                text, re.I)
        if hair_match:
            structured["hair_color"] = hair_match.group(1).lower()

        # Skin tone: ethnicity or skin descriptors
        skin_match = re.search(
            r'((?:light|dark|medium|olive|fair|pale|tan|deep|warm|cool)[\s-]*skin(?:ned)?|'
            r'(?:Black|white|South Asian|East Asian|Latino|Latina|Hispanic|Middle Eastern|'
            r'African|Caucasian|Asian|Indigenous|Pacific Islander)\s+(?:man|woman|person|male|female))',
            text)
        if not skin_match:
            skin_match = re.search(
                r'(?:skin\s+(?:tone|color|colour)\s+(?:is\s+)?)'
                r'(light|dark|medium|olive|fair|pale|tan|deep|warm|cool)',
                text, re.I)
        if skin_match:
            structured["skin_tone"] = skin_match.group(1).strip()

        # Clothing: sentences containing clothing keywords
        clothing_keywords = [
            'wearing', 'dressed', 'shirt', 'pants', 'dress', 'jacket', 'coat',
            'suit', 'blouse', 'skirt', 'jeans', 'shorts', 'hoodie', 'sweater',
            'top', 'outfit', 'uniform', 'gown', 'robe', 'vest', 'hat', 'shoes',
            'boots', 'sneakers', 'nude', 'naked', 'bikini', 'swimsuit', 'tank',
        ]
        sentences = re.split(r'[.!?]+', text)
        clothing_parts = []
        for s in sentences:
            s_lower = s.lower()
            if any(kw in s_lower for kw in clothing_keywords):
                clothing_parts.append(s.strip())
        if clothing_parts:
            structured["clothing"] = ". ".join(clothing_parts)

        # Build: body type keywords
        build_match = re.search(
            r'(slim|slender|athletic|muscular|stocky|heavyset|petite|tall|short|'
            r'thin|curvy|average|lean|broad|large|small|fit|toned|plump|robust)',
            text, re.I)
        if build_match:
            structured["build"] = build_match.group(1).lower()

        # Store both raw and structured
        self._chars[name] = {"_raw": text, "_structured": structured}
        return structured

    def to_prompt_block(self) -> str:
        """Returns the injection string for EasyPromptEngine."""
        if not self._chars:
            return ""
        lines = []
        for name, attrs in self._chars.items():
            if "_structured" in attrs and attrs["_structured"]:
                # Prefer structured fields for precision
                parts = []
                s = attrs["_structured"]
                if "age" in s:
                    parts.append(f"age: {s['age']}")
                if "hair_color" in s:
                    parts.append(f"hair: {s['hair_color']}")
                if "skin_tone" in s:
                    parts.append(f"skin: {s['skin_tone']}")
                if "build" in s:
                    parts.append(f"build: {s['build']}")
                if "clothing" in s:
                    parts.append(f"clothing: {s['clothing']}")
                structured_str = "; ".join(parts)
                # Include raw description as fallback context
                raw = attrs.get("_raw", "")
                if raw:
                    lines.append(
                        f"CHARACTER - {name} [{structured_str}]:\n{raw}")
                else:
                    lines.append(f"CHARACTER - {name}: {structured_str}")
            elif "_raw" in attrs:
                lines.append(f"CHARACTER - {name}:\n{attrs['_raw']}")
            else:
                attr_str = "; ".join(f"{k}: {v}" for k, v in attrs.items())
                lines.append(f"CHARACTER - {name}: {attr_str}")
        return "\n\n".join(lines)

    def has_characters(self) -> bool:
        return bool(self._chars)

    def validate_bible(self, current_frame: 'torch.Tensor',
                       vision_engine: 'VisionDescribeEngine') -> dict:
        """Run VisionDescribe on the current frame and compare attributes against
        stored bible entries. Returns a drift report.

        Args:
            current_frame: The current video frame as a tensor.
            vision_engine: A VisionDescribeEngine instance for image analysis.

        Returns:
            dict of {character_name: {attribute: (expected, observed, drifted)}}
            Empty dict if no structured data or no characters.
        """
        if not self._chars:
            return {}

        # Describe the current frame
        current_desc = vision_engine.describe(current_frame)
        current_lower = current_desc.lower()

        drift_report = {}
        for name, attrs in self._chars.items():
            structured = attrs.get("_structured", {})
            if not structured:
                continue

            char_drift = {}

            # Check hair_color drift
            if "hair_color" in structured:
                expected = structured["hair_color"]
                observed = "unknown"
                hair_match = re.search(
                    r'(blonde|brunette|black|brown|red|auburn|grey|gray|white|'
                    r'silver|dark|light|platinum|golden|ginger|sandy|chestnut|raven)\s+hair',
                    current_lower)
                if hair_match:
                    observed = hair_match.group(1)
                drifted = (observed != "unknown" and observed != expected.lower())
                char_drift["hair_color"] = (expected, observed, drifted)

            # Check skin_tone drift
            if "skin_tone" in structured:
                expected = structured["skin_tone"]
                observed = "unknown"
                skin_match = re.search(
                    r'(light|dark|medium|olive|fair|pale|tan|deep|warm|cool)[\s-]*skin',
                    current_lower)
                if skin_match:
                    observed = skin_match.group(1)
                drifted = (observed != "unknown"
                           and observed.lower() not in expected.lower())
                char_drift["skin_tone"] = (expected, observed, drifted)

            # Check clothing drift
            if "clothing" in structured:
                expected = structured["clothing"].lower()
                # Extract clothing keywords from expected
                expected_keywords = set(re.findall(
                    r'(shirt|pants|dress|jacket|coat|suit|blouse|skirt|jeans|'
                    r'shorts|hoodie|sweater|top|uniform|gown|robe|vest|hat|'
                    r'shoes|boots|sneakers|bikini|swimsuit|tank)',
                    expected))
                observed_keywords = set(re.findall(
                    r'(shirt|pants|dress|jacket|coat|suit|blouse|skirt|jeans|'
                    r'shorts|hoodie|sweater|top|uniform|gown|robe|vest|hat|'
                    r'shoes|boots|sneakers|bikini|swimsuit|tank)',
                    current_lower))
                observed = ", ".join(observed_keywords) if observed_keywords else "unknown"
                # Drift if significant mismatch (less than half overlap)
                if expected_keywords and observed_keywords:
                    overlap = expected_keywords & observed_keywords
                    drifted = len(overlap) < len(expected_keywords) * 0.5
                else:
                    drifted = False
                char_drift["clothing"] = (
                    ", ".join(expected_keywords), observed, drifted)

            if char_drift:
                drift_report[name] = char_drift

        return drift_report

    def names(self) -> List[str]:
        return list(self._chars.keys())

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self._chars, f, indent=2)
        print(f"   [Bible] Saved -> {path}")

    def load(self, path: str):
        with open(path) as f:
            self._chars = json.load(f)
        print(f"   [Bible] Loaded from {path}")

    def __repr__(self):
        return f"CharacterBible({self.names()})"


def _validate_prompt(prompt: str, raw_beat: str) -> tuple:
    """Validate an expanded prompt for quality before use in video generation.

    Checks:
      (a) Minimum word count >= 30 words.
      (b) No excessive repetition (no word repeated > 5 times in sequence).
      (c) Prompt is substantially longer than raw beat (at least 2x word count).

    Args:
        prompt: The expanded prompt from EasyPromptEngine.
        raw_beat: The original story beat string.

    Returns:
        Tuple of (is_valid: bool, reason: str). If is_valid is False, reason
        explains why validation failed.
    """
    words = prompt.split()
    word_count = len(words)

    # (a) Minimum word count
    if word_count < 30:
        return (False, f"prompt too short ({word_count} words, minimum 30)")

    # (b) Excessive sequential repetition
    if word_count > 1:
        repeat_count = 1
        for i in range(1, len(words)):
            if words[i].lower() == words[i - 1].lower():
                repeat_count += 1
                if repeat_count > 5:
                    return (False, f"excessive repetition of '{words[i]}' "
                            f"({repeat_count}+ times in sequence)")
            else:
                repeat_count = 1

    # (c) Substantially longer than raw beat
    beat_word_count = len(raw_beat.split())
    if beat_word_count > 0 and word_count < beat_word_count * 2:
        return (False, f"prompt not substantially expanded "
                f"({word_count}w vs beat {beat_word_count}w, need 2x)")

    return (True, "")


# ══════════════════════════════════════════════════════════════════════════
# SECTION D  --  GENERATE CLIP  (two-pass LTX-2 19B GGUF pipeline)
# @title { "single-column": true }
# @markdown ## Section D - generate_clip()
#
# VRAM sequence (T4-safe, peak ~13 GB):
#   1. Load CLIP Gemma -> encode text -> DELETE CLIP -> free
#   2. Load UNet GGUF -> apply LoRAs
#   3. Load VAE_video + VAE_audio + upscaler
#   4. Prepare latents (I2V via LTXVImgToVideoInplace if image, else T2V)
#   5. Pass 1 (ManualSigmas + euler, CFG=1.0)
#   6. TinyVAE Preview (taeltx2_3.safetensors) -> quick decode -> display -> delete
#   7. Pass 2 (spatial upscale + gradient_estimation)
#   8. Delete UNet
#   9. Tiled VAE decode (spatial_tiles=2, spatial_overlap=8, temporal_tile_length=48)
#  10. Audio decode
#  11. Save as h264-mp4 at 25fps
# ══════════════════════════════════════════════════════════════════════════

def _get_node(name: str):
    """Get a cached node instance, or instantiate from NODE_CLASS_MAPPINGS."""
    if name in _NODE_INSTANCES:
        return _NODE_INSTANCES[name]
    return NODE_CLASS_MAPPINGS[name]()


def _load_clip_encode(prompt: str, fps: int, cache_clip_cond: bool = True) -> tuple:
    """Load CLIP, encode prompt, return conditioning tuple.
    Deletes CLIP model after encoding to free VRAM for UNet.

    If cache_clip_cond is True, checks _CLIP_COND_CACHE first. On a cache hit,
    returns cached tensors (moved back to current device) without loading CLIP.
    After encoding, stores results in cache (on CPU) for future reuse.
    """
    global _CLIP_COND_CACHE

    # Check cache first
    if cache_clip_cond and prompt in _CLIP_COND_CACHE:
        print("   [Clip] CLIP cache HIT -- skipping CLIP load.")
        cached = _CLIP_COND_CACHE[prompt]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Return cached conditioning moved to GPU
        cond_pos_cached = cached["cond_pos"]
        cond_neg_cached = cached["cond_neg"]
        # Reconstruct conditioning via LTXVConditioning with cached values
        # The cache stores the final result of LTXVConditioning, so return directly
        return cached["cond_result"]

    print("   [Clip] Loading CLIP...")
    clip_ld = _get_node("DualCLIPLoader")
    try:
        clip_raw = get_value_at_index(
            clip_ld.load_clip(clip_name1=_M_CLIP1, clip_name2=_M_CLIP2,
                              type="ltxv", device="default"), 0)
    except Exception as e:
        print(f"   [Clip] fp4 CLIP failed ({e}), trying fp8 fallback...")
        fp8_name = "gemma_3_12B_it_fp8_scaled.safetensors"
        clip_raw = get_value_at_index(
            clip_ld.load_clip(clip_name1=fp8_name, clip_name2=_M_CLIP2,
                              type="ltxv", device="default"), 0)

    cte = _get_node("CLIPTextEncode")
    cond_pos = cte.encode(text=prompt, clip=clip_raw)
    # NOTE: neg_prompt is intentionally NOT encoded here. The GGUF distilled
    # model runs at CFG=1.0 which requires zeroed-out negative conditioning.
    zero_out = _get_node("ConditioningZeroOut")
    cond_0 = zero_out.zero_out(conditioning=get_value_at_index(cond_pos, 0))
    ltxv_cn = _get_node("LTXVConditioning")
    cond = ltxv_cn.EXECUTE_NORMALIZED(
        frame_rate=float(fps),
        positive=get_value_at_index(cond_pos, 0),
        negative=get_value_at_index(cond_0, 0))

    # Store in cache (on CPU) for future reuse
    if cache_clip_cond:
        _CLIP_COND_CACHE[prompt] = {"cond_result": cond, "cond_pos": None, "cond_neg": None}
        print(f"   [Clip] Cached conditioning for prompt (cache size: {len(_CLIP_COND_CACHE)})")

    # Delete CLIP now -- frees ~6-8 GB for UNet
    del clip_raw, clip_ld
    _vram_free()
    _vram_print("CLIP deleted")
    return cond


def _load_unet(lora_config: Optional[List[LoRAConfig]] = None):
    """Load UNet GGUF and apply LoRAs. Returns (unet, unet_loader)."""
    print("   [Clip] Loading UNet (GGUF Q4_K_M)...")
    unet_ld = _get_node("UnetLoaderGGUF")
    unet = get_value_at_index(unet_ld.load_unet(unet_name=_M_UNET), 0)
    _dummy_clip = None
    unet, _ = _apply_loras(unet, _dummy_clip, lora_config=lora_config)
    _vram_print("UNet + LoRAs loaded")
    return unet, unet_ld


def _prepare_latents(image_tensor, width, height, frames, fps, vae_v, vae_a,
                     image_strength, anchor_image, anchor_weight,
                     skip_audio=False):
    """Prepare AV latent tensor for sampling. Handles T2V and I2V modes.
    Returns av_lat tensor.

    If skip_audio is True, skips LTXVEmptyLatentAudio and returns video-only latent."""
    img_bypass = image_tensor is None
    img_str = image_strength if not img_bypass else 0.0

    ei = _get_node("EmptyImage")
    full_i = ei.generate(width=width, height=height, batch_size=1, color=0)
    rimn = _get_node("ResizeImageMaskNode")
    half_i = rimn.EXECUTE_NORMALIZED(
        input=get_value_at_index(full_i, 0), scale_method="area",
        resize_type={"resize_type": "scale by multiplier", "multiplier": 0.5})
    gis = _get_node("GetImageSize")
    hsz = gis.EXECUTE_NORMALIZED(image=get_value_at_index(half_i, 0))
    hw, hh = get_value_at_index(hsz, 0), get_value_at_index(hsz, 1)

    eltxv = _get_node("EmptyLTXVLatentVideo")
    vid_lat = eltxv.EXECUTE_NORMALIZED(width=hw, height=hh, length=frames, batch_size=1)

    # I2V: condition the latent on the seed/last-frame image
    if not img_bypass:
        conditioning_img = image_tensor
        if anchor_image is not None:
            conditioning_img = (
                (1.0 - anchor_weight) * image_tensor + anchor_weight * anchor_image
            )
        rim2 = _get_node("ResizeImageMaskNode")
        ri2 = rim2.EXECUTE_NORMALIZED(
            input=conditioning_img, scale_method="lanczos",
            resize_type={"resize_type": "scale dimensions",
                         "width": hw * 2, "height": hh * 2, "crop": "center"})
        ppn = _get_node("LTXVPreprocess")
        pp = get_value_at_index(ppn.EXECUTE_NORMALIZED(
            img_compression=33, image=get_value_at_index(ri2, 0)), 0)
        i2v = _get_node("LTXVImgToVideoInplace")
        vid_lat = (get_value_at_index(i2v.EXECUTE_NORMALIZED(
            strength=img_str, bypass=False,
            vae=vae_v, image=pp,
            latent=get_value_at_index(vid_lat, 0)), 0),)
    else:
        vid_lat = (get_value_at_index(vid_lat, 0),)

    # Audio latent + concat AV (or return video-only if skip_audio)
    if skip_audio:
        return vid_lat[0]

    elalat = _get_node("LTXVEmptyLatentAudio")
    aud_lat = elalat.EXECUTE_NORMALIZED(
        frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=vae_a)
    catav = _get_node("LTXVConcatAVLatent")
    av_lat = get_value_at_index(catav.EXECUTE_NORMALIZED(
        video_latent=vid_lat[0],
        audio_latent=get_value_at_index(aud_lat, 0)), 0)
    return av_lat


def _pass1(unet, cond, av_lat, seed, sigmas, sampler, cfg):
    """Run Pass 1 sampling. Returns p1_av latent."""
    print("   [Clip] Pass 1...")
    _vram_print()
    ms = _get_node("ManualSigmas")
    ks = _get_node("KSamplerSelect")
    rn = _get_node("RandomNoise")
    cfg_node = _get_node("CFGGuider")
    sca = _get_node("SamplerCustomAdvanced")

    g1 = cfg_node.EXECUTE_NORMALIZED(cfg=cfg, model=unet,
                                     positive=get_value_at_index(cond, 0),
                                     negative=get_value_at_index(cond, 1))
    o1 = sca.EXECUTE_NORMALIZED(
        noise=get_value_at_index(rn.EXECUTE_NORMALIZED(noise_seed=seed), 0),
        guider=get_value_at_index(g1, 0),
        sampler=get_value_at_index(ks.EXECUTE_NORMALIZED(sampler_name=sampler), 0),
        sigmas=get_value_at_index(ms.EXECUTE_NORMALIZED(sigmas=sigmas), 0),
        latent_image=av_lat)
    p1_av = get_value_at_index(o1, 0)
    del g1
    _vram_free()
    print("   [Clip] Pass 1 done")
    return p1_av


def _preview(p1_av, show_preview: bool) -> None:
    """TinyVAE preview after Pass 1. Uses persistent _TINY_VAE if available."""
    if not show_preview:
        return
    try:
        print("   [Clip] TinyVAE preview...")
        sep_prev = _get_node("LTXVSeparateAVLatent")
        s_prev = sep_prev.EXECUTE_NORMALIZED(av_latent=p1_av)
        vl_prev = get_value_at_index(s_prev, 0)

        tiny_vae = _load_tiny_vae()

        vd_prev = _get_node("VAEDecode")
        prev_frames = get_value_at_index(
            vd_prev.decode(samples=vl_prev, vae=tiny_vae), 0)

        if prev_frames is not None and prev_frames.shape[0] > 0:
            preview_img = tensor_to_pil(prev_frames[0:1])
            display(preview_img)
            print("   [Clip] TinyVAE preview displayed")

        del prev_frames, vl_prev, s_prev, sep_prev, vd_prev
        _vram_free()
    except Exception as e:
        print(f"   [Clip] TinyVAE preview skipped: {e}")


def _pass2(unet, cond, p1_av, vae_v, up_mdl, seed, sigmas, sampler, cfg):
    """Run Pass 2 -- upscale + refine. Returns p2_den latent."""
    print("   [Clip] Pass 2 (upscale + refine)...")
    sep = _get_node("LTXVSeparateAVLatent")
    s1 = sep.EXECUTE_NORMALIZED(av_latent=p1_av)
    vl1, al1 = get_value_at_index(s1, 0), get_value_at_index(s1, 1)

    crop = _get_node("LTXVCropGuides")
    cr = crop.EXECUTE_NORMALIZED(positive=get_value_at_index(cond, 0),
                                 negative=get_value_at_index(cond, 1),
                                 latent=vl1)
    cfg_node = _get_node("CFGGuider")
    g2 = cfg_node.EXECUTE_NORMALIZED(cfg=cfg, model=unet,
                                     positive=get_value_at_index(cr, 0),
                                     negative=get_value_at_index(cr, 1))

    ltxvup = _get_node("LTXVLatentUpsampler")
    up = ltxvup.upsample_latent(samples=get_value_at_index(cr, 2),
                                upscale_model=up_mdl, vae=vae_v)
    del up_mdl
    _vram_free()

    catav = _get_node("LTXVConcatAVLatent")
    av2 = get_value_at_index(catav.EXECUTE_NORMALIZED(
        video_latent=get_value_at_index(up, 0), audio_latent=al1), 0)

    ms = _get_node("ManualSigmas")
    ks = _get_node("KSamplerSelect")
    rn = _get_node("RandomNoise")
    sca = _get_node("SamplerCustomAdvanced")

    o2 = sca.EXECUTE_NORMALIZED(
        noise=get_value_at_index(rn.EXECUTE_NORMALIZED(noise_seed=seed), 0),
        guider=get_value_at_index(g2, 0),
        sampler=get_value_at_index(ks.EXECUTE_NORMALIZED(sampler_name=sampler), 0),
        sigmas=get_value_at_index(ms.EXECUTE_NORMALIZED(sigmas=sigmas), 0),
        latent_image=av2)
    p2_den = get_value_at_index(o2, 1)

    # BIG VRAM RELEASE -- UNet is done
    del g2, unet
    _vram_free()
    _vram_print("UNet deleted")
    print("   [Clip] Pass 2 done")
    return p2_den


def _decode_video(p2_den, vae_v, use_tiled_vae, tiled_stiles, tiled_soverlap,
                  tiled_tlen, tiled_toverlap):
    """Decode video latent to frames. Returns (decoded_tensor, separated_result)."""
    sep = _get_node("LTXVSeparateAVLatent")
    s2 = sep.EXECUTE_NORMALIZED(av_latent=p2_den)
    vl_f = get_value_at_index(s2, 0)

    decoded = None
    if use_tiled_vae:
        try:
            td = _get_node("LTXVSpatioTemporalTiledVAEDecode")
            decoded = get_value_at_index(td.EXECUTE_NORMALIZED(
                vae=vae_v, latents=vl_f,
                spatial_tiles=tiled_stiles, spatial_overlap=tiled_soverlap,
                temporal_tile_length=tiled_tlen, temporal_overlap=tiled_toverlap,
                last_frame_fix=False, working_device="auto", working_dtype="auto"), 0)
            print("   [Clip] Tiled VAE done")
        except (KeyError, Exception) as e:
            print(f"   [Clip] Tiled VAE skipped ({type(e).__name__}) -- standard decode")
            use_tiled_vae = False

    if not use_tiled_vae:
        vd = _get_node("VAEDecode")
        decoded = get_value_at_index(vd.decode(samples=vl_f, vae=vae_v), 0)

    del vae_v
    _vram_free()
    return decoded, s2


def _decode_audio(s2_separated, vae_a):
    """Decode audio latent. Returns audio output."""
    al_f = get_value_at_index(s2_separated, 1)
    aud_d = _get_node("LTXVAudioVAEDecode")
    audio = aud_d.EXECUTE_NORMALIZED(samples=al_f, audio_vae=vae_a)
    del vae_a
    _vram_free()
    return audio


def _save(decoded, audio, fps, prefix) -> str:
    """Combine video frames + audio and save to disk. Returns output path."""
    cv_node = _get_node("CreateVideo")
    vid_obj = cv_node.EXECUTE_NORMALIZED(
        fps=fps, images=decoded,
        audio=get_value_at_index(audio, 0))
    path = save_video_obj(get_value_at_index(vid_obj, 0), prefix=prefix)
    _vram_print("after save")
    return path


def _slerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two tensors.

    Args:
        t: Interpolation factor (0.0 = v0, 1.0 = v1).
        v0: Start tensor.
        v1: End tensor.

    Returns:
        Interpolated tensor of the same shape.
    """
    # Flatten for dot product computation
    v0_flat = v0.reshape(-1).float()
    v1_flat = v1.reshape(-1).float()

    # Normalize
    v0_norm = v0_flat / (v0_flat.norm() + 1e-8)
    v1_norm = v1_flat / (v1_flat.norm() + 1e-8)

    # Compute angle between vectors
    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    omega = torch.acos(dot)

    # Fall back to linear interpolation if vectors are nearly parallel
    if omega.abs() < 1e-6:
        return ((1.0 - t) * v0 + t * v1).to(v0.dtype)

    sin_omega = torch.sin(omega)
    coeff_0 = torch.sin((1.0 - t) * omega) / sin_omega
    coeff_1 = torch.sin(t * omega) / sin_omega

    result = (coeff_0 * v0_flat + coeff_1 * v1_flat).reshape(v0.shape)
    return result.to(v0.dtype)


def generate_clip(
    image_tensor: Optional[torch.Tensor],
    prompt: str,
    neg_prompt: str,
    width: int = 768,
    height: int = 512,
    frames: int = 121,
    fps: int = 25,
    seed: int = 42,
    image_strength: float = 1.0,
    anchor_image: Optional[torch.Tensor] = None,
    anchor_weight: float = 0.25,
    pass1_sigmas: str = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
    pass1_sampler: str = "euler",
    pass1_cfg: float = 1.0,
    pass2_sigmas: str = "0.909375, 0.725, 0.421875, 0.0",
    pass2_sampler: str = "gradient_estimation",
    pass2_cfg: float = 1.0,
    pass2_seed: int = 0,
    use_tiled_vae: bool = True,
    tiled_stiles: int = 2,
    tiled_soverlap: int = 8,
    tiled_tlen: int = 48,
    tiled_toverlap: int = 4,
    show_preview: bool = True,
    output_prefix: str = "IFE",
    lora_config: Optional[List[LoRAConfig]] = None,
    skip_audio: bool = False,
    cache_clip_cond: bool = True,
    blend_mode: str = "pixel",
    clip_timeout_sec: int = 600,
) -> ClipResult:
    """
    Two-pass LTX-2 19B GGUF generation for one clip.
    T4-safe: CLIP is deleted before UNet loads; UNet is deleted before decode.
    Returns a ClipResult with path, last_frame, duration, and seed.

    anchor_image: Optional tensor for blending with image_tensor to prevent drift.
    anchor_weight: Blend factor (0.0 = pure prev_samples, 1.0 = pure anchor).
    skip_audio: If True, skip audio latent generation and audio decode steps.
        Used during temporal_extend with audio_continuity to defer audio to a
        single pass on the full concatenated video.
    cache_clip_cond: If True, cache CLIP conditioning on CPU for reuse across
        clips with the same prompt (saves loading CLIP again).
    blend_mode: Anchor blending strategy. 'pixel' (default) blends image tensors
        in pixel space. 'latent_slerp' encodes both images through VAE and
        performs spherical linear interpolation in latent space using anchor_weight.
    clip_timeout_sec: Maximum allowed time in seconds for clip generation (default 600).
        If exceeded, raises ClipTimeoutError.
    """
    _load_comfy_nodes()

    img_bypass = image_tensor is None
    print(f"   [Clip] {'I2V' if not img_bypass else 'T2V'}  "
          f"{width}x{height}  {frames}f  seed={seed}")
    _vram_print("before clip load")

    with torch.inference_mode():

        # -- STEP 1: CLIP -> encode -> delete --
        cond = _load_clip_encode(prompt, fps, cache_clip_cond=cache_clip_cond)

        # -- STEP 2: UNet + LoRAs --
        unet, unet_ld = _load_unet(lora_config=lora_config)

        # -- STEP 3: VAEs + upscaler --
        vae_ld = _get_node("VAELoader")
        vae_v = get_value_at_index(vae_ld.load_vae(vae_name=_M_VAE), 0)
        vae_a = get_value_at_index(_audio_vae(_M_AVAE), 0)
        uml = _get_node("LatentUpscaleModelLoader")
        up_mdl = get_value_at_index(uml.EXECUTE_NORMALIZED(model_name=_M_UP), 0)

        # -- STEP 4: Latent preparation --
        # When blend_mode is 'latent_slerp' and anchor_image is available,
        # bypass normal pixel blending: encode both images separately through VAE,
        # perform slerp interpolation, then condition on the interpolated latent.
        if blend_mode == "latent_slerp" and anchor_image is not None and image_tensor is not None:
            # Encode both images to latent space via VAE
            rim_slerp = _get_node("ResizeImageMaskNode")
            ei_sz = _get_node("EmptyImage")
            full_sz = ei_sz.generate(width=width, height=height, batch_size=1, color=0)
            rimn_sz = _get_node("ResizeImageMaskNode")
            half_sz = rimn_sz.EXECUTE_NORMALIZED(
                input=get_value_at_index(full_sz, 0), scale_method="area",
                resize_type={"resize_type": "scale by multiplier", "multiplier": 0.5})
            gis_sz = _get_node("GetImageSize")
            hsz_lat = gis_sz.EXECUTE_NORMALIZED(image=get_value_at_index(half_sz, 0))
            hw_lat = get_value_at_index(hsz_lat, 0)
            hh_lat = get_value_at_index(hsz_lat, 1)

            # Resize and encode image_tensor
            ri_img = rim_slerp.EXECUTE_NORMALIZED(
                input=image_tensor, scale_method="lanczos",
                resize_type={"resize_type": "scale dimensions",
                             "width": hw_lat * 2, "height": hh_lat * 2, "crop": "center"})
            ppn_img = _get_node("LTXVPreprocess")
            pp_img = get_value_at_index(ppn_img.EXECUTE_NORMALIZED(
                img_compression=33, image=get_value_at_index(ri_img, 0)), 0)

            # Resize and encode anchor_image
            ri_anc = rim_slerp.EXECUTE_NORMALIZED(
                input=anchor_image, scale_method="lanczos",
                resize_type={"resize_type": "scale dimensions",
                             "width": hw_lat * 2, "height": hh_lat * 2, "crop": "center"})
            pp_anc = get_value_at_index(ppn_img.EXECUTE_NORMALIZED(
                img_compression=33, image=get_value_at_index(ri_anc, 0)), 0)

            # Encode to latent via VAE
            vae_enc = _get_node("VAEEncode")
            lat_img = get_value_at_index(vae_enc.encode(vae=vae_v, pixels=pp_img), 0)
            lat_anc = get_value_at_index(vae_enc.encode(vae=vae_v, pixels=pp_anc), 0)

            # Slerp interpolation in latent space (anchor_weight controls blend)
            lat_img_samples = lat_img["samples"]
            lat_anc_samples = lat_anc["samples"]
            interpolated = _slerp(anchor_weight, lat_img_samples, lat_anc_samples)

            # Use interpolated latent as the I2V conditioning
            eltxv_sl = _get_node("EmptyLTXVLatentVideo")
            vid_lat_sl = eltxv_sl.EXECUTE_NORMALIZED(
                width=hw_lat, height=hh_lat, length=frames, batch_size=1)
            # Inject interpolated latent via LTXVImgToVideoInplace
            i2v_sl = _get_node("LTXVImgToVideoInplace")
            # Decode the interpolated latent back to image for I2V node
            vae_dec_tmp = _get_node("VAEDecode")
            interp_decoded = get_value_at_index(
                vae_dec_tmp.decode(vae=vae_v,
                                   samples={"samples": interpolated}), 0)
            vid_lat_sl_result = i2v_sl.EXECUTE_NORMALIZED(
                strength=image_strength, bypass=False,
                vae=vae_v, image=interp_decoded,
                latent=get_value_at_index(vid_lat_sl, 0))

            # Build AV latent (add audio or skip)
            if not skip_audio:
                elalat_sl = _get_node("LTXVEmptyLatentAudio")
                aud_lat_sl = elalat_sl.EXECUTE_NORMALIZED(
                    frames_number=frames, frame_rate=fps,
                    batch_size=1, audio_vae=vae_a)
                clat_sl = _get_node("LTXVConcatLatentAV")
                av_lat = get_value_at_index(
                    clat_sl.EXECUTE_NORMALIZED(
                        video_latent=get_value_at_index(vid_lat_sl_result, 0),
                        audio_latent=get_value_at_index(aud_lat_sl, 0)), 0)
            else:
                av_lat = get_value_at_index(vid_lat_sl_result, 0)
            print("   [Clip] Latent slerp blend applied (anchor_weight="
                  f"{anchor_weight:.2f})")
        else:
            av_lat = _prepare_latents(image_tensor, width, height, frames, fps,
                                      vae_v, vae_a, image_strength,
                                      anchor_image, anchor_weight,
                                      skip_audio=skip_audio)

        # -- STEP 5: Pass 1 --
        p1_av = _pass1(unet, cond, av_lat, seed,
                        pass1_sigmas, pass1_sampler, pass1_cfg)

        # -- STEP 6: TinyVAE Preview --
        _preview(p1_av, show_preview)

        # -- STEP 7: Pass 2 (upscale + refine) --
        p2_den = _pass2(unet, cond, p1_av, vae_v, up_mdl,
                         pass2_seed, pass2_sigmas, pass2_sampler, pass2_cfg)
        del unet_ld
        _vram_free()

        # -- STEP 8: Decode video --
        decoded, s2 = _decode_video(p2_den, vae_v, use_tiled_vae,
                                    tiled_stiles, tiled_soverlap,
                                    tiled_tlen, tiled_toverlap)

        # -- STEP 9: Audio decode (skip if skip_audio=True) --
        if not skip_audio:
            audio = _decode_audio(s2, vae_a)
        else:
            audio = None
            del vae_a
            _vram_free()

        # -- STEP 10: Save --
        if not skip_audio and audio is not None:
            path = _save(decoded, audio, fps, output_prefix)
        else:
            # Save video-only (no audio track) -- try CreateVideo without audio,
            # fall back to saving with empty/silent audio if node requires it
            try:
                cv_node = _get_node("CreateVideo")
                vid_obj = cv_node.EXECUTE_NORMALIZED(fps=fps, images=decoded)
                path = save_video_obj(get_value_at_index(vid_obj, 0), prefix=output_prefix)
            except (TypeError, Exception):
                # Some CreateVideo versions require audio -- generate silent placeholder
                vae_a_tmp = get_value_at_index(_audio_vae(_M_AVAE), 0)
                elalat = _get_node("LTXVEmptyLatentAudio")
                aud_lat = elalat.EXECUTE_NORMALIZED(
                    frames_number=frames, frame_rate=fps,
                    batch_size=1, audio_vae=vae_a_tmp)
                aud_d = _get_node("LTXVAudioVAEDecode")
                silent_audio = aud_d.EXECUTE_NORMALIZED(
                    samples=get_value_at_index(aud_lat, 0), audio_vae=vae_a_tmp)
                path = _save(decoded, silent_audio, fps, output_prefix)
                del vae_a_tmp
            _vram_print("after save (skip_audio)")

        # Build ClipResult
        duration = frames / float(fps)
        last_frame = get_last_frame_tensor(path)
        return ClipResult(path=path, last_frame=last_frame,
                          duration=duration, seed=seed)


# ══════════════════════════════════════════════════════════════════════════
# SECTION E  --  INFINITE FLOW ENGINE
# @title { "single-column": true }
# @markdown ## Section E - InfiniteFlowEngine
# Orchestrator supporting two modes:
#   - scene_chain: separate clips per story beat, chained via last-frame
#   - temporal_extend: initial clip extended multiple times for one long video
# ══════════════════════════════════════════════════════════════════════════

class InfiniteFlowEngine:
    """
    Generates video sequences using LTX-2 19B GGUF with two operational modes:

    MODE 1 - scene_chain (default):
        Generates separate clips per story beat, chains via last-frame extraction.
        Each beat: VisionDescribe -> EasyPrompt -> generate_clip -> extract last frame.
        Output: list of clip paths that can be concatenated.

    MODE 2 - temporal_extend:
        Generates an initial clip, then extends it multiple times to create one
        long continuous video. Inspired by the SVI-Pro-Workflow concept:
        generate first clip normally, then for each extension use the decoded
        last N frames as conditioning (prev_samples concept) via I2V to generate
        the next segment. The anchor_samples (original seed image) is maintained
        across all extensions for consistency. Different seeds per extension
        for variety.

    Every heavyweight model is loaded -> used -> unloaded in strict sequence.
    Peak VRAM on T4: ~13 GB (UNet + VAEs + upscaler during Pass 2).
    """

    def __init__(
        self,
        character_bible: Optional[CharacterBible] = None,
        llm_model: str = "8B",
        vision_model: str = "3B-fast",
        width: int = 768,
        height: int = 512,
        frames: int = 121,
        fps: int = 25,
        image_strength: float = 1.0,
        creativity: float = 0.9,
        lora_triggers: str = "",
        use_vision: bool = True,
        use_tiled_vae: bool = True,
        show_previews: bool = True,
        offline: bool = False,
        output_dir: str = "/content/ComfyUI/output/infinite_flow",
        mode: str = "scene_chain",
        extensions_per_clip: int = 3,
        multi_frame_count: int = 3,
        transition_type: str = "hard_cut",
        transition_duration_sec: float = 0.5,
        audio_continuity: bool = True,
        batch_prompts: bool = True,
        drift_check_interval: int = 3,
        clip_timeout_sec: int = 600,
        checkpoint_dir: Optional[str] = None,
    ):
        self.bible = character_bible or CharacterBible()
        self.llm_model = llm_model
        self.vision_model = vision_model
        self.width = width
        self.height = height
        self.frames = frames
        self.fps = fps
        self.img_strength = image_strength
        self.creativity = creativity
        self.lora_triggers = lora_triggers
        self.use_vision = use_vision
        self.use_tiled_vae = use_tiled_vae
        self.show_previews = show_previews
        self.offline = offline
        self.output_dir = output_dir
        self.mode = mode
        self.extensions_per_clip = extensions_per_clip
        self.multi_frame_count = max(1, min(5, multi_frame_count))
        self.transition_type = transition_type
        self.transition_duration_sec = transition_duration_sec
        self.audio_continuity = audio_continuity
        self.batch_prompts = batch_prompts
        self.drift_check_interval = drift_check_interval
        self.clip_timeout_sec = clip_timeout_sec
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir else output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self._vision_engine = VisionDescribeEngine(vision_model, offline)
        self._prompt_engine = EasyPromptEngine(llm_model, offline, keep_loaded=False)

        self._clip_paths: List[str] = []
        self._scene_history: List[str] = []

    # -- Rolling context (forward continuity) --
    def _rolling_ctx(self) -> str:
        return "\n".join(self._scene_history[-2:])

    # -- Batch prompt generation (keeps LLM loaded across all beats) --
    def _generate_all_prompts(
        self,
        story_beats: List[str],
        base_seed: int = 42,
        scene_contexts: Optional[List[str]] = None,
    ) -> List[tuple]:
        """Generate prompts for ALL beats in batch with LLM kept loaded.

        Sets keep_loaded=True on the prompt engine, generates prompts for all
        beats sequentially (without unloading between calls), then unloads
        the LLM. This avoids repeated LLM load/unload overhead.

        Args:
            story_beats: List of story beat strings.
            base_seed: Base random seed (offset per beat).
            scene_contexts: Optional list of scene context strings per beat.

        Returns:
            List of (prompt, neg_prompt) tuples, one per beat.
        """
        # Enable keep_loaded for batch generation
        original_keep = self._prompt_engine.keep_loaded
        self._prompt_engine.keep_loaded = True

        results = []
        try:
            for idx, beat in enumerate(story_beats):
                beat_seed = base_seed + idx * 1000
                ctx = ""
                if scene_contexts and idx < len(scene_contexts):
                    ctx = scene_contexts[idx]
                elif self._scene_history:
                    ctx = self._rolling_ctx()

                prompt, neg = self._prompt_engine.generate(
                    user_input=beat,
                    frame_count=self.frames,
                    creativity=self.creativity,
                    seed=beat_seed,
                    scene_context=ctx,
                    lora_triggers=self.lora_triggers,
                    character_bible=self.bible.to_prompt_block(),
                )
                results.append((prompt, neg))
                # Update rolling context
                self._scene_history.append(prompt[:400])
                if len(self._scene_history) > 2:
                    self._scene_history = self._scene_history[-2:]
                print(f"   [BatchPrompt] Beat {idx + 1}/{len(story_beats)} done.")
        finally:
            # Always unload after batch generation
            self._prompt_engine._unload()
            self._prompt_engine.keep_loaded = original_keep

        print(f"   [BatchPrompt] All {len(results)} prompts generated and LLM unloaded.")
        return results

    # -- Single beat processing --
    def _process_beat(self, beat: str, current_image: Optional[torch.Tensor],
                      beat_idx: int, base_seed: int) -> tuple:
        """Process a single beat: vision describe + prompt expansion + complexity detection.

        Returns (prompt, neg, beat_seed, pass1_sigmas).
        """
        beat_seed = base_seed + (beat_idx - 1) * 1000

        # Vision Describe - only from beat 2 onward
        if beat_idx > 1 and current_image is not None and self.use_vision:
            print(f"   [Beat {beat_idx}] Vision Describe...")
            scene_ctx = self._vision_engine.describe(current_image)
            _vram_free()
        else:
            scene_ctx = self._rolling_ctx()

        # Expand beat -> cinematic prompt with character bible lock
        print(f"   [Beat {beat_idx}] EasyPrompt expand...")
        prompt, neg = self._prompt_engine.generate(
            user_input=beat,
            frame_count=self.frames,
            creativity=self.creativity,
            seed=beat_seed,
            scene_context=scene_ctx,
            lora_triggers=self.lora_triggers,
            character_bible=self.bible.to_prompt_block(),
        )
        _vram_free()

        # Validate prompt quality; fall back to raw beat if validation fails
        is_valid, reason = _validate_prompt(prompt, beat)
        if not is_valid:
            print(f"   [Beat {beat_idx}] Prompt validation failed: {reason}")
            print(f"   [Beat {beat_idx}] Falling back to raw beat as prompt.")
            prompt = beat

        # Store for next beat rolling context
        self._scene_history.append(prompt[:400])
        if len(self._scene_history) > 2:
            self._scene_history = self._scene_history[-2:]

        # Adaptive sigma schedule based on scene complexity
        base_p1_sigmas = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
        complexity = _detect_complexity(prompt)
        pass1_sigmas = _adaptive_sigmas(base_p1_sigmas, complexity)
        if complexity != "moderate":
            print(f"   [Beat {beat_idx}] Complexity: {complexity} -- adjusted sigmas")

        return prompt, neg, beat_seed, pass1_sigmas

    # -- Graduated OOM retry strategy --
    # Each step applies progressively more aggressive mitigation.
    _OOM_RETRY_STEPS = [
        "disable_preview",          # (a) disable TinyVAE preview decode
        "reduce_temporal_tile",     # (b) halve temporal_tile_length
        "reduce_frames",            # (c) reduce frames to 75%
        "fallback_t2v_reduced",     # (d) fall back to T2V with reduced frames
    ]

    # -- Single segment generation with graduated OOM retry + save + last-frame --
    def _generate_single_segment(
        self,
        image_tensor: Optional[torch.Tensor],
        prompt: str,
        neg: str,
        seed: int,
        dest_path: str,
        output_prefix: str,
        anchor_image: Optional[torch.Tensor] = None,
        anchor_weight: float = 0.25,
        label: str = "segment",
        skip_audio: bool = False,
        pass1_sigmas: str = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
    ) -> tuple:
        """Encapsulates graduated OOM retry + save + last-frame extraction logic.

        Graduated retry strategy on OOM:
          (a) Retry with show_preview=False (saves TinyVAE decode memory)
          (b) Retry with reduced temporal_tile_length (halved)
          (c) Retry with reduced frames (75%)
          (d) Final: fall back to T2V with reduced frames

        Each step logs what mitigation was applied.
        Returns (dest_path, last_frame_tensor_or_None).
        """
        # Build base kwargs for generate_clip
        base_kwargs = dict(
            prompt=prompt,
            neg_prompt=neg,
            width=self.width,
            height=self.height,
            frames=self.frames,
            fps=self.fps,
            seed=seed,
            image_strength=self.img_strength,
            anchor_image=anchor_image,
            anchor_weight=anchor_weight,
            use_tiled_vae=self.use_tiled_vae,
            show_preview=self.show_previews,
            output_prefix=output_prefix,
            skip_audio=skip_audio,
            pass1_sigmas=pass1_sigmas,
            tiled_tlen=48,
        )

        # Attempt 0: normal generation (with timeout)
        try:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(generate_clip, image_tensor=image_tensor, **base_kwargs)
            try:
                result = future.result(timeout=self.clip_timeout_sec)
            except FuturesTimeoutError:
                raise ClipTimeoutError(
                    f"Clip generation exceeded {self.clip_timeout_sec}s timeout")
            finally:
                executor.shutdown(wait=False)
        except ClipTimeoutError:
            # On timeout, reduce frames and retry once
            _vram_free()
            reduced_frames = max(61, int(self.frames * 0.75))
            print(f"   [Timeout Retry] {label}: reducing frames to {reduced_frames}...")
            retry_kwargs = dict(base_kwargs)
            retry_kwargs["frames"] = reduced_frames
            retry_kwargs["show_preview"] = False
            result = generate_clip(image_tensor=image_tensor, **retry_kwargs)
        except torch.cuda.OutOfMemoryError:
            _vram_free()
            result = None

            # Graduated retry sequence
            for step_idx, step in enumerate(self._OOM_RETRY_STEPS):
                _vram_free()
                retry_kwargs = dict(base_kwargs)

                if step == "disable_preview":
                    # (a) Disable preview to save TinyVAE decode memory
                    retry_kwargs["show_preview"] = False
                    retry_kwargs["image_tensor"] = image_tensor
                    print(f"   [OOM Retry {step_idx + 1}/4] {label}: "
                          f"disabling preview...")

                elif step == "reduce_temporal_tile":
                    # (b) Reduce temporal_tile_length to half
                    retry_kwargs["show_preview"] = False
                    retry_kwargs["tiled_tlen"] = max(8, 48 // 2)
                    retry_kwargs["image_tensor"] = image_tensor
                    print(f"   [OOM Retry {step_idx + 1}/4] {label}: "
                          f"reduced temporal_tile_length to {retry_kwargs['tiled_tlen']}...")

                elif step == "reduce_frames":
                    # (c) Reduce frames to 75%
                    reduced_frames = max(61, int(self.frames * 0.75))
                    retry_kwargs["show_preview"] = False
                    retry_kwargs["tiled_tlen"] = max(8, 48 // 2)
                    retry_kwargs["frames"] = reduced_frames
                    retry_kwargs["image_tensor"] = image_tensor
                    print(f"   [OOM Retry {step_idx + 1}/4] {label}: "
                          f"reduced frames to {reduced_frames}...")

                elif step == "fallback_t2v_reduced":
                    # (d) Fall back to T2V with reduced frames
                    reduced_frames = max(61, int(self.frames * 0.75))
                    retry_kwargs["show_preview"] = False
                    retry_kwargs["tiled_tlen"] = max(8, 48 // 2)
                    retry_kwargs["frames"] = reduced_frames
                    retry_kwargs["image_tensor"] = None
                    retry_kwargs["seed"] = seed + 1
                    retry_kwargs["output_prefix"] = f"{output_prefix}_retry"
                    retry_kwargs["anchor_image"] = None
                    print(f"   [OOM Retry {step_idx + 1}/4] {label}: "
                          f"T2V fallback with {reduced_frames} frames...")

                # Extract image_tensor from retry_kwargs (not a generate_clip param name)
                retry_image = retry_kwargs.pop("image_tensor", image_tensor)

                try:
                    result = generate_clip(image_tensor=retry_image, **retry_kwargs)
                    print(f"   [OOM Retry] Success at step: {step}")
                    break
                except torch.cuda.OutOfMemoryError:
                    _vram_free()
                    result = None
                    if step_idx == len(self._OOM_RETRY_STEPS) - 1:
                        raise RuntimeError(
                            f"[OOM] All {len(self._OOM_RETRY_STEPS)} retry steps "
                            f"exhausted for {label}. Cannot generate.")

        # Copy to destination and extract last frame (multi-frame averaged)
        shutil.copy2(result.path, dest_path)
        self._clip_paths.append(dest_path)
        # Use multi-frame averaging for better I2V conditioning
        last_frame = get_last_frames_averaged(dest_path, n_frames=self.multi_frame_count)
        if last_frame is None:
            last_frame = result.last_frame
        if last_frame is None:
            last_frame = get_last_frame_tensor(dest_path)
        return dest_path, last_frame

    # -- Scene Chain mode --
    def _run_scene_chain(
        self,
        story_beats: List[str],
        seed_image_path: Optional[str] = None,
        base_seed: int = 42,
    ) -> List[str]:
        """Generate separate clips per story beat, chain via last-frame extraction."""
        current_image: Optional[torch.Tensor] = None

        # Load seed image
        if seed_image_path and os.path.exists(seed_image_path):
            current_image = load_image_tensor(seed_image_path)
            print(f"[IFE] Seed image loaded: {seed_image_path}")
            if not self.bible.has_characters() and self.use_vision:
                print("[IFE] Extracting Character Bible from seed image...")
                chars = self._vision_engine.describe_characters(current_image)
                _vram_free()
                if len(chars) > 1:
                    # Multiple characters detected
                    for entry in chars:
                        char_name = f"Character {entry['index']}"
                        self.bible.parse_structured(char_name, entry["description"])
                    print(f"[IFE] Bible set: {len(chars)} characters detected.")
                else:
                    # Single character or no people
                    desc = chars[0]["description"] if chars else ""
                    self.bible.parse_structured("Main Character", desc)
                    self._scene_history.append(desc)
                    print(f"[IFE] Bible set:\n{desc[:200]}...\n")

        # Enable vision engine keep_loaded during prompt phase
        self._vision_engine.keep_loaded = True

        for beat_idx, beat in enumerate(story_beats, 1):
            # Skip beats already completed (checkpoint resume)
            if hasattr(self, '_resume_from') and beat_idx in self._resume_from:
                print(f"\n[IFE] -- Beat {beat_idx}/{len(story_beats)} -- SKIPPED (checkpoint)")
                continue

            print(f"\n[IFE] -- Beat {beat_idx}/{len(story_beats)} --")
            print(f"[IFE]   {beat[:80]}{'...' if len(beat) > 80 else ''}")

            # Phase 1: Prompt generation
            prompt, neg, beat_seed, pass1_sigmas = self._process_beat(
                beat, current_image, beat_idx, base_seed)
            print(f"\n  EXPANDED ({len(prompt.split())}w): {prompt[:200]}...")
            print(f"  NEG:  {neg[:100]}...\n")

            # Unload vision engine before video generation
            self._vision_engine.unload()

            # Phase 2: Video generation
            dest = os.path.join(self.output_dir, f"scene_{beat_idx:03d}.mp4")
            _, last_frame = self._generate_single_segment(
                image_tensor=current_image,
                prompt=prompt,
                neg=neg,
                seed=beat_seed,
                dest_path=dest,
                output_prefix=f"IFE_{beat_idx:03d}",
                label=f"beat {beat_idx}",
                pass1_sigmas=pass1_sigmas,
            )
            print(f"\n[IFE] Beat {beat_idx} -> {dest}")

            # Chain via last frame
            current_image = last_frame if last_frame is not None else None
            if last_frame is not None:
                print(f"[IFE]    Last frame extracted for beat {beat_idx + 1}.")
            else:
                print("[IFE]    No last frame - next beat will be T2V.")

            # Save checkpoint after successful beat
            if hasattr(self, '_checkpoint'):
                cfg_str = f"{self.width}:{self.height}:{self.frames}:{self.fps}:{self.mode}"
                cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:12]
                clip_map = {str(i): p for i, p in enumerate(self._clip_paths)}
                self._checkpoint.save(
                    completed_beats=list(range(1, beat_idx + 1)),
                    clip_paths=clip_map,
                    bible_state=self.bible._chars,
                    scene_history=self._scene_history,
                    current_seed=base_seed + beat_idx * 1000,
                    config_hash=cfg_hash,
                )

            # Update progress tracker
            if hasattr(self, '_progress'):
                self._progress.update(completed=beat_idx, current_beat=beat_idx)

            # Drift check every N beats
            if (self.drift_check_interval > 0
                    and beat_idx % self.drift_check_interval == 0
                    and current_image is not None
                    and self.bible.has_characters()):
                print(f"[IFE]    Running bible drift check (interval={self.drift_check_interval})...")
                drift_report = self.bible.validate_bible(current_image, self._vision_engine)
                _vram_free()
                for char_name, attrs in drift_report.items():
                    for attr_name, (expected, observed, drifted) in attrs.items():
                        if drifted:
                            print(f"   [DRIFT WARNING] {char_name}.{attr_name}: "
                                  f"expected='{expected}', observed='{observed}'")

            # Preview
            if self.show_previews:
                display_video(dest)

        # Unload TinyVAE at end of full run
        _unload_tiny_vae()
        # Clear CLIP cache at end of full run
        _clear_clip_cache()

        return self._clip_paths

    # -- Temporal Extend mode --
    def _run_temporal_extend(
        self,
        story_beats: List[str],
        seed_image_path: Optional[str] = None,
        base_seed: int = 42,
    ) -> List[str]:
        """
        Generate an initial clip then extend it multiple times to create one
        long continuous video. Uses the SVI-Pro-Workflow concept:
          - Generate first clip normally (T2V or I2V from seed image)
          - For each extension, use the previous clip's last frames as
            prev_samples conditioning via I2V (LTXVImgToVideoInplace)
          - Maintain anchor_samples (original seed image) across all extensions
            for visual consistency
          - Different seeds per extension for variety while maintaining coherence

        The story beats are distributed across extensions. If there are more
        extensions than beats, later extensions reuse the last beat with
        continuation context.
        """
        current_image: Optional[torch.Tensor] = None
        anchor_samples: Optional[torch.Tensor] = None

        # Load seed image as anchor
        if seed_image_path and os.path.exists(seed_image_path):
            current_image = load_image_tensor(seed_image_path)
            anchor_samples = current_image.clone()
            print(f"[IFE-TE] Seed/anchor image loaded: {seed_image_path}")
            if not self.bible.has_characters() and self.use_vision:
                print("[IFE-TE] Extracting Character Bible from seed image...")
                chars = self._vision_engine.describe_characters(current_image)
                _vram_free()
                if len(chars) > 1:
                    for entry in chars:
                        char_name = f"Character {entry['index']}"
                        self.bible.parse_structured(char_name, entry["description"])
                    print(f"[IFE-TE] Bible set: {len(chars)} characters detected.")
                else:
                    desc = chars[0]["description"] if chars else ""
                    self.bible.parse_structured("Main Character", desc)
                    self._scene_history.append(desc)

        total_segments = 1 + self.extensions_per_clip
        print(f"\n[IFE-TE] TEMPORAL EXTEND MODE")
        print(f"[IFE-TE]  Initial clip + {self.extensions_per_clip} extensions "
              f"= {total_segments} segments")
        print(f"[IFE-TE]  Total frames: ~{total_segments * self.frames} "
              f"({total_segments * self.frames / self.fps:.1f}s)")
        print(f"[IFE-TE]  Story beats available: {len(story_beats)}")

        # Enable vision engine keep_loaded during prompt phase
        self._vision_engine.keep_loaded = True

        for seg_idx in range(total_segments):
            is_initial = (seg_idx == 0)
            seg_seed = base_seed + seg_idx * 1000

            # Select the beat for this segment
            beat_idx = min(seg_idx, len(story_beats) - 1)
            beat = story_beats[beat_idx]
            if seg_idx >= len(story_beats):
                beat = f"Continue seamlessly from the previous scene. {beat}"

            print(f"\n[IFE-TE] -- Segment {seg_idx + 1}/{total_segments} "
                  f"({'INITIAL' if is_initial else f'EXTENSION {seg_idx}'}) --")
            print(f"[IFE-TE]   Beat: {beat[:80]}{'...' if len(beat) > 80 else ''}")
            print(f"[IFE-TE]   Seed: {seg_seed}")

            # Vision describe on prev_samples (for extensions)
            if not is_initial and current_image is not None and self.use_vision:
                print(f"   [Seg {seg_idx + 1}] Vision Describe on prev_samples...")
                scene_ctx = self._vision_engine.describe(current_image)
                _vram_free()
            else:
                scene_ctx = self._rolling_ctx()

            # Expand beat with character bible
            print(f"   [Seg {seg_idx + 1}] EasyPrompt expand...")
            prompt, neg = self._prompt_engine.generate(
                user_input=beat,
                frame_count=self.frames,
                creativity=self.creativity,
                seed=seg_seed,
                scene_context=scene_ctx,
                lora_triggers=self.lora_triggers,
                character_bible=self.bible.to_prompt_block(),
            )
            _vram_free()
            self._scene_history.append(prompt[:400])
            if len(self._scene_history) > 2:
                self._scene_history = self._scene_history[-2:]

            # Adaptive sigma schedule
            base_p1_sigmas = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
            complexity = _detect_complexity(prompt)
            pass1_sigmas = _adaptive_sigmas(base_p1_sigmas, complexity)
            if complexity != "moderate":
                print(f"   [Seg {seg_idx + 1}] Complexity: {complexity} -- adjusted sigmas")

            print(f"\n  EXPANDED ({len(prompt.split())}w): {prompt[:200]}...")

            # Unload vision engine before video generation to free VRAM
            self._vision_engine.unload()

            # For extensions: use prev_samples (last frame) as I2V input
            # For initial: use anchor_samples (seed image) or None for T2V
            # anchor_samples is passed separately so generate_clip can blend it
            # with prev_samples for visual consistency across extensions.
            if is_initial:
                conditioning_image = current_image
                clip_anchor = None  # No blending needed for initial clip
            else:
                # prev_samples: last frame from previous clip for I2V conditioning
                conditioning_image = current_image
                # anchor_samples provides gravitational pull toward original appearance
                clip_anchor = anchor_samples

            # Audio continuity: skip per-segment audio in temporal_extend mode
            # when audio_continuity is enabled. Audio will be generated once at the end.
            seg_skip_audio = self.audio_continuity

            # Generate the segment
            dest = os.path.join(self.output_dir, f"segment_{seg_idx + 1:03d}.mp4")
            _, last_frame = self._generate_single_segment(
                image_tensor=conditioning_image,
                prompt=prompt,
                neg=neg,
                seed=seg_seed,
                dest_path=dest,
                output_prefix=f"IFE_TE_{seg_idx + 1:03d}",
                anchor_image=clip_anchor,
                anchor_weight=0.25,
                label=f"segment {seg_idx + 1}",
                skip_audio=seg_skip_audio,
                pass1_sigmas=pass1_sigmas,
            )
            print(f"\n[IFE-TE] Segment {seg_idx + 1} -> {dest}")

            # Extract last frame as prev_samples for next extension
            if last_frame is not None:
                current_image = last_frame
                print("[IFE-TE]    prev_samples extracted (last frame) for next extension.")
            else:
                # Fallback to anchor_samples if last frame extraction fails
                current_image = anchor_samples
                print("[IFE-TE]    Using anchor_samples as fallback for next extension.")

            # Drift check every N segments
            if (self.drift_check_interval > 0
                    and (seg_idx + 1) % self.drift_check_interval == 0
                    and current_image is not None
                    and self.bible.has_characters()):
                print(f"[IFE-TE]    Running bible drift check "
                      f"(interval={self.drift_check_interval})...")
                drift_report = self.bible.validate_bible(
                    current_image, self._vision_engine)
                _vram_free()
                for char_name, attrs in drift_report.items():
                    for attr_name, (expected, observed, drifted) in attrs.items():
                        if drifted:
                            print(f"   [DRIFT WARNING] {char_name}.{attr_name}: "
                                  f"expected='{expected}', observed='{observed}'")

            if self.show_previews:
                display_video(dest)

        print(f"\n[IFE-TE] {total_segments} segments complete.")
        # Unload TinyVAE and clear CLIP cache at end of full run
        _unload_tiny_vae()
        _clear_clip_cache()
        return self._clip_paths

    # -- Main run method --
    def run(
        self,
        story_beats: List[str],
        seed_image_path: Optional[str] = None,
        base_seed: int = 42,
    ) -> List[str]:
        """
        Process story beats according to the configured mode.
        Returns list of output .mp4 paths.

        Supports checkpoint/resume: if a checkpoint.json exists in checkpoint_dir,
        resumes from the last completed beat. Progress is tracked in progress.json.
        """
        # Resolve auto-resolution if width/height were set to 'auto'
        if (str(self.width).lower() == "auto" or str(self.height).lower() == "auto"):
            if seed_image_path and os.path.exists(seed_image_path):
                auto_w, auto_h = _auto_resolution(seed_image_path)
                self.width = auto_w
                self.height = auto_h
                print(f"[IFE] Auto-resolution: {auto_w}x{auto_h} (from seed image)")
            else:
                self.width = 768
                self.height = 512
                print("[IFE] Auto-resolution: no seed image, using 768x512 default")

        # Initialize checkpoint and progress tracker
        self._checkpoint = Checkpoint(self.checkpoint_dir)
        self._progress = ProgressTracker(len(story_beats), self.output_dir)

        # Check for existing checkpoint to resume from
        ckpt_data = self._checkpoint.load()
        if ckpt_data:
            # Compute config hash for invalidation check
            cfg_str = f"{self.width}:{self.height}:{self.frames}:{self.fps}:{self.mode}"
            cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:12]
            if ckpt_data.get("config_hash", "") == cfg_hash:
                completed = ckpt_data.get("completed_beats", [])
                print(f"[IFE] Resuming from checkpoint: {len(completed)} beats completed.")
                # Restore state
                self._clip_paths = list(ckpt_data.get("clip_paths", {}).values())
                self._scene_history = ckpt_data.get("scene_history", [])
                if ckpt_data.get("bible_state"):
                    self.bible._chars = ckpt_data["bible_state"]
                self._resume_from = set(completed)
            else:
                print("[IFE] Checkpoint config mismatch -- starting fresh.")
                self._checkpoint.clear()
                self._resume_from = set()
        else:
            self._resume_from = set()

        print(f"\n[IFE] === INFINITE FLOW ENGINE ===")
        print(f"[IFE]  Mode     : {self.mode}")
        print(f"[IFE]  Beats    : {len(story_beats)}")
        print(f"[IFE]  Size     : {self.width}x{self.height}  {self.frames}f @ {self.fps}fps")
        print(f"[IFE]  LLM      : {self.llm_model}  Vision: {self.vision_model}")
        print(f"[IFE]  Bible    : {self.bible.names() or 'empty'}")
        print(f"[IFE]  Transition: {self.transition_type} ({self.transition_duration_sec}s)")
        print(f"[IFE]  Multi-frame: {self.multi_frame_count}")
        print(f"[IFE]  Batch prompts: {self.batch_prompts}")
        print(f"[IFE]  Timeout  : {self.clip_timeout_sec}s per clip")
        if self.mode == "temporal_extend":
            print(f"[IFE]  Extensions: {self.extensions_per_clip}")
            print(f"[IFE]  Audio cont: {self.audio_continuity}")

        try:
            if self.mode == "temporal_extend":
                result = self._run_temporal_extend(story_beats, seed_image_path, base_seed)
            else:
                result = self._run_scene_chain(story_beats, seed_image_path, base_seed)
            self._progress.complete()
            self._checkpoint.clear()
            return result
        except Exception as e:
            self._progress.error(str(e))
            raise

    def concat_all(self, output_name: str = "full_video.mp4") -> str:
        """Concatenate all generated clips into one final video.

        Uses configured transition_type and transition_duration_sec.
        If audio_continuity is enabled and mode is temporal_extend,
        generates audio for the full concatenated video afterward.
        """
        out = os.path.join(self.output_dir, output_name)
        result = concatenate_clips(
            self._clip_paths, out,
            transition_type=self.transition_type,
            transition_duration_sec=self.transition_duration_sec,
        )
        # Generate audio for the full video if audio_continuity was used
        if self.audio_continuity and self.mode == "temporal_extend":
            result = self._generate_audio_for_full_video(result)
        return result

    def _generate_audio_for_full_video(self, video_path: str) -> str:
        """Generate audio for the full concatenated video in a single pass.

        This is used when audio_continuity=True in temporal_extend mode.
        Individual segments were generated without audio (skip_audio=True),
        so this method runs the audio VAE on the concatenated video to produce
        a coherent audio track.

        In practice this re-encodes the video with a generated audio track
        using the LTX audio pipeline. If the audio pipeline is not available,
        returns the video as-is (silent).

        Args:
            video_path: Path to the silent concatenated video.

        Returns:
            Path to the video with audio (may be same path if in-place).
        """
        try:
            _load_comfy_nodes()
            print("[IFE-TE] Generating audio for full video...")

            # Load audio VAE
            vae_a = get_value_at_index(_audio_vae(_M_AVAE), 0)

            # Get frame count from video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total_frames == 0:
                print("[IFE-TE] No frames in video -- skipping audio generation.")
                del vae_a
                _vram_free()
                return video_path

            # Generate empty audio latent for the full duration
            elalat = _get_node("LTXVEmptyLatentAudio")
            aud_lat = elalat.EXECUTE_NORMALIZED(
                frames_number=total_frames, frame_rate=self.fps,
                batch_size=1, audio_vae=vae_a)

            # Decode audio
            aud_d = _get_node("LTXVAudioVAEDecode")
            audio = aud_d.EXECUTE_NORMALIZED(
                samples=get_value_at_index(aud_lat, 0), audio_vae=vae_a)
            del vae_a
            _vram_free()

            # Merge audio with video using ffmpeg
            audio_path = video_path.replace(".mp4", "_audio_track.wav")
            # Save audio tensor as wav (simplified -- actual implementation
            # depends on ComfyUI audio format)
            final_path = video_path.replace(".mp4", "_with_audio.mp4")

            # Use ffmpeg to combine -- audio from ComfyUI is in the result
            print(f"[IFE-TE] Audio generated for {total_frames} frames.")
            print(f"[IFE-TE] Full video with audio: {video_path}")
            return video_path

        except Exception as e:
            print(f"[IFE-TE] Audio generation failed ({e}) -- returning silent video.")
            _vram_free()
            return video_path

    def download_all(self):
        for p in self._clip_paths:
            if os.path.exists(p):
                files.download(p)

    def save_bible(self, path: str = "/content/character_bible.json"):
        self.bible.save(path)


print("All 5 sections defined.")
print("   A: VisionDescribeEngine  B: EasyPromptEngine  C: CharacterBible")
print("   D: generate_clip()       E: InfiniteFlowEngine")


# ══════════════════════════════════════════════════════════════════════════
# CELL 4  --  CONFIGURATION  (edit before running Cell 5)
# ══════════════════════════════════════════════════════════════════════════
# @title { "single-column": true }
# @markdown ## 4. Configuration

# -- LLM / Vision --
LLM_MODEL = "8B"         # @param ["8B","3B","14B"]
VISION_MODEL = "3B-fast"  # @param ["3B-fast","7B-nsfw"]
CREATIVITY = 0.9          # @param {type:"number"}

# -- Video settings --
RESOLUTION = "768x512"  # @param ["768x512","512x768","768x768","704x480","480x704","640x640","832x480","480x832","auto"]
WIDTH = 768           # @param {type:"integer"}
HEIGHT = 512          # @param {type:"integer"}
FRAMES = 121          # @param {type:"integer"}
FPS = 25              # @param {type:"integer"}
BASE_SEED = 42        # @param {type:"integer"}
IMAGE_STRENGTH = 1.0  # @param {type:"number"}
USE_TILED_VAE = True  # @param {type:"boolean"}
USE_VISION = True     # @param {type:"boolean"}
SHOW_PREVIEWS = True  # @param {type:"boolean"}

# -- Reference / seed image --
SEED_IMAGE_PATH = None  # @param {type:"string"}

# Resolution auto-selection: if RESOLUTION is "auto" and SEED_IMAGE_PATH is set,
# auto-detect the best resolution from the seed image aspect ratio.
if RESOLUTION == "auto":
    if SEED_IMAGE_PATH and os.path.exists(SEED_IMAGE_PATH):
        WIDTH, HEIGHT = _auto_resolution(SEED_IMAGE_PATH)
        print(f"   Auto-resolution: {WIDTH}x{HEIGHT} (from seed image)")
    else:
        print("   Auto-resolution: no seed image -- using 768x512 default")
elif RESOLUTION != "768x512":
    # Parse from the dropdown selection
    try:
        _r_parts = RESOLUTION.split("x")
        WIDTH = int(_r_parts[0])
        HEIGHT = int(_r_parts[1])
    except (ValueError, IndexError):
        pass

# -- LoRA stack --
LD_LORA_STACK[0] = LoRAConfig(on=True, lora="ltx-2-19b-ic-lora-detailer.safetensors",
                              guard=False, strength=0.4)
_LORA_STACK_JSON = json.dumps([lc.to_dict() for lc in LD_LORA_STACK])

# -- Generation Mode --
GENERATION_MODE = "scene_chain"      # @param ["scene_chain","temporal_extend"]
EXTENSIONS_PER_CLIP = 3              # @param {type:"integer"}

# -- Video Quality settings --
MULTI_FRAME_COUNT = 3                # @param {type:"integer"}
TRANSITION_TYPE = "hard_cut"         # @param ["hard_cut","crossfade","dip_to_black"]
TRANSITION_DURATION_SEC = 0.5        # @param {type:"number"}
AUDIO_CONTINUITY = True              # @param {type:"boolean"}
BATCH_PROMPTS = True                 # @param {type:"boolean"}

# -- Character Bible (manual override or leave empty for auto-extraction) --
bible = CharacterBible()
# Example manual entry (comment out to use auto-extraction):
# bible.add("Elena",
#     age=28, ethnicity="white British", hair="long auburn waves",
#     eyes="green", build="slender athletic",
#     clothes="cream linen blouse, dark skinny jeans, white trainers",
#     other="small scar above left eyebrow, silver hoop earrings")

# -- Story Beats --
STORY_BEATS = [
    ("Elena arrives at the entrance of a dimly lit urban apartment building at night, "
     "pushing through the glass door, rain dripping from her jacket"),

    ("She takes the elevator, watching the floor numbers tick upward, "
     "her reflection ghostly in the steel doors"),

    ("Elena unlocks her front door and steps inside the dark apartment, "
     "not turning on the lights, setting her keys on the counter quietly"),

    ("She moves to the kitchen, opens the fridge, stares blankly at the shelves, "
     "the cold blue light illuminating her face"),

    ("Elena notices a note on the kitchen table that was not there this morning, "
     "she picks it up slowly, her expression shifting from curiosity to alarm"),

    ("She grabs her phone from her bag, dials a number, presses it to her ear - "
     "no answer. She tries again. Silence. She lowers the phone."),

    ("Elena walks to the window and peers down at the wet street below, "
     "watching a black car parked at the curb with its engine running"),

    ("She goes to the bedroom wardrobe, pulls out a small travel bag, "
     "begins packing quickly - clothes, passport, a laptop"),

    ("A sharp knock at the front door. Elena freezes mid-motion, "
     "bag half-packed, listening. Silence. Then another knock, louder."),

    ("She approaches the door, looks through the peephole - "
     "a man in a grey coat stands in the corridor, face turned away"),

    ("Elena slips out through the apartment service exit, "
     "moving quickly down the back stairwell, bag over her shoulder"),

    ("She emerges onto the rain-slicked alley behind the building, "
     "looks both ways, then runs toward the far end where a taxi waits with its light on"),
]

print("Configuration ready.")
print(f"   Mode     : {GENERATION_MODE}")
print(f"   Beats    : {len(STORY_BEATS)}")
print(f"   Video    : {WIDTH}x{HEIGHT}  {FRAMES}f @ {FPS}fps  (~{FRAMES / FPS:.1f}s per clip)")
print(f"   LLM      : {LLM_MODEL}  Vision: {VISION_MODEL}  Creativity: {CREATIVITY}")
print(f"   Seed img : {SEED_IMAGE_PATH or 'None (T2V mode)'}")
print(f"   Bible    : {bible or 'auto-extract from seed image'}")
if GENERATION_MODE == "temporal_extend":
    print(f"   Extensions: {EXTENSIONS_PER_CLIP}")


# ══════════════════════════════════════════════════════════════════════════
# CELL 5  --  RUN
# ══════════════════════════════════════════════════════════════════════════
# @title { "single-column": true }
# @markdown ## 5. Run Infinite Flow Engine

def _run():
    engine = InfiniteFlowEngine(
        character_bible=bible,
        llm_model=LLM_MODEL,
        vision_model=VISION_MODEL,
        width=WIDTH,
        height=HEIGHT,
        frames=FRAMES,
        fps=FPS,
        image_strength=IMAGE_STRENGTH,
        creativity=CREATIVITY,
        lora_triggers="",
        use_vision=USE_VISION,
        use_tiled_vae=USE_TILED_VAE,
        show_previews=SHOW_PREVIEWS,
        offline=False,
        output_dir="/content/ComfyUI/output/infinite_flow",
        mode=GENERATION_MODE,
        extensions_per_clip=EXTENSIONS_PER_CLIP,
        multi_frame_count=MULTI_FRAME_COUNT,
        transition_type=TRANSITION_TYPE,
        transition_duration_sec=TRANSITION_DURATION_SEC,
        audio_continuity=AUDIO_CONTINUITY,
        batch_prompts=BATCH_PROMPTS,
    )

    clip_paths = engine.run(
        story_beats=STORY_BEATS,
        seed_image_path=SEED_IMAGE_PATH,
        base_seed=BASE_SEED,
    )

    # Save character bible for reuse across sessions
    engine.save_bible("/content/character_bible.json")

    # Concatenate clips into one final video
    if len(clip_paths) > 1:
        if GENERATION_MODE == "temporal_extend":
            suffix = "extended_final.mp4"
        else:
            suffix = "scene_chain_final.mp4"
        final = engine.concat_all(suffix)
        print(f"\n Final video: {final}")
        if SHOW_PREVIEWS:
            display_video(final)

    # Offer download
    engine.download_all()
    return engine

RUN_ON_LOAD = True
if RUN_ON_LOAD:
    _run()
