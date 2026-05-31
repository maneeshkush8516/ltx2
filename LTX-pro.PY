# ======================================================================
# LTX-2 PRO  --  Production Video Generation for Google Colab T4 (15 GB)
# Model: LTX-2 19B Q4_K_M GGUF | Strategy: load -> use -> unload
# Peak VRAM: ~13 GB | Cells: 1-Setup 2-Models 3-Helpers 4-Config
#   5-Vision+LLM 6-generate_clip 7-Engine 8-UserConfig 9-Run
# ======================================================================


# ======================================================================
# CELL 1  --  ENVIRONMENT SETUP  (run once per Colab session)
# ======================================================================
# @title 1. Install Environment

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

# Pinned ComfyUI build (skip if already cloned)
if not os.path.isdir("/content/ComfyUI"):
    try:
        _shell("git clone --branch ComfyUI_22_01_2026_v0.10.0 https://github.com/Isi-dev/ComfyUI.git")
    except subprocess.CalledProcessError:
        try:
            _shell("git clone https://github.com/Isi-dev/ComfyUI.git")
        except subprocess.CalledProcessError:
            _shell("git clone https://github.com/comfyanonymous/ComfyUI.git")

_shell("pip install -r /content/ComfyUI/requirements.txt -q")
clear_output()

os.chdir("/content/ComfyUI/custom_nodes")

# Pinned custom nodes with idempotent checks
if not os.path.isdir("/content/ComfyUI/custom_nodes/ComfyUI_KJNodes"):
    try:
        _shell("git clone --branch kj_1.2.6 https://github.com/Isi-dev/ComfyUI_KJNodes")
    except subprocess.CalledProcessError:
        _shell("git clone https://github.com/Isi-dev/ComfyUI_KJNodes")

if not os.path.isdir("/content/ComfyUI/custom_nodes/ComfyUI_GGUF"):
    try:
        _shell("git clone --branch ComfyUI_GGUF_22_01_2026 https://github.com/Isi-dev/ComfyUI_GGUF.git")
    except subprocess.CalledProcessError:
        _shell("git clone https://github.com/Isi-dev/ComfyUI_GGUF.git")

if not os.path.isdir("/content/ComfyUI/custom_nodes/ComfyUI-LTXVideo"):
    _shell("git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git")

if not os.path.isdir("/content/ComfyUI/custom_nodes/LTX2EasyPrompt-LD"):
    _shell("git clone https://github.com/seanhan19911990-source/LTX2EasyPrompt-LD.git")

if not os.path.isdir("/content/ComfyUI/custom_nodes/LTX2-Master-Loader"):
    _shell("git clone https://github.com/seanhan19911990-source/LTX2-Master-Loader.git")

if not os.path.isdir("/content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite"):
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
print("Cell 1 complete: Environment ready.")


# ======================================================================
# CELL 2  --  MODEL DOWNLOADS  (run once; skips cached files)
# ======================================================================
# @title 2. Download Model Weights

import os, subprocess
from pathlib import Path


def _dl(url: str, dest: str, filename: str = None) -> str:
    """Download a file with aria2c, skip if cached and >1MB."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    fn = filename or url.split("/")[-1].split("?")[0]
    full = os.path.join(dest, fn)
    if os.path.exists(full) and os.path.getsize(full) > 1_000_000:
        print(f"  cached  {fn}")
        return fn
    cmd = ["aria2c", "--console-log-level=error", "--summary-interval=0", "--quiet",
           "-c", "-x", "16", "-s", "16", "-k", "1M", "-d", dest, "-o", fn, url]
    print(f"  dl {fn}...", end=" ", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED: {r.stderr.strip()}")
        return None
    print("done.")
    return fn


KIJAI  = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main"
CORG   = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
LTX_HF = "https://huggingface.co/Lightricks"
UNET_D = "/content/ComfyUI/models/unet"
TE_D   = "/content/ComfyUI/models/text_encoders"
VAE_D  = "/content/ComfyUI/models/vae"
UPD    = "/content/ComfyUI/models/latent_upscale_models"
LR_D   = "/content/ComfyUI/models/loras"

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


# -- Optional LLM/Vision model downloaders --
LLM_MODELS_DIR = "/content/ComfyUI/models/llm"


def download_llm_model(model_size: str = "3B", target_dir: str = LLM_MODELS_DIR) -> str:
    """Download an LLM model for EasyPromptEngine using huggingface_hub."""
    from huggingface_hub import snapshot_download

    model_ids = {
        "3B": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "8B": "mlabonne/NeuralDaredevil-8B-abliterated",
        "14B": "huihui-ai/Huihui-Qwen3-14B-abliterated-v2",
    }
    if model_size not in model_ids:
        print(f"   Unknown LLM size: {model_size}. Available: {list(model_ids.keys())}")
        return ""

    hf_id = model_ids[model_size]
    local_dir = os.path.join(target_dir, model_size)

    if os.path.isdir(local_dir) and any(
        f.endswith((".safetensors", ".bin", ".gguf"))
        for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))
    ):
        print(f"   LLM {model_size} already cached at {local_dir}")
        return local_dir

    print(f"   Downloading LLM {model_size} ({hf_id})...")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(hf_id, local_dir=local_dir,
                                 local_dir_use_symlinks=False,
                                 resume_download=True, max_workers=8)
        print(f"   LLM {model_size} downloaded to: {path}")
        return path
    except Exception as e:
        print(f"   LLM download failed: {e}")
        return ""


def download_vision_model(model_key: str = "3B-fast", target_dir: str = LLM_MODELS_DIR) -> str:
    """Download a Vision model for VisionDescribeEngine using huggingface_hub."""
    from huggingface_hub import snapshot_download

    model_ids = {
        "3B-fast": "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
        "7B-nsfw": "prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it",
    }
    if model_key not in model_ids:
        print(f"   Unknown Vision model: {model_key}. Available: {list(model_ids.keys())}")
        return ""

    hf_id = model_ids[model_key]
    local_dir = os.path.join(target_dir, f"vision_{model_key}")

    if os.path.isdir(local_dir) and any(
        f.endswith((".safetensors", ".bin"))
        for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))
    ):
        print(f"   Vision {model_key} already cached at {local_dir}")
        return local_dir

    print(f"   Downloading Vision model {model_key} ({hf_id})...")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(hf_id, local_dir=local_dir,
                                 local_dir_use_symlinks=False,
                                 resume_download=True, max_workers=8)
        print(f"   Vision {model_key} downloaded to: {path}")
        return path
    except Exception as e:
        print(f"   Vision download failed: {e}")
        return ""


print("\nCell 2 complete: All models ready.")


# ======================================================================
# CELL 3  --  CORE IMPORTS & HELPERS
# ======================================================================
# @title 3. Imports, Helpers, Node Loader

import gc, re, json, time, shutil, warnings, asyncio
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, List, Any, Union, Sequence, Mapping
from base64 import b64encode
from IPython.display import display, HTML, clear_output

try:
    from google.colab import files as colab_files
except ImportError:
    colab_files = None

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


def _vram_print(tag=""):
    if not torch.cuda.is_available():
        return
    u = torch.cuda.memory_allocated() / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    filled = int(20 * u / t) if t else 0
    bar = "#" * filled + "." * (20 - filled)
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
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame).float().unsqueeze(0) / 255.0


# -- Video helpers --
def display_video(path: str):
    if not path or not os.path.exists(path):
        return
    data = b64encode(open(path, "rb").read()).decode()
    display(HTML(
        '<video width=720 controls autoplay loop muted>'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4"></video>'))


def save_video_obj(video_obj, prefix="PRO") -> str:
    from comfy_api.latest import Types
    w, h = video_obj.get_dimensions()
    folder, fname, ctr, _, _ = folder_paths.get_save_image_path(
        f"video/{prefix}", folder_paths.get_output_directory(), w, h)
    ext = Types.VideoContainer.get_extension("auto")
    path = os.path.join(folder, f"{fname}_{ctr:05}_.{ext}")
    video_obj.save_to(path, format=Types.VideoContainer("auto"),
                      codec="auto", metadata=None)
    return path


def concatenate_clips(clip_paths: List[str], output_path: str) -> str:
    """ffmpeg concat all clips into one final video."""
    list_file = "/tmp/concat_list.txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                    "-c", "copy", output_path], check=True, capture_output=True)
    print(f"   Concatenated -> {output_path}")
    return output_path


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
            print(f"   Warning: Node failures: {failed}")

    try:
        asyncio.run(_l())
    except RuntimeError:
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(_l())
    _NODES_LOADED = True


def _audio_vae(name: str):
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS["VAELoaderKJ"]().load_vae(
            vae_name=name, device="main_device", weight_dtype="fp16")
    return NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=name)


# -- LoRA stack (applied in generate_clip) --
LD_LORA_STACK = [
    {"on": True, "lora": "ltx-2-19b-ic-lora-detailer.safetensors", "guard": False, "strength": 0.4},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
]
_LORA_STACK_JSON = json.dumps(LD_LORA_STACK)


def _apply_loras(unet, clip_model):
    """Apply LD_LORA_STACK to UNet. Tries LTX2MasterLoaderLD first, fallback to loop."""
    active = [s for s in LD_LORA_STACK
              if s.get("on") and s.get("lora") not in (None, "None", "")]
    if not active:
        return unet, clip_model

    if "LTX2MasterLoaderLD" in NODE_CLASS_MAPPINGS:
        try:
            node = NODE_CLASS_MAPPINGS["LTX2MasterLoaderLD"]()
            result = getattr(node, node.FUNCTION)(
                model=unet, clip=clip_model, stack_data=_LORA_STACK_JSON)
            print(f"   [LoRA] {len(active)} slot(s) via LTX2MasterLoaderLD")
            return get_value_at_index(result, 0), clip_model
        except Exception as e:
            print(f"   [LoRA] MasterLoader failed ({e}) -- manual fallback")

    for slot in active:
        name, strength = slot["lora"], slot["strength"]
        try:
            unet = LoraLoaderModelOnly().load_lora_model_only(unet, name, strength)[0]
            print(f"   [LoRA] {name} @ {strength}")
        except Exception as e:
            print(f"   [LoRA] skip {name}: {e}")
    return unet, clip_model


print("Cell 3 complete: Core utilities ready.")


# ======================================================================
# CELL 4  --  CONFIGURATION
# ======================================================================
# @title 4. Configuration

import torch

# -- Video settings --
WIDTH = 768
HEIGHT = 512
FRAMES = 121
FPS = 25
SEED = 42
POSITIVE_PROMPT = (
    "Cinematic establishing shot of a woman walking through a misty forest "
    "at golden hour, soft volumetric lighting filtering through the trees, "
    "smooth steadicam tracking movement, film grain, shallow depth of field."
)
NEGATIVE_PROMPT = (
    "blurry, out of focus, low quality, worst quality, jpeg artifacts, "
    "static, no motion, frozen, duplicate, watermark, text, signature, "
    "poorly drawn, bad anatomy, deformed, disfigured, extra limbs, "
    "missing limbs, overexposed, underexposed, grainy, noise, flickering"
)

# -- Model filenames (must match Cell 2 downloads) --
# These are module-level so generate_clip() can reference them
# _M_UNET, _M_CLIP1, _M_CLIP2, _M_VAE, _M_AVAE, _M_TAEV, _M_UP
# are already set in Cell 2

# -- Pass 1 settings --
PASS1_SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
PASS1_SAMPLER = "euler"
PASS1_CFG = 1.0

# -- Pass 2 settings --
PASS2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"
PASS2_SAMPLER = "gradient_estimation"
PASS2_CFG = 1.0
PASS2_SEED = 0

# -- Tiled VAE settings --
USE_TILED_VAE = True
TILED_SPATIAL_TILES = 2
TILED_SPATIAL_OVERLAP = 8
TILED_TEMPORAL_LENGTH = 48
TILED_TEMPORAL_OVERLAP = 4

# -- LLM/Vision (disabled by default for T4 safety) --
USE_LLM_PROMPT = False
LLM_MODEL = "3B"
USE_VISION = False
OFFLINE_LLM_PATH = ""
OFFLINE_VISION_PATH = ""

# -- Image-to-Video --
SEED_IMAGE_PATH = None
IMAGE_STRENGTH = 1.0

# -- Generation --
CREATIVITY = 0.9
LORA_TRIGGERS = ""
SHOW_PREVIEWS = True
OUTPUT_DIR = "/content/ComfyUI/output/ltx_pro"

# -- GPU tier auto-detection --
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_mem / 1024**3
    if vram_gb < 14:
        print("WARNING: Less than 14GB VRAM detected. Some features may OOM.")
    elif vram_gb < 20:
        print("T4/L4 detected. LLM/Vision features should stay disabled.")
    else:
        print(f"GPU: {props.name} ({vram_gb:.0f}GB) - all features available.")
else:
    print("No CUDA GPU detected. This script requires a GPU runtime.")

print("Cell 4 complete: Configuration set.")
print(f"   Video: {WIDTH}x{HEIGHT} {FRAMES}f @ {FPS}fps (~{FRAMES/FPS:.1f}s)")
print(f"   LLM: {'ON ('+LLM_MODEL+')' if USE_LLM_PROMPT else 'OFF'}")
print(f"   Vision: {'ON' if USE_VISION else 'OFF'}")
print(f"   I2V: {'ON' if SEED_IMAGE_PATH else 'OFF (T2V mode)'}")


# ======================================================================
# CELL 5  --  OPTIONAL: VisionDescribe + EasyPrompt
# ======================================================================
# @title 5. Vision + LLM Engines (optional, loaded on demand)


class VisionDescribeEngine:
    """Analyses an image and returns a 100-130 word scene description.
    Loads model -> describes -> fully unloads to free VRAM."""

    MODEL_OPTIONS = {
        "3B-fast": "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
        "7B-nsfw": "prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it",
    }

    PROMPT = (
        "Describe this image in one paragraph, 100-130 words. "
        "Start with style (photorealistic/anime/3D). For people: state ethnicity, "
        "age, hair, body type, clothing, pose, camera framing, lighting, setting. "
        "One flowing paragraph, no bullets. If no person, describe environment."
    )

    def __init__(self, model_key: str = "3B-fast", offline: bool = False):
        self.model_key = model_key
        self.offline = offline

    def describe(self, image: Union[Image.Image, torch.Tensor]) -> str:
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from huggingface_hub import snapshot_download
            from qwen_vl_utils import process_vision_info
        except ImportError as ie:
            print(f"   [VisionDescribe] Missing dependency: {ie}")
            return ""

        if isinstance(image, torch.Tensor):
            image = tensor_to_pil(image)

        hf_id = self.MODEL_OPTIONS[self.model_key]
        if OFFLINE_VISION_PATH:
            source = OFFLINE_VISION_PATH
            self.offline = True
        elif not self.offline:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                source = snapshot_download(hf_id)
            except Exception:
                source = hf_id
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            source = hf_id

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"   [VisionDescribe] Loading {self.model_key}...")
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

        # Full cleanup
        del out, inputs, model, processor
        _vram_free()
        print(f"   [VisionDescribe] done ({len(desc.split())} words).")
        return desc


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
    """Expands a simple story beat into a dense cinematic LTX-2 prompt.
    Loads LLM -> generates -> cleans output -> unloads to free VRAM."""

    MODELS = {
        "8B": "mlabonne/NeuralDaredevil-8B-abliterated",
        "3B": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "14B": "huihui-ai/Huihui-Qwen3-14B-abliterated-v2",
    }

    SYSTEM_PROMPT = (
        "You are a cinematic prompt writer for LTX-2 video generation. "
        "Expand the user's idea into 8-12 sentences of dense flowing prose. "
        "Include: video style, camera angle, character description (specific age), "
        "scene/environment, action in present tense, camera movement as prose, "
        "max 2 ambient sounds. Output ONLY the expanded prompt. No preamble."
    )

    _CLEAN_RE = [
        (re.compile(r"<think>.*?</think>", re.DOTALL), ""),
        (re.compile(r"^(Sure!?|Certainly!?|Here(?:'s| is).*?:)[^\n]*\n?", re.IGNORECASE), ""),
        (re.compile(r"\s*(assistant|user|system|<\|[^|>]*\|>)\s*$", re.IGNORECASE), ""),
        (re.compile(r"\s*\n+Note:.*$", re.DOTALL), ""),
        (re.compile(r"\s*\(Note:.*$", re.DOTALL | re.IGNORECASE), ""),
        (re.compile(r"\s*\n+(Please let me know|Let me revise|Done\.|I hope|Thank you).*$",
                    re.DOTALL | re.IGNORECASE), ""),
        (re.compile(r"\n{3,}"), "\n\n"),
    ]

    def __init__(self, model_size: str = "3B", offline: bool = False,
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

        if OFFLINE_LLM_PATH:
            source = OFFLINE_LLM_PATH
            self.offline = True
        elif not self.offline:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                source = snapshot_download(hf_id)
            except Exception:
                source = hf_id
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            source = hf_id

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"   [EasyPrompt] Loading {key}...")
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
                  "<|im_end|>", "<end_of_turn>", "[/INST]"]
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
        """Returns (positive_prompt, negative_prompt)."""
        if self.model_size == "disabled":
            return user_input, _build_neg(user_input, user_input)

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

        ordinal = {2: "2nd", 3: "3rd"}.get(action_count, f"{action_count}th")
        pacing = (
            f"This clip is {real_seconds:.0f}s. Write EXACTLY {action_count} "
            f"distinct actions. HARD STOP after the {ordinal} action. "
            f"Write ~{token_budget} tokens."
        )

        bible_clause = ""
        if character_bible.strip():
            bible_clause = (
                f"\n[CHARACTER BIBLE -- NON-NEGOTIABLE: Every character attribute below "
                f"MUST remain exactly as described. Do NOT alter hair, age, skin, "
                f"clothing, or any other attribute.]\n{character_bible.strip()}\n"
            )

        if scene_context.strip():
            effective = (
                f"[SCENE CONTEXT FROM IMAGE]\n{scene_context.strip()}\n\n"
                f"[USER DIRECTION]\n{user_input.strip()}"
            )
        else:
            effective = user_input.strip()

        lora_clause = (f"\n[LORA: Begin prompt with: {lora_triggers.strip()}]"
                       if lora_triggers.strip() else "")

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
        del out, input_ids
        neg = _build_neg(result, user_input)

        if not self.keep_loaded:
            self._unload()

        print(f"   [EasyPrompt] done ({len(result.split())} words).")
        return result, neg


# -- Character Bible --
class CharacterBible:
    """Cross-scene character consistency record."""

    def __init__(self):
        self._chars: dict = {}

    def add(self, name: str, **attributes):
        self._chars[name] = dict(attributes)

    def extract_from_description(self, name: str, description: str):
        self._chars[name] = {"_raw": description.strip()}

    def to_prompt_block(self) -> str:
        if not self._chars:
            return ""
        lines = []
        for name, attrs in self._chars.items():
            if "_raw" in attrs:
                lines.append(f"CHARACTER -- {name}:\n{attrs['_raw']}")
            else:
                attr_str = "; ".join(f"{k}: {v}" for k, v in attrs.items())
                lines.append(f"CHARACTER -- {name}: {attr_str}")
        return "\n\n".join(lines)

    def has_characters(self) -> bool:
        return bool(self._chars)

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


print("Cell 5 complete: VisionDescribe + EasyPrompt + CharacterBible ready.")


# ======================================================================
# CELL 6  --  generate_clip()  (two-pass LTX-2 19B GGUF pipeline)
# ======================================================================
# @title 6. Generate Clip (Core Pipeline)


def generate_clip(
    image_tensor: Optional[torch.Tensor],
    prompt: str,
    neg_prompt: str,
    width: int = 768,   height: int = 512,
    frames: int = 121,  fps: int = 25,
    seed: int = 42,     image_strength: float = 1.0,
    pass1_sigmas: str = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
    pass1_sampler: str = "euler",   pass1_cfg: float = 1.0,
    pass2_sigmas: str = "0.909375, 0.725, 0.421875, 0.0",
    pass2_sampler: str = "gradient_estimation",  pass2_cfg: float = 1.0,
    pass2_seed: int = 0,
    use_tiled_vae: bool = True,  tiled_stiles: int = 2,
    tiled_soverlap: int = 8,     tiled_tlen: int = 48,
    tiled_toverlap: int = 4,     output_prefix: str = "PRO",
) -> str:
    """Two-pass LTX-2 19B GGUF generation for one clip.
    T4-safe: CLIP is deleted before UNet loads; UNet is deleted before decode."""
    _load_comfy_nodes()

    img_bypass = image_tensor is None
    img_str = image_strength if not img_bypass else 0.0

    print(f"   [Clip] {'I2V' if not img_bypass else 'T2V'}  "
          f"{width}x{height}  {frames}f  seed={seed}")
    _vram_print("before clip load")

    with torch.inference_mode():
        # Model variables for OOM cleanup (None so finally block can always reference them)
        clip_raw = unet = vae_v = vae_a = up_mdl = None

        try:
            # -- STEP 1: CLIP -> encode -> delete --
            print("   [Clip] Loading CLIP...")
            clip_ld = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
            clip_raw = get_value_at_index(
                clip_ld.load_clip(clip_name1=_M_CLIP1, clip_name2=_M_CLIP2,
                                  type="ltxv", device="default"), 0)

            cte = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cond_pos = cte.encode(text=prompt, clip=clip_raw)
            zero_out = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
            cond_0 = zero_out.zero_out(conditioning=get_value_at_index(cond_pos, 0))
            ltxv_cn = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
            cond = ltxv_cn.EXECUTE_NORMALIZED(
                frame_rate=float(fps),
                positive=get_value_at_index(cond_pos, 0),
                negative=get_value_at_index(cond_0, 0))

            # Delete CLIP now -- frees ~6-8 GB for UNet
            del clip_raw
            clip_raw = None
            _vram_free()
            _vram_print("CLIP deleted")

            # -- STEP 2: UNet + LoRAs --
            print("   [Clip] Loading UNet (GGUF Q4_K_M)...")
            unet_ld = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            unet = get_value_at_index(unet_ld.load_unet(unet_name=_M_UNET), 0)
            # clip=None is safe here: IC/camera LoRAs are model-only (no text-encoder layers)
            _dummy_clip = None
            unet, _ = _apply_loras(unet, _dummy_clip)
            _vram_print("UNet + LoRAs loaded")

            # -- STEP 3: VAEs + upscaler (peak window ~7.5GB, proven T4-safe) --
            vae_ld = NODE_CLASS_MAPPINGS["VAELoader"]()
            vae_v = get_value_at_index(vae_ld.load_vae(vae_name=_M_VAE), 0)
            vae_a = get_value_at_index(_audio_vae(_M_AVAE), 0)
            uml = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
            up_mdl = get_value_at_index(uml.EXECUTE_NORMALIZED(model_name=_M_UP), 0)

            # -- STEP 4: Latent preparation --
            ei = NODE_CLASS_MAPPINGS["EmptyImage"]()
            full_i = ei.generate(width=width, height=height, batch_size=1, color=0)
            rimn = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
            half_i = rimn.EXECUTE_NORMALIZED(
                input=get_value_at_index(full_i, 0), scale_method="area",
                resize_type={"resize_type": "scale by multiplier", "multiplier": 0.5})
            gis = NODE_CLASS_MAPPINGS["GetImageSize"]()
            hsz = gis.EXECUTE_NORMALIZED(image=get_value_at_index(half_i, 0))
            hw, hh = get_value_at_index(hsz, 0), get_value_at_index(hsz, 1)

            eltxv = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
            vid_lat = eltxv.EXECUTE_NORMALIZED(width=hw, height=hh, length=frames, batch_size=1)

            # I2V: condition the latent on the seed/last-frame image
            if not img_bypass:
                rim2 = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
                ri2 = rim2.EXECUTE_NORMALIZED(
                    input=image_tensor, scale_method="lanczos",
                    resize_type={"resize_type": "scale dimensions",
                                 "width": hw * 2, "height": hh * 2, "crop": "center"})
                ppn = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
                pp = get_value_at_index(ppn.EXECUTE_NORMALIZED(
                    img_compression=33, image=get_value_at_index(ri2, 0)), 0)
                i2v = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
                vid_lat = (get_value_at_index(i2v.EXECUTE_NORMALIZED(
                    strength=img_str, bypass=False,
                    vae=vae_v, image=pp,
                    latent=get_value_at_index(vid_lat, 0)), 0),)
            else:
                vid_lat = (get_value_at_index(vid_lat, 0),)

            # Audio latent + concat AV
            elalat = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
            aud_lat = elalat.EXECUTE_NORMALIZED(
                frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=vae_a)
            catav = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
            av_lat = get_value_at_index(catav.EXECUTE_NORMALIZED(
                video_latent=vid_lat[0],
                audio_latent=get_value_at_index(aud_lat, 0)), 0)

            # -- STEP 5: Pass 1 --
            print("   [Clip] Pass 1...")
            _vram_print()
            ms = NODE_CLASS_MAPPINGS["ManualSigmas"]()
            ks = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            rn = NODE_CLASS_MAPPINGS["RandomNoise"]()
            cfg_node = NODE_CLASS_MAPPINGS["CFGGuider"]()
            sca = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

            g1 = cfg_node.EXECUTE_NORMALIZED(cfg=pass1_cfg, model=unet,
                                             positive=get_value_at_index(cond, 0),
                                             negative=get_value_at_index(cond, 1))
            o1 = sca.EXECUTE_NORMALIZED(
                noise=get_value_at_index(rn.EXECUTE_NORMALIZED(noise_seed=seed), 0),
                guider=get_value_at_index(g1, 0),
                sampler=get_value_at_index(ks.EXECUTE_NORMALIZED(sampler_name=pass1_sampler), 0),
                sigmas=get_value_at_index(ms.EXECUTE_NORMALIZED(sigmas=pass1_sigmas), 0),
                latent_image=av_lat)
            p1_av = get_value_at_index(o1, 0)
            del g1
            _vram_free()
            print("   [Clip] Pass 1 done.")

            # -- STEP 6: Pass 2 --
            print("   [Clip] Pass 2 (upscale + refine)...")
            sep = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
            s1 = sep.EXECUTE_NORMALIZED(av_latent=p1_av)
            vl1, al1 = get_value_at_index(s1, 0), get_value_at_index(s1, 1)

            crop = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
            cr = crop.EXECUTE_NORMALIZED(positive=get_value_at_index(cond, 0),
                                         negative=get_value_at_index(cond, 1),
                                         latent=vl1)
            g2 = cfg_node.EXECUTE_NORMALIZED(cfg=pass2_cfg, model=unet,
                                             positive=get_value_at_index(cr, 0),
                                             negative=get_value_at_index(cr, 1))

            ltxvup = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
            up = ltxvup.upsample_latent(samples=get_value_at_index(cr, 2),
                                        upscale_model=up_mdl, vae=vae_v)
            del up_mdl
            up_mdl = None
            _vram_free()

            av2 = get_value_at_index(catav.EXECUTE_NORMALIZED(
                video_latent=get_value_at_index(up, 0), audio_latent=al1), 0)

            o2 = sca.EXECUTE_NORMALIZED(
                noise=get_value_at_index(rn.EXECUTE_NORMALIZED(noise_seed=pass2_seed), 0),
                guider=get_value_at_index(g2, 0),
                sampler=get_value_at_index(ks.EXECUTE_NORMALIZED(sampler_name=pass2_sampler), 0),
                sigmas=get_value_at_index(ms.EXECUTE_NORMALIZED(sigmas=pass2_sigmas), 0),
                latent_image=av2)
            p2_den = get_value_at_index(o2, 1)

            # BIG VRAM RELEASE -- UNet is done
            del g2, unet
            unet = None
            _vram_free()
            _vram_print("UNet deleted")
            print("   [Clip] Pass 2 done.")

            # -- STEP 7: Decode video --
            s2 = sep.EXECUTE_NORMALIZED(av_latent=p2_den)
            vl_f = get_value_at_index(s2, 0)
            al_f = get_value_at_index(s2, 1)

            decoded = None
            if use_tiled_vae:
                try:
                    td = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
                    decoded = get_value_at_index(td.EXECUTE_NORMALIZED(
                        vae=vae_v, latents=vl_f,
                        spatial_tiles=tiled_stiles, spatial_overlap=tiled_soverlap,
                        temporal_tile_length=tiled_tlen, temporal_overlap=tiled_toverlap,
                        last_frame_fix=False, working_device="auto",
                        working_dtype="auto"), 0)
                    print("   [Clip] Tiled VAE decode done.")
                except (KeyError, Exception) as e:
                    print(f"   [Clip] Tiled VAE skipped ({type(e).__name__}) -- standard decode")
                    use_tiled_vae = False

            if not use_tiled_vae:
                vd = NODE_CLASS_MAPPINGS["VAEDecode"]()
                decoded = get_value_at_index(vd.decode(samples=vl_f, vae=vae_v), 0)

            del vae_v
            vae_v = None
            _vram_free()

            # -- STEP 8: Decode audio --
            aud_d = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
            audio = aud_d.EXECUTE_NORMALIZED(samples=al_f, audio_vae=vae_a)
            del vae_a
            vae_a = None
            _vram_free()

            # -- STEP 9: Save --
            cv_node = NODE_CLASS_MAPPINGS["CreateVideo"]()
            vid_obj = cv_node.EXECUTE_NORMALIZED(
                fps=fps, images=decoded,
                audio=get_value_at_index(audio, 0))
            path = save_video_obj(get_value_at_index(vid_obj, 0), prefix=output_prefix)
            _vram_print("after save")
            print(f"   [Clip] Saved -> {path}")
            return path

        finally:
            # Ensure all GPU models are freed on OOM or unexpected exit
            for _m in (clip_raw, unet, vae_v, vae_a, up_mdl):
                if _m is not None:
                    del _m
            _vram_free()


print("Cell 6 complete: generate_clip() ready.")


# ======================================================================
# CELL 7  --  InfiniteFlowEngine
# ======================================================================
# @title 7. Infinite Flow Engine (Multi-Clip Orchestrator)


class InfiniteFlowEngine:
    """Multi-clip orchestrator with last-frame chaining and optional Vision/LLM.
    Peak VRAM on T4: ~13 GB (UNet + VAEs + upscaler during Pass 2)."""

    def __init__(
        self,
        character_bible: Optional[CharacterBible] = None,
        llm_model: str = "3B",
        vision_model: str = "3B-fast",
        use_llm: bool = False,
        use_vision: bool = False,
        width: int = 768,
        height: int = 512,
        frames: int = 121,
        fps: int = 25,
        image_strength: float = 1.0,
        creativity: float = 0.9,
        lora_triggers: str = "",
        use_tiled_vae: bool = True,
        offline: bool = False,
        output_dir: str = "/content/ComfyUI/output/ltx_pro",
        show_previews: bool = True,
        pass1_sigmas: str = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
        pass1_sampler: str = "euler",
        pass1_cfg: float = 1.0,
        pass2_sigmas: str = "0.909375, 0.725, 0.421875, 0.0",
        pass2_sampler: str = "gradient_estimation",
        pass2_cfg: float = 1.0,
        pass2_seed: int = 0,
        tiled_stiles: int = 2,
        tiled_soverlap: int = 8,
        tiled_tlen: int = 48,
        tiled_toverlap: int = 4,
    ):
        self.bible = character_bible or CharacterBible()
        self.llm_model = llm_model
        self.vision_model = vision_model
        self.use_llm = use_llm
        self.use_vision = use_vision
        self.width = width
        self.height = height
        self.frames = frames
        self.fps = fps
        self.img_strength = image_strength
        self.creativity = creativity
        self.lora_triggers = lora_triggers
        self.use_tiled_vae = use_tiled_vae
        self.offline = offline
        self.output_dir = output_dir
        self.show_previews = show_previews
        self.pass1_sigmas = pass1_sigmas
        self.pass1_sampler = pass1_sampler
        self.pass1_cfg = pass1_cfg
        self.pass2_sigmas = pass2_sigmas
        self.pass2_sampler = pass2_sampler
        self.pass2_cfg = pass2_cfg
        self.pass2_seed = pass2_seed
        self.tiled_stiles = tiled_stiles
        self.tiled_soverlap = tiled_soverlap
        self.tiled_tlen = tiled_tlen
        self.tiled_toverlap = tiled_toverlap

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._clip_paths: List[str] = []
        self._scene_history: List[str] = []

    def _rolling_ctx(self) -> str:
        return "\n".join(self._scene_history[-2:])

    def _process_beat(self, beat: str, current_image: Optional[torch.Tensor],
                      beat_idx: int, base_seed: int) -> tuple:
        beat_seed = base_seed + (beat_idx - 1) * 1000

        # Vision Describe (only from beat 2 onward if enabled)
        scene_ctx = ""
        if beat_idx > 1 and current_image is not None and self.use_vision:
            print(f"   [Beat {beat_idx}] Vision Describe...")
            try:
                ve = VisionDescribeEngine(self.vision_model, self.offline)
                scene_ctx = ve.describe(current_image)
                del ve
                _vram_free()
            except Exception as e:
                print(f"   [Beat {beat_idx}] Vision failed: {e}")
                scene_ctx = self._rolling_ctx()
        else:
            scene_ctx = self._rolling_ctx()

        # Expand beat with LLM (if enabled)
        if self.use_llm:
            print(f"   [Beat {beat_idx}] EasyPrompt expand...")
            try:
                pe = EasyPromptEngine(self.llm_model, self.offline, keep_loaded=False)
                prompt, neg = pe.generate(
                    user_input=beat,
                    frame_count=self.frames,
                    creativity=self.creativity,
                    seed=beat_seed,
                    scene_context=scene_ctx,
                    lora_triggers=self.lora_triggers,
                    character_bible=self.bible.to_prompt_block(),
                )
                del pe
                _vram_free()
            except Exception as e:
                print(f"   [Beat {beat_idx}] LLM failed: {e}, using raw beat")
                prompt = beat
                neg = _build_neg(beat, beat)
        else:
            prompt = beat
            neg = _build_neg(beat, beat)

        self._scene_history.append(prompt[:400])
        if len(self._scene_history) > 2:
            self._scene_history = self._scene_history[-2:]

        return prompt, neg, beat_seed

    def run(
        self,
        story_beats: List[str],
        seed_image_path: Optional[str] = None,
        base_seed: int = 42,
    ) -> List[str]:
        """Process each beat. Returns list of output .mp4 paths."""
        current_image: Optional[torch.Tensor] = None

        if seed_image_path and os.path.exists(seed_image_path):
            current_image = load_image_tensor(seed_image_path)
            print(f"[Engine] Seed image loaded: {seed_image_path}")

            # Auto-populate CharacterBible from seed image
            if not self.bible.has_characters() and self.use_vision:
                print("[Engine] Extracting Character Bible from seed image...")
                try:
                    ve = VisionDescribeEngine(self.vision_model, self.offline)
                    desc = ve.describe(current_image)
                    del ve
                    _vram_free()
                    self.bible.extract_from_description("Main Character", desc)
                    self._scene_history.append(desc)
                except Exception as e:
                    print(f"[Engine] Vision extraction failed: {e}")

        print(f"\n[Engine] === LTX-2 PRO ===")
        print(f"[Engine]  Beats   : {len(story_beats)}")
        print(f"[Engine]  Size    : {self.width}x{self.height}  {self.frames}f @ {self.fps}fps")
        print(f"[Engine]  LLM     : {'ON ('+self.llm_model+')' if self.use_llm else 'OFF'}")
        print(f"[Engine]  Vision  : {'ON' if self.use_vision else 'OFF'}")
        print(f"[Engine]  Bible   : {self.bible.names() or 'empty'}")
        print(f"[Engine]  Mode    : {'I2V' if current_image is not None else 'T2V'}\n")

        for beat_idx, beat in enumerate(story_beats, 1):
            print(f"\n[Engine] -- Beat {beat_idx}/{len(story_beats)} --")
            print(f"[Engine]   {beat[:80]}{'...' if len(beat) > 80 else ''}")

            # Phase 1: Prompt generation
            prompt, neg, beat_seed = self._process_beat(
                beat, current_image, beat_idx, base_seed)

            print(f"\n  PROMPT ({len(prompt.split())}w): {prompt[:200]}...")
            print(f"  NEG: {neg[:100]}...\n")

            # Phase 2: Video generation
            try:
                clip_path = generate_clip(
                    image_tensor=current_image,
                    prompt=prompt,
                    neg_prompt=neg,
                    width=self.width,
                    height=self.height,
                    frames=self.frames,
                    fps=self.fps,
                    seed=beat_seed,
                    image_strength=self.img_strength,
                    pass1_sigmas=self.pass1_sigmas,
                    pass1_sampler=self.pass1_sampler,
                    pass1_cfg=self.pass1_cfg,
                    pass2_sigmas=self.pass2_sigmas,
                    pass2_sampler=self.pass2_sampler,
                    pass2_cfg=self.pass2_cfg,
                    pass2_seed=self.pass2_seed,
                    use_tiled_vae=self.use_tiled_vae,
                    tiled_stiles=self.tiled_stiles,
                    tiled_soverlap=self.tiled_soverlap,
                    tiled_tlen=self.tiled_tlen,
                    tiled_toverlap=self.tiled_toverlap,
                    output_prefix=f"PRO_{beat_idx:03d}",
                )
            except torch.cuda.OutOfMemoryError:
                # gc + empty_cache reclaims cached allocations; the try/finally in
                # generate_clip already deleted any model tensors still in scope.
                _vram_free()
                print(f"   OOM on beat {beat_idx}. Retrying with reduced frames in T2V...")
                reduced_frames = max(49, self.frames - 24)
                clip_path = generate_clip(
                    image_tensor=None,
                    prompt=prompt,
                    neg_prompt=neg,
                    width=self.width,
                    height=self.height,
                    frames=reduced_frames,
                    fps=self.fps,
                    seed=beat_seed + 1,
                    use_tiled_vae=False,
                    output_prefix=f"PRO_{beat_idx:03d}_retry",
                )

            # Copy to output dir
            dest = os.path.join(self.output_dir, f"scene_{beat_idx:03d}.mp4")
            shutil.copy2(clip_path, dest)
            self._clip_paths.append(dest)
            print(f"\n[Engine] Beat {beat_idx} -> {dest}")

            # Extract last frame for chaining
            last_frame = get_last_frame_tensor(dest)
            current_image = last_frame if last_frame is not None else None
            if last_frame is not None:
                print(f"[Engine]   Last frame extracted for beat {beat_idx + 1}.")
            else:
                print(f"[Engine]   No last frame -- next beat will be T2V.")

            # Preview
            if self.show_previews:
                display_video(dest)

        print(f"\n[Engine] === {len(story_beats)} scenes complete ===")
        print(f"[Engine]  Output dir: {self.output_dir}")
        return self._clip_paths

    def concat_all(self, output_name: str = "full_video.mp4") -> str:
        """Concatenate all generated clips into one final video."""
        out = os.path.join(self.output_dir, output_name)
        return concatenate_clips(self._clip_paths, out)

    def download_all(self):
        if colab_files is None:
            print("   (Not in Colab, skipping download)")
            return
        for p in self._clip_paths:
            if os.path.exists(p):
                colab_files.download(p)


print("Cell 7 complete: InfiniteFlowEngine ready.")


# ======================================================================
# CELL 8  --  USER CONFIGURATION (edit before running)
# ======================================================================
# @title 8. Story Configuration

# -- Story Beats (each string = one video clip) --
STORY_BEATS = [
    "A woman steps out of a taxi onto a rain-slicked city street at night, "
    "neon signs reflecting on the wet pavement, she looks up at a towering building",

    "She pushes through the glass doors of the building lobby, shaking rain "
    "from her coat, the security guard nods as she passes",

    "Inside the elevator, she watches the floor numbers climb, "
    "her reflection ghostly in the polished steel doors, tension in her jaw",
]

# -- LLM/Vision Toggles (disabled by default for T4 safety) --
USE_LLM_PROMPT = False
USE_VISION = False

# -- Character Bible (optional, for cross-scene consistency) --
bible = CharacterBible()
# Example:
# bible.add("Maya",
#     age=32, ethnicity="East Asian", hair="short black bob",
#     build="athletic", clothes="dark trench coat, red scarf")

# -- LoRA Stack Config (edit slots in LD_LORA_STACK from Cell 3) --
# LD_LORA_STACK[0] = {"on": True, "lora": "ltx-2-19b-ic-lora-detailer.safetensors",
#                      "guard": False, "strength": 0.4}
# LD_LORA_STACK[1] = {"on": True, "lora": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
#                      "guard": False, "strength": 0.6}

# -- Print Summary --
print("Cell 8 complete: User configuration set.")
print(f"   Beats: {len(STORY_BEATS)}")
print(f"   LLM: {'ON ('+LLM_MODEL+')' if USE_LLM_PROMPT else 'OFF (raw beats)'}")
print(f"   Vision: {'ON' if USE_VISION else 'OFF'}")
print(f"   Bible: {bible.names() or 'empty'}")
print(f"   Output: {OUTPUT_DIR}")


# ======================================================================
# CELL 9  --  RUN
# ======================================================================
# @title 9. Generate Video


def _run():
    """Main execution: creates engine, runs all beats, concatenates output."""
    engine = InfiniteFlowEngine(
        character_bible=bible,
        llm_model=LLM_MODEL,
        vision_model="3B-fast",
        use_llm=USE_LLM_PROMPT,
        use_vision=USE_VISION,
        width=WIDTH,
        height=HEIGHT,
        frames=FRAMES,
        fps=FPS,
        image_strength=IMAGE_STRENGTH,
        creativity=CREATIVITY,
        lora_triggers=LORA_TRIGGERS,
        use_tiled_vae=USE_TILED_VAE,
        offline=bool(OFFLINE_LLM_PATH or OFFLINE_VISION_PATH),
        output_dir=OUTPUT_DIR,
        show_previews=SHOW_PREVIEWS,
        pass1_sigmas=PASS1_SIGMAS,
        pass1_sampler=PASS1_SAMPLER,
        pass1_cfg=PASS1_CFG,
        pass2_sigmas=PASS2_SIGMAS,
        pass2_sampler=PASS2_SAMPLER,
        pass2_cfg=PASS2_CFG,
        pass2_seed=PASS2_SEED,
        tiled_stiles=TILED_SPATIAL_TILES,
        tiled_soverlap=TILED_SPATIAL_OVERLAP,
        tiled_tlen=TILED_TEMPORAL_LENGTH,
        tiled_toverlap=TILED_TEMPORAL_OVERLAP,
    )

    clip_paths = engine.run(
        story_beats=STORY_BEATS,
        seed_image_path=SEED_IMAGE_PATH,
        base_seed=SEED,
    )

    # Concatenate all scenes
    if len(clip_paths) > 1:
        final = engine.concat_all("final_video.mp4")
        print(f"\nFinal video: {final}")
        if SHOW_PREVIEWS:
            display_video(final)
    elif len(clip_paths) == 1:
        print(f"\nSingle clip: {clip_paths[0]}")
        if SHOW_PREVIEWS:
            display_video(clip_paths[0])

    print("\nGeneration complete.")
    return engine, clip_paths


try:
    engine, output_clips = _run()

except torch.cuda.OutOfMemoryError:
    _vram_free()
    print("\nCUDA OOM -- try: LLM_MODEL='3B', FRAMES=97, USE_TILED_VAE=True, USE_VISION=False")
except Exception as e:
    import traceback
    print(f"\n{type(e).__name__}: {e}")
    traceback.print_exc()

print("Cell 9 complete.")
