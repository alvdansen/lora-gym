"""
Wan 2.2 T2V 14B LoRA Training — Local Edition (Lightning-Optimized)
====================================================================
Trains a character identity LoRA against Lightning-merged DiT weights
on a local GPU. Recommended: 48GB VRAM (A6000, RTX 6000 Ada).
Also works on 24GB cards (RTX 3090/4090) — add --blocks_to_swap 20
for video training if you hit OOM.

Adapted from the RunPod pipeline. Training will be slower than cloud
A100s but produces identical results.

Usage:
  # Train high-noise expert
  python train_local_t2v_lightning.py --noise_level high

  # Train low-noise expert
  python train_local_t2v_lightning.py --noise_level low

  # Custom base directory (where setup_local.py put everything)
  python train_local_t2v_lightning.py --noise_level high --base_dir D:/my_training

  # Custom dataset config
  python train_local_t2v_lightning.py --noise_level high --dataset_config my_dataset.toml

  # Skip Lightning merge (saves ~40-50GB RAM during merge step)
  python train_local_t2v_lightning.py --noise_level high --skip_lightning

  # Resume from a specific checkpoint
  python train_local_t2v_lightning.py --noise_level high --resume_from ./outputs/my-lora-e25.safetensors

  # Full custom config
  python train_local_t2v_lightning.py --noise_level high --lr 5e-5 --dim 32 --alpha 32 --epochs 30

At inference: load Lightning LoRA + your character LoRA together in ComfyUI.
"""

import os
import sys
import subprocess
import datetime
import argparse
import glob
import platform
from pathlib import Path

# =============================================================================
# ██████  CONFIG  ██████
# =============================================================================
# These are defaults — override any of them via command-line arguments.

# Default base directory — override with --base_dir
DEFAULT_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_training")

OUTPUT_NAME = "my-wan22-t2v-lightning"

# --- Lightning LoRA to merge into base DiT ---
# Downloads from HuggingFace and bakes into the DiT weights before training.
# Set repo_id to None to skip merge for that expert.
LIGHTNING_LORA = {
    "high": {
        "repo_id": "lightx2v/Wan2.2-Lightning",
        "filename": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
    },
    "low": {
        "repo_id": "lightx2v/Wan2.2-Lightning",
        "filename": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
    },
}
LIGHTNING_MERGE_STRENGTH = 1.0

# --- Shared Hyperparameters ---
LR_SCHEDULER        = "polynomial"
LR_SCHEDULER_POWER  = "2"
MIN_LR_RATIO        = "0.01"
OPTIMIZER           = "adamw8bit"
SEED                = "42"
DISCRETE_FLOW_SHIFT = "3.0"       # T2V standard

# --- Per-Expert Hyperparameters ---
# High and low noise experts can have different training configs.
# The high-noise expert handles the initial denoising (coarse structure),
# while the low-noise expert handles refinement (fine detail).
EXPERT_CONFIG = {
    "high": {
        "learning_rate": "1e-4",
        "network_dim":   "8",
        "network_alpha": "8",
        "max_epochs":    "50",
        "save_every":    "5",
    },
    "low": {
        "learning_rate": "8e-5",
        "network_dim":   "42",
        "network_alpha": "42",
        "max_epochs":    "50",
        "save_every":    "5",
    },
}

# --- Model filenames (downloaded by setup_local.py) ---
MODEL_FILENAMES = {
    "dit_high": "wan2.2_t2v_high_noise_14B_fp16.safetensors",
    "dit_low":  "wan2.2_t2v_low_noise_14B_fp16.safetensors",
    "vae":      "wan_2.1_vae.safetensors",
    "t5":       "models_t5_umt5-xxl-enc-bf16.pth",
    "t5_alt":   "models_t5_umt5-xxl-enc-bf16.safetensors",  # ComfyUI copies are safetensors
}


# =============================================================================
# Path Resolution
# =============================================================================

def resolve_paths(base_dir):
    """Build all paths from the base directory."""
    models_dir = os.path.join(base_dir, "models")

    # T5 can be either .pth (HuggingFace download) or .safetensors (ComfyUI copy)
    t5_path = os.path.join(models_dir, MODEL_FILENAMES["t5"])
    if not os.path.exists(t5_path):
        t5_alt = os.path.join(models_dir, MODEL_FILENAMES["t5_alt"])
        if os.path.exists(t5_alt):
            t5_path = t5_alt

    return {
        "models_dir":      models_dir,
        "datasets_dir":    os.path.join(base_dir, "datasets"),
        "outputs_dir":     os.path.join(base_dir, "outputs"),
        "merged_dits_dir": os.path.join(base_dir, "merged_dits"),
        "lightning_cache": os.path.join(base_dir, "lightning_loras"),
        "resume_dir":      os.path.join(base_dir, "resume_checkpoints"),
        "dit_high":        os.path.join(models_dir, MODEL_FILENAMES["dit_high"]),
        "dit_low":         os.path.join(models_dir, MODEL_FILENAMES["dit_low"]),
        "vae":             os.path.join(models_dir, MODEL_FILENAMES["vae"]),
        "t5":              t5_path,
    }


def find_musubi_tuner(base_dir):
    """Auto-detect musubi-tuner installation. Returns path or None."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "musubi-tuner"),
        os.path.join(script_dir, "local_training", "musubi-tuner"),
        os.path.join(script_dir, "musubi-tuner"),
    ]

    for path in candidates:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, ".git")):
            return path

    return None


# =============================================================================
# Hardware Checks
# =============================================================================

def check_hardware(skip_lightning=False):
    """Print hardware info and warnings before training."""
    print("Hardware check:")

    # GPU VRAM
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) == 2:
                    name, vram_mb = parts[0], int(parts[1])
                    vram_gb = vram_mb / 1024
                    print(f"  GPU: {name} ({vram_gb:.0f} GB VRAM)")
                    if vram_gb < 24:
                        print(f"  WARNING: 24 GB VRAM recommended. You have {vram_gb:.0f} GB.")
                        print(f"  Training may OOM. Consider reducing resolution or batch size.")
    except FileNotFoundError:
        print("  WARNING: nvidia-smi not found — cannot check GPU")

    # System RAM
    ram_gb = _get_system_ram_gb()
    if ram_gb is not None:
        print(f"  System RAM: {ram_gb:.0f} GB")
        if ram_gb < 48 and not skip_lightning:
            print(f"  WARNING: Lightning merge loads the ~28GB DiT on CPU.")
            print(f"  With {ram_gb:.0f} GB RAM this may fail. Use --skip_lightning if needed.")
    else:
        print("  System RAM: could not detect")

    print()


def _get_system_ram_gb():
    """Get total system RAM in GB, or None if detection fails."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    if platform.system() == "Windows":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024**3)
        except Exception:
            pass

    return None


# =============================================================================
# TOML Validation
# =============================================================================

def validate_dataset_config(dataset_config):
    """
    Parse the TOML dataset config and verify that referenced directories exist.
    Returns True if valid, False if not.
    """
    # Python 3.11+ has tomllib built-in; fall back to tomli for 3.10
    tomllib = None
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            print(f"  WARNING: Cannot validate TOML (install 'tomli' package for Python <3.11)")
            return True

    try:
        with open(dataset_config, "rb") as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f"  ERROR: Cannot parse {dataset_config}: {e}")
        return False

    datasets = config.get("datasets", [])
    if not datasets:
        print(f"  ERROR: No [[datasets]] entries found in {dataset_config}")
        return False

    all_ok = True
    for i, ds in enumerate(datasets):
        for key in ["image_directory", "video_directory"]:
            path = ds.get(key)
            if path and not os.path.isdir(path):
                print(f"  ERROR: {key} does not exist: {path}")
                print(f"         (in [[datasets]] entry #{i+1} of {os.path.basename(dataset_config)})")
                all_ok = False

    if all_ok:
        print(f"  Dataset config OK: {len(datasets)} dataset(s), all paths exist")
    else:
        print()
        print(f"  Fix the paths in {dataset_config} and try again.")
        print(f"  Use forward slashes even on Windows (e.g., C:/path/to/datasets/character/images)")

    return all_ok


# =============================================================================
# LoRA -> Base Model Merge Logic
# =============================================================================
# Merges Lightning LoRA delta weights directly into the DiT safetensors so
# the character LoRA trains against the Lightning-modified denoising path.
#
# Formula per layer: W_merged = W_base + strength * (alpha/rank) * (up @ down)
#
# Runs on CPU to avoid OOM — the 14B DiT is ~28GB in fp16.
# =============================================================================

def merge_lora_into_dit(dit_path, lora_path, output_path, strength=1.0):
    """
    Merge a LoRA's learned deltas into the base DiT weights.
    Returns True on success, False on failure.
    """
    import torch
    from safetensors.torch import load_file, save_file

    print(f"Loading base DiT: {dit_path}")
    print(f"  (this is ~28GB for 14B fp16 — loading on CPU)")
    base_sd = load_file(dit_path, device="cpu")
    print(f"  Loaded {len(base_sd)} base model keys")

    print(f"Loading Lightning LoRA: {lora_path}")
    lora_sd = load_file(lora_path, device="cpu")
    print(f"  Loaded {len(lora_sd)} LoRA keys")

    # Diagnostic: show sample keys
    base_key_list = sorted(base_sd.keys())
    print(f"\n  First 5 base keys: {base_key_list[:5]}")
    down_keys = [k for k in sorted(lora_sd.keys()) if ".lora_down.weight" in k]
    if down_keys:
        print(f"  Sample LoRA key:   {down_keys[0]}")

    base_keys = set(base_sd.keys())

    # Find all LoRA down/up pairs and map them to base model keys
    lora_pairs = {}
    unmapped_keys = []

    for key in lora_sd:
        if ".lora_down.weight" not in key:
            continue

        down_key = key
        up_key = key.replace(".lora_down.weight", ".lora_up.weight")
        alpha_key = key.replace(".lora_down.weight", ".alpha")

        if up_key not in lora_sd:
            print(f"  WARNING: Found down without up: {key}")
            continue

        # Try multiple key prefix variations to find the matching base key
        module_path = key.replace(".lora_down.weight", "")
        candidates = [
            f"{module_path}.weight",
            module_path,
        ]
        # Strip common prefixes
        for prefix in ["diffusion_model.", "lora_unet_", "lora_te_"]:
            if module_path.startswith(prefix):
                stripped = module_path[len(prefix):]
                candidates.extend([f"{stripped}.weight", stripped])
        # Try adding prefix
        if not module_path.startswith("diffusion_model."):
            candidates.extend([
                f"diffusion_model.{module_path}.weight",
                f"diffusion_model.{module_path}",
            ])

        base_key = None
        for candidate in candidates:
            if candidate in base_keys:
                base_key = candidate
                break

        if base_key is None:
            unmapped_keys.append(key)
            if len(unmapped_keys) <= 3:
                print(f"  WARNING: Cannot map: {key}")
            continue

        alpha = lora_sd[alpha_key].item() if alpha_key in lora_sd else None
        lora_pairs[base_key] = {"down": lora_sd[down_key], "up": lora_sd[up_key], "alpha": alpha}

    print(f"\n  Found {len(lora_pairs)} LoRA layer pairs to merge")
    if unmapped_keys:
        print(f"  WARNING: {len(unmapped_keys)} keys could not be mapped")
    if len(lora_pairs) == 0:
        print("  ERROR: No LoRA pairs mapped — aborting merge")
        return False

    # Apply deltas
    merged_count = 0
    for base_key, pair in lora_pairs.items():
        down = pair["down"].float()
        up = pair["up"].float()
        alpha = pair["alpha"]
        rank = down.shape[0]

        scale = strength * (alpha / rank) if alpha is not None else strength
        base_weight = base_sd[base_key].float()

        if down.dim() == 2 and up.dim() == 2:
            delta = (up @ down) * scale
        elif down.dim() == 5 and up.dim() == 5:
            if up.shape[2:] == (1, 1, 1):
                down_2d = down.reshape(rank, -1)
                up_2d = up.reshape(up.shape[0], rank)
                delta_2d = (up_2d @ down_2d) * scale
                delta = delta_2d.reshape(base_weight.shape)
            else:
                print(f"  WARNING: Complex Conv3d for {base_key}, skipping")
                continue
        elif down.dim() == 3 and up.dim() == 3:
            down_2d = down.reshape(rank, -1)
            up_2d = up.reshape(up.shape[0], rank)
            delta_2d = (up_2d @ down_2d) * scale
            delta = delta_2d.reshape(base_weight.shape)
        else:
            print(f"  WARNING: Unexpected dims for {base_key}, skipping")
            continue

        base_sd[base_key] = (base_weight + delta).to(base_sd[base_key].dtype)
        merged_count += 1

    print(f"  Merged {merged_count}/{len(lora_pairs)} layers (strength={strength})")

    # Save
    print(f"  Saving merged DiT to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(base_sd, output_path)
    merged_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"  Saved ({merged_size_gb:.1f} GB)")

    del base_sd, lora_sd
    return True


# =============================================================================
# Main Training Function
# =============================================================================

def train(args):
    """
    Run the full training pipeline:
    1. (Optional) Download and merge Lightning LoRA into DiT
    2. Cache latents
    3. Cache text encoder outputs
    4. Train LoRA
    """
    noise_level = args.noise_level
    is_high = noise_level == "high"
    label = "HIGH-NOISE" if is_high else "LOW-NOISE"
    base_dir = os.path.abspath(args.base_dir)
    paths = resolve_paths(base_dir)

    # --- Resolve hyperparameters (CLI overrides > per-expert defaults) ---
    expert = EXPERT_CONFIG[noise_level]
    lr          = args.lr or expert["learning_rate"]
    scheduler   = args.scheduler or LR_SCHEDULER
    power       = args.power or LR_SCHEDULER_POWER
    min_lr      = args.min_lr or MIN_LR_RATIO
    optimizer   = args.optimizer or OPTIMIZER
    dim         = args.dim or expert["network_dim"]
    alpha       = args.alpha or expert["network_alpha"]
    epochs      = args.epochs or expert["max_epochs"]
    save_every  = args.save_every or expert["save_every"]
    seed        = args.seed or SEED
    flow_shift  = args.flow_shift or DISCRETE_FLOW_SHIFT
    output_name = args.output_name or OUTPUT_NAME

    # --- Resolve dataset config ---
    if args.dataset_config:
        dataset_config = os.path.abspath(args.dataset_config)
    else:
        # Look for the config next to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_config = os.path.join(script_dir, "wan22-dataset-config-local.toml")

    # --- Resolve model paths (CLI overrides > base_dir defaults) ---
    dit_key = "dit_high" if is_high else "dit_low"
    dit_path = args.dit or paths[dit_key]
    vae_path = args.vae or paths["vae"]
    t5_path  = args.t5 or paths["t5"]

    missing = []
    for name, path in [("DiT", dit_path), ("VAE", vae_path), ("T5", t5_path)]:
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")
    if missing:
        print("ERROR: Model files not found.")
        print("Either run setup_local.py first, or use --dit / --vae / --t5 to point to existing files.")
        print("(If you have ComfyUI, run: setup_local.py --comfyui_dir /path/to/ComfyUI)")
        for m in missing:
            print(m)
        print("\nNOTE: DiT must be fp16/bf16. ComfyUI's fp8_scaled versions won't work.")
        sys.exit(1)

    if not os.path.exists(dataset_config):
        print(f"ERROR: Dataset config not found: {dataset_config}")
        print(f"Expected wan22-dataset-config-local.toml next to this script,")
        print(f"or specify one with --dataset_config.")
        sys.exit(1)

    # --- Validate TOML: check that dataset directories exist ---
    print(f"Validating dataset config: {dataset_config}")
    if not validate_dataset_config(dataset_config):
        sys.exit(1)
    print()

    # --- Hardware warnings ---
    check_hardware(skip_lightning=args.skip_lightning)

    # --- Find musubi-tuner ---
    musubi_dir = find_musubi_tuner(base_dir)
    if musubi_dir is None:
        print("ERROR: musubi-tuner not found.")
        print(f"  Searched:")
        print(f"    {os.path.join(base_dir, 'musubi-tuner')}")
        print(f"  Run setup_local.py first to clone it.")
        sys.exit(1)

    print(f"Using musubi-tuner: {musubi_dir}")

    # --- Ensure musubi-tuner is importable ---
    # If musubi-tuner isn't pip-installed (common if Python version is too new),
    # add its src/ directory to PYTHONPATH so subprocess calls can import it.
    src_dir = os.path.join(musubi_dir, "src")
    if os.path.isdir(src_dir):
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if src_dir not in current_pythonpath:
            os.environ["PYTHONPATH"] = src_dir + os.pathsep + current_pythonpath if current_pythonpath else src_dir
            print(f"  Added {src_dir} to PYTHONPATH")

    # Force UTF-8 for subprocess output — musubi-tuner has Japanese text that crashes Windows cp1252
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # --- Detect musubi-tuner repo structure ---
    if os.path.exists(os.path.join(musubi_dir, "src", "musubi_tuner", "wan_train_network.py")):
        SCRIPT_PREFIX = os.path.join("src", "musubi_tuner") + os.sep
        print("Detected new musubi-tuner repo structure")
    else:
        SCRIPT_PREFIX = ""
        print("Detected classic musubi-tuner repo structure")

    # --- Timestamped output ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{output_name}-{noise_level}-{timestamp}"
    run_output_dir = os.path.join(paths["outputs_dir"], output_name)
    os.makedirs(run_output_dir, exist_ok=True)

    # =================================================================
    # Resume from checkpoint (optional)
    # =================================================================
    resume_weights = None
    os.makedirs(paths["resume_dir"], exist_ok=True)

    if args.resume_from:
        # Explicit path provided via CLI
        if os.path.exists(args.resume_from):
            resume_weights = args.resume_from
            print(f"\n  RESUME: Using explicitly specified checkpoint:")
            print(f"    {resume_weights}")
        else:
            print(f"\n  WARNING: --resume_from path not found: {args.resume_from}")
            print(f"  Training will start from scratch.")
    else:
        # Auto-detect: check resume_dir for .safetensors files
        candidates = sorted(
            glob.glob(os.path.join(paths["resume_dir"], "*.safetensors")),
            key=os.path.getmtime,
            reverse=True,  # newest first
        )
        if candidates:
            resume_weights = candidates[0]
            print(f"\n  RESUME: Auto-detected checkpoint in {paths['resume_dir']}:")
            print(f"    {resume_weights}")
            if len(candidates) > 1:
                print(f"    ({len(candidates)} files found — using newest by modification time)")
        else:
            print(f"\n  No resume checkpoint found. Starting from scratch.")
            print(f"  (To resume: drop a .safetensors in {paths['resume_dir']}/ or use --resume_from)")

    if resume_weights:
        size_mb = os.path.getsize(resume_weights) / (1024 * 1024)
        print(f"    File size: {size_mb:.1f} MB")
        print(f"    NOTE: Loading weights only (no optimizer state) — LR schedule starts fresh.")

    print("=" * 60)
    print(f"  Wan 2.2 T2V 14B — {label} Expert (Local)")
    print(f"  Output: {name}")
    print(f"  Dir:    {run_output_dir}")
    print(f"  LR: {lr} | Scheduler: {scheduler}")
    print(f"  Dim: {dim} | Alpha: {alpha}")
    print(f"  Epochs: {epochs} | Flow Shift: {flow_shift}")
    print(f"  Lightning merge strength: {LIGHTNING_MERGE_STRENGTH}")
    if resume_weights:
        print(f"  ** RESUMING from: {os.path.basename(resume_weights)} **")
    print("=" * 60)

    # =================================================================
    # Lightning LoRA Merge (optional)
    # =================================================================
    lightning_cfg = LIGHTNING_LORA.get(noise_level, {})
    has_lightning = (
        lightning_cfg.get("repo_id")
        and lightning_cfg.get("filename")
    )

    if has_lightning and not args.skip_lightning:
        print(f"\n{'='*60}")
        print(f"  LIGHTNING MERGE — {label} expert")
        print(f"  Repo: {lightning_cfg['repo_id']}")
        print(f"  File: {lightning_cfg['filename']}")
        print(f"  Strength: {LIGHTNING_MERGE_STRENGTH}")
        print(f"{'='*60}")

        # Check for cached merged DiT
        os.makedirs(paths["merged_dits_dir"], exist_ok=True)
        merge_tag = f"wan22_t2v_{noise_level}_lightning_s{LIGHTNING_MERGE_STRENGTH}"
        merged_dit_path = os.path.join(paths["merged_dits_dir"], f"{merge_tag}.safetensors")

        if os.path.exists(merged_dit_path):
            print(f"  [CACHED] Merged DiT exists: {merged_dit_path}")
            dit_path = merged_dit_path
        else:
            # Download Lightning LoRA
            try:
                import huggingface_hub
                os.makedirs(paths["lightning_cache"], exist_ok=True)

                # Check local cache first
                lora_filename = os.path.basename(lightning_cfg["filename"])
                local_lora = os.path.join(paths["lightning_cache"], lora_filename)

                if os.path.exists(local_lora):
                    print(f"  [CACHED] Lightning LoRA: {local_lora}")
                    lightning_lora_path = local_lora
                else:
                    print(f"  [DOWNLOADING] Lightning LoRA...")
                    downloaded = huggingface_hub.hf_hub_download(
                        repo_id=lightning_cfg["repo_id"],
                        filename=lightning_cfg["filename"],
                    )
                    # Copy to our cache for future runs
                    import shutil
                    shutil.copy2(downloaded, local_lora)
                    lightning_lora_path = local_lora
                    print(f"  Downloaded and cached: {local_lora}")

            except Exception as e:
                print(f"  ERROR downloading Lightning LoRA: {e}")
                print(f"  Training with vanilla DiT instead.")
                lightning_lora_path = None

            if lightning_lora_path:
                success = merge_lora_into_dit(
                    dit_path, lightning_lora_path, merged_dit_path, LIGHTNING_MERGE_STRENGTH
                )
                if success:
                    print(f"  Merge complete. Using merged DiT for training.")
                    dit_path = merged_dit_path
                else:
                    print(f"  WARNING: Merge failed. Training with vanilla DiT.")
    elif args.skip_lightning:
        print(f"\n  Lightning merge SKIPPED (--skip_lightning flag)")
    else:
        print(f"\n  Lightning LoRA not configured for {label} expert.")
        print(f"  Training against vanilla DiT.")

    # =================================================================
    # Step 1: Cache latents (uses vanilla VAE — not affected by merge)
    # =================================================================
    print("\n" + "=" * 60)
    print("  Step 1: Caching latents...")
    print("=" * 60)
    subprocess.run([
        sys.executable,
        os.path.join(SCRIPT_PREFIX, "wan_cache_latents.py") if SCRIPT_PREFIX else "wan_cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", vae_path,
        "--vae_cache_cpu",
    ], check=True, cwd=musubi_dir)

    # =================================================================
    # Step 2: Cache text encoder outputs
    # =================================================================
    print("\n" + "=" * 60)
    print("  Step 2: Caching text encoder outputs...")
    print("=" * 60)
    subprocess.run([
        sys.executable,
        os.path.join(SCRIPT_PREFIX, "wan_cache_text_encoder_outputs.py") if SCRIPT_PREFIX else "wan_cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--t5", t5_path,
        "--batch_size", "16",
        "--fp8_t5",
    ], check=True, cwd=musubi_dir)

    # =================================================================
    # Step 3: Train LoRA
    # =================================================================
    print("\n" + "=" * 60)
    print(f"  Step 3: Training {label} expert...")
    print(f"  DiT: {dit_path}")
    if paths["merged_dits_dir"] in dit_path:
        print(f"  ** Training against Lightning-merged weights **")
    print("=" * 60)

    # T2V boundary: 875 (I2V would be 900)
    if is_high:
        min_ts, max_ts = "875", "1000"
    else:
        min_ts, max_ts = "0", "875"

    # Use sys.executable -m accelerate instead of bare "accelerate" command
    # to avoid PATH issues on Windows
    train_script = os.path.join(SCRIPT_PREFIX, "wan_train_network.py") if SCRIPT_PREFIX else "wan_train_network.py"

    train_cmd = [
        sys.executable, "-m", "accelerate.commands.accelerate_cli", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "fp16",       # Wan 2.2 weights are fp16
        train_script,
        "--task", "t2v-A14B",
        "--dit", dit_path,
        "--vae", vae_path,
        "--t5", t5_path,
        "--dataset_config", dataset_config,
        "--sdpa",
        "--mixed_precision", "fp16",       # Wan 2.2 = fp16 (NOT bf16)
        "--fp8_base",
        "--fp8_scaled",                    # Valid for Wan 2.2 only
        "--vae_cache_cpu",
        # --- Wan 2.2 T2V specific ---
        "--min_timestep", min_ts,
        "--max_timestep", max_ts,
        "--preserve_distribution_shape",
        # --- Optimizer ---
        "--optimizer_type", optimizer,
        "--optimizer_args", "weight_decay=0.01",
        # --- Learning Rate ---
        "--learning_rate", lr,
        # --- LR Scheduler ---
        "--lr_scheduler", scheduler,
        "--lr_scheduler_min_lr_ratio", min_lr,
        "--lr_scheduler_power", power,
        # --- Memory ---
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        # --- LoRA Config ---
        "--network_module", "networks.lora_wan",
        "--network_dim", dim,
        "--network_alpha", alpha,
        # --- Timestep / Flow ---
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", flow_shift,
        # --- Training Duration ---
        "--max_train_epochs", epochs,
        "--save_every_n_epochs", save_every,
        # --- Output ---
        "--seed", seed,
        "--output_dir", run_output_dir,
        "--output_name", name,
        # --- Logging ---
        "--log_with", "tensorboard",
        "--logging_dir", os.path.join(run_output_dir, "logs"),
    ]

    # --- Optional: blocks_to_swap (for 24GB cards) ---
    if args.blocks_to_swap:
        train_cmd += ["--blocks_to_swap", args.blocks_to_swap]
        print(f"\n  >> --blocks_to_swap {args.blocks_to_swap} (offloading transformer blocks to CPU)")

    # --- Optional: network_args (e.g., loraplus_lr_ratio=4) ---
    if args.network_args:
        train_cmd += ["--network_args"] + args.network_args
        print(f"\n  >> --network_args {' '.join(args.network_args)}")

    # --- Resume: inject --network_weights if we have a checkpoint ---
    if resume_weights:
        train_cmd += ["--network_weights", resume_weights]
        print(f"\n  >> --network_weights {resume_weights}")

    print(f"\nTraining command:\n{' '.join(train_cmd)}\n")
    result = subprocess.run(train_cmd, cwd=musubi_dir)
    print(f"\nTraining exit code: {result.returncode}")

    if result.returncode != 0:
        print(f"ERROR: Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"{'='*60}")
    print(f"\nContents of {run_output_dir}:")
    for root, dirs, files in os.walk(run_output_dir):
        level = root.replace(run_output_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in sorted(files):
            filepath = os.path.join(root, f)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"{subindent}{f} ({size_mb:.1f} MB)")

    print(f"\nCheckpoints saved to: {run_output_dir}")
    print(f"Copy .safetensors files to your ComfyUI/models/loras/ folder for inference.")


# =============================================================================
# CLI Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Wan 2.2 T2V LoRA Training — Local Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train high-noise expert with defaults
  python train_local_t2v_lightning.py --noise_level high

  # Train low-noise with custom learning rate
  python train_local_t2v_lightning.py --noise_level low --lr 5e-5

  # Custom base directory
  python train_local_t2v_lightning.py --noise_level high --base_dir D:/my_training

  # Train without Lightning merge (saves RAM)
  python train_local_t2v_lightning.py --noise_level high --skip_lightning

  # Resume from a specific checkpoint
  python train_local_t2v_lightning.py --noise_level high --resume_from ./outputs/my-lora-e25.safetensors

  # Full custom config
  python train_local_t2v_lightning.py --noise_level high --lr 5e-5 --dim 32 --alpha 32 --epochs 30
        """
    )

    # Required
    parser.add_argument("--noise_level", required=True, choices=["high", "low"],
                        help="Which noise expert to train: 'high' or 'low'")

    # Local-specific args
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR,
                        help="Base directory where setup_local.py put models/musubi-tuner "
                             f"(default: ./local_training/)")
    parser.add_argument("--dataset_config", default=None,
                        help="Path to dataset TOML config "
                             "(default: wan22-dataset-config-local.toml next to this script)")

    # Optional overrides (defaults vary by expert — see EXPERT_CONFIG)
    parser.add_argument("--lr", default=None, help="Learning rate (default: 8e-5)")
    parser.add_argument("--scheduler", default=None, help=f"LR scheduler (default: {LR_SCHEDULER})")
    parser.add_argument("--power", default=None, help=f"Scheduler power (default: {LR_SCHEDULER_POWER})")
    parser.add_argument("--min_lr", default=None, help=f"Min LR ratio (default: {MIN_LR_RATIO})")
    parser.add_argument("--optimizer", default=None, help=f"Optimizer (default: {OPTIMIZER})")
    parser.add_argument("--dim", default=None, help="LoRA rank/dim (default varies by expert)")
    parser.add_argument("--alpha", default=None, help="LoRA alpha (default varies by expert)")
    parser.add_argument("--epochs", default=None, help="Max epochs (default: 50)")
    parser.add_argument("--save_every", default=None, help="Save interval (default: 5)")
    parser.add_argument("--seed", default=None, help=f"Random seed (default: {SEED})")
    parser.add_argument("--flow_shift", default=None, help=f"Discrete flow shift (default: {DISCRETE_FLOW_SHIFT})")
    parser.add_argument("--output_name", default=None, help=f"Output name prefix (default: {OUTPUT_NAME})")

    # Model path overrides — point to existing files instead of downloading
    parser.add_argument("--dit", default=None,
                        help="Path to DiT model (fp16/bf16 safetensors). "
                             "Overrides the default path from --base_dir. "
                             "NOTE: must be fp16/bf16, NOT fp8 — musubi-tuner quantizes during loading.")
    parser.add_argument("--vae", default=None,
                        help="Path to VAE model (e.g., your ComfyUI wan_2.1_vae.safetensors)")
    parser.add_argument("--t5", default=None,
                        help="Path to T5 text encoder (.pth or .safetensors)")

    # Lightning merge control
    parser.add_argument("--skip_lightning", action="store_true",
                        help="Skip Lightning LoRA merge — train against vanilla DiT. "
                             "Use this if you don't have enough system RAM (~48GB needed for merge). "
                             "You can still use Lightning LoRA at inference — weights will be "
                             "slightly mismatched which may reduce character consistency.")

    # Memory management for smaller GPUs
    parser.add_argument("--blocks_to_swap", default=None,
                        help="Number of transformer blocks to offload to CPU (0-39). "
                             "Use 20-36 on 24GB GPUs (3090/4090) if video training OOMs. "
                             "Higher = less VRAM but slower. Not needed on 48GB+ GPUs.")

    # Extra training args
    parser.add_argument("--network_args", nargs="+", default=None,
                        help="Extra network args (e.g., loraplus_lr_ratio=4)")
    # Resume from checkpoint
    parser.add_argument("--resume_from", default=None,
                        help="Path to a .safetensors LoRA checkpoint to resume training from. "
                             "Uses --network_weights (weights only, no optimizer state) to avoid "
                             "the known LR scheduling bug. If not specified, auto-checks "
                             "the resume_checkpoints/ folder for the latest .safetensors file.")

    return parser.parse_args()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)

    # Find musubi-tuner
    musubi_dir = find_musubi_tuner(base_dir)
    if musubi_dir is None:
        print(f"ERROR: musubi-tuner not found.")
        print(f"  Searched: {os.path.join(base_dir, 'musubi-tuner')}")
        print(f"  Run setup_local.py first.")
        sys.exit(1)

    print(f"Working directory: {musubi_dir}")
    train(args)
