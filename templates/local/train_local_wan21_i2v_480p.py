"""
Wan 2.1 I2V 14B 480P LoRA Training — Local Edition
=====================================================
Trains a character identity LoRA against the Wan 2.1 Image-to-Video 480P
DiT on a local GPU. Recommended: 48GB VRAM (A6000, RTX 6000 Ada).
Also works on 24GB cards (RTX 3090/4090) — add --blocks_to_swap 20
for video training if you hit OOM.

This is a single-model pipeline (no dual experts, no noise level split).
Requires a CLIP encoder model in addition to DiT, VAE, and T5.

Adapted from the RunPod pipeline. Training will be slower than cloud
A100s but produces identical results.

Usage:
  # Train with defaults
  python train_local_wan21_i2v_480p.py

  # Custom base directory (where setup_local.py put everything)
  python train_local_wan21_i2v_480p.py --base_dir D:/my_training

  # Custom dataset config
  python train_local_wan21_i2v_480p.py --dataset_config my_dataset.toml

  # Point to specific model files
  python train_local_wan21_i2v_480p.py --dit /path/to/dit.safetensors --clip /path/to/clip.pth

  # Resume from a specific checkpoint
  python train_local_wan21_i2v_480p.py --resume_from ./outputs/my-lora-e25.safetensors

  # Full custom config
  python train_local_wan21_i2v_480p.py --lr 5e-5 --dim 32 --alpha 32 --epochs 30

At inference: load your character LoRA in ComfyUI with the Wan 2.1 I2V 480P model.
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

OUTPUT_NAME = "my-wan21-i2v-480p"

# --- Hyperparameters ---
LEARNING_RATE       = "8e-5"
LR_SCHEDULER        = "polynomial"
LR_SCHEDULER_POWER  = "2"
MIN_LR_RATIO        = "0.01"
OPTIMIZER           = "adamw8bit"
NETWORK_DIM         = "24"
NETWORK_ALPHA       = "24"
MAX_EPOCHS          = "50"
SAVE_EVERY          = "5"
SEED                = "42"
DISCRETE_FLOW_SHIFT = "5.0"       # I2V standard

# --- Model filenames (NOT downloaded by setup_local.py by default) ---
MODEL_FILENAMES = {
    "dit":    "wan2.1_i2v_480p_14B_bf16.safetensors",
    "vae":    "wan_2.1_vae.safetensors",
    "t5":     "models_t5_umt5-xxl-enc-bf16.pth",
    "t5_alt": "models_t5_umt5-xxl-enc-bf16.safetensors",  # ComfyUI copies are safetensors
    "clip":   "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
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
        "models_dir":   models_dir,
        "datasets_dir": os.path.join(base_dir, "datasets"),
        "outputs_dir":  os.path.join(base_dir, "outputs"),
        "resume_dir":   os.path.join(base_dir, "resume_checkpoints"),
        "dit":          os.path.join(models_dir, MODEL_FILENAMES["dit"]),
        "vae":          os.path.join(models_dir, MODEL_FILENAMES["vae"]),
        "t5":           t5_path,
        "clip":         os.path.join(models_dir, MODEL_FILENAMES["clip"]),
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

def check_hardware():
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
# Main Training Function
# =============================================================================

def train(args):
    """
    Run the full training pipeline:
    1. Cache latents (with --i2v flag)
    2. Cache text encoder outputs
    3. Train LoRA (with --clip flag)
    """
    base_dir = os.path.abspath(args.base_dir)
    paths = resolve_paths(base_dir)

    # --- Resolve hyperparameters (CLI overrides > defaults) ---
    lr          = args.lr or LEARNING_RATE
    scheduler   = args.scheduler or LR_SCHEDULER
    power       = args.power or LR_SCHEDULER_POWER
    min_lr      = args.min_lr or MIN_LR_RATIO
    optimizer   = args.optimizer or OPTIMIZER
    dim         = args.dim or NETWORK_DIM
    alpha       = args.alpha or NETWORK_ALPHA
    epochs      = args.epochs or MAX_EPOCHS
    save_every  = args.save_every or SAVE_EVERY
    seed        = args.seed or SEED
    flow_shift  = args.flow_shift or DISCRETE_FLOW_SHIFT
    output_name = args.output_name or OUTPUT_NAME

    # --- Resolve dataset config ---
    if args.dataset_config:
        dataset_config = os.path.abspath(args.dataset_config)
    else:
        # Look for the config next to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_config = os.path.join(script_dir, "wan21-i2v-dataset-config-local.toml")

    # --- Resolve model paths (CLI overrides > base_dir defaults) ---
    dit_path  = args.dit or paths["dit"]
    vae_path  = args.vae or paths["vae"]
    t5_path   = args.t5 or paths["t5"]
    clip_path = args.clip or paths["clip"]

    missing = []
    for name, path in [("DiT", dit_path), ("VAE", vae_path), ("T5", t5_path), ("CLIP", clip_path)]:
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")
    if missing:
        print("ERROR: Model files not found.")
        print("Wan 2.1 I2V models are not downloaded by setup_local.py by default.")
        print("Download from HuggingFace or use --dit / --clip flags.")
        print("(VAE and T5 may already exist if you ran setup_local.py for T2V.)")
        for m in missing:
            print(m)
        print("\nNOTE: DiT must be bf16. ComfyUI's fp8 versions won't work for training.")
        sys.exit(1)

    if not os.path.exists(dataset_config):
        print(f"ERROR: Dataset config not found: {dataset_config}")
        print(f"Expected wan21-i2v-dataset-config-local.toml next to this script,")
        print(f"or specify one with --dataset_config.")
        sys.exit(1)

    # --- Validate TOML: check that dataset directories exist ---
    print(f"Validating dataset config: {dataset_config}")
    if not validate_dataset_config(dataset_config):
        sys.exit(1)
    print()

    # --- Hardware warnings ---
    check_hardware()

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
    name = f"{output_name}-{timestamp}"
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
    print(f"  Wan 2.1 I2V 14B 480P LoRA Training (Local)")
    print(f"  Output: {name}")
    print(f"  Dir:    {run_output_dir}")
    print(f"  LR: {lr} | Scheduler: {scheduler}")
    print(f"  Dim: {dim} | Alpha: {alpha}")
    print(f"  Epochs: {epochs} | Flow Shift: {flow_shift}")
    if resume_weights:
        print(f"  ** RESUMING from: {os.path.basename(resume_weights)} **")
    print("=" * 60)

    # =================================================================
    # Step 1: Cache latents (with --i2v flag for I2V pipeline)
    # =================================================================
    print("\n" + "=" * 60)
    print("  Step 1: Caching latents (I2V mode)...")
    print("=" * 60)
    subprocess.run([
        sys.executable,
        os.path.join(SCRIPT_PREFIX, "wan_cache_latents.py") if SCRIPT_PREFIX else "wan_cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", vae_path,
        "--vae_cache_cpu",
        "--i2v",
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
    # Step 3: Train LoRA (with --clip for I2V)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"  Step 3: Training Wan 2.1 I2V 480P LoRA...")
    print(f"  DiT:  {dit_path}")
    print(f"  CLIP: {clip_path}")
    print("=" * 60)

    # Use sys.executable -m accelerate instead of bare "accelerate" command
    # to avoid PATH issues on Windows
    train_script = os.path.join(SCRIPT_PREFIX, "wan_train_network.py") if SCRIPT_PREFIX else "wan_train_network.py"

    train_cmd = [
        sys.executable, "-m", "accelerate.commands.accelerate_cli", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",        # Wan 2.1 weights are bf16
        train_script,
        "--task", "i2v-14B",
        "--dit", dit_path,
        "--vae", vae_path,
        "--t5", t5_path,
        "--clip", clip_path,
        "--dataset_config", dataset_config,
        "--sdpa",
        "--mixed_precision", "bf16",        # Wan 2.1 = bf16 (NOT fp16)
        "--fp8_base",
        "--vae_cache_cpu",
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
        description="Wan 2.1 I2V 14B 480P LoRA Training — Local Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with defaults
  python train_local_wan21_i2v_480p.py

  # Custom base directory
  python train_local_wan21_i2v_480p.py --base_dir D:/my_training

  # Point to specific model files
  python train_local_wan21_i2v_480p.py --dit /path/to/dit.safetensors --clip /path/to/clip.pth

  # Resume from a specific checkpoint
  python train_local_wan21_i2v_480p.py --resume_from ./outputs/my-lora-e25.safetensors

  # Full custom config
  python train_local_wan21_i2v_480p.py --lr 5e-5 --dim 32 --alpha 32 --epochs 30
        """
    )

    # Local-specific args
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR,
                        help="Base directory where setup_local.py put models/musubi-tuner "
                             f"(default: ./local_training/)")
    parser.add_argument("--dataset_config", default=None,
                        help="Path to dataset TOML config "
                             "(default: wan21-i2v-dataset-config-local.toml next to this script)")

    # Optional overrides
    parser.add_argument("--lr", default=None, help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--scheduler", default=None, help=f"LR scheduler (default: {LR_SCHEDULER})")
    parser.add_argument("--power", default=None, help=f"Scheduler power (default: {LR_SCHEDULER_POWER})")
    parser.add_argument("--min_lr", default=None, help=f"Min LR ratio (default: {MIN_LR_RATIO})")
    parser.add_argument("--optimizer", default=None, help=f"Optimizer (default: {OPTIMIZER})")
    parser.add_argument("--dim", default=None, help=f"LoRA rank/dim (default: {NETWORK_DIM})")
    parser.add_argument("--alpha", default=None, help=f"LoRA alpha (default: {NETWORK_ALPHA})")
    parser.add_argument("--epochs", default=None, help=f"Max epochs (default: {MAX_EPOCHS})")
    parser.add_argument("--save_every", default=None, help=f"Save interval (default: {SAVE_EVERY})")
    parser.add_argument("--seed", default=None, help=f"Random seed (default: {SEED})")
    parser.add_argument("--flow_shift", default=None, help=f"Discrete flow shift (default: {DISCRETE_FLOW_SHIFT})")
    parser.add_argument("--output_name", default=None, help=f"Output name prefix (default: {OUTPUT_NAME})")

    # Model path overrides — point to existing files instead of downloading
    parser.add_argument("--dit", default=None,
                        help="Path to DiT model (bf16 safetensors). "
                             "Overrides the default path from --base_dir. "
                             "NOTE: must be bf16, NOT fp8 — musubi-tuner quantizes during loading.")
    parser.add_argument("--vae", default=None,
                        help="Path to VAE model (e.g., your ComfyUI wan_2.1_vae.safetensors)")
    parser.add_argument("--t5", default=None,
                        help="Path to T5 text encoder (.pth or .safetensors)")
    parser.add_argument("--clip", default=None,
                        help="Path to CLIP encoder model (.pth). "
                             "Required for I2V training. "
                             "Default: models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")

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
