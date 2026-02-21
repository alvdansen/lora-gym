"""
Wan 2.1 T2V 14B LoRA Training -- Local Edition
================================================
Trains a character identity LoRA against the Wan 2.1 T2V 14B DiT on a
local GPU. Recommended: 48GB VRAM (A6000, RTX 6000 Ada).
Also works on 24GB cards (RTX 3090/4090) -- add --blocks_to_swap 20
for video training if you hit OOM.

Wan 2.1 differences from Wan 2.2:
  - Single model (no dual high/low noise experts, no --noise_level)
  - bf16 mixed precision (not fp16)
  - fp8_base quantization but NO fp8_scaled (fp8_scaled is Wan 2.2 MoE only)
  - No min_timestep / max_timestep / preserve_distribution_shape
  - No Lightning LoRA merge

Usage:
  python train_local_wan21_t2v_14b.py
  python train_local_wan21_t2v_14b.py --base_dir D:/my_training
  python train_local_wan21_t2v_14b.py --lr 5e-5 --dim 32 --alpha 32 --epochs 30
"""

import os, sys, subprocess, datetime, argparse, glob, platform
from pathlib import Path

DEFAULT_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_training")
OUTPUT_NAME = "my-wan21-t2v-14b"
LEARNING_RATE = "8e-5"
LR_SCHEDULER = "polynomial"
LR_SCHEDULER_POWER = "2"
MIN_LR_RATIO = "0.01"
OPTIMIZER = "adamw8bit"
NETWORK_DIM = "24"
NETWORK_ALPHA = "24"
MAX_EPOCHS = "50"
SAVE_EVERY = "5"
SEED = "42"
DISCRETE_FLOW_SHIFT = "3.0"
MODEL_FILENAMES = {"dit": "wan2.1_t2v_14B_bf16.safetensors", "vae": "wan_2.1_vae.safetensors", "t5": "models_t5_umt5-xxl-enc-bf16.pth", "t5_alt": "models_t5_umt5-xxl-enc-bf16.safetensors"}

def resolve_paths(base_dir):
    models_dir = os.path.join(base_dir, "models")
    t5_path = os.path.join(models_dir, MODEL_FILENAMES["t5"])
    if not os.path.exists(t5_path):
        t5_alt = os.path.join(models_dir, MODEL_FILENAMES["t5_alt"])
        if os.path.exists(t5_alt): t5_path = t5_alt
    return {"models_dir": models_dir, "datasets_dir": os.path.join(base_dir, "datasets"), "outputs_dir": os.path.join(base_dir, "outputs"), "resume_dir": os.path.join(base_dir, "resume_checkpoints"), "dit": os.path.join(models_dir, MODEL_FILENAMES["dit"]), "vae": os.path.join(models_dir, MODEL_FILENAMES["vae"]), "t5": t5_path}

def find_musubi_tuner(base_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(base_dir, "musubi-tuner"), os.path.join(script_dir, "local_training", "musubi-tuner"), os.path.join(script_dir, "musubi-tuner")]:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, ".git")): return path
    return None

def validate_dataset_config(dataset_config):
    tomllib = None
    try: import tomllib
    except ImportError:
        try: import tomli as tomllib
        except ImportError: return True
    try:
        with open(dataset_config, "rb") as f: config = tomllib.load(f)
    except Exception: return False
    datasets = config.get("datasets", [])
    if not datasets: return False
    all_ok = True
    for ds in datasets:
        for key in ["image_directory", "video_directory"]:
            path = ds.get(key)
            if path and not os.path.isdir(path): all_ok = False
    return all_ok

def train(args):
    base_dir = os.path.abspath(args.base_dir)
    paths = resolve_paths(base_dir)
    lr = args.lr or LEARNING_RATE
    scheduler = args.scheduler or LR_SCHEDULER
    power = args.power or LR_SCHEDULER_POWER
    min_lr = args.min_lr or MIN_LR_RATIO
    optimizer = args.optimizer or OPTIMIZER
    dim = args.dim or NETWORK_DIM
    alpha = args.alpha or NETWORK_ALPHA
    epochs = args.epochs or MAX_EPOCHS
    save_every = args.save_every or SAVE_EVERY
    seed = args.seed or SEED
    flow_shift = args.flow_shift or DISCRETE_FLOW_SHIFT
    output_name = args.output_name or OUTPUT_NAME
    if args.dataset_config: dataset_config = os.path.abspath(args.dataset_config)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_config = os.path.join(script_dir, "wan21-dataset-config-local.toml")
    dit_path = args.dit or paths["dit"]
    vae_path = args.vae or paths["vae"]
    t5_path = args.t5 or paths["t5"]
    missing = []
    for name, path in [("DiT", dit_path), ("VAE", vae_path), ("T5", t5_path)]:
        if not os.path.exists(path): missing.append(f"  {name}: {path}")
    if missing:
        print("ERROR: Model files not found.")
        for m in missing: print(m)
        print("\nWan 2.1 models are NOT downloaded by setup_local.py by default.")
        print("Download wan2.1_t2v_14B_bf16.safetensors from HuggingFace.")
        sys.exit(1)
    if not os.path.exists(dataset_config):
        print(f"ERROR: Dataset config not found: {dataset_config}")
        sys.exit(1)
    validate_dataset_config(dataset_config)
    musubi_dir = find_musubi_tuner(base_dir)
    if musubi_dir is None:
        print("ERROR: musubi-tuner not found. Run setup_local.py first.")
        sys.exit(1)
    src_dir = os.path.join(musubi_dir, "src")
    if os.path.isdir(src_dir):
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if src_dir not in current_pythonpath:
            os.environ["PYTHONPATH"] = src_dir + os.pathsep + current_pythonpath if current_pythonpath else src_dir
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if os.path.exists(os.path.join(musubi_dir, "src", "musubi_tuner", "wan_train_network.py")):
        SCRIPT_PREFIX = os.path.join("src", "musubi_tuner") + os.sep
    else: SCRIPT_PREFIX = ""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{output_name}-{timestamp}"
    run_output_dir = os.path.join(paths["outputs_dir"], output_name)
    os.makedirs(run_output_dir, exist_ok=True)
    resume_weights = None
    os.makedirs(paths["resume_dir"], exist_ok=True)
    if args.resume_from:
        if os.path.exists(args.resume_from): resume_weights = args.resume_from
    else:
        candidates = sorted(glob.glob(os.path.join(paths["resume_dir"], "*.safetensors")), key=os.path.getmtime, reverse=True)
        if candidates: resume_weights = candidates[0]
    print("=" * 60)
    print(f"  Wan 2.1 T2V 14B LoRA Training (Local)")
    print(f"  LR: {lr} | Dim: {dim} | Epochs: {epochs}")
    print("=" * 60)
    # Cache latents
    subprocess.run([sys.executable, os.path.join(SCRIPT_PREFIX, "wan_cache_latents.py") if SCRIPT_PREFIX else "wan_cache_latents.py", "--dataset_config", dataset_config, "--vae", vae_path, "--vae_cache_cpu"], check=True, cwd=musubi_dir)
    # Cache text encoder
    subprocess.run([sys.executable, os.path.join(SCRIPT_PREFIX, "wan_cache_text_encoder_outputs.py") if SCRIPT_PREFIX else "wan_cache_text_encoder_outputs.py", "--dataset_config", dataset_config, "--t5", t5_path, "--batch_size", "16", "--fp8_t5"], check=True, cwd=musubi_dir)
    # Train
    train_script = os.path.join(SCRIPT_PREFIX, "wan_train_network.py") if SCRIPT_PREFIX else "wan_train_network.py"
    train_cmd = [sys.executable, "-m", "accelerate.commands.accelerate_cli", "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16", train_script, "--task", "t2v-14B", "--dit", dit_path, "--vae", vae_path, "--t5", t5_path, "--dataset_config", dataset_config, "--sdpa", "--mixed_precision", "bf16", "--fp8_base", "--vae_cache_cpu", "--optimizer_type", optimizer, "--optimizer_args", "weight_decay=0.01", "--learning_rate", lr, "--lr_scheduler", scheduler, "--lr_scheduler_min_lr_ratio", min_lr, "--lr_scheduler_power", power, "--gradient_checkpointing", "--max_data_loader_n_workers", "2", "--persistent_data_loader_workers", "--network_module", "networks.lora_wan", "--network_dim", dim, "--network_alpha", alpha, "--timestep_sampling", "shift", "--discrete_flow_shift", flow_shift, "--max_train_epochs", epochs, "--save_every_n_epochs", save_every, "--seed", seed, "--output_dir", run_output_dir, "--output_name", name, "--log_with", "tensorboard", "--logging_dir", os.path.join(run_output_dir, "logs")]
    if args.blocks_to_swap: train_cmd += ["--blocks_to_swap", args.blocks_to_swap]
    if args.network_args: train_cmd += ["--network_args"] + args.network_args
    if resume_weights: train_cmd += ["--network_weights", resume_weights]
    result = subprocess.run(train_cmd, cwd=musubi_dir)
    if result.returncode != 0: sys.exit(result.returncode)
    print(f"\nCheckpoints saved to: {run_output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Wan 2.1 T2V 14B LoRA Training -- Local Edition")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--lr", default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--power", default=None)
    parser.add_argument("--min_lr", default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--dim", default=None)
    parser.add_argument("--alpha", default=None)
    parser.add_argument("--epochs", default=None)
    parser.add_argument("--save_every", default=None)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--flow_shift", default=None)
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--dit", default=None)
    parser.add_argument("--vae", default=None)
    parser.add_argument("--t5", default=None)
    parser.add_argument("--blocks_to_swap", default=None)
    parser.add_argument("--network_args", nargs="+", default=None)
    parser.add_argument("--resume_from", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
