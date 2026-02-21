"""
Wan 2.2 I2V 14B LoRA Training — Local Edition (Lightning-Optimized)
====================================================================
Trains a character identity LoRA against Lightning-merged DiT weights
on a local GPU. Recommended: 48GB VRAM (A6000, RTX 6000 Ada).
Also works on 24GB cards (RTX 3090/4090) — add --blocks_to_swap 20
for video training if you hit OOM.

Adapted from the RunPod pipeline. Training will be slower than cloud
A100s but produces identical results.

Usage:
  python train_local_i2v_lightning.py --noise_level high
  python train_local_i2v_lightning.py --noise_level low
  python train_local_i2v_lightning.py --noise_level high --skip_lightning

At inference: load Lightning LoRA + your character LoRA together in ComfyUI.
"""

import os, sys, subprocess, datetime, argparse, glob, platform
from pathlib import Path

DEFAULT_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_training")
OUTPUT_NAME = "my-wan22-i2v-lightning"
LIGHTNING_LORA = {"high": {"repo_id": "lightx2v/Wan2.2-Lightning", "filename": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors"}, "low": {"repo_id": "lightx2v/Wan2.2-Lightning", "filename": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors"}}
LIGHTNING_MERGE_STRENGTH = 1.0
LR_SCHEDULER = "polynomial"
LR_SCHEDULER_POWER = "2"
MIN_LR_RATIO = "0.01"
OPTIMIZER = "adamw8bit"
SEED = "42"
DISCRETE_FLOW_SHIFT = "5.0"  # I2V standard
EXPERT_CONFIG = {"high": {"learning_rate": "1e-4", "network_dim": "8", "network_alpha": "8", "max_epochs": "50", "save_every": "5"}, "low": {"learning_rate": "8e-5", "network_dim": "42", "network_alpha": "42", "max_epochs": "50", "save_every": "5"}}
MODEL_FILENAMES = {"dit_high": "wan2.2_i2v_high_noise_14B_fp16.safetensors", "dit_low": "wan2.2_i2v_low_noise_14B_fp16.safetensors", "vae": "wan_2.1_vae.safetensors", "t5": "models_t5_umt5-xxl-enc-bf16.pth", "t5_alt": "models_t5_umt5-xxl-enc-bf16.safetensors"}

def resolve_paths(base_dir):
    models_dir = os.path.join(base_dir, "models")
    t5_path = os.path.join(models_dir, MODEL_FILENAMES["t5"])
    if not os.path.exists(t5_path):
        t5_alt = os.path.join(models_dir, MODEL_FILENAMES["t5_alt"])
        if os.path.exists(t5_alt): t5_path = t5_alt
    return {"models_dir": models_dir, "datasets_dir": os.path.join(base_dir, "datasets"), "outputs_dir": os.path.join(base_dir, "outputs"), "merged_dits_dir": os.path.join(base_dir, "merged_dits"), "lightning_cache": os.path.join(base_dir, "lightning_loras"), "resume_dir": os.path.join(base_dir, "resume_checkpoints"), "dit_high": os.path.join(models_dir, MODEL_FILENAMES["dit_high"]), "dit_low": os.path.join(models_dir, MODEL_FILENAMES["dit_low"]), "vae": os.path.join(models_dir, MODEL_FILENAMES["vae"]), "t5": t5_path}

def find_musubi_tuner(base_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(base_dir, "musubi-tuner"), os.path.join(script_dir, "local_training", "musubi-tuner"), os.path.join(script_dir, "musubi-tuner")]:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, ".git")): return path
    return None

def check_hardware(skip_lightning=False):
    print("Hardware check:")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) == 2:
                    name, vram_mb = parts[0], int(parts[1])
                    print(f"  GPU: {name} ({vram_mb/1024:.0f} GB VRAM)")
    except FileNotFoundError: print("  WARNING: nvidia-smi not found")
    print()

def _get_system_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError: pass
    return None

def validate_dataset_config(dataset_config):
    tomllib = None
    try: import tomllib
    except ImportError:
        try: import tomli as tomllib
        except ImportError: return True
    try:
        with open(dataset_config, "rb") as f: config = tomllib.load(f)
    except Exception as e:
        print(f"  ERROR: Cannot parse {dataset_config}: {e}")
        return False
    datasets = config.get("datasets", [])
    if not datasets: return False
    all_ok = True
    for i, ds in enumerate(datasets):
        for key in ["image_directory", "video_directory"]:
            path = ds.get(key)
            if path and not os.path.isdir(path): all_ok = False
    return all_ok

def merge_lora_into_dit(dit_path, lora_path, output_path, strength=1.0):
    import torch
    from safetensors.torch import load_file, save_file
    print(f"Loading base DiT: {dit_path}")
    base_sd = load_file(dit_path, device="cpu")
    print(f"Loading Lightning LoRA: {lora_path}")
    lora_sd = load_file(lora_path, device="cpu")
    base_keys = set(base_sd.keys())
    lora_pairs = {}
    for key in lora_sd:
        if ".lora_down.weight" not in key: continue
        up_key = key.replace(".lora_down.weight", ".lora_up.weight")
        alpha_key = key.replace(".lora_down.weight", ".alpha")
        if up_key not in lora_sd: continue
        module_path = key.replace(".lora_down.weight", "")
        candidates = [f"{module_path}.weight", module_path]
        for prefix in ["diffusion_model.", "lora_unet_", "lora_te_"]:
            if module_path.startswith(prefix):
                stripped = module_path[len(prefix):]
                candidates.extend([f"{stripped}.weight", stripped])
        if not module_path.startswith("diffusion_model."):
            candidates.extend([f"diffusion_model.{module_path}.weight", f"diffusion_model.{module_path}"])
        base_key = None
        for c in candidates:
            if c in base_keys: base_key = c; break
        if base_key is None: continue
        alpha = lora_sd[alpha_key].item() if alpha_key in lora_sd else None
        lora_pairs[base_key] = {"down": lora_sd[key], "up": lora_sd[up_key], "alpha": alpha}
    if len(lora_pairs) == 0: return False
    merged_count = 0
    for base_key, pair in lora_pairs.items():
        down, up, alpha = pair["down"].float(), pair["up"].float(), pair["alpha"]
        rank = down.shape[0]
        scale = strength * (alpha / rank) if alpha is not None else strength
        base_weight = base_sd[base_key].float()
        if down.dim() == 2 and up.dim() == 2: delta = (up @ down) * scale
        elif down.dim() == 5 and up.dim() == 5 and up.shape[2:] == (1, 1, 1):
            delta = ((up.reshape(up.shape[0], rank) @ down.reshape(rank, -1)) * scale).reshape(base_weight.shape)
        elif down.dim() == 3 and up.dim() == 3:
            delta = ((up.reshape(up.shape[0], rank) @ down.reshape(rank, -1)) * scale).reshape(base_weight.shape)
        else: continue
        base_sd[base_key] = (base_weight + delta).to(base_sd[base_key].dtype)
        merged_count += 1
    print(f"  Merged {merged_count}/{len(lora_pairs)} layers")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(base_sd, output_path)
    del base_sd, lora_sd
    return True

def train(args):
    noise_level = args.noise_level
    is_high = noise_level == "high"
    label = "HIGH-NOISE" if is_high else "LOW-NOISE"
    base_dir = os.path.abspath(args.base_dir)
    paths = resolve_paths(base_dir)
    expert = EXPERT_CONFIG[noise_level]
    lr = args.lr or expert["learning_rate"]
    scheduler = args.scheduler or LR_SCHEDULER
    power = args.power or LR_SCHEDULER_POWER
    min_lr = args.min_lr or MIN_LR_RATIO
    optimizer = args.optimizer or OPTIMIZER
    dim = args.dim or expert["network_dim"]
    alpha = args.alpha or expert["network_alpha"]
    epochs = args.epochs or expert["max_epochs"]
    save_every = args.save_every or expert["save_every"]
    seed = args.seed or SEED
    flow_shift = args.flow_shift or DISCRETE_FLOW_SHIFT
    output_name = args.output_name or OUTPUT_NAME
    if args.dataset_config: dataset_config = os.path.abspath(args.dataset_config)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_config = os.path.join(script_dir, "wan22-dataset-config-local.toml")
    dit_key = "dit_high" if is_high else "dit_low"
    dit_path = args.dit or paths[dit_key]
    vae_path = args.vae or paths["vae"]
    t5_path = args.t5 or paths["t5"]
    missing = []
    for name, path in [("DiT", dit_path), ("VAE", vae_path), ("T5", t5_path)]:
        if not os.path.exists(path): missing.append(f"  {name}: {path}")
    if missing:
        print("ERROR: Model files not found. Run setup_local.py --include_i2v first.")
        for m in missing: print(m)
        sys.exit(1)
    if not os.path.exists(dataset_config):
        print(f"ERROR: Dataset config not found: {dataset_config}")
        sys.exit(1)
    validate_dataset_config(dataset_config)
    check_hardware(skip_lightning=args.skip_lightning)
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
    name = f"{output_name}-{noise_level}-{timestamp}"
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
    print(f"  Wan 2.2 I2V 14B — {label} Expert (Local)")
    print(f"  LR: {lr} | Dim: {dim} | Epochs: {epochs}")
    print("=" * 60)
    # Lightning merge
    lightning_cfg = LIGHTNING_LORA.get(noise_level, {})
    if lightning_cfg.get("repo_id") and not args.skip_lightning:
        os.makedirs(paths["merged_dits_dir"], exist_ok=True)
        merge_tag = f"wan22_i2v_{noise_level}_lightning_s{LIGHTNING_MERGE_STRENGTH}"
        merged_dit_path = os.path.join(paths["merged_dits_dir"], f"{merge_tag}.safetensors")
        if os.path.exists(merged_dit_path):
            dit_path = merged_dit_path
        else:
            try:
                import huggingface_hub, shutil
                os.makedirs(paths["lightning_cache"], exist_ok=True)
                lora_filename = os.path.basename(lightning_cfg["filename"])
                local_lora = os.path.join(paths["lightning_cache"], lora_filename)
                if not os.path.exists(local_lora):
                    downloaded = huggingface_hub.hf_hub_download(repo_id=lightning_cfg["repo_id"], filename=lightning_cfg["filename"])
                    shutil.copy2(downloaded, local_lora)
                if merge_lora_into_dit(dit_path, local_lora, merged_dit_path, LIGHTNING_MERGE_STRENGTH):
                    dit_path = merged_dit_path
            except Exception as e:
                print(f"  ERROR: Lightning merge failed: {e}")
    # Cache latents
    subprocess.run([sys.executable, os.path.join(SCRIPT_PREFIX, "wan_cache_latents.py") if SCRIPT_PREFIX else "wan_cache_latents.py", "--dataset_config", dataset_config, "--vae", vae_path, "--vae_cache_cpu"], check=True, cwd=musubi_dir)
    # Cache text encoder
    subprocess.run([sys.executable, os.path.join(SCRIPT_PREFIX, "wan_cache_text_encoder_outputs.py") if SCRIPT_PREFIX else "wan_cache_text_encoder_outputs.py", "--dataset_config", dataset_config, "--t5", t5_path, "--batch_size", "16", "--fp8_t5"], check=True, cwd=musubi_dir)
    # Train
    if is_high: min_ts, max_ts = "900", "1000"
    else: min_ts, max_ts = "0", "900"
    train_script = os.path.join(SCRIPT_PREFIX, "wan_train_network.py") if SCRIPT_PREFIX else "wan_train_network.py"
    train_cmd = [sys.executable, "-m", "accelerate.commands.accelerate_cli", "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "fp16", train_script, "--task", "i2v-A14B", "--dit", dit_path, "--vae", vae_path, "--t5", t5_path, "--dataset_config", dataset_config, "--sdpa", "--mixed_precision", "fp16", "--fp8_base", "--fp8_scaled", "--vae_cache_cpu", "--min_timestep", min_ts, "--max_timestep", max_ts, "--preserve_distribution_shape", "--optimizer_type", optimizer, "--optimizer_args", "weight_decay=0.01", "--learning_rate", lr, "--lr_scheduler", scheduler, "--lr_scheduler_min_lr_ratio", min_lr, "--lr_scheduler_power", power, "--gradient_checkpointing", "--max_data_loader_n_workers", "2", "--persistent_data_loader_workers", "--network_module", "networks.lora_wan", "--network_dim", dim, "--network_alpha", alpha, "--timestep_sampling", "shift", "--discrete_flow_shift", flow_shift, "--max_train_epochs", epochs, "--save_every_n_epochs", save_every, "--seed", seed, "--output_dir", run_output_dir, "--output_name", name, "--log_with", "tensorboard", "--logging_dir", os.path.join(run_output_dir, "logs")]
    if args.blocks_to_swap: train_cmd += ["--blocks_to_swap", args.blocks_to_swap]
    if args.network_args: train_cmd += ["--network_args"] + args.network_args
    if resume_weights: train_cmd += ["--network_weights", resume_weights]
    result = subprocess.run(train_cmd, cwd=musubi_dir)
    if result.returncode != 0: sys.exit(result.returncode)
    print(f"\nCheckpoints saved to: {run_output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Wan 2.2 I2V LoRA Training — Local Edition (Lightning)")
    parser.add_argument("--noise_level", required=True, choices=["high", "low"])
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
    parser.add_argument("--skip_lightning", action="store_true")
    parser.add_argument("--blocks_to_swap", default=None)
    parser.add_argument("--network_args", nargs="+", default=None)
    parser.add_argument("--resume_from", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
