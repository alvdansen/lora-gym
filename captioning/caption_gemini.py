"""
Gemini Auto-Captioner for LoRA Training
=========================================
Captions all images AND videos in your dataset folders using Gemini.
Outputs .txt files alongside each file with LoRA-optimized captions.

Caption format:
  "strawberryman dancing in a full body view, straight on"
  + unique actions, movements, expressions — NO character appearance.

For images: describes pose, framing, expression.
For videos: describes motion, actions, transitions.

Usage:
  1. python -m pip install google-generativeai
  2. Set GEMINI_API_KEY as environment variable (see below)
  3. Set IMAGES_DIR and VIDEOS_DIR to your dataset folders
  4. Set CHARACTER_NAME to your character's trigger word
  5. python caption_gemini.py

Get a Gemini API key free at: https://aistudio.google.com/apikey

Set API key in PowerShell:
  $env:GEMINI_API_KEY = "INSERT_YOUR_GEMINI_API_KEY_HERE"
"""

import os
import sys
import time
from pathlib import Path

# =============================================================================
# ██████  CONFIG — EDIT THESE  ██████
# =============================================================================

# Your Gemini API key — set as environment variable GEMINI_API_KEY
# In PowerShell: $env:GEMINI_API_KEY = "your-key-here"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "INSERT_YOUR_GEMINI_API_KEY_HERE")

# Dataset folders — set these to your actual paths
IMAGES_DIR = r"datasets\annika\images"    # <-- Your images folder
VIDEOS_DIR = r"datasets\annika\videos"    # <-- Your videos folder

# Character trigger word — goes at the start of every caption
CHARACTER_NAME = "annika"

# Gemini model — 2.5 Flash is fast, cheap, and good at vision
GEMINI_MODEL = "gemini-2.0-flash"

# Skip files that already have a .txt caption file next to them
SKIP_EXISTING = True

# File extensions to process
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

# =============================================================================
# Caption prompts
# =============================================================================
# Two separate prompts — images describe a static pose/frame,
# videos describe motion and action over time.

SYSTEM_PROMPT_BASE = f"""You are a captioning assistant for AI model training data. Your captions must follow a strict format.

RULES:
1. Every caption starts with "{CHARACTER_NAME}" followed by what they are doing
2. Include the camera view: "full body view", "medium shot", "close up", "upper body view", etc.
3. Include the camera angle: "straight on", "from above", "from below", "from the side", "three quarter view", etc.
4. After the initial line, describe ONLY:
   - Specific movements or poses (arms raised, head tilted, stepping forward)
   - Facial expressions (smiling, frowning, looking surprised)
   - Hair or accessory motion/position (hair blowing, scarf draped)
   - Emotions or states (laughing, shouting, looking pensive)
   - Environmental interaction (picking up an object, leaning on a wall)
5. Do NOT describe the character's physical appearance, clothing colors, body type, or features
6. Do NOT describe the background or setting unless the character directly interacts with it
7. Keep the caption to 1-3 sentences maximum
8. Use simple, direct language — no flowery prose
9. Write as a single continuous caption, not a list

EXAMPLES OF GOOD CAPTIONS:
"{CHARACTER_NAME} walking forward in a full body view, straight on, arms swinging naturally with a slight bounce in their step"
"{CHARACTER_NAME} sitting and turning to look over their shoulder in a medium shot, three quarter view, expression shifting from neutral to surprised"
"{CHARACTER_NAME} jumping and landing in a full body view, from the side, hair bouncing on impact with arms spread for balance"

EXAMPLES OF BAD CAPTIONS (do not do this):
"{CHARACTER_NAME} is a red character with blocky features wearing a green hat, walking through a forest" (describes appearance + setting)
"{CHARACTER_NAME}, a voxel-style figure, stands in a colorful meadow" (describes art style + setting)"""

VIDEO_USER_PROMPT = """Caption this video following the rules exactly. Focus on the motion and actions happening over time. Output ONLY the caption text, nothing else — no quotes, no labels, no explanation."""

IMAGE_USER_PROMPT = """Caption this image following the rules exactly. Describe the pose, framing, and any expression or gesture. Output ONLY the caption text, nothing else — no quotes, no labels, no explanation."""


# =============================================================================
# Script
# =============================================================================

def validate_config():
    """Check config before doing anything."""
    if "INSERT" in GEMINI_API_KEY:
        print("ERROR: Gemini API key not set.")
        print()
        print("Option 1 — Set as environment variable in PowerShell:")
        print('  $env:GEMINI_API_KEY = "your-key-here"')
        print()
        print("Option 2 — Edit GEMINI_API_KEY in this script directly.")
        print()
        print("Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    found_any = False
    for label, path in [("IMAGES_DIR", IMAGES_DIR), ("VIDEOS_DIR", VIDEOS_DIR)]:
        if path and os.path.isdir(path):
            found_any = True
        elif path:
            print(f"WARNING: {label} not found: {path}")

    if not found_any:
        print("ERROR: Neither IMAGES_DIR nor VIDEOS_DIR exist.")
        print("Edit the paths at the top of this script.")
        sys.exit(1)


def collect_files():
    """Gather all images and videos to caption."""
    files = []

    # Collect images
    if IMAGES_DIR and os.path.isdir(IMAGES_DIR):
        img_dir = Path(IMAGES_DIR)
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(("image", f))

    # Collect videos
    if VIDEOS_DIR and os.path.isdir(VIDEOS_DIR):
        vid_dir = Path(VIDEOS_DIR)
        for f in sorted(vid_dir.iterdir()):
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(("video", f))

    return files


def caption_image(model, image_path):
    """
    Caption a single image.
    Images are small enough to pass directly to Gemini — no upload needed.
    """
    import google.generativeai as genai
    from PIL import Image

    img = Image.open(image_path)
    response = model.generate_content(
        [img, IMAGE_USER_PROMPT],
        request_options={"timeout": 60},
    )
    return response.text.strip()


def caption_video(model, video_path):
    """
    Caption a single video.
    Videos must be uploaded to Gemini's Files API first,
    then referenced in the prompt. Gemini needs time to process them.
    """
    import google.generativeai as genai

    # Upload to Gemini
    video_file = genai.upload_file(
        path=str(video_path),
        display_name=video_path.stem,
    )

    # Wait for Gemini to process the video
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise RuntimeError(f"Gemini rejected video: {video_file.state.name}")

    # Generate caption
    response = model.generate_content(
        [video_file, VIDEO_USER_PROMPT],
        request_options={"timeout": 120},
    )

    # Clean up uploaded file from Gemini's servers
    try:
        genai.delete_file(video_file.name)
    except Exception:
        pass

    return response.text.strip()


def clean_caption(caption):
    """Fix common Gemini quirks."""
    # Remove wrapping quotes
    if caption.startswith('"') and caption.endswith('"'):
        caption = caption[1:-1]
    if caption.startswith("'") and caption.endswith("'"):
        caption = caption[1:-1]

    # Ensure it starts with character name
    if not caption.lower().startswith(CHARACTER_NAME.lower()):
        print(f"  WARNING: Caption missing trigger word, prepending '{CHARACTER_NAME}'")
        caption = f"{CHARACTER_NAME} {caption}"

    return caption


def main():
    validate_config()

    # --- Import and configure Gemini ---
    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: google-generativeai not installed.")
        print("Run: python -m pip install google-generativeai")
        sys.exit(1)

    # Pillow is needed for images
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed.")
        print("Run: python -m pip install Pillow")
        sys.exit(1)

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT_BASE,
    )

    # --- Collect files ---
    all_files = collect_files()

    if not all_files:
        print("No images or videos found.")
        print(f"  IMAGES_DIR: {IMAGES_DIR} (looking for {', '.join(IMAGE_EXTENSIONS)})")
        print(f"  VIDEOS_DIR: {VIDEOS_DIR} (looking for {', '.join(VIDEO_EXTENSIONS)})")
        sys.exit(1)

    # Filter already-captioned
    if SKIP_EXISTING:
        to_process = [(t, f) for t, f in all_files if not f.with_suffix(".txt").exists()]
        skipped = len(all_files) - len(to_process)
        if skipped > 0:
            print(f"Skipping {skipped} already-captioned files")
    else:
        to_process = all_files

    n_images = sum(1 for t, _ in to_process if t == "image")
    n_videos = sum(1 for t, _ in to_process if t == "video")
    print(f"To caption: {n_images} images + {n_videos} videos = {len(to_process)} total")
    print(f"Character: {CHARACTER_NAME}")
    print(f"Model: {GEMINI_MODEL}")
    print("-" * 50)

    # --- Process ---
    success = 0
    failed = 0

    for i, (file_type, file_path) in enumerate(to_process, 1):
        caption_path = file_path.with_suffix(".txt")
        label = "IMG" if file_type == "image" else "VID"
        print(f"\n[{i}/{len(to_process)}] [{label}] {file_path.name}")

        # Retry loop — handles 429 rate limit errors automatically
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if file_type == "image":
                    print(f"  Captioning...", end=" ", flush=True)
                    caption = caption_image(model, file_path)
                else:
                    print(f"  Uploading...", end=" ", flush=True)
                    caption = caption_video(model, file_path)

                caption = clean_caption(caption)

                # Save the .txt file next to the image/video
                caption_path.write_text(caption, encoding="utf-8")
                print(f"done.")
                print(f"  → {caption[:120]}{'...' if len(caption) > 120 else ''}")
                success += 1

                # Wait between requests to stay under rate limit
                # Free tier is 20 RPM — 10 seconds between requests to be safe
                if i < len(to_process):
                    time.sleep(10)

                break  # Success — exit retry loop

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = 45 * (attempt + 1)  # 45s, 90s, 135s, 180s, 225s
                    print(f"  Rate limited. Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        print(f"  FAILED after {max_retries} retries")
                        failed += 1
                else:
                    print(f"ERROR: {e}")
                    failed += 1
                    time.sleep(10)
                    break  # Non-rate-limit error, don't retry

    # --- Summary ---
    print("\n" + "=" * 50)
    print(f"Done! {success} captioned, {failed} failed, "
          f"{len(all_files) - len(to_process)} skipped")
    if IMAGES_DIR and os.path.isdir(IMAGES_DIR):
        print(f"Image captions in: {IMAGES_DIR}")
    if VIDEOS_DIR and os.path.isdir(VIDEOS_DIR):
        print(f"Video captions in: {VIDEOS_DIR}")


if __name__ == "__main__":
    main()
