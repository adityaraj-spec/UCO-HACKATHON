"""
organize_asvspoof_correct.py

Reads ASVspoof 2019 LA and PA datasets.
Extracts bonafide (real) and spoof (fake) files.
Samples speaker-balanced subsets from each train partition.
Copies training data into your existing data/sources/real and data/sources/fake folders.
Copies all dev/eval files into data/held_out/ folders.
"""

import os
import shutil
import random
import math
from collections import defaultdict

random.seed(42)

# ============================================================
# TARGET SYSTEM PATHS (Verified on system)
# ============================================================

# LA paths
LA_BASE         = "E:/ASVSPOOFDATASET/archive/LA/LA"
LA_TRAIN_FLAC   = f"{LA_BASE}/ASVspoof2019_LA_train/flac"
LA_DEV_FLAC     = f"{LA_BASE}/ASVspoof2019_LA_dev/flac"
LA_EVAL_FLAC    = f"{LA_BASE}/ASVspoof2019_LA_eval/flac"

LA_TRAIN_LABELS = f"{LA_BASE}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
LA_DEV_LABELS   = f"{LA_BASE}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
LA_EVAL_LABELS  = f"{LA_BASE}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

# PA paths
PA_BASE         = "E:/ASVSPOOFDATASET/archive/PA/PA"
PA_TRAIN_FLAC   = f"{PA_BASE}/ASVspoof2019_PA_train/flac"
PA_DEV_FLAC     = f"{PA_BASE}/ASVspoof2019_PA_dev/flac"
PA_EVAL_FLAC    = f"{PA_BASE}/ASVspoof2019_PA_eval/flac"

PA_TRAIN_LABELS = f"{PA_BASE}/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt"
PA_DEV_LABELS   = f"{PA_BASE}/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt"
PA_EVAL_LABELS  = f"{PA_BASE}/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt"

# Project Workspace Output folders
TRAIN_REAL      = "data/sources/real"       # for model training (appended)
TRAIN_FAKE      = "data/sources/fake"       # for model training (appended)

HELD_OUT_LA     = "data/held_out/LA"        # you test manually
HELD_OUT_PA     = "data/held_out/PA"        # you test manually


# ============================================================
# READ LABEL FILE
# ============================================================

def read_label_file(label_file_path, partition_type):
    bonafide_by_speaker = defaultdict(list)
    spoof_by_speaker    = defaultdict(list)

    if not os.path.exists(label_file_path):
        print(f"  MISSING: {label_file_path}")
        return bonafide_by_speaker, spoof_by_speaker

    with open(label_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            speaker  = parts[0]
            filename = parts[1]
            label    = parts[4]
            extra    = parts[3] if partition_type == 'LA' else f"{parts[2]}_{parts[3]}"

            entry = {
                'filename' : filename,
                'speaker'  : speaker,
                'extra'    : extra,
                'partition': partition_type
            }

            if label == 'bonafide':
                bonafide_by_speaker[speaker].append(entry)
            else:
                spoof_by_speaker[speaker].append(entry)

    print(f"  {os.path.basename(label_file_path)}")
    print(f"    Speakers: {len(bonafide_by_speaker)} real, {len(spoof_by_speaker)} fake")
    print(f"    Files:    {sum(len(v) for v in bonafide_by_speaker.values())} real, "
          f"{sum(len(v) for v in spoof_by_speaker.values())} fake")

    return bonafide_by_speaker, spoof_by_speaker


# ============================================================
# SPEAKER BALANCED SAMPLE
# ============================================================

def speaker_balanced_sample(by_speaker_dict, target_count):
    if not by_speaker_dict:
        return []

    speakers    = list(by_speaker_dict.keys())
    per_speaker = math.ceil(target_count / len(speakers))
    selected    = []

    for speaker in speakers:
        files = by_speaker_dict[speaker].copy()
        random.shuffle(files)
        selected.extend(files[:per_speaker])

    random.shuffle(selected)
    return selected[:target_count]


# ============================================================
# COPY FILES
# ============================================================

def copy_files(entries, audio_folder, output_folder, prefix):
    os.makedirs(output_folder, exist_ok=True)
    copied  = 0
    missing = 0

    for entry in entries:
        src = None
        for ext in ['.flac', '.wav']:
            candidate = os.path.join(audio_folder, entry['filename'] + ext)
            if os.path.exists(candidate):
                src = candidate
                ext_used = ext
                break

        if src is None:
            missing += 1
            continue

        # Prefix so process_phase1.py identifies source types appropriately
        dst_name = f"{prefix}_{entry['speaker']}_{entry['extra']}_{entry['filename']}{ext_used}"
        dst      = os.path.join(output_folder, dst_name)
        shutil.copy2(src, dst)
        copied += 1

    if missing > 0:
        print(f"    WARNING: {missing} files not found")

    return copied


# ============================================================
# COPY HELD-OUT
# ============================================================

def copy_held_out(
    label_file,
    audio_folder,
    output_base,
    split_name,
    partition_type
):
    print(f"\n  Copying held-out: {partition_type} {split_name}")

    bonafide_by_spk, spoof_by_spk = read_label_file(label_file, partition_type)

    all_real = []
    for entries in bonafide_by_spk.values():
        all_real.extend(entries)

    all_fake = []
    for entries in spoof_by_spk.values():
        all_fake.extend(entries)

    real_out = os.path.join(output_base, split_name, "real")
    fake_out = os.path.join(output_base, split_name, "fake")

    prefix_real = f"{partition_type.lower()}_{split_name}_real"
    prefix_fake = f"{partition_type.lower()}_{split_name}_fake"

    print(f"    Copying {len(all_real)} real files...")
    real_copied = copy_files(all_real, audio_folder, real_out, prefix_real)

    print(f"    Copying {len(all_fake)} fake files...")
    fake_copied = copy_files(all_fake, audio_folder, fake_out, prefix_fake)

    print(f"    Held-out {partition_type} {split_name}: "
          f"{real_copied} real + {fake_copied} fake saved")

    return real_copied, fake_copied


# ============================================================
# COUNT FILES IN FOLDER
# ============================================================

def count_files(folder):
    if not os.path.exists(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if f.endswith('.wav') or f.endswith('.flac')
    ])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("PhaseGuard — ASVspoof Strict Train / Held-Out Separation")
    print("="*60)

    print(f"\nExisting training files:")
    print(f"  Real : {count_files(TRAIN_REAL)}")
    print(f"  Fake : {count_files(TRAIN_FAKE)}")

    total_train_real = 0
    total_train_fake = 0

    print("\n" + "="*60)
    print("PART 1 — TRAINING DATA (LA train + PA train only)")
    print("="*60)

    # --- LA Train ---
    print("\n--- LA Train ---")
    bonafide_by_spk, spoof_by_spk = read_label_file(LA_TRAIN_LABELS, 'LA')

    print(f"\n  Sampling real (target 400)...")
    real_sel = speaker_balanced_sample(bonafide_by_spk, 400)
    r = copy_files(real_sel, LA_TRAIN_FLAC, TRAIN_REAL, "la_train_real")

    print(f"  Sampling fake (target 600)...")
    fake_sel = speaker_balanced_sample(spoof_by_spk, 600)
    f = copy_files(fake_sel, LA_TRAIN_FLAC, TRAIN_FAKE, "la_train_fake")
    print(f"  Copied: {r} real, {f} fake")
    total_train_real += r
    total_train_fake += f

    # --- PA Train ---
    print("\n--- PA Train ---")
    bonafide_by_spk, spoof_by_spk = read_label_file(PA_TRAIN_LABELS, 'PA')

    print(f"\n  Sampling real (target 300)...")
    real_sel = speaker_balanced_sample(bonafide_by_spk, 300)
    r = copy_files(real_sel, PA_TRAIN_FLAC, TRAIN_REAL, "pa_train_real")

    print(f"  Sampling fake (target 400)...")
    fake_sel = speaker_balanced_sample(spoof_by_spk, 400)
    f = copy_files(fake_sel, PA_TRAIN_FLAC, TRAIN_FAKE, "pa_train_fake")
    print(f"  Copied: {r} real, {f} fake")
    total_train_real += r
    total_train_fake += f

    print("\n" + "="*60)
    print("PART 2 — HELD-OUT DATA (dev + eval, ALL files)")
    print("="*60)

    total_held_real = 0
    total_held_fake = 0

    # LA Dev
    r, f = copy_held_out(
        LA_DEV_LABELS, LA_DEV_FLAC,
        HELD_OUT_LA, 'dev', 'LA'
    )
    total_held_real += r
    total_held_fake += f

    # LA Eval
    r, f = copy_held_out(
        LA_EVAL_LABELS, LA_EVAL_FLAC,
        HELD_OUT_LA, 'eval', 'LA'
    )
    total_held_real += r
    total_held_fake += f

    # PA Dev
    r, f = copy_held_out(
        PA_DEV_LABELS, PA_DEV_FLAC,
        HELD_OUT_PA, 'dev', 'PA'
    )
    total_held_real += r
    total_held_fake += f

    # PA Eval
    r, f = copy_held_out(
        PA_EVAL_LABELS, PA_EVAL_FLAC,
        HELD_OUT_PA, 'eval', 'PA'
    )
    total_held_real += r
    total_held_fake += f

    final_real  = count_files(TRAIN_REAL)
    final_fake  = count_files(TRAIN_FAKE)
    total_train = final_real + final_fake

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print(f"\nTRAINING DATA:")
    print(f"  Real folder : {final_real} files")
    print(f"  Fake folder : {final_fake} files")
    print(f"  Total       : {total_train} files")
    if total_train > 0:
        print(f"  Balance     : {final_real/total_train*100:.1f}% real / {final_fake/total_train*100:.1f}% fake")

    print(f"\nHELD-OUT DATA:")
    print(f"  Total held-out: {total_held_real} real + {total_held_fake} fake")
