"""
organize_sources_subfolders.py

Organizes the training files inside data/sources/real/ and data/sources/fake/
into the clean subfolder structure requested by the user:
- data/sources/fake/la_fake/
- data/sources/fake/pa_fake/
- data/sources/fake/wavefake/
- data/sources/real/la_real/
- data/sources/real/pa_real/
- data/sources/real/common_voice/

Cleans up other directories to keep only those used for training/testing.
"""

import os
import shutil

sources_fake = "data/sources/fake"
sources_real = "data/sources/real"

# Define target directories
fake_dirs = {
    'la_fake': os.path.join(sources_fake, 'la_fake'),
    'pa_fake': os.path.join(sources_fake, 'pa_fake'),
    'wavefake': os.path.join(sources_fake, 'wavefake')
}

real_dirs = {
    'la_real': os.path.join(sources_real, 'la_real'),
    'pa_real': os.path.join(sources_real, 'pa_real'),
    'common_voice': os.path.join(sources_real, 'common_voice')
}

# Create the target folders
for d in list(fake_dirs.values()) + list(real_dirs.values()):
    os.makedirs(d, exist_ok=True)

# Helper functions to classify files
def get_fake_target(filename):
    if filename.startswith("la_train_fake"):
        return fake_dirs['la_fake']
    elif filename.startswith("pa_train_fake"):
        return fake_dirs['pa_fake']
    else:
        return fake_dirs['wavefake']

def get_real_target(filename):
    if filename.startswith("la_train_real"):
        return real_dirs['la_real']
    elif filename.startswith("pa_train_real"):
        return real_dirs['pa_real']
    else:
        return real_dirs['common_voice']

# Move all fake files to their respective target folders
print("Organizing fake files...")
fake_moved = 0
for root, dirs, files in os.walk(sources_fake):
    # Skip target directories to prevent moving files inside them
    if any(os.path.abspath(root) == os.path.abspath(target) or os.path.abspath(root).startswith(os.path.abspath(target) + os.sep) for target in fake_dirs.values()):
        continue
    for f in files:
        if f.lower().endswith(('.wav', '.flac')):
            src_path = os.path.join(root, f)
            dst_dir = get_fake_target(f)
            dst_path = os.path.join(dst_dir, f)
            # Avoid conflict if file already in correct place (e.g. root check skipped)
            if src_path != dst_path:
                shutil.move(src_path, dst_path)
                fake_moved += 1

# Move all real files to their respective target folders
print("Organizing real files...")
real_moved = 0
for root, dirs, files in os.walk(sources_real):
    if any(os.path.abspath(root) == os.path.abspath(target) or os.path.abspath(root).startswith(os.path.abspath(target) + os.sep) for target in real_dirs.values()):
        continue
    for f in files:
        if f.lower().endswith(('.wav', '.flac')):
            src_path = os.path.join(root, f)
            dst_dir = get_real_target(f)
            dst_path = os.path.join(dst_dir, f)
            if src_path != dst_path:
                shutil.move(src_path, dst_path)
                real_moved += 1

print(f"Moved: {fake_moved} fake files, {real_moved} real files.")

# Clean up empty/unneeded subfolders
def cleanup_empty_folders(base_path, keep_dirs):
    for root, dirs, files in os.walk(base_path, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            # Check if this dir path is one of the keep paths
            if any(os.path.abspath(dir_path) == os.path.abspath(keep) for keep in keep_dirs):
                continue
            # Delete if empty or contains only non-audio junk files
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                else:
                    # check if contains any wav/flac
                    has_audio = False
                    for r, ds, fs in os.walk(dir_path):
                        if any(fn.lower().endswith(('.wav', '.flac')) for fn in fs):
                            has_audio = True
                            break
                    if not has_audio:
                        shutil.rmtree(dir_path)
            except Exception as e:
                pass

cleanup_empty_folders(sources_fake, fake_dirs.values())
cleanup_empty_folders(sources_real, real_dirs.values())
print("Cleanup complete.")
