# ========== [ENHANCED] DATA CONFIGURATION ==========
CONFIG = {
    # ========== [NEW] Data Source ==========
    'use_official_splits': True,       # Use SoccerNet official train/val/test splits

    # ========== [NEW] Custom Loss Weights ==========
    'use_repetition_penalty': True,
    'use_coverage_loss': True,
    'use_diversity_loss': True,
    'repetition_weight': 0.1,
    'coverage_weight': 0.1,
    'diversity_weight': 0.05,

    # ========== [MODIFIED] Data subset for training ==========
    'num_train_events': 3000,            # Number of events for training (None = all)
    'num_val_events': 200,               # Number of events for validation (None = all)
    'num_test_events': 200,             # Number of events for testing (None = all)

    # Video processing
    'clip_duration': 20,               # seconds (±10 around timestamp)
    'num_frames': 16,                  # frames per clip

    # Feature extraction
    'vit_model': 'google/vit-base-patch16-224',  # Pre-trained ViT
    'feature_dim': 768,                # ViT output dimension
    'use_cached_features': True,       # Load if already extracted

    # Vocabulary
    'vocab_size': 5000,                # Max vocabulary size
    'min_word_freq': 2,                # Minimum word frequency to include
    'max_caption_length': 30,          # Max words per caption

    # Model architecture
    'decoder_layers': 6,               # Number of transformer decoder layers
    'num_heads': 8,                    # Attention heads (1 for baseline, 8 for improved)
    'd_model': 512,                    # Model dimension
    'dim_feedforward': 2048,           # FFN dimension
    'dropout': 0.1,                    # Dropout rate

    # ========== [ENHANCED] Training ==========
    'batch_size': 8,                   # Batch size per step
    'gradient_accumulation_steps': 4,  # Effective batch = batch_size * this
    'num_epochs': 50,                   # Training epochs
    'learning_rate': 1e-4,             # Learning rate
    'weight_decay': 1e-5,              # Weight decay
    'use_mixed_precision': True,       # FP16 training (saves 50% memory)
    'gradient_clip_val': 1.0,          # Gradient clipping threshold
    'warmup_steps': 500,               # LR warmup steps

    # ========== [NEW] Checkpointing ==========
    'save_checkpoint_every': 50,       # Save every N events processed
    'resume_from_checkpoint': True,    # Auto-resume if checkpoint exists
    'keep_n_checkpoints': 2,           # Keep last N checkpoints + best

    # ========== [NEW] Early Stopping ==========
    'early_stopping_patience': 5,      # Stop if no improvement for N epochs
    'early_stopping_metric': 'bleu',   # Metric to monitor

    # ========== [ENHANCED] Evaluation ==========
    'beam_size': 3,                    # Beam search size for inference
    'compute_cider': True,             # Compute CIDEr score
    'compute_meteor': True,            # Compute METEOR score
    'compute_rouge': True,             # Compute ROUGE score
    'compute_perplexity': True,        # Compute perplexity

    # Mode
    'dry_run': False,                   # ⚠️ Set to False for full training
}

# Print configuration
print("="*70)
print("⚡ IMPROVED BASELINE 2 CONFIGURATION")
print("="*70)
print(f"\n🔧 MODE: {'DRY RUN (minimal data)' if CONFIG['dry_run'] else 'FULL TRAINING'}")

print(f"\n📊 DATA:")
print(f"  Official splits: {'YES ✓' if CONFIG['use_official_splits'] else 'NO (random)'}")
print(f"  Train events: {CONFIG['num_train_events'] if CONFIG['num_train_events'] else 'ALL'}")
print(f"  Val events: {CONFIG['num_val_events'] if CONFIG['num_val_events'] else 'ALL'}")
print(f"  Test events: {CONFIG['num_test_events'] if CONFIG['num_test_events'] else 'ALL'}")
print(f"  Frames per clip: {CONFIG['num_frames']}")

print(f"\n🏗️ MODEL:")
print(f"  Decoder layers: {CONFIG['decoder_layers']}")
print(f"  Attention heads: {CONFIG['num_heads']}")
print(f"  Model dim: {CONFIG['d_model']}")

print(f"\n🎯 TRAINING:")
print(f"  Epochs: {CONFIG['num_epochs']}")
print(f"  Batch size: {CONFIG['batch_size']} x {CONFIG['gradient_accumulation_steps']} = {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']} (effective)")
print(f"  Learning rate: {CONFIG['learning_rate']}")

print("\n" + "="*70)
print("💡 To run full training: Set CONFIG['dry_run'] = False")
print("="*70)

"""## Import Libraries & Setup Paths"""

from torch.cuda.amp import autocast, GradScaler  # [NEW] Mixed precision
import time
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTImageProcessor #<-------------------------
from PIL import Image
import cv2
from tqdm import tqdm
from collections import Counter
import pickle
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)

# [NEW] Install advanced metrics
try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    print("✓ Advanced metrics available")
except ImportError:
    print("⚠️ Installing advanced metrics...")
    !pip install pycocoevalcap -q
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    print("✓ Advanced metrics installed")

print("✓ Libraries imported")

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Mount Drive (if in Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("✓ Google Drive mounted")
except:
    print("✓ Running locally")

# Setup paths (reusing from previous work)
# BASE_DIR = "/content/drive/MyDrive/NEU - MS CS/3_SEM/CS - 7150 (DeepLearning)/final_project"
# BASE_DIR = '/content/drive/MyDrive/NEU - MS CS/3_SEM/CS - 7150 (DeepLearning)/final_project_v2'

# Old location (for loading existing features)
OLD_BASE_DIR = '/content/drive/MyDrive/NEU - MS CS/3_SEM/CS - 7150 (DeepLearning)/final_project'
OLD_FEATURES_DIR = os.path.join(OLD_BASE_DIR, 'data', 'features')

# New location (for saving this run's outputs)
BASE_DIR = '/content/drive/MyDrive/NEU - MS CS/3_SEM/CS - 7150 (DeepLearning)/final_project_v2_improved'

DATA_DIR = os.path.join(BASE_DIR, 'data')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TEMP_VIDEO_DIR = '/content/temp_videos'

# Create directories
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
# [NEW] Checkpoint directory
CHECKPOINT_DIR = os.path.join(MODELS_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# At very top of notebook (first cell after imports)
from google.colab import output
output.enable_custom_widget_manager()

"""---

- Memory Management Utilities
"""

"""
Memory Management Utilities
Add this as a new cell after imports in your notebook
"""

import torch
import shutil

def get_gpu_memory():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9
    return 0, 0

def print_gpu_memory(prefix=""):
    """Print GPU memory status"""
    if torch.cuda.is_available():
        allocated, reserved = get_gpu_memory()
        print(f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def clear_gpu_memory():
    """Aggressive GPU memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_drive_space(base_dir):
    """Get available Drive space in GB"""
    try:
        total, used, free = shutil.disk_usage(base_dir)
        return free / 1e9
    except:
        return None

def check_drive_space(base_dir, min_gb=0.5):
    """Check if enough Drive space available"""
    free_gb = get_drive_space(base_dir)
    if free_gb is not None and free_gb < min_gb:
        print(f"⚠️ WARNING: Only {free_gb:.2f}GB free on Drive (need {min_gb}GB)")
        return False
    return True

print("✓ Memory management utilities loaded")

"""- Checkpoint Management Utilities"""

"""
Checkpoint Management Utilities
Add this as a new cell after memory utilities
"""

import torch
import os
import pickle
from datetime import datetime

def get_checkpoint_path(checkpoint_dir, name='latest'):
    """Get checkpoint file path"""
    return os.path.join(checkpoint_dir, f'checkpoint_{name}.pt')

def save_checkpoint(model, optimizer, scaler, epoch, events_processed,
                   best_metric, vocab, train_history, checkpoint_dir, name='latest'):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'events_processed': events_processed,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_metric': best_metric,
        'vocab': vocab,
        'train_history': train_history,
        'timestamp': datetime.now().isoformat()
    }

    path = get_checkpoint_path(checkpoint_dir, name)
    torch.save(checkpoint, path)

    # Also save vocabulary separately for easy access
    if name == 'best':
        vocab_path = os.path.join(checkpoint_dir, 'vocab.pkl')
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)

    return path

def load_checkpoint(model, optimizer=None, scaler=None, checkpoint_dir=None, name='latest', device='cuda'):
    """Load training checkpoint"""
    path = get_checkpoint_path(checkpoint_dir, name)

    if not os.path.exists(path):
        return None

    print(f"\\n📂 Loading checkpoint from {name}...")
    checkpoint = torch.load(path, map_location=device)

    # model.load_state_dict(checkpoint['model_state_dict'])

    # if optimizer is not None:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # if scaler is not None and checkpoint['scaler_state_dict'] is not None:
    #     scaler.load_state_dict(checkpoint['scaler_state_dict'])

    print(f"  ✓ Resumed from epoch {checkpoint['epoch']}")
    print(f"  ✓ Events processed: {len(checkpoint['events_processed'])}")
    print(f"  ✓ Best metric: {checkpoint['best_metric']:.4f}")

    return checkpoint

def cleanup_old_checkpoints(checkpoint_dir, keep_n=2):
    """Keep only N latest checkpoints plus best"""
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_epoch_') and f.endswith('.pt'):
            path = os.path.join(checkpoint_dir, f)
            checkpoints.append((path, os.path.getmtime(path)))

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x[1], reverse=True)

    # Remove old checkpoints (keep only keep_n latest)
    for path, _ in checkpoints[keep_n:]:
        try:
            os.remove(path)
            print(f"  🗑️ Removed old checkpoint: {os.path.basename(path)}")
        except:
            pass

print("✓ Checkpoint management utilities loaded")

"""- Training Utilities"""

"""
Training Utilities
Add this as a new cell after checkpoint utilities
"""

class WarmupScheduler:
    """Learning rate warmup scheduler"""
    def __init__(self, optimizer, warmup_steps, initial_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.initial_lr * (self.step_count / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=3, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

print("✓ Training utilities loaded")

"""## Check Existing Data"""

"""
Check what data we already have to avoid re-downloading/re-processing
"""

print("="*70)
print("CHECKING EXISTING DATA")
print("="*70)

# Check processed annotations
# annotations_file = os.path.join(METADATA_DIR, 'processed_annotations.csv')

# Load annotations from OLD location
OLD_METADATA_DIR = os.path.join(OLD_BASE_DIR, 'data', 'metadata')
annotations_file = os.path.join(OLD_METADATA_DIR, 'processed_annotations.csv')

if os.path.exists(annotations_file):
    df = pd.read_csv(annotations_file)
    print(f"\n✓ Processed annotations found: {len(df)} events")
    print(f"  Unique matches: {df['match_id'].nunique()}")
else:
    print(f"\n✗ Processed annotations not found")
    df = None

# Check sample matches
sample_matches_file = os.path.join(METADATA_DIR, 'sample_matches.txt')
if os.path.exists(sample_matches_file):
    with open(sample_matches_file, 'r') as f:
        sample_matches = [line.strip() for line in f.readlines()]
    print(f"\n✓ Sample matches found: {len(sample_matches)} matches")
    for i, match in enumerate(sample_matches, 1):
        print(f"  {i}. {match[:60]}...")
else:
    print(f"\n✗ Sample matches not found")
    sample_matches = []

# Check videos in runtime
video_files = []
if os.path.exists(TEMP_VIDEO_DIR):
    for root, dirs, files in os.walk(TEMP_VIDEO_DIR):
        for file in files:
            if file.endswith(('.mkv', '.mp4', '.avi')):
                video_files.append(os.path.join(root, file))

if len(video_files) > 0:
    print(f"\n✓ Videos in runtime: {len(video_files)} files")
    total_size = sum(os.path.getsize(vf) for vf in video_files) / (1024**2)
    print(f"  Total size: {total_size:.2f} MB")
else:
    print(f"\n✗ No videos in runtime (will need to download)")

# Check extracted features
feature_files = []
if os.path.exists(FEATURES_DIR):
    feature_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('.pt') or f.endswith('.npy')]

if len(feature_files) > 0:
    print(f"\n✓ Extracted features found: {len(feature_files)} files")
    for f in feature_files[:5]:
        print(f"  - {f}")
    if len(feature_files) > 5:
        print(f"  ... and {len(feature_files) - 5} more")
else:
    print(f"\n✗ No extracted features found")

# [NEW] Check Drive space
free_space = get_drive_space(BASE_DIR)
if free_space:
    print(f"\n💾 Google Drive space: {free_space:.2f}GB free")
    if free_space < 0.5:
        print(f"⚠️ WARNING: Low on Drive space!")

"""---"""

# Load features from old location
train_features_data = torch.load(os.path.join(OLD_FEATURES_DIR, 'train_features.pt'))
val_features_data = torch.load(os.path.join(OLD_FEATURES_DIR, 'val_features.pt'))
test_features_data = torch.load(os.path.join(OLD_FEATURES_DIR, 'test_features.pt'))

# Create dataframes from features (captions are already inside)
train_df = pd.DataFrame([{'match_id': x['match_id'], 'caption': x['caption'], 'timestamp': x['timestamp']} for x in train_features_data])
val_df = pd.DataFrame([{'match_id': x['match_id'], 'caption': x['caption'], 'timestamp': x['timestamp']} for x in val_features_data])
test_df = pd.DataFrame([{'match_id': x['match_id'], 'caption': x['caption'], 'timestamp': x['timestamp']} for x in test_features_data])

print(f"✓ Train: {len(train_df)} events")
print(f"✓ Val: {len(val_df)} events")
print(f"✓ Test: {len(test_df)} events")

"""### Official Splits Loading"""

# !pip install SoccerNet -q

# from SoccerNet.Downloader import SoccerNetDownloader
# import SoccerNet

# print(TEMP_VIDEO_DIR)  # Is this in /content/ or /content/drive/?

# df = pd.read_csv(annotations_file)
# print(df['split'].unique())
# df.head()

# def load_official_splits():
#     """Load splits using official SoccerNet approach"""
#     print("="*70)
#     print("LOADING OFFICIAL SOCCERNET SPLITS")
#     print("="*70)

#     # Download all caption data
#     mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=TEMP_VIDEO_DIR)
#     mySoccerNetDownloader.downloadGames(
#         files=["Labels-caption.json"],
#         split=["train", "valid", "test"],
#         task="caption")

#     # Import official split definitions
#     from SoccerNet.Downloader import getListGames

#     listGames_train = getListGames("train", task="caption")
#     listGames_valid = getListGames("valid", task="caption")
#     listGames_test = getListGames("test", task="caption")

#     print(f"\n📊 Official splits:")
#     print(f"  Train games: {len(listGames_train)}")
#     print(f"  Valid games: {len(listGames_valid)}")
#     print(f"  Test games: {len(listGames_test)}")

#     # Parse JSONs for specific games only
#     def parse_jsons(game_list, split_name):
#         data = []
#         for game in game_list:
#             json_path = os.path.join(TEMP_VIDEO_DIR, game, "Labels-caption.json")
#             if os.path.exists(json_path):
#                 with open(json_path) as f:
#                     captions = json.load(f)
#                     for ann in captions['annotations']:
#                         data.append({
#                             'match_id': game,
#                             'caption': ann['anonymized'],
#                             'timestamp': ann['position'],
#                             'split': split_name
#                         })
#         return pd.DataFrame(data)

#     train_df = parse_jsons(listGames_train, "train")
#     val_df = parse_jsons(listGames_valid, "valid")
#     test_df = parse_jsons(listGames_test, "test")

#     # Apply limits
#     if CONFIG['num_train_events']:
#         train_df = train_df.head(CONFIG['num_train_events'])
#     if CONFIG['num_val_events']:
#         val_df = val_df.head(CONFIG['num_val_events'])
#     if CONFIG['num_test_events']:
#         test_df = test_df.head(CONFIG['num_test_events'])

#     print(f"\n✓ Train: {len(train_df)} events")
#     print(f"✓ Val: {len(val_df)} events")
#     print(f"✓ Test: {len(test_df)} events")

#     return train_df, val_df, test_df

"""# Section 2: Data Selection & Preparation

"""

# # Make sure this cell ran:
# train_df, val_df, test_df = load_official_splits()

train_df

val_df

test_df

# # Get existing video files
# video_files = []
# if os.path.exists(TEMP_VIDEO_DIR):
#     for root, dirs, files in os.walk(TEMP_VIDEO_DIR):
#         video_files.extend([os.path.join(root, f) for f in files if f.endswith('.mkv')])

# """
# Determine which matches we need videos for
# Check if we already have them, or need to download
# """

# print("="*70)
# print("VIDEO REQUIREMENTS CHECK")
# print("="*70)

# # Matches we need for training/validation
# # required_matches = list(set(train_matches + val_matches))
# required_matches = list(set(
#     list(train_df['match_id'].unique()) +
#     list(val_df['match_id'].unique()) +
#     list(test_df['match_id'].unique())
# ))

# print(f"\n🎬 Videos needed: {len(required_matches)} matches")
# for i, match in enumerate(required_matches, 1):
#     print(f"  {i}. {match}")

# # Check which we already have
# have_videos = []
# need_videos = []

# for match in required_matches:
#     match_has_video = False
#     for vf in video_files:
#         if match.replace('/', os.sep) in vf:
#             match_has_video = True
#             break

#     if match_has_video:
#         have_videos.append(match)
#     else:
#         need_videos.append(match)

# print(f"\n✅ Already have videos: {len(have_videos)}")
# for match in have_videos:
#     print(f"  ✓ {match[:60]}...")

# print(f"\n❌ Need to download: {len(need_videos)}")
# for match in need_videos:
#     print(f"  ✗ {match[:60]}...")

# # Estimate download size
# if len(need_videos) > 0:
#     estimated_size_mb = len(need_videos) * 320  # ~320 MB per match (2 halves)
#     print(f"\n📦 Estimated download size: ~{estimated_size_mb} MB ({estimated_size_mb/1024:.2f} GB)")
#     print(f"   Time estimate: ~{len(need_videos) * 4} minutes")

# print("\n" + "="*70)

"""---

# Section 3: ON THE FLY - > Download → Extract → Delete
"""

# import shutil
# shutil.rmtree(FEATURES_DIR)
# os.makedirs(FEATURES_DIR, exist_ok=True)
# print("✓ Cache cleared")

# print("TEST")
# print(len(train_df))

# """
# Preview: Check how many matches need downloading
# """

# print("="*70)
# print("DOWNLOAD PREVIEW")
# print("="*70)

# # Get unique matches needed
# train_matches = train_df['match_id'].unique()
# val_matches = val_df['match_id'].unique()
# test_matches = test_df['match_id'].unique()
# all_matches = list(set(list(train_matches) + list(val_matches) + list(test_matches)))

# print(f"\n📊 Total unique matches: {len(all_matches)}")
# print(f"  Train: {len(train_matches)} matches, {len(train_df)} events")
# print(f"  Val: {len(val_matches)} matches, {len(val_df)} events")
# print(f"  Test: {len(test_matches)} matches, {len(test_df)} events")

# # Check how many already have features cached
# cached_matches = 0
# for match in all_matches:
#     # Check if any event from this match has cached features
#     match_cached = False
#     for split, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
#         events = df[df['match_id'] == match]
#         for _, event in events.iterrows():
#             event_id = f"{event['match_id']}_{event['timestamp']}"
#             feature_path = os.path.join(FEATURES_DIR, f"{split}_{event_id.replace('/', '_')}.pt")
#             if os.path.exists(feature_path):
#                 match_cached = True
#                 break
#         if match_cached:
#             break
#     if match_cached:
#         cached_matches += 1

# need_download = len(all_matches) - cached_matches

# print(f"\n✅ Already cached: {cached_matches} matches")
# print(f"❌ Need to download: {need_download} matches")
# print(f"\n📦 Estimated:")
# print(f"  Download size: ~{need_download * 320} MB (~{need_download * 320 / 1024:.1f} GB)")
# print(f"  Download time: ~{need_download * 2} minutes (~{need_download * 2 / 60:.1f} hours)")
# print(f"  Features size: ~{len(train_df + val_df + test_df) * 0.2:.0f} MB (~{len(train_df + val_df + test_df) * 0.2 / 1024:.1f} GB)")

# print("\n" + "="*70)

# """
# MERGED: Download videos on-the-fly, extract features, delete videos
# Keeps disk usage minimal (~500MB max)
# """

# print("="*70)
# print("ON-THE-FLY VIDEO PROCESSING WITH FEATURE EXTRACTION")
# print("="*70)

# # Load ViT model first
# print("\n🔧 Loading Vision Transformer...")
# from transformers import ViTImageProcessor, ViTModel

# vit_processor = ViTImageProcessor.from_pretrained(CONFIG['vit_model'])
# vit_model = ViTModel.from_pretrained(CONFIG['vit_model'])
# vit_model = vit_model.to(device)
# vit_model.eval()
# print("✓ ViT model loaded")

# # Initialize downloader
# PASSWORD = "s0cc3rn3t"
# os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
# mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=TEMP_VIDEO_DIR)
# mySoccerNetDownloader.password = PASSWORD

# # Feature extraction function (same as before)
# def extract_clip_features_robust(video_file, timestamp_sec, clip_duration=20, num_frames=16):
#     """Extract ViT features from a video clip"""
#     try:
#         cap = cv2.VideoCapture(video_file)
#         if not cap.isOpened():
#             return None

#         fps = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         start_time = max(0, timestamp_sec - clip_duration/2)
#         end_time = timestamp_sec + clip_duration/2

#         start_frame = int(start_time * fps)
#         end_frame = int(end_time * fps)
#         end_frame = min(end_frame, total_frames - 1)

#         frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

#         frames = []
#         for frame_idx in frame_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#             ret, frame = cap.read()
#             if ret:
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 pil_frame = Image.fromarray(frame_rgb)
#                 frames.append(pil_frame)
#             else:
#                 frames.append(None)

#         cap.release()

#         if None in frames or len(frames) != num_frames:
#             return None

#         frame_features = []
#         with torch.no_grad():
#             for frame in frames:
#                 inputs = vit_processor(images=frame, return_tensors="pt").to(device)
#                 outputs = vit_model(**inputs)
#                 cls_feature = outputs.last_hidden_state[:, 0, :]
#                 frame_features.append(cls_feature)

#         features = torch.cat(frame_features, dim=0)
#         return features.cpu()

#     except Exception as e:
#         print(f"\n⚠️ Feature extraction failed: {e}")
#         return None

# # Function to process one match
# def process_match_on_the_fly(match_id, events_df, split_name):
#     """Download match, extract features for all events, delete match"""

#     match_video_dir = os.path.join(TEMP_VIDEO_DIR, match_id)
#     video_files = [
#         os.path.join(match_video_dir, "1_224p.mkv"),
#         os.path.join(match_video_dir, "2_224p.mkv")
#     ]

#     # Check if we need to download (no features exist for this match's events)
#     events_for_match = events_df[events_df['match_id'] == match_id]
#     all_cached = True

#     for _, event in events_for_match.iterrows():
#         event_id = f"{event['match_id']}_{event['timestamp']}"
#         feature_path = os.path.join(FEATURES_DIR, f"{split_name}_{event_id.replace('/', '_')}.pt")
#         if not os.path.exists(feature_path):
#             all_cached = False
#             break

#     if all_cached:
#         # All features cached, skip download
#         features_data = []
#         for _, event in events_for_match.iterrows():
#             event_id = f"{event['match_id']}_{event['timestamp']}"
#             feature_path = os.path.join(FEATURES_DIR, f"{split_name}_{event_id.replace('/', '_')}.pt")
#             features = torch.load(feature_path, map_location='cpu')
#             features_data.append({
#                 'features': features,
#                 'caption': event['caption'],
#                 'match_id': event['match_id'],
#                 'timestamp': event['timestamp']
#             })
#         return features_data, 0

#     # Download videos
#     try:
#         # Suppress download progress bars
#         import sys
#         from io import StringIO

#         old_stdout = sys.stdout
#         sys.stdout = StringIO()  # Redirect stdout
#         # mySoccerNetDownloader.downloadGame(
#         #     files=["1_224p.mkv", "2_224p.mkv"],
#         #     game=match_id
#         #     )
#         try:
#             # Use verbose=False if supported
#             mySoccerNetDownloader.downloadGame(
#                 files=["1_224p.mkv", "2_224p.mkv"],
#                 game=match_id,
#                 verbose=False  # Try this
#             )
#         except TypeError:
#             # If verbose not supported, download normally
#             mySoccerNetDownloader.downloadGame(
#                 files=["1_224p.mkv", "2_224p.mkv"],
#                 game=match_id
#             )
#         sys.stdout = old_stdout  # Restore stdout
#     except Exception as e:
#         sys.stdout = old_stdout  # Restore on error
#         print(f"\n⚠️ Download failed for {match_id}: {e}")
#         return [], len(events_for_match)

#     # Extract features for all events in this match
#     features_data = []
#     failed = 0

#     for _, event in events_for_match.iterrows():
#         event_id = f"{event['match_id']}_{event['timestamp']}"
#         feature_path = os.path.join(FEATURES_DIR, f"{split_name}_{event_id.replace('/', '_')}.pt")

#         # Find which half
#         video_path = None
#         for half_file in video_files:
#             if os.path.exists(half_file):
#                 video_path = half_file
#                 break

#         if video_path is None:
#             failed += 1
#             continue

#         # Extract features
#         features = extract_clip_features_robust(video_path, int(event['timestamp']) / 1000)

#         if features is not None:
#             torch.save(features, feature_path)
#             features_data.append({
#                 'features': features,
#                 'caption': event['caption'],
#                 'match_id': event['match_id'],
#                 'timestamp': event['timestamp']
#             })
#         else:
#             failed += 1

#     # DELETE VIDEOS
#     for video_file in video_files:
#         if os.path.exists(video_file):
#             os.remove(video_file)
#             print(f"  🗑️ Deleted: {os.path.basename(video_file)}")

#     return features_data, failed

# # ========== PROCESS EACH SPLIT ==========
# # train_features_file = os.path.join(FEATURES_DIR, 'train_features.pt')
# # val_features_file = os.path.join(FEATURES_DIR, 'val_features.pt')
# # test_features_file = os.path.join(FEATURES_DIR, 'test_features.pt')

# # Load features from OLD location (already extracted)
# train_features_file = os.path.join(OLD_FEATURES_DIR, 'train_features.pt')
# val_features_file = os.path.join(OLD_FEATURES_DIR, 'val_features.pt')
# test_features_file = os.path.join(OLD_FEATURES_DIR, 'test_features.pt')

# # TRAIN
# if CONFIG['use_cached_features'] and os.path.exists(train_features_file):
#     print("\n✓ Loading cached train features...")
#     train_features_data = torch.load(train_features_file)
#     print(f"  Train: {len(train_features_data)} events")
# else:
#     print("\n⚙️ Processing TRAIN matches...")
#     train_features_data = []
#     train_failed = 0
#     train_matches = train_df['match_id'].unique()

#     # for i, match in enumerate(tqdm(train_matches, desc="Train matches"), 1):
#     print(f"\n📊 Processing {len(train_matches)} train matches...")
#     for i, match in enumerate(train_matches, 1):
#         if i % 10 == 0 or i == len(train_matches):
#             print(f"  Progress: {i}/{len(train_matches)} matches ({i/len(train_matches)*100:.1f}%)")
#         features, failed = process_match_on_the_fly(match, train_df, "train")
#         train_features_data.extend(features)
#         train_failed += failed

#         if i % 10 == 0:
#             torch.cuda.empty_cache()

#     torch.save(train_features_data, train_features_file)
#     print(f"\n✓ Train: {len(train_features_data)} extracted, {train_failed} failed")

# # VAL
# if CONFIG['use_cached_features'] and os.path.exists(val_features_file):
#     print("\n✓ Loading cached val features...")
#     val_features_data = torch.load(val_features_file)
#     print(f"  Val: {len(val_features_data)} events")
# else:
#     print("\n⚙️ Processing VAL matches...")
#     val_features_data = []
#     val_failed = 0
#     val_matches = val_df['match_id'].unique()

#     # for i, match in enumerate(tqdm(val_matches, desc="Val matches"), 1):
#     print(f"\n📊 Processing {len(val_matches)} train matches...")
#     for i, match in enumerate(val_matches, 1):
#         if i % 10 == 0 or i == len(val_matches):
#             print(f"  Progress: {i}/{len(val_matches)} matches ({i/len(val_matches)*100:.1f}%)")
#         features, failed = process_match_on_the_fly(match, val_df, "val")
#         val_features_data.extend(features)
#         val_failed += failed

#         if i % 10 == 0:
#             torch.cuda.empty_cache()

#     torch.save(val_features_data, val_features_file)
#     print(f"\n✓ Val: {len(val_features_data)} extracted, {val_failed} failed")

# # TEST
# if CONFIG['use_cached_features'] and os.path.exists(test_features_file):
#     print("\n✓ Loading cached test features...")
#     test_features_data = torch.load(test_features_file)
#     print(f"  Test: {len(test_features_data)} events")
# else:
#     print("\n⚙️ Processing TEST matches...")
#     test_features_data = []
#     test_failed = 0
#     test_matches = test_df['match_id'].unique()

#     # for i, match in enumerate(tqdm(test_matches, desc="Test matches"), 1):
#     print(f"\n📊 Processing {len(test_matches)} train matches...")
#     for i, match in enumerate(test_matches, 1):
#         if i % 10 == 0 or i == len(test_matches):
#             print(f"  Progress: {i}/{len(test_matches)} matches ({i/len(test_matches)*100:.1f}%)")
#         features, failed = process_match_on_the_fly(match, test_df, "test")
#         test_features_data.extend(features)
#         test_failed += failed

#         if i % 10 == 0:
#             torch.cuda.empty_cache()

#     torch.save(test_features_data, test_features_file)
#     print(f"\n✓ Test: {len(test_features_data)} extracted, {test_failed} failed")

# print("\n✅ All features ready! Max disk usage: ~500MB")

"""---

---

# Section 5: Vocabulary & Tokenization

### - Build Vocabulary from Captions
"""

import nltk
nltk.download('punkt_tab')

"""
Build vocabulary from training captions
Convert words to integer IDs for model training
Keep semantic content (players, teams, scores) but clean formatting
"""

print("="*70)
print("BUILDING VOCABULARY (SMART CLEANING)")
print("="*70)

import re
import string

# Special tokens
SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
}

def clean_caption(caption):
    """
    Smart cleaning:
    - Keep [PLAYER] → 'player' (remove brackets, keep word)
    - Keep [TEAM] → 'team'
    - Keep parentheses content (team names)
    - Keep meaningful punctuation context
    - Remove only formatting noise
    """
    # Lowercase
    caption = caption.lower()

    # Replace [WORD] with just WORD (keep the semantic token)
    # [PLAYER] → player, [TEAM] → team, [REFEREE] → referee
    caption = re.sub(r'\[([^\]]+)\]', r'\1', caption)

    # Keep parentheses content but remove the parentheses
    # (Arsenal) → Arsenal
    caption = re.sub(r'\(([^\)]+)\)', r'\1', caption)

    # Remove only problematic punctuation (dots, commas at end)
    # But keep hyphens in scores like "2-1"
    caption = caption.replace('.', ' ')
    caption = caption.replace(',', ' ')
    caption = caption.replace('!', ' ')
    caption = caption.replace('?', ' ')

    # Remove extra whitespace
    caption = ' '.join(caption.split())

    return caption

# Collect all words from training captions
print("\n📝 Analyzing training captions...")

all_words = []
cleaned_captions = []

for sample in train_features_data:
    original = sample['caption']
    cleaned = clean_caption(original)
    cleaned_captions.append(cleaned)

    words = word_tokenize(cleaned)
    all_words.extend(words)

# Show example of cleaning
print(f"\n📋 Example of caption cleaning:")
print(f"  Original: {train_features_data[0]['caption'][:100]}...")
print(f"  Cleaned:  {cleaned_captions[0][:100]}...")

if len(train_features_data) > 1:
    print(f"\n  Original: {train_features_data[1]['caption'][:100]}...")
    print(f"  Cleaned:  {cleaned_captions[1][:100]}...")

print(f"\n  Total words (with repeats): {len(all_words)}")

# Count word frequencies
word_freq = Counter(all_words)
print(f"  Unique words: {len(word_freq)}")

# Filter by minimum frequency (but don't remove numbers or important tokens)
filtered_words = {word: count for word, count in word_freq.items()
                  if count >= CONFIG['min_word_freq']}

print(f"  After filtering (min_freq={CONFIG['min_word_freq']}): {len(filtered_words)}")

# Sort by frequency
sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)

# Take top vocab_size words
vocab_words = [word for word, count in sorted_words[:CONFIG['vocab_size'] - len(SPECIAL_TOKENS)]]

print(f"  Final vocabulary size: {len(vocab_words) + len(SPECIAL_TOKENS)}")

# Build mappings
word2idx = SPECIAL_TOKENS.copy()
for i, word in enumerate(vocab_words):
    word2idx[word] = i + len(SPECIAL_TOKENS)

idx2word = {idx: word for word, idx in word2idx.items()}

print(f"\n📊 Vocabulary statistics:")
print(f"  Total vocabulary: {len(word2idx)}")
print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
print(f"  Regular words: {len(vocab_words)}")

# Show top 30 most frequent words (more context)
print(f"\n🔝 Top 30 most frequent words:")
for i, (word, count) in enumerate(sorted_words[:30], 1):
    print(f"  {i:2d}. '{word}': {count} times")

# Show important semantic tokens
semantic_tokens = [w for w in vocab_words if w in ['player', 'team', 'referee',
                                                     'goal', 'shot', 'pass', 'corner',
                                                     'penalty', 'foul', 'kick', 'striker',
                                                     'midfielder', 'defender', 'goalkeeper',
                                                     'arsenal', 'barcelona', 'milan']]
print(f"\n⚽ Important semantic words found: {semantic_tokens[:20]}")

# Save vocabulary AND cleaned captions
vocab_file = os.path.join(FEATURES_DIR, 'vocabulary.pkl')
with open(vocab_file, 'wb') as f:
    pickle.dump({
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': len(word2idx),
        'cleaned_captions_train': cleaned_captions
    }, f)

# train_features_data['caption']
# # []['caption']#[:101]

cleaned_captions[10]#[:100]

"""---

# Section 6: Tokenize Captions & Create Dataset

###  Tokenize All Captions
"""

"""
Convert captions to token sequences (numbers)
Add <SOS>, <EOS>, and padding
"""

print("="*70)
print("TOKENIZING CAPTIONS")
print("="*70)

def tokenize_caption(caption, word2idx, max_length):
    """
    Convert caption text to sequence of token IDs

    Args:
        caption: Cleaned caption text (string)
        word2idx: Word to index mapping
        max_length: Maximum sequence length

    Returns:
        tokens: List of token IDs with <SOS>, <EOS>, and <PAD>
    """
    # Tokenize words
    words = word_tokenize(caption.lower())  # from nltk.tokenize import word_tokenize

    # Convert words to IDs (use <UNK> for words not in vocab)
    token_ids = []
    for word in words:
        if word in word2idx:
            token_ids.append(word2idx[word])
        else:
            token_ids.append(word2idx['<UNK>'])  # Unknown word

    # Add <SOS> at start and <EOS> at end
    token_ids = [word2idx['<SOS>']] + token_ids + [word2idx['<EOS>']]

    # Truncate if too long
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length-1] + [word2idx['<EOS>']]

    # Pad if too short
    while len(token_ids) < max_length:
        token_ids.append(word2idx['<PAD>'])

    return token_ids

# Load vocabulary
vocab_file = os.path.join(FEATURES_DIR, 'vocabulary.pkl')
with open(vocab_file, 'rb') as f:
    vocab_data = pickle.load(f)
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']
    cleaned_captions_train = vocab_data['cleaned_captions_train']

print(f"\n✓ Vocabulary loaded: {len(word2idx)} words")

# Tokenize training captions
print(f"\n🔢 Tokenizing {len(cleaned_captions_train)} training captions...")

train_tokenized = []
caption_lengths = []

for caption in cleaned_captions_train:
    tokens = tokenize_caption(caption, word2idx, CONFIG['max_caption_length'])
    train_tokenized.append(tokens)

    # Track actual length (excluding padding)
    actual_length = len([t for t in tokens if t != word2idx['<PAD>']])
    caption_lengths.append(actual_length)

# Show examples
print(f"\n📝 Tokenization examples:")

for i in range(min(3, len(cleaned_captions_train))):
    caption = cleaned_captions_train[i]
    tokens = train_tokenized[i]

    print(f"\n  Example {i+1}:")
    print(f"    Text: {caption[:80]}...")
    print(f"    Tokens: {tokens[:15]}...")  # Show first 15 tokens

    # Decode back to verify
    decoded = ' '.join([idx2word[t] for t in tokens if t not in [word2idx['<PAD>']]])
    print(f"    Decoded: {decoded[:80]}...")

# Add tokenized captions to feature data
print(f"\n💾 Adding tokenized captions to feature data...")

for i, sample in enumerate(train_features_data):
    sample['tokens'] = train_tokenized[i]
    sample['cleaned_caption'] = cleaned_captions_train[i]

print(f"✓ Tokenized captions added to {len(train_features_data)} training samples")

# Also need to tokenize validation captions
print(f"\n🔢 Tokenizing validation captions...")

val_tokenized = []
for sample in val_features_data:
    cleaned = clean_caption(sample['caption'])
    tokens = tokenize_caption(cleaned, word2idx, CONFIG['max_caption_length'])
    val_tokenized.append(tokens)

    sample['tokens'] = tokens
    sample['cleaned_caption'] = cleaned

"""---

# Section 7: PyTorch Dataset & DataLoader

### Define Custom Dataset Class
"""

"""
PyTorch Dataset class for video captioning
Handles loading features and tokenized captions in batches
"""

from torch.utils.data import Dataset, DataLoader

class SoccerCaptionDataset(Dataset):
    """
    Custom Dataset for soccer event captioning

    Returns:
        features: Video features [num_frames, feature_dim]
        tokens: Tokenized caption [max_caption_length]
        caption: Original caption text (for reference)
    """

    def __init__(self, features_data):
        """
        Args:
            features_data: List of dicts with 'features', 'tokens', 'caption'
        """
        self.data = features_data

    def __len__(self):
        """Return total number of samples"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get one sample

        Args:
            idx: Sample index

        Returns:
            features: Tensor [num_frames, feature_dim]
            tokens: Tensor [max_caption_length]
            caption: String (for visualization/debugging)
        """
        sample = self.data[idx]

        # Get features (already a tensor, but might be on CPU)
        features = sample['features']  # [16, 768]

        # Convert tokens list to tensor
        tokens = torch.tensor(sample['tokens'], dtype=torch.long)  # [30]

        # Get caption for reference
        caption = sample.get('cleaned_caption', sample['caption'])

        return features, tokens, caption

# Create datasets
train_dataset = SoccerCaptionDataset(train_features_data)
val_dataset = SoccerCaptionDataset(val_features_data)

# Tokenize test captions first
for sample in test_features_data:
    caption = sample['caption']
    tokens = [word2idx.get(word, word2idx['<UNK>']) for word in caption.split()]
    tokens = [word2idx['<SOS>']] + tokens + [word2idx['<EOS>']]

    # Pad or truncate
    if len(tokens) < CONFIG['max_caption_length']:
        tokens = tokens + [word2idx['<PAD>']] * (CONFIG['max_caption_length'] - len(tokens))
    else:
        tokens = tokens[:CONFIG['max_caption_length']]

    sample['tokens'] = tokens  # Add tokens field

# Now create test dataset
test_dataset = SoccerCaptionDataset(test_features_data)

# Test dataset
print(f"\n🧪 Testing dataset...")
sample_features, sample_tokens, sample_caption = train_dataset[0]

# Decode tokens to verify
decoded_words = [idx2word[t.item()] for t in sample_tokens if t.item() != word2idx['<PAD>']]
print(f"    Decoded:        {' '.join(decoded_words)[:80]}...")

"""# Create DataLoaders"""

"""
Create DataLoaders for batching and shuffling
DataLoader handles:
- Batching (combining multiple samples)
- Shuffling (randomize order each epoch)
- Multi-processing (parallel data loading)
"""

print("="*70)
print("CREATING DATALOADERS")
print("="*70)

print(f"\n⚙️  Configuration:")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Shuffle train: Yes")
print(f"  Shuffle val: No")

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,      # Shuffle training data each epoch
    num_workers=0,     # 0 for Colab (avoid multiprocessing issues)
    pin_memory=True if torch.cuda.is_available() else False  # Faster GPU transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,     # Don't shuffle validation
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,     # Don't shuffle test data
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

# Calculate batches
train_batches = len(train_dataset) // CONFIG['batch_size']
if len(train_dataset) % CONFIG['batch_size'] != 0:
    train_batches += 1

val_batches = len(val_dataset) // CONFIG['batch_size']
if len(val_dataset) % CONFIG['batch_size'] != 0:
    val_batches += 1

# Test dataloader
print(f"\n🧪 Testing dataloader...")

for batch_features, batch_tokens, batch_captions in train_loader:
    print(f"\n  First batch:")
    print(f"    Features shape: {batch_features.shape}")  # [batch_size, 16, 768]
    print(f"    Tokens shape:   {batch_tokens.shape}")    # [batch_size, 30]
    print(f"    Num captions:   {len(batch_captions)}")

    print(f"\n  Sample from batch:")
    print(f"    Caption: {batch_captions[0][:80]}...")
    print(f"    Tokens:  {batch_tokens[0][:10].tolist()}...")

    break  # Only test first batch

"""---

# Section 8: Build Decoder Model

### Define Transformer Decoder
"""

"""
Transformer Decoder for Caption Generation
Takes video features as input, generates caption tokens autoregressively
"""

import torch.nn.functional as F

class CaptionDecoder(nn.Module):
    """
    Transformer-based decoder for video captioning

    Architecture:
    - Token embedding layer
    - Positional encoding
    - Transformer decoder layers
    - Output projection to vocabulary
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 dim_feedforward, dropout, max_seq_length, feature_dim):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension (512)
            num_heads: Number of attention heads (1 for baseline)
            num_layers: Number of decoder layers (4)
            dim_feedforward: FFN dimension (2048)
            dropout: Dropout rate (0.1)
            max_seq_length: Max caption length (30)
            feature_dim: Video feature dimension (768 from ViT)
        """
        super(CaptionDecoder, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding: Convert token IDs to vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding: Add position information
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        self.frame_pos_embedding = nn.Embedding(16, d_model)  # 16 frames

        # Project video features to model dimension
        # ViT features are 768-dim, we need d_model (512)
        self.feature_projection = nn.Linear(feature_dim, d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Input: [batch, seq, features]
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Output projection: Convert decoder output to vocabulary logits
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        print(f"\n🏗️  Model Architecture:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Model dim (d_model): {d_model}")
        print(f"  Attention heads: {num_heads}")
        print(f"  Decoder layers: {num_layers}")
        print(f"  FFN dimension: {dim_feedforward}")
        print(f"  Dropout: {dropout}")

    # def forward(self, video_features, target_tokens):
    def forward(self, video_features, target_tokens, return_attention=False):
        """
        Forward pass

        Args:
            video_features: [batch, num_frames, feature_dim] - from ViT
            target_tokens: [batch, seq_length] - tokenized captions
            return_attention: if True, also return attention weights

        Returns:
            logits: [batch, seq_length, vocab_size] - predictions
            attention_weights (optional): [batch, num_heads, tgt_len, src_len]
        """
        batch_size = target_tokens.size(0)
        seq_length = target_tokens.size(1)

        # 1. Project video features to model dimension
        memory = self.feature_projection(video_features)

        # 2. Add temporal positional encoding to frames
        frame_positions = torch.arange(video_features.size(1), device=video_features.device)
        frame_positions = frame_positions.unsqueeze(0).expand(batch_size, -1)
        memory = memory + self.frame_pos_embedding(frame_positions)

        memory = self.dropout(memory)

        # 3. Embed target tokens
        token_embeds = self.token_embedding(target_tokens)

        # 4. Add positional encoding
        positions = torch.arange(seq_length, device=target_tokens.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(positions)

        decoder_input = self.dropout(token_embeds + pos_embeds)

        # 5. Create causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(target_tokens.device)

        # 6. Pass through transformer decoder
        if return_attention:
            # Hack: Enable attention output by temporarily modifying layers
            attention_weights_list = []

            # Manually iterate through decoder layers to capture attention
            output = decoder_input
            for layer in self.transformer_decoder.layers:
                # Self-attention (we don't need this one)
                output2 = layer.self_attn(output, output, output, attn_mask=tgt_mask)[0]
                output = output + layer.dropout1(output2)
                output = layer.norm1(output)

                # Cross-attention (this is what we want!)
                output2, attn_weights = layer.multihead_attn(
                    output, memory, memory,
                    need_weights=True,
                    average_attn_weights=False  # Get per-head weights
                )
                attention_weights_list.append(attn_weights)

                output = output + layer.dropout2(output2)
                output = layer.norm2(output)

                # Feedforward
                output2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
                output = output + layer.dropout3(output2)
                output = layer.norm3(output)

            decoder_output = output

            # Average attention across layers: [batch, heads, tgt_len, src_len]
            attention_weights = torch.stack(attention_weights_list, dim=0).mean(dim=0)
        else:
            decoder_output = self.transformer_decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=tgt_mask
            )
            attention_weights = None

        # 7. Project to vocabulary
        logits = self.output_projection(decoder_output)

        if return_attention:
            return logits, attention_weights
        return logits


    def generate(self, video_features, max_length, start_token, end_token, device):
        """
        Generate caption autoregressively (for inference)

        Args:
            video_features: [1, num_frames, feature_dim]
            max_length: Maximum caption length
            start_token: <SOS> token ID
            end_token: <EOS> token ID
            device: CPU or CUDA

        Returns:
            generated_tokens: List of token IDs
        """
        self.eval()

        with torch.no_grad():
            # Start with <SOS> token
            generated = [start_token]

            for _ in range(max_length - 1):
                # Current sequence
                tokens = torch.tensor([generated], dtype=torch.long).to(device)

                # Get predictions
                logits = self.forward(video_features, tokens)  # [1, len, vocab_size]

                # Get next token (greedy - take highest probability)
                next_token = logits[0, -1, :].argmax().item()

                generated.append(next_token)

                # Stop if <EOS> generated
                if next_token == end_token:
                    break

            return generated

    def beam_search_generate(self, video_features, max_length, start_token, end_token,
                            device, beam_size=3):
        """
        Beam search decoding with repetition penalty
        """
        self.eval()

        # Start with <SOS>
        sequences = [[start_token]]  # List of token lists
        scores = [0.0]               # Log probability scores

        # Project video features once
        memory = self.feature_projection(video_features)
        frame_positions = torch.arange(video_features.size(1), device=device)
        frame_positions = frame_positions.unsqueeze(0)
        memory = memory + self.frame_pos_embedding(frame_positions)

        for _ in range(max_length - 1):
            all_candidates = []

            for seq, score in zip(sequences, scores):
                # Skip if already ended
                if seq[-1] == end_token:
                    all_candidates.append((seq, score))
                    continue

                # Prepare input
                input_tokens = torch.tensor([seq], device=device)
                seq_len = input_tokens.size(1)

                # Get embeddings
                token_embeds = self.token_embedding(input_tokens)
                positions = torch.arange(seq_len, device=device).unsqueeze(0)
                pos_embeds = self.pos_embedding(positions)
                decoder_input = token_embeds + pos_embeds

                # Causal mask
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

                # Decode
                decoder_output = self.transformer_decoder(
                    tgt=decoder_input,
                    memory=memory,
                    tgt_mask=tgt_mask
                )

                # Get next token probabilities
                logits = self.output_projection(decoder_output[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                # Repetition penalty: reduce score for recent tokens
                if len(seq) >= 3:
                    for tok in seq[-3:]:
                        log_probs[tok] -= 0.5

                # Get top beam_size candidates
                topk_probs, topk_ids = torch.topk(log_probs, beam_size)

                for prob, tok_id in zip(topk_probs, topk_ids):
                    new_seq = seq + [tok_id.item()]
                    new_score = score + prob.item()
                    all_candidates.append((new_seq, new_score))

            # Keep top beam_size sequences
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            sequences = [c[0] for c in all_candidates[:beam_size]]
            scores = [c[1] for c in all_candidates[:beam_size]]

            # Stop if all beams ended
            if all(seq[-1] == end_token for seq in sequences):
                break

        # Return best sequence (exclude <SOS>)
        return sequences[0]

print("="*70)
print("BUILDING CAPTION DECODER MODEL")
print("="*70)

# Create model
model = CaptionDecoder(
    vocab_size=len(word2idx),
    d_model=CONFIG['d_model'],
    num_heads=CONFIG['num_heads'],
    num_layers=CONFIG['decoder_layers'],
    dim_feedforward=CONFIG['dim_feedforward'],
    dropout=CONFIG['dropout'],
    max_seq_length=CONFIG['max_caption_length'],
    feature_dim=CONFIG['feature_dim']
)

# Move to GPU
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test forward pass
with torch.no_grad():
    test_features = torch.randn(2, 16, 768).to(device)  # Batch of 2
    test_tokens = torch.randint(0, len(word2idx), (2, 30)).to(device)

    test_output = model(test_features, test_tokens)

    print(f"  Input features: {test_features.shape}")
    print(f"  Input tokens: {test_tokens.shape}")
    print(f"  Output logits: {test_output.shape}")
    print(f"    [batch_size, seq_length, vocab_size]")
    print(f"    [{test_output.size(0)}, {test_output.size(1)}, {test_output.size(2)}]")

"""---

# Custom Loss Functions
"""

"""
Custom Loss Functions for Improved Caption Generation
"""

class CaptionLossWithRegularization(nn.Module):
    """
    Combined loss: CrossEntropy + Repetition + Coverage + Diversity
    """

    def __init__(self, vocab_size, pad_idx,
                 use_repetition=True, use_coverage=True, use_diversity=True,
                 repetition_weight=0.1, coverage_weight=0.1, diversity_weight=0.05):
        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        # Flags
        self.use_repetition = use_repetition
        self.use_coverage = use_coverage
        self.use_diversity = use_diversity

        # Weights
        self.repetition_weight = repetition_weight
        self.coverage_weight = coverage_weight
        self.diversity_weight = diversity_weight

        print(f"✓ Custom Loss initialized:")
        print(f"  Repetition penalty: {use_repetition} (weight={repetition_weight})")
        print(f"  Coverage loss: {use_coverage} (weight={coverage_weight})")
        print(f"  Diversity loss: {use_diversity} (weight={diversity_weight})")

    def soft_repetition_penalty(self, logits):
        """
        Soft repetition penalty using probability distribution.
        Penalizes when model keeps assigning high probability to same tokens.

        logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len, vocab_size = logits.shape

        if seq_len < 3:
            return torch.tensor(0.0, device=logits.device)

        # Get probability distributions
        probs = torch.softmax(logits, dim=-1)  # [batch, seq_len, vocab]

        penalty = 0.0

        # For each position, check similarity with previous positions
        for t in range(2, seq_len):
            # Current position distribution
            curr_prob = probs[:, t, :]  # [batch, vocab]

            # Previous positions (look back 2 steps)
            for prev_t in range(max(0, t-2), t):
                prev_prob = probs[:, prev_t, :]  # [batch, vocab]

                # KL divergence as similarity measure (high = different, low = similar)
                # We want to penalize similarity, so use negative KL or cosine similarity
                similarity = torch.sum(curr_prob * prev_prob, dim=-1)  # [batch]
                penalty = penalty + similarity.mean()

        # Normalize by number of comparisons
        num_comparisons = sum(min(2, t) for t in range(2, seq_len))
        if num_comparisons > 0:
            penalty = penalty / num_comparisons

        return penalty

    def coverage_loss(self, attention_weights):
        """
        Ensure all frames receive attention.

        attention_weights: [batch, num_heads, tgt_len, src_len (16 frames)]
        """
        if attention_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else 'cuda')

        # Average across heads and target positions
        # [batch, num_heads, tgt_len, 16] -> [batch, 16]
        frame_coverage = attention_weights.mean(dim=1).sum(dim=1)  # [batch, 16]

        # Penalize low coverage (want each frame to get some attention)
        # Using negative log: low attention = high penalty
        coverage_loss = -torch.log(frame_coverage + 1e-8).mean()

        return coverage_loss

    def diversity_loss(self, attention_weights):
        """
        Encourage different attention heads to attend differently.

        attention_weights: [batch, num_heads, tgt_len, src_len]
        """
        if attention_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else 'cuda')

        batch_size, num_heads, tgt_len, src_len = attention_weights.shape

        if num_heads < 2:
            return torch.tensor(0.0, device=attention_weights.device)

        # Flatten attention for each head: [batch, num_heads, tgt_len * src_len]
        attn_flat = attention_weights.view(batch_size, num_heads, -1)

        # Compute pairwise cosine similarity between heads
        total_similarity = 0.0
        num_pairs = 0

        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                # Cosine similarity between head i and head j
                head_i = attn_flat[:, i, :]  # [batch, tgt_len * src_len]
                head_j = attn_flat[:, j, :]  # [batch, tgt_len * src_len]

                similarity = torch.nn.functional.cosine_similarity(head_i, head_j, dim=-1)
                total_similarity = total_similarity + similarity.mean()
                num_pairs += 1

        # Average similarity across all pairs
        diversity_loss = total_similarity / num_pairs if num_pairs > 0 else torch.tensor(0.0)

        return diversity_loss

    def forward(self, logits, targets, attention_weights=None):
        """
        Compute combined loss.

        logits: [batch, seq_len, vocab_size]
        targets: [batch, seq_len]
        attention_weights: [batch, num_heads, tgt_len, src_len] or None
        """
        batch_size, seq_len, vocab_size = logits.shape

        # 1. Main cross-entropy loss
        # Shift: predict next token (ignore last prediction, ignore first target)
        ce_loss = self.ce_loss(
            logits[:, :-1, :].contiguous().view(-1, vocab_size),
            targets[:, 1:].contiguous().view(-1)
        )

        # 2. Repetition penalty
        rep_loss = torch.tensor(0.0, device=logits.device)
        if self.use_repetition:
            rep_loss = self.soft_repetition_penalty(logits)

        # 3. Coverage loss
        cov_loss = torch.tensor(0.0, device=logits.device)
        if self.use_coverage and attention_weights is not None:
            cov_loss = self.coverage_loss(attention_weights)

        # 4. Diversity loss
        div_loss = torch.tensor(0.0, device=logits.device)
        if self.use_diversity and attention_weights is not None:
            div_loss = self.diversity_loss(attention_weights)

        # Combined loss
        total_loss = (
            ce_loss
            + self.repetition_weight * rep_loss
            + self.coverage_weight * cov_loss
            - self.diversity_weight * div_loss  # Negative: minimize similarity
        )

        # Return all components for logging
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'rep_loss': rep_loss.item(),
            'cov_loss': cov_loss.item() if attention_weights is not None else 0.0,
            'div_loss': div_loss.item() if attention_weights is not None else 0.0,
            'total_loss': total_loss.item()
        }


print("✓ Custom loss functions defined")

"""---

# Section 9: Training Loop

### Setup Training (Optimizer, Loss, Scheduler)
"""

print(word2idx.keys())

"""
Setup training components:
- Optimizer (Adam)
- Loss function (Cross Entropy)
- Learning rate scheduler
"""
# ========== [ENHANCED] Training Setup ==========

# criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
criterion = CaptionLossWithRegularization(
    vocab_size=len(word2idx),
    pad_idx=word2idx['<PAD>'],
    use_repetition=CONFIG['use_repetition_penalty'],
    use_coverage=CONFIG['use_coverage_loss'],
    use_diversity=CONFIG['use_diversity_loss'],
    repetition_weight=CONFIG['repetition_weight'],
    coverage_weight=CONFIG['coverage_weight'],
    diversity_weight=CONFIG['diversity_weight']
)

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                              weight_decay=CONFIG['weight_decay'])

# [NEW] Mixed precision scaler
scaler = GradScaler() if CONFIG['use_mixed_precision'] else None

# [NEW] Learning rate scheduler
warmup_scheduler = WarmupScheduler(optimizer, CONFIG['warmup_steps'], CONFIG['learning_rate'])

# [NEW] Early stopping
early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'], mode='max')

# Training state
start_epoch = 0
events_processed = set()
best_val_bleu = 0.0
train_history = {'train_loss': [], 'val_loss': [], 'val_bleu': [], 'learning_rates': []}

"""#### Training Loop"""

# Define checkpoint directory
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'models', 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CONFIG['resume_from_checkpoint'] = False

# [NEW] Try to resume from checkpoint
if CONFIG['resume_from_checkpoint']:
    checkpoint = load_checkpoint(model, optimizer, scaler, checkpoint_dir=CHECKPOINT_DIR, name='latest')
    if checkpoint:
        # ========== CHECK VOCAB SIZE FIRST ==========
        checkpoint_vocab_size = checkpoint['model_state_dict']['token_embedding.weight'].shape[0]
        current_vocab_size = len(word2idx)

        if checkpoint_vocab_size != current_vocab_size:
            print("\n⚠️ WARNING: Vocabulary size changed!")
            print(f"  Checkpoint vocab: {checkpoint_vocab_size}")
            print(f"  Current vocab: {current_vocab_size}")
            print("  Cannot resume - starting fresh training...")
            start_epoch = 0
            best_val_bleu = 0.0
            train_history = {'train_loss': [], 'val_loss': [], 'val_bleu': [], 'learning_rates': []}
            events_processed = set()
        else:
            # Vocab matches, check data
            current_train_size = len(train_df)
            current_data_hash = hash(str(sorted(train_df['match_id'].tolist())))

            checkpoint_train_size = checkpoint.get('train_size', current_train_size)
            checkpoint_data_hash = checkpoint.get('data_hash', current_data_hash)

            if checkpoint_train_size != current_train_size or checkpoint_data_hash != current_data_hash:
                print("\n⚠️ WARNING: Training data has changed!")
                print(f"  Checkpoint: {checkpoint_train_size} samples")
                print(f"  Current:    {current_train_size} samples")
                response = input("Continue with old checkpoint anyway? (yes/no): ")
                if response.lower() != 'yes':
                    print("Starting fresh training...")
                    start_epoch = 0
                    best_val_bleu = 0.0
                    train_history = {'train_loss': [], 'val_loss': [], 'val_bleu': [], 'learning_rates': []}
                    events_processed = set()
                else:
                    start_epoch = checkpoint['epoch'] + 1
                    best_val_bleu = checkpoint.get('best_metric', 0.0)
                    train_history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_bleu': [], 'learning_rates': []})
                    events_processed = set(checkpoint.get('events_processed', []))
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
            else:
                # Everything matches, resume normally
                start_epoch = checkpoint['epoch'] + 1
                best_val_bleu = checkpoint.get('best_metric', 0.0)
                train_history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_bleu': [], 'learning_rates': []})
                events_processed = set(checkpoint.get('events_processed', []))
                # NOW load the weights (everything matched)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"  ✓ Resumed from epoch {checkpoint['epoch']}")
    else:
        # No checkpoint found
        start_epoch = 0
        best_val_bleu = 0.0
        train_history = {'train_loss': [], 'val_loss': [], 'val_bleu': [], 'learning_rates': []}
        events_processed = set()
else:
    # Resume disabled
    start_epoch = 0
    best_val_bleu = 0.0
    train_history = {'train_loss': [], 'val_loss': [], 'val_bleu': [], 'learning_rates': []}
    events_processed = set()

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

# ========== [ENHANCED] Main Training Loop ==========

for epoch in range(start_epoch, CONFIG['num_epochs']):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch+1}/{CONFIG['num_epochs']}")
    print(f"{'='*70}")

    # TRAINING PHASE
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

    for batch_idx, batch in enumerate(progress_bar):
        # features = batch['features'].to(device)
        # captions = batch['caption'].to(device)
        features, tokens, captions = batch
        features = features.to(device)
        tokens = tokens.to(device)
        # Forward pass with mixed precision
        if CONFIG['use_mixed_precision']:
            with autocast():
                logits, attn_weights = model(features, tokens, return_attention=True)
                loss, loss_components = criterion(logits, tokens, attn_weights)
                loss = loss / CONFIG['gradient_accumulation_steps']

            scaler.scale(loss).backward()
        else:
            logits, attn_weights = model(features, tokens, return_attention=True)
            loss, loss_components = criterion(logits, tokens, attn_weights)
            loss = loss / CONFIG['gradient_accumulation_steps']
            loss.backward()

        # Log loss components every 50 batches
        if batch_idx % 50 == 0:
            print(f"\n  CE: {loss_components['ce_loss']:.4f} | "
                  f"Rep: {loss_components['rep_loss']:.4f} | "
                  f"Cov: {loss_components['cov_loss']:.4f} | "
                  f"Div: {loss_components['div_loss']:.4f}")

        # Gradient accumulation
        if (batch_idx + 1) % CONFIG['gradient_accumulation_steps'] == 0:
            if CONFIG['use_mixed_precision']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip_val'])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip_val'])
                optimizer.step()

            optimizer.zero_grad()
            warmup_scheduler.step()

        train_loss += loss.item() * CONFIG['gradient_accumulation_steps']

        progress_bar.set_postfix({
            'loss': f"{loss.item() * CONFIG['gradient_accumulation_steps']:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

        # Periodic memory cleanup
        if (batch_idx + 1) % 20 == 0:
            clear_gpu_memory()

    avg_train_loss = train_loss / len(train_loader)
    train_history['train_loss'].append(avg_train_loss)
    train_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

    print(f"\n📊 Training Loss: {avg_train_loss:.4f}")
    print_gpu_memory("  ")

    # VALIDATION PHASE
    print(f"\n{'='*50}")
    print("VALIDATION")
    print(f"{'='*50}")

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            features, tokens, captions = batch
            features = features.to(device)
            tokens = tokens.to(device)

            if CONFIG['use_mixed_precision']:
                with autocast():
                    logits, attn_weights = model(features, tokens, return_attention=True)
                    loss, _ = criterion(logits, tokens, attn_weights)
            else:
                logits, attn_weights = model(features, tokens, return_attention=True)
                loss, _ = criterion(logits, tokens, attn_weights)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    train_history['val_loss'].append(avg_val_loss)

    print(f"\n📊 Validation Loss: {avg_val_loss:.4f}")

    # Quick BLEU evaluation
    print(f"\nComputing validation BLEU...")
    val_bleu = 0.0
    num_val_samples = min(50, len(val_dataset))

    for i in range(num_val_samples):
        sample = val_dataset[i]
        features, tokens, caption = sample
        features = features.unsqueeze(0).to(device)
        # features = sample['features'].unsqueeze(0).to(device)

        generated_ids = [word2idx['<SOS>']]

        for _ in range(CONFIG['max_caption_length'] - 1):
            captions_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)

            with torch.no_grad():
                if CONFIG['use_mixed_precision']:
                    with autocast():
                        logits = model(features, captions_tensor)
                else:
                    logits = model(features, captions_tensor)

                next_token = logits[:, -1, :].argmax(dim=-1).item()

            if next_token == word2idx['<EOS>']:
                break

            generated_ids.append(next_token)

        generated_words = [idx2word.get(idx, '<unk>') for idx in generated_ids[1:]]
        generated_caption = ' '.join(generated_words)

        # reference = [word_tokenize(sample['caption_text'])]
        features, tokens, caption = sample
        reference = [word_tokenize(caption)]
        hypothesis = word_tokenize(generated_caption)

        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
        val_bleu += bleu

    avg_val_bleu = val_bleu / num_val_samples
    train_history['val_bleu'].append(avg_val_bleu)

    print(f"📊 Validation BLEU: {avg_val_bleu:.4f}")

    # CHECKPOINTING
    print(f"\n{'='*50}")
    print("CHECKPOINTING")
    print(f"{'='*50}")

    # Save latest
    save_checkpoint(model, optimizer, scaler, epoch, list(events_processed),
                   best_val_bleu, vocab_data, train_history, checkpoint_dir=CHECKPOINT_DIR, name='latest')
    print(f"✓ Latest checkpoint saved")

    # Save epoch checkpoint
    save_checkpoint(model, optimizer, scaler, epoch, list(events_processed),
                   best_val_bleu, vocab_data, train_history, checkpoint_dir=CHECKPOINT_DIR, name=f'epoch_{epoch+1}')
    print(f"✓ Epoch {epoch+1} checkpoint saved")

    # Save best model
    if avg_val_bleu > best_val_bleu:
        best_val_bleu = avg_val_bleu
        save_checkpoint(model, optimizer, scaler, epoch, list(events_processed),
                       best_val_bleu, vocab_data, train_history, checkpoint_dir=CHECKPOINT_DIR, name='best')
        print(f"🎉 New best model! BLEU: {best_val_bleu:.4f}")

    cleanup_old_checkpoints(keep_n=CONFIG['keep_n_checkpoints'], checkpoint_dir=CHECKPOINT_DIR)

    # Early stopping check
    if early_stopping(avg_val_bleu):
        print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
        break

    clear_gpu_memory()

print(f"\n{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")
print(f"✓ Best validation BLEU: {best_val_bleu:.4f}")

"""---

# Section 10: Evaluation & Results

#### Load Best Model & Generate Captions
"""

# Re-tokenize test data with NEW vocabulary
print("Re-tokenizing test data with new vocabulary...")
for sample in test_features_data:
    caption = sample['caption']
    cleaned = clean_caption(caption)
    tokens = tokenize_caption(cleaned, word2idx, CONFIG['max_caption_length'])
    sample['tokens'] = tokens
    sample['cleaned_caption'] = cleaned

# Recreate test dataset
test_dataset = SoccerCaptionDataset(test_features_data)
print(f"✓ Test dataset re-tokenized: {len(test_dataset)} samples")

"""
Load best model and generate captions for test samples
"""

# Import smoothing function
from nltk.translate.bleu_score import SmoothingFunction
smoothing = SmoothingFunction().method1
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

print("="*70)
print("LOADING BEST MODEL FOR EVALUATION")
print("="*70)

# Load best model
# checkpoint = torch.load(best_model_path)
checkpoint = load_checkpoint(model, checkpoint_dir=CHECKPOINT_DIR, name='best')
if checkpoint:
    print(f"\n✓ Best model loaded (BLEU: {checkpoint.get('best_metric', 0):.4f})")
else:
    print("⚠️ No checkpoint found, using current model")
# model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\n✓ Best model loaded (Epoch {checkpoint['epoch'] + 1})")
# print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
if 'val_loss' in checkpoint:
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
if 'best_metric' in checkpoint:
    print(f"  Best BLEU: {checkpoint['best_metric']:.4f}")

# print(f"\n✓ Metrics saved to {metrics_path}")
print("\n" + "="*70)
print("GENERATING CAPTIONS ON TEST SET")
print("="*70)

# Generate captions for all test samples
print(f"\n🎬 Generating captions for {len(test_dataset)} test samples...")

generated_captions = []
ground_truth_captions = []
bleu_scores_baseline2 = []

test_features_list = []
test_tokens_list = []

for i in tqdm(range(len(test_dataset)), desc="Generating captions"):
    features, tokens, caption = test_dataset[i]

    # Move to device and add batch dimension
    features = features.unsqueeze(0).to(device)  # [1, 16, 768]

    # Generate caption
    # generated_tokens = model.generate(
    #     features,
    #     max_length=CONFIG['max_caption_length'],
    #     start_token=word2idx['<SOS>'],
    #     end_token=word2idx['<EOS>'],
    #     device=device
    # )
    generated_tokens = model.beam_search_generate(features, max_length=CONFIG['max_caption_length'], start_token=word2idx['<SOS>'], end_token=word2idx['<EOS>'], device=device, beam_size=3)


    # Decode tokens to text
    generated_words = [idx2word[t] for t in generated_tokens
                      if t not in [word2idx['<PAD>'], word2idx['<SOS>'], word2idx['<EOS>']]]
    generated_caption = ' '.join(generated_words)

    # Ground truth (remove special tokens)
    gt_words = [idx2word[t.item()] for t in tokens
               if t.item() not in [word2idx['<PAD>'], word2idx['<SOS>'], word2idx['<EOS>']]]
    gt_caption = ' '.join(gt_words)

    # Calculate BLEU score
    reference = word_tokenize(gt_caption)
    candidate = word_tokenize(generated_caption)
    bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing)

    generated_captions.append(generated_caption)
    ground_truth_captions.append(gt_caption)
    bleu_scores_baseline2.append(bleu)

    test_features_list.append(features)
    test_tokens_list.append(tokens)

print(f"\n✓ Caption generation complete!")

# Calculate statistics
print("\n" + "="*70)
print("BASELINE 2 EVALUATION RESULTS")
print("="*70)

print(f"\n📊 BLEU Score Statistics:")
print(f"  Mean BLEU-4:   {np.mean(bleu_scores_baseline2):.4f}")
print(f"  Median BLEU-4: {np.median(bleu_scores_baseline2):.4f}")
print(f"  Std Dev:       {np.std(bleu_scores_baseline2):.4f}")
print(f"  Min:           {np.min(bleu_scores_baseline2):.4f}")
print(f"  Max:           {np.max(bleu_scores_baseline2):.4f}")

#---------------------------------------------------------------------------------------

# ========== [NEW] Compute All Metrics ==========
print(f"\n{'='*70}")
print("COMPUTING METRICS")
print(f"{'='*70}")

# Prepare data for advanced metrics
# gts_dict = {i: [test_ground_truth[i]] for i in range(len(test_generated))}
# res_dict = {i: [test_generated[i]] for i in range(len(test_generated))}
gts_dict = {i: [ground_truth_captions[i]] for i in range(len(generated_captions))}
res_dict = {i: [generated_captions[i]] for i in range(len(generated_captions))}

# BLEU
avg_bleu = np.mean(bleu_scores_baseline2)
print(f"\n📊 BLEU-4: {avg_bleu:.4f}")

# CIDEr
if CONFIG['compute_cider']:
    try:
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts_dict, res_dict)
        print(f"📊 CIDEr: {cider_score:.4f}")
    except Exception as e:
        print(f"⚠️ CIDEr failed: {e}")
        cider_score = None

# METEOR
if CONFIG['compute_meteor']:
    try:
        meteor_scorer = Meteor()
        meteor_score, _ = meteor_scorer.compute_score(gts_dict, res_dict)
        print(f"📊 METEOR: {meteor_score:.4f}")
    except Exception as e:
        print(f"⚠️ METEOR failed: {e}")
        meteor_score = None

# ROUGE
if CONFIG['compute_rouge']:
    try:
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts_dict, res_dict)
        print(f"📊 ROUGE-L: {rouge_score:.4f}")
    except Exception as e:
        print(f"⚠️ ROUGE failed: {e}")
        rouge_score = None

# Perplexity
if CONFIG['compute_perplexity']:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Use simple CE loss for perplexity calculation
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'], reduction='sum')

    with torch.no_grad():
        for batch in test_loader:
            features, tokens, captions = batch
            features = features.to(device)
            tokens = tokens.to(device)

            if CONFIG['use_mixed_precision']:
                with autocast():
                    logits = model(features, tokens)  # Full sequence
                    loss = ce_loss_fn(logits[:, :-1, :].reshape(-1, len(word2idx)),
                                      tokens[:, 1:].reshape(-1))
            else:
                logits = model(features, tokens)  # Full sequence
                loss = ce_loss_fn(logits[:, :-1, :].reshape(-1, len(word2idx)),
                                  tokens[:, 1:].reshape(-1))

            non_pad = (tokens[:, 1:] != word2idx['<PAD>']).sum().item()
            total_loss += loss.item()
            total_tokens += non_pad

    perplexity = np.exp(total_loss / total_tokens)
    print(f"📊 Perplexity: {perplexity:.2f}")

# Save metrics
metrics_summary = {
    'bleu': avg_bleu,
    'cider': cider_score if CONFIG['compute_cider'] else None,
    'meteor': meteor_score if CONFIG['compute_meteor'] else None,
    'rouge': rouge_score if CONFIG['compute_rouge'] else None,
    'perplexity': perplexity if CONFIG['compute_perplexity'] else None,
    'num_test_samples': len(test_dataset)
}

os.makedirs(RESULTS_DIR, exist_ok=True)
metrics_path = os.path.join(RESULTS_DIR, 'baseline2_metrics_summary.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print(f"\n✓ Metrics saved to {metrics_path}")

# Show some examples
print(f"\n📝 Sample Predictions:")

for i in range(min(5, len(generated_captions))):
    print(f"\n  Sample {i+1}:")
    print(f"    Ground Truth: {ground_truth_captions[i][:80]}...")
    print(f"    Generated:    {generated_captions[i][:80]}...")
    print(f"    BLEU:         {bleu_scores_baseline2[i]:.4f}")

print("\n" + "="*70)
