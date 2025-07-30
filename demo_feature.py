#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess
import torch
import numpy as np
import os

from SyncNetInstance import *

def analyze_features(features, save_path):
    """
    Analyze extracted features and provide detailed statistics
    """
    print("\n" + "="*60)
    print("FEATURE ANALYSIS REPORT")
    print("="*60)
    
    # Basic info
    print("\n=== BASIC INFO ===")
    print(f"Features shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")
    print(f"Features device: {features.device}")
    print(f"Number of dimensions: {features.dim()}")
    print(f"Total elements: {features.numel()}")
    print(f"Memory usage: {features.numel() * features.element_size() / (1024*1024):.2f} MB")
    
    # Statistical info
    print("\n=== STATISTICAL INFO ===")
    feats_np = features.numpy()
    print(f"Min value: {feats_np.min():.6f}")
    print(f"Max value: {feats_np.max():.6f}")
    print(f"Mean value: {feats_np.mean():.6f}")
    print(f"Std deviation: {feats_np.std():.6f}")
    print(f"Number of zeros: {np.sum(feats_np == 0)}")
    print(f"Number of NaN values: {np.sum(np.isnan(feats_np))}")
    print(f"Number of infinite values: {np.sum(np.isinf(feats_np))}")
    
    # Feature analysis
    print("\n=== FEATURE ANALYSIS ===")
    num_frames, features_per_frame = features.shape
    print(f"Features per frame: {features_per_frame}")
    print(f"Total frames processed: {num_frames}")
    print(f"Video duration estimate: {num_frames/25:.2f} seconds (assuming 25fps)")
    
    # Feature vector analysis
    print("\n=== FEATURE VECTOR ANALYSIS ===")
    first_frame = feats_np[0]
    non_zero_count = np.sum(first_frame != 0)
    l2_norm = np.linalg.norm(first_frame, ord=2)
    l1_norm = np.linalg.norm(first_frame, ord=1)
    print(f"First frame feature vector stats:")
    print(f"  - Non-zero elements: {non_zero_count}/{features_per_frame}")
    print(f"  - L2 norm: {l2_norm:.6f}")
    print(f"  - L1 norm: {l1_norm:.6f}")
    
    # Temporal analysis
    print("\n=== TEMPORAL ANALYSIS ===")
    frame_norms = np.linalg.norm(feats_np, axis=1, ord=2)
    print(f"Frame-wise L2 norms:")
    print(f"  - Min norm: {frame_norms.min():.6f}")
    print(f"  - Max norm: {frame_norms.max():.6f}")
    print(f"  - Mean norm: {frame_norms.mean():.6f}")
    print(f"  - Std norm: {frame_norms.std():.6f}")
    
    # Feature diversity
    print("\n=== FEATURE DIVERSITY ===")
    feature_stds = np.std(feats_np, axis=0)
    low_variance_count = np.sum(feature_stds < 0.01)
    high_variance_count = np.sum(feature_stds > 1.0)
    most_active_dim = np.argmax(feature_stds)
    least_active_dim = np.argmin(feature_stds)
    
    print(f"Feature dimension statistics:")
    print(f"  - Dimensions with low variance (<0.01): {low_variance_count}")
    print(f"  - Dimensions with high variance (>1.0): {high_variance_count}")
    print(f"  - Most active feature dimension: {most_active_dim} (std: {feature_stds[most_active_dim]:.6f})")
    print(f"  - Least active feature dimension: {least_active_dim} (std: {feature_stds[least_active_dim]:.6f})")
    
    # File info
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path) / (1024*1024)
        print(f"\n=== FILE INFO ===")
        print(f"Saved to: {save_path}")
        print(f"File size: {file_size:.2f} MB")
    
    print("="*60)

# ==================== LOAD PARAMS ====================

parser = argparse.ArgumentParser(description = "SyncNet Feature Extractor");

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='Path to pre-trained SyncNet model');
parser.add_argument('--batch_size', type=int, default='1', help='Fixed to 1 for memory efficiency');
parser.add_argument('--vshift', type=int, default='15', help='Time shift for sync analysis');
parser.add_argument('--videofile', type=str, default="data/example.avi", help='Input video file');
parser.add_argument('--tmp_dir', type=str, default="data", help='Temporary directory');
parser.add_argument('--save_as', type=str, default="data/features.pt", help='Output feature file path');
parser.add_argument('--analyze', action='store_true', help='Enable detailed feature analysis');

opt = parser.parse_args();

# ==================== RUN EVALUATION ====================

# Check if CUDA is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[INFO] Using device: {device}')

s = SyncNetInstance(device=device);

s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);

print(f'[INFO] Extracting features from: {opt.videofile}')
feats = s.extract_feature(opt, videofile=opt.videofile)

print(f'[INFO] Saving features to: {opt.save_as}')
torch.save(feats, opt.save_as)

# Perform detailed analysis
if opt.analyze:
    analyze_features(feats, opt.save_as)
else:
    print(f'[INFO] Features extracted and saved successfully!')
    print(f'[INFO] Feature shape: {feats.shape}')
    print(f'[INFO] Use --analyze flag for detailed feature analysis')
