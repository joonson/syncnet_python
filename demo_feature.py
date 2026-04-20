#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess
import torch

from SyncNetInstance import *

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--videofile', type=str, default="data/example.avi", help='');
parser.add_argument('--tmp_dir', type=str, default="data", help='');
parser.add_argument('--save_as', type=str, default="data/features.pt", help='');

opt = parser.parse_args();


# ==================== RUN EVALUATION ====================

# Check if CUDA is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[INFO] Using device: {device}')

s = SyncNetInstance(device=device);

s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);

feats = s.extract_feature(opt, videofile=opt.videofile)

torch.save(feats, opt.save_as)
