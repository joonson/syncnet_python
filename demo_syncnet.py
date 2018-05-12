#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess

from SyncNetInstance import *

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, default="data/syncnetl2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--videofile', type=str, default="", help='');
parser.add_argument('--tmp_dir', type=str, default="~", help='');

opt = parser.parse_args();


# ==================== RUN EVALUATION ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);

s.evaluate(opt, videofile=opt.videofile)
