#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip

from SyncNetInstance import *

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnetl2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_dir', type=str, default='/mnt/hdd1/krdemo4', help='');
parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))


# ==================== LOAD MODEL ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);

# ==================== GET OFFSETS ====================

with open(os.path.join(opt.work_dir,opt.reference,'tracks.pckl'), 'rb') as fil:
    tracks = pickle.load(fil)

dists = []
offsets = []
confs = []
for ii, track in enumerate(tracks):
    offset, conf, dist = s.evaluate(opt,videofile=os.path.join(opt.crop_dir,opt.reference,'%05d.avi'%ii))
    offsets.append(offset)
    dists.append(dist)
    confs.append(conf)
      
# ==================== PRINT RESULTS TO FILE ====================

with open(os.path.join(opt.work_dir,opt.reference,'offsets.txt'), 'w') as fil:
    fil.write('FILENAME\tOFFSET\tCONF\n')
    for ii, track in enumerate(tracks):
      fil.write('%05d.avi\t%d\t%.3f\n'%(ii, offsets[ii], confs[ii]))
      
with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
    pickle.dump(dists, fil)
