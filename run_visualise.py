#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import numpy
import time, pdb, argparse, subprocess, pickle, os, glob
import cv2

from scipy import signal

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--data_dir', 	type=str, default='data/work', help='');
parser.add_argument('--videofile', 	type=str, default='', help='');
parser.add_argument('--reference', 	type=str, default='', help='');
parser.add_argument('--frame_rate', type=int, default=25, help='Frame rate');
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))
setattr(opt,'frames_dir',os.path.join(opt.data_dir,'pyframes'))

# ==================== LOAD FILES ====================

with open(os.path.join(opt.work_dir,opt.reference,'tracks.pckl'), 'rb') as fil:
    tracks = pickle.load(fil, encoding='latin1')

with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'rb') as fil:
    dists = pickle.load(fil, encoding='latin1')

flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
flist.sort()

# ==================== SMOOTH FACES ====================

faces = [[] for i in range(len(flist))]

for tidx, track in enumerate(tracks):

	mean_dists 	=  numpy.mean(numpy.stack(dists[tidx],1),1)
	minidx 		= numpy.argmin(mean_dists,0)
	minval 		= mean_dists[minidx] 
	
	fdist   	= numpy.stack([dist[minidx] for dist in dists[tidx]])
	fdist   	= numpy.pad(fdist, (3,3), 'constant', constant_values=10)

	fconf   = numpy.median(mean_dists) - fdist
	fconfm  = signal.medfilt(fconf,kernel_size=9)

	for fidx, frame in enumerate(track['track']['frame'].tolist()) :
		faces[frame].append({'track': tidx, 'conf':fconfm[fidx], 's':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})

# ==================== ADD DETECTIONS TO VIDEO ====================

first_image = cv2.imread(flist[0])

fw = first_image.shape[1]
fh = first_image.shape[0]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vOut = cv2.VideoWriter(os.path.join(opt.avi_dir,opt.reference,'video_only.avi'), fourcc, opt.frame_rate, (fw,fh))

for fidx, fname in enumerate(flist):

	image = cv2.imread(fname)

	for face in faces[fidx]:

		clr = max(min(face['conf']*25,255),0)

		cv2.rectangle(image,(int(face['x']-face['s']),int(face['y']-face['s'])),(int(face['x']+face['s']),int(face['y']+face['s'])),(0,clr,255-clr),3)
		cv2.putText(image,'Track %d, Conf %.3f'%(face['track'],face['conf']), (int(face['x']-face['s']),int(face['y']-face['s'])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

	vOut.write(image)

	print('Frame %d'%fidx)

vOut.release()

# ========== COMBINE AUDIO AND VIDEO FILES ==========

command = ("ffmpeg -y -i %s -i %s -c:v copy -c:a copy %s" % (os.path.join(opt.avi_dir,opt.reference,'video_only.avi'),os.path.join(opt.avi_dir,opt.reference,'audio.wav'),os.path.join(opt.avi_dir,opt.reference,'video_out.avi'))) #-async 1 
output = subprocess.call(command, shell=True, stdout=None)


