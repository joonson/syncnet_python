#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import numpy
import time, pdb, argparse, subprocess, pickle, os
import cv2

from scipy import signal

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_dir', type=str, default='data/work', help='');
parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))


# ==================== LOAD FILES ====================

with open(os.path.join(opt.work_dir,opt.reference,'tracks.pckl'), 'rb') as fil:
    tracks = pickle.load(fil, encoding='latin1')

with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'rb') as fil:
    dists = pickle.load(fil, encoding='latin1')

# ==================== SMOOTH FACES ====================

faces = [ [] for i in range(1000000) ]

for ii, track in enumerate(tracks):

	mean_dists =  numpy.mean(numpy.stack(dists[ii],1),1)
	minidx = numpy.argmin(mean_dists,0)
	minval = mean_dists[minidx] 
	
	fdist   	= numpy.stack([dist[minidx] for dist in dists[ii]])
	fdist   	= numpy.pad(fdist, (3,3), 'constant', constant_values=10)

	fconf   = numpy.median(mean_dists) - fdist
	fconfm  = signal.medfilt(fconf,kernel_size=9)

	for ij, frame in enumerate(track[0][0].tolist()) :
		faces[frame].append([ii, fconfm[ij], track[1][0][ij], track[1][1][ij], track[1][2][ij]])

# ==================== ADD DETECTIONS TO VIDEO ====================

cap = cv2.VideoCapture(os.path.join(opt.avi_dir,opt.reference,'video.avi'))
fw = int(cap.get(3))
fh = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vOut = cv2.VideoWriter(os.path.join(opt.avi_dir,opt.reference,'video_only.avi'), fourcc, cap.get(5), (fw,fh))

frame_num=0

while True:
	ret, image = cap.read()
	if ret == 0:
		break

	for face in faces[frame_num]:

		clr = max(min(face[1]*30,255),0)

		cv2.rectangle(image,(int(face[3]-face[2]),int(face[4]-face[2])),(int(face[3]+face[2]),int(face[4]+face[2])),(0,clr,255-clr),3)
		cv2.putText(image,'Track %d, L2 Dist %.3f'%(face[0],face[1]), (int(face[3]-face[2]),int(face[4]-face[2])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

	vOut.write(image)

	print('Frame %d'%frame_num)

	frame_num+=1

cap.release()
vOut.release()

# ========== CROP AUDIO FILE ==========

command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.avi_dir,opt.reference,'audio_only.avi'))) 
output = subprocess.call(command, shell=True, stdout=None)

# ========== COMBINE AUDIO AND VIDEO FILES ==========

command = ("ffmpeg -y -i %s -i %s -c:v copy -c:a copy %s" % (os.path.join(opt.avi_dir,opt.reference,'video_only.avi'),os.path.join(opt.avi_dir,opt.reference,'audio_only.avi'),os.path.join(opt.avi_dir,opt.reference,'video_out.avi'))) #-async 1 
output = subprocess.call(command, shell=True, stdout=None)


