#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
import numpy
import subprocess
import time
import argparse
import pdb
import cv2
import python_speech_features

from scipy.io import wavfile
from TwoStreamMFCC import *


# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--gpu_id', type=int, default='0', help='');
parser.add_argument('--initial_model', type=str, default="data/syncnet.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--video', type=str, default="", help='');

args = parser.parse_args();


# ==================== Get OFFSET ====================

def calc_pdist(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1

    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift)).data

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.cosine_similarity(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    meandist = torch.mean(torch.stack(dists,1),1)

    return meandist, dists

# ==================== MAIN DEF ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, GPU_ID = 0, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__();

        self.__GPU_ID__ = GPU_ID;
        self.__S__ = S(self.__GPU_ID__, num_layers_in_fc_layers = num_layers_in_fc_layers);

    def evaluate(self, videopath, batch_size=50, vshift=10):

        self.__S__.eval();
        
        # ========== ==========
        # Load video 
        # ========== ==========
        cap = cv2.VideoCapture(videopath)

        frame_num = 1;
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break

            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image_np)

        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Load audio
        # ========== ==========

        audiotmp = '/dev/shm/audio.wav'

        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (videopath,audiotmp))
        output = subprocess.call(command, shell=True, stdout=None)

        sample_rate, audio = wavfile.read(audiotmp)
        mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])

        cc = numpy.expand_dims(numpy.expand_dims(mfcc,axis=0),axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        if float(len(audio))/float(len(images)) != 640 :
            print("Mismatch between the number of audio and video frames. Type 'cont' to continue.")
            pdb.set_trace()
        
        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = len(images)-7
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0,lastframe,batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lip(im_in.cuda(self.__GPU_ID__));
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.__S__.forward_aud(cc_in.cuda(self.__GPU_ID__))
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        print('Compute time %.3f sec.' % (time.time()-tS))

        mdist, dists = calc_pdist(im_feat,cc_feat,vshift=vshift)

        maxval, maxidx = torch.max(mdist,0)

        offset = vshift-maxidx
        conf   = maxval-torch.median(mdist)

        print('AV offset %d, conf %.3f.' % (offset,conf))


    def loadParameters(self, path):
        loaded_state = torch.load(path);

        self_state = self.__S__.state_dict();

        for name, param in loaded_state.items():

            self_state[name].copy_(param);

# ==================== MAKE DIRECTORIES ====================

s = SyncNetInstance(GPU_ID = args.gpu_id);

if(args.initial_model != ""):
    s.loadParameters(args.initial_model);
    print("Model %s loaded."%args.initial_model);

s.evaluate(args.video, batch_size=args.batch_size, vshift=args.vshift)
