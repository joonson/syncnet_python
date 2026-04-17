#!/usr/bin/env python3

import sys, time, os, argparse, pickle, subprocess, glob, cv2, logging
import numpy as np
import torch
from shutil import rmtree

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from scenedetect import open_video, SceneManager, ContentDetector
from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

from detectors import S3FD

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "FaceTracker")
parser.add_argument('--data_dir',       type=str, default='data/work', help='Output direcotry')
parser.add_argument('--videofile',      type=str, default='',   help='Input video file')
parser.add_argument('--reference',      type=str, default='',   help='Video reference')
parser.add_argument('--facedet_scale',  type=float, default=0.25, help='Scale factor for face detection')
parser.add_argument('--crop_scale',     type=float, default=0.40, help='Scale bounding box')
parser.add_argument('--min_track',      type=int, default=100,  help='Minimum facetrack duration')
parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate')
parser.add_argument('--num_failed_det', type=int, default=25,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--min_face_size',  type=int, default=100,  help='Minimum face size in pixels')
parser.add_argument('--overwrite',      action='store_true',    help='Overwrite existing output directories')
opt = parser.parse_args()

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))
setattr(opt,'frames_dir',os.path.join(opt.data_dir,'pyframes'))

# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========

def bb_intersection_over_union(boxA, boxB):

  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interArea = max(0, xB - xA) * max(0, yB - yA)

  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

  iou = interArea / float(boxAArea + boxBArea - interArea)

  return iou

# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========

def track_shot(opt,scenefaces):

  iouThres  = 0.5     # Minimum IOU between consecutive face detections
  tracks    = []

  while True:
    track     = []
    for framefaces in scenefaces:
      for face in framefaces:
        if track == []:
          track.append(face)
          framefaces.remove(face)
        elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
          iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
          if iou > iouThres:
            track.append(face)
            framefaces.remove(face)
            continue
        else:
          break

    if track == []:
      break
    elif len(track) > opt.min_track:

      framenum    = np.array([ f['frame'] for f in track ])
      bboxes      = np.array([np.array(f['bbox']) for f in track])

      frame_i   = np.arange(framenum[0],framenum[-1]+1)

      bboxes_i    = []
      for ij in range(0,4):
        interpfn  = interp1d(framenum, bboxes[:,ij])
        bboxes_i.append(interpfn(frame_i))
      bboxes_i  = np.stack(bboxes_i, axis=1)

      if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > opt.min_face_size:
        tracks.append({'frame':frame_i,'bbox':bboxes_i})

  return tracks

# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========

def crop_video(opt,track,cropfile):

  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, opt.frame_rate, (224,224))

  dets = {'x':[], 'y':[], 's':[]}

  for det in track['bbox']:

    dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2)
    dets['y'].append((det[1]+det[3])/2) # crop center x
    dets['x'].append((det[0]+det[2])/2) # crop center y

  # Smooth detections
  dets['s'] = signal.medfilt(dets['s'],kernel_size=13)
  dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
  dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

  for fidx, frame in enumerate(track['frame']):

    cs  = opt.crop_scale

    bs  = dets['s'][fidx]   # Detection box size
    bsi = int(bs*(1+2*cs))  # Pad videos by this amount

    image = cv2.imread(flist[frame])

    frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
    my  = dets['y'][fidx]+bsi  # BBox center Y
    mx  = dets['x'][fidx]+bsi  # BBox center X

    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]

    vOut.write(cv2.resize(face,(224,224)))

  audiotmp    = os.path.join(opt.tmp_dir,opt.reference,'audio.wav')
  audiostart  = (track['frame'][0])/opt.frame_rate
  audioend    = (track['frame'][-1]+1)/opt.frame_rate

  vOut.release()

  # ========== CROP AUDIO FILE ==========

  logger.info('Cropping audio track for %s', cropfile)
  command = ["ffmpeg", "-y", "-loglevel", "error", "-i",
             os.path.join(opt.avi_dir, opt.reference, 'audio.wav'),
             "-ss", "%.3f" % audiostart, "-to", "%.3f" % audioend,
             audiotmp]
  subprocess.run(command, check=True)

  sample_rate, audio = wavfile.read(audiotmp)

  # ========== COMBINE AUDIO AND VIDEO FILES ==========

  logger.info('Merging audio and video for %s', cropfile)
  command = ["ffmpeg", "-y", "-loglevel", "error", "-i", cropfile+'t.avi', "-i", audiotmp,
             "-c:v", "copy", "-c:a", "copy", cropfile+'.avi']
  subprocess.run(command, check=True)

  logger.info('Written %s', cropfile)

  os.remove(cropfile+'t.avi')

  logger.info('Mean pos: x %.2f y %.2f s %.2f', np.mean(dets['x']),np.mean(dets['y']),np.mean(dets['s']))

  return {'track':track, 'proc_track':dets}

# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

def inference_video(opt):

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  DET = S3FD(device=device)

  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()

  dets = []

  with tqdm(enumerate(flist), total=len(flist), desc='Detecting faces') as pbar:
    for fidx, fname in pbar:

      image = cv2.imread(fname)

      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])

      dets.append([])
      for bbox in bboxes:
        dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

      pbar.set_postfix(dets=len(dets[-1]))

  savepath = os.path.join(opt.work_dir,opt.reference,'faces.pckl')

  with open(savepath, 'wb') as fil:
    pickle.dump(dets, fil)

  return dets

# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========

def scene_detect(opt):

  video_path = os.path.join(opt.avi_dir,opt.reference,'video.avi')
  video = open_video(video_path)

  scene_manager = SceneManager()
  scene_manager.add_detector(ContentDetector())
  scene_manager.detect_scenes(video)

  scene_list = scene_manager.get_scene_list()

  savepath = os.path.join(opt.work_dir,opt.reference,'scene.pckl')

  if not scene_list:
    scene_list = [(video.base_timecode, video.base_timecode + video.duration)]

  with open(savepath, 'wb') as fil:
    pickle.dump(scene_list, fil)

  logger.info('%s - scenes detected %d', video_path, len(scene_list))

  return scene_list


# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========

# ========== DELETE EXISTING DIRECTORIES ==========

for d in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
  path = os.path.join(d, opt.reference)
  if os.path.exists(path):
    if not opt.overwrite:
      sys.exit(f"Output directory already exists: {path}. Use --overwrite to overwrite.")
    logger.warning('Overwriting existing directory: %s', path)
    rmtree(path)

# ========== MAKE NEW DIRECTORIES ==========

for d in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
  os.makedirs(os.path.join(d, opt.reference), exist_ok=True)

# ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

logger.info('Converting video to 25fps: %s', opt.videofile)
command = ["ffmpeg", "-y", "-loglevel", "error", "-i", opt.videofile, "-qscale:v", "2", "-async", "1", "-r", "25",
           os.path.join(opt.avi_dir, opt.reference, 'video.avi')]
subprocess.run(command, check=True)

logger.info('Extracting frames from video')
command = ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(opt.avi_dir, opt.reference, 'video.avi'),
           "-qscale:v", "2", "-threads", "1", "-f", "image2",
           os.path.join(opt.frames_dir, opt.reference, '%06d.jpg')]
subprocess.run(command, check=True)

logger.info('Extracting audio from video')
command = ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(opt.avi_dir, opt.reference, 'video.avi'),
           "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
           os.path.join(opt.avi_dir, opt.reference, 'audio.wav')]
subprocess.run(command, check=True)

# ========== FACE DETECTION ==========

faces = inference_video(opt)

# ========== SCENE DETECTION ==========

scene = scene_detect(opt)

# ========== FACE TRACKING ==========

alltracks = []
vidtracks = []

for shot in scene:

  if shot[1].get_frames() - shot[0].get_frames() >= opt.min_track :
    alltracks.extend(track_shot(opt,faces[shot[0].get_frames():shot[1].get_frames()]))

# ========== FACE TRACK CROP ==========

for ii, track in enumerate(alltracks):
  vidtracks.append(crop_video(opt,track,os.path.join(opt.crop_dir,opt.reference,'%05d'%ii)))

# ========== SAVE RESULTS ==========

savepath = os.path.join(opt.work_dir,opt.reference,'tracks.pckl')

with open(savepath, 'wb') as fil:
  pickle.dump(vidtracks, fil)

rmtree(os.path.join(opt.tmp_dir,opt.reference))
