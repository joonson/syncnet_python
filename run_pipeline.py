#!/usr/bin/python

import sys, time, os, pdb, argparse, pickle, subprocess, glob
import numpy as np
import tensorflow as tf
import cv2
from shutil import rmtree

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from utils import label_map_util
from scipy.io import wavfile
from scipy import signal

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "FaceTracker");
parser.add_argument('--data_dir',       type=str, default='data/work', help='Output direcotry');
parser.add_argument('--videofile',      type=str, default='', help='Input video file');
parser.add_argument('--reference',      type=str, default='', help='Name of the video');
parser.add_argument('--crop_scale',     type=float, default=0.5, help='Scale bounding box');
parser.add_argument('--min_track',      type=int, default=100, help='Minimum facetrack duration');
parser.add_argument('--frame_rate',     type=int, default=25, help='Frame rate');
parser.add_argument('--num_failed_det', type=int, default=25, help='Number of missed detections allowed');
parser.add_argument('--min_face_size',  type=float, default=0.03, help='Minimum size of faces');
opt = parser.parse_args();

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

      if np.mean(bboxes_i[:,3]-bboxes_i[:,1]) > opt.min_face_size:
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

  first_image = cv2.imread(flist[track['frame'][0]])

  fw = first_image.shape[1]
  fh = first_image.shape[0]

  dets = {'x':[], 'y':[], 's':[]}

  for det in track['bbox']:

    dets['s'].append(((det[3]-det[1])*fw+(det[2]-det[0])*fh)/4) # H+W / 4
    dets['x'].append((det[1]+det[3])*fw/2) # crop center x 
    dets['y'].append((det[0]+det[2])*fh/2) # crop center y

  # Smooth detections
  dets['s'] = signal.medfilt(dets['s'],kernel_size=7)   
  dets['x'] = signal.medfilt(dets['x'],kernel_size=5)
  dets['y'] = signal.medfilt(dets['y'],kernel_size=5)

  for fidx, frame in enumerate(track['frame']):

    cs  = opt.crop_scale

    bs  = dets['s'][fidx]   # Detection box size
    bsi = int(bs*(1+2*cs))  # Pad videos by this amount 

    image = cv2.imread(flist[frame])
    
    frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(0,0))
    my  = dets['y'][fidx]+bsi  # BBox center Y
    mx  = dets['x'][fidx]+bsi  # BBox center X

    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
    
    vOut.write(cv2.resize(face,(224,224)))

  audiotmp    = os.path.join(opt.tmp_dir,opt.reference,'audio.wav')
  audiostart  = (track['frame'][0])/opt.frame_rate
  audioend    = (track['frame'][-1]+1)/opt.frame_rate

  vOut.release()

  # ========== CROP AUDIO FILE ==========

  command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(opt.avi_dir,opt.reference,'audio.wav'),audiostart,audioend,audiotmp)) 
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    pdb.set_trace()

  sample_rate, audio = wavfile.read(audiotmp)

  # ========== COMBINE AUDIO AND VIDEO FILES ==========

  command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile,audiotmp,cropfile))
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    pdb.set_trace()

  print('Written %s'%cropfile)

  os.remove(cropfile+'t.avi')

  return {'track':track, 'proc_track':dets}

# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

def inference_video(opt):


  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_CKPT = './protos/frozen_inference_graph_face.pb'

  # List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = './protos/face_label_map.pbtxt'

  NUM_CLASSES = 2
  MIN_CONF = 0.3

  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()

  detection_graph = tf.Graph()
  with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')

  dets = []

  with detection_graph.as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=detection_graph, config=config) as sess:
      
      for fidx, fname in enumerate(flist):
        
        image = cv2.imread(fname)

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        
        score = scores[0]

        dets.append([]);
        for index in range(0,len(score)):
          if score[index] > MIN_CONF:
            dets[-1].append({'frame':fidx, 'bbox':boxes[0][index].tolist(), 'conf':score[index]})

        print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),fidx,len(dets[-1]),(1/elapsed_time))) 

      savepath = os.path.join(opt.work_dir,opt.reference,'faces.pckl')

      with open(savepath, 'wb') as fil:
        pickle.dump(dets, fil)

  return dets

# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========

def scene_detect(opt):

  video_manager = VideoManager([os.path.join(opt.avi_dir,opt.reference,'video.avi')])
  stats_manager = StatsManager()
  scene_manager = SceneManager(stats_manager)
  # Add ContentDetector algorithm (constructor takes detector options like threshold).
  scene_manager.add_detector(ContentDetector())
  base_timecode = video_manager.get_base_timecode()

  video_manager.set_downscale_factor()

  video_manager.start()

  scene_manager.detect_scenes(frame_source=video_manager)

  scene_list = scene_manager.get_scene_list(base_timecode)

  savepath = os.path.join(opt.work_dir,'scene.pckl')

  if scene_list == []:
    scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]

  with open(savepath, 'wb') as fil:
    pickle.dump(scene_list, fil)

  print('%s - scenes detected %d'%(os.path.join(opt.avi_dir,opt.reference,'video.avi'),len(scene_list)))

  return scene_list
    

# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========

# ========== DELETE EXISTING DIRECTORIES ==========

if os.path.exists(os.path.join(opt.work_dir,opt.reference)):
  rmtree(os.path.join(opt.work_dir,opt.reference))

if os.path.exists(os.path.join(opt.crop_dir,opt.reference)):
  rmtree(os.path.join(opt.crop_dir,opt.reference))

if os.path.exists(os.path.join(opt.avi_dir,opt.reference)):
  rmtree(os.path.join(opt.avi_dir,opt.reference))

if os.path.exists(os.path.join(opt.frames_dir,opt.reference)):
  rmtree(os.path.join(opt.frames_dir,opt.reference))

if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
  rmtree(os.path.join(opt.tmp_dir,opt.reference))

# ========== MAKE NEW DIRECTORIES ==========

os.makedirs(os.path.join(opt.work_dir,opt.reference))
os.makedirs(os.path.join(opt.crop_dir,opt.reference))
os.makedirs(os.path.join(opt.avi_dir,opt.reference))
os.makedirs(os.path.join(opt.frames_dir,opt.reference))
os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

# ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

command = ("ffmpeg -y -i %s -async 1 -qscale:v 4 -r 25 %s" % (opt.videofile,os.path.join(opt.avi_dir,opt.reference,'video.avi'))) #-async 1  -deinterlace
output = subprocess.call(command, shell=True, stdout=None)

command = ("ffmpeg -y -i %s -threads 1 -f image2 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.frames_dir,opt.reference,'%06d.jpg'))) 
output = subprocess.call(command, shell=True, stdout=None)

command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.avi_dir,opt.reference,'audio.wav'))) 
output = subprocess.call(command, shell=True, stdout=None)

# ========== FACE DETECTION ==========

faces = inference_video(opt)

# ========== SCENE DETECTION ==========

scene = scene_detect(opt)

# ========== FACE TRACKING ==========

alltracks = []
vidtracks = []

for shot in scene:

  if shot[1].frame_num - shot[0].frame_num >= opt.min_track :
    alltracks.extend(track_shot(opt,faces[shot[0].frame_num:shot[1].frame_num]))

# ========== FACE TRACK CROP ==========

for ii, track in enumerate(alltracks):
  vidtracks.append(crop_video(opt,track,os.path.join(opt.crop_dir,opt.reference,'%05d'%ii)))

# ========== SAVE RESULTS ==========

savepath = os.path.join(opt.work_dir,opt.reference,'tracks.pckl')

with open(savepath, 'wb') as fil:
  pickle.dump(vidtracks, fil)

rmtree(os.path.join(opt.tmp_dir,opt.reference))
