# SyncNet

This repository contains the demo for the audio-to-video synchronisation network (SyncNet). This network can be used for audio-visual synchronisation tasks including: 
1. Removing temporal lags between the audio and visual streams in a video;
2. Determining who is speaking amongst multiple faces in a video. 

The model can be used for research purposes under <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution License</a>. Please cite the paper below if you make use of the software. 

## Prerequisites
The following packages are required to run the SyncNet demo:
```
python (2.7.12)
pytorch (0.4.0)
numpy (1.14.3)
scipy (1.0.1)
opencv-python (3.4.0) - via opencv-contrib-python
python_speech_features (0.6)
cuda (8.0)
ffmpeg (3.4.2)
```

In addition to above, these are required to run the full pipeline:
```
tensorflow (1.2, 1.4)
pyscenedetect (0.5) 
```

The demo has been tested with the package versions shown above, but may also work on other versions.

## Demo

SyncNet demo:
```
python demo_syncnet.py --videofile data/example.avi --tmp_dir /path/to/temp/directory
```

Check that this script returns:
```
AV offset:      3 
Min dist:       5.353
Confidence:     10.021
```

Full pipeline:
```
sh download_model.sh
python run_pipeline.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_syncnet.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_visualise.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
```

Outputs:
```
$DATA_DIR/pycrop/$REFERENCE/*.avi - cropped face tracks
$DATA_DIR/pywork/$REFERENCE/offsets.txt - audio-video offset values
$DATA_DIR/pyavi/$REFERENCE/video_out.avi - output video (as shown below)
```
<p align="center">
  <img src="img/ex1.jpg" width="45%"/>
  <img src="img/ex2.jpg" width="45%"/>
</p>

## Publications
 
```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```
