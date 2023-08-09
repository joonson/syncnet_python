# SyncNet with RetinaFace for face cropping.

This repository is a fork of the [original SyncNet repository](https://github.com/joonson/syncnet_python) with the addition of Retinaface face detection, instead of S3FD. 
Retinaface is a more accurate and robust face detection algorithm compared to the Haar cascades face detection used in the original repository. 

## Changes

- Added Retinaface face detection for more accurate and robust face detection
- Added a .devcontainer.json and a Dockerfile_retinaface to pre-install all the python dependencies
- Updated the `README.md` file to reflect the changes

This repository contains the demo for the audio-to-video synchronisation network (SyncNet). This network can be used for audio-visual synchronisation tasks including: 
1. Removing temporal lags between the audio and visual streams in a video;
2. Determining who is speaking amongst multiple faces in a video. 

Please cite the paper below if you make use of the software. 

## Dependencies
```
pip install -r requirements.txt
```

In addition, `ffmpeg` is required.


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

## check_synch_videos.py

This script is a batch script that checks a folder with videos and creates a CSV file with quantitative results in terms of video and audio sync offset. The script uses the SyncNetInstance module to check the sync between audio and video.

### Requirements

- Python 3.x
- SyncNetInstance module

### Usage

To use the script, simply run it from the command line and specify the folder containing the videos you want to check for video-audio sync, as well as the path to the CSV file where the results will be saved. For example:

```
python check_synch_videos.py --folder ../data/input/avspeech/train/ --results ../data/preprocessing/avspeechdataset_results.csv
```

This will check all the videos in the specified folder and save the results to the specified CSV file.

### Output

The script creates a CSV file with the following columns:

- `video_file`: the name of the video file
- `av_offset`: the number of frames audio-video offset 
- `min_dist`: the minimum distance between the audio and video signals
- `confidence`: the confidence score of the sync detection algorithm

If the script encounters an error while processing a video file, the `av_offset`, `min_dist`, and `confidence` columns will be set to "error".

You can decide then to use the av_offset value to sync correct the videos, but depending on your purpose and the size of your dataset, you can opt to simply filter out data that has an av_offset larger than a threshold.


## AVdataset_downloader.py
This script is a modified version of an existing script that downloads high quality head talking videos datasets to train Wav2Lip or similar type of machine learning algorithms. The original code can be found here(add link).

To use the script, simply run it from the command line and specify the dataset you want to download. For example:

python avdataset_downloader.py --dataset train

This will download the GRID dataset to the current directory.

### License

This script is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.


## Publications
 
```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```


