# syncnet_python

## Prerequisites
The following packages are required to run the SyncNet demo:
```
python (2.7.12)
pytorch (0.4.0)
numpy (1.14.3)
scipy (1.0.1)
opencv-python (2.4.9)
python_speech_features (0.6)
cuda (8.0)
ffmpeg (3.4.2)
```

The demo has been tested with the package versions shown above, but may also work on other versions.

## Demo
SyncNet demo:
```
python demo_syncnet.py --videofile data/example.avi
```
Check that this script returns:
```
AV offset:      4 
Min dist:       6.568
Confidence:     9.889
```


## Citation
Please cite the papers below if you make use of the software. 
```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
@InProceedings{Chung17a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Lip Reading in Profile",
  booktitle    = "British Machine Vision Conference",
  year         = "2017",
}
```
