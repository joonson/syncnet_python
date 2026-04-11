# SyncNet model

mkdir -p data
wget http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model -O data/syncnet_v2.model
wget http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi -O data/example.avi

# For the pre-processing pipeline
mkdir -p detectors/s3fd/weights
wget https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth -O detectors/s3fd/weights/sfd_face.pth