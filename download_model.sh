# SyncNet model

mkdir data
wget http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnetl2.model -O data/syncnetl2.model
wget http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi -O data/example.avi

# For the pre-processing pipeline

wget http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/face_detection_tf.zip -O facedet.zip

mkdir protos
unzip facedet.zip -d protos/
rm -f facedet.zip

cat /dev/null > protos/__init__.py
cat /dev/null > utils/__init__.py