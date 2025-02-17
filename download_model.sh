# SyncNet model

# check SYNCNET_MODEL_DIR is set
if [ -z ${SYNCNET_MODEL_DIR+x} ]; then
    echo "SYNCNET_MODEL_DIR is unset"
    exit 1
fi

mkdir -p ${SYNCNET_MODEL_DIR}

syncnet_path=${SYNCNET_MODEL_DIR}/syncnet_v2.model
if [ ! -f ${syncnet_path} ]; then
    wget http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model -O $syncnet_path
fi

sfd_path=${SYNCNET_MODEL_DIR}/s3fd_convertor.pth
if [ ! -f ${sfd_path} ]; then
    wget https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth -O $sfd_path
fi