#!/bin/bash

# NOTE: DO NOT CHANGE THE CODE BELOW #
basedir=/media/faststorage           #
if [ ! -d $basedir ] ; then          #
    basedir=/media/datapart          #
fi                                   #
mkdir -p $basedir/$USER/tmp          #
# NOTE: DO NOT CHANGE THE CODE ABOVE #

# Change with your project/container name (no spaces)
project_name=nm_gnn

docker build --build-arg="USERID=$(id -u)" \
    --build-arg="GROUPID=$(id -g)" \
    --build-arg="USERNAME=$USER" \
    --build-arg="REPO_DIR=$(pwd)" \
    --no-cache \
    -t $USER/${project_name}_pyg:24.05-py3 .
docker run -it -d -h $(hostname)_docker --name ${project_name}_${USER}_24.05 \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -u $(id -u):$(id -g) \
    -v /home/$USER:/home/$USER \
    -v /media:/media \
    -v $basedir/$USER/tmp:/tmp/ray \
    -v /mnt:/mnt \
    -w /home/$USER/ \
    $USER/${project_name}_pyg:24.05-py3 bash
