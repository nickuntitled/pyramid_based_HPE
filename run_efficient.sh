#!/bin/bash
sudo docker run -it --shm-size 32G --rm --gpus all -v /media/user/FCCA96CFCA968614/Nick/nick_proposed_technique:/workspace -v /media/user/FCCA96CFCA968614/Nick/datasets:/workspace/datasets nick_public525 /bin/bash
