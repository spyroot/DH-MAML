docker run --privileged --gpus all -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/code spyroot/meta_critic:latest bash
