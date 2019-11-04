    
dockerfile := "docker/Dockerfile.cpu"
docker_image_tag_name := "juniorxsound/tf-unet:latest"
runtime :=

# If NVIDIA SMI is intalled use the GPU docker file and the nvidia-docker-2 runtime
ifneq (, $(shell which nvidia-smi))
	runtime = --gpus all
	dockerfile = "docker/Dockerfile.gpu"
endif

build:
	docker build -f ./$(dockerfile) -t $(docker_image_tag_name) ./

build-clean:
	docker build --no-cache -f ./$(dockerfile) -t $(docker_image_tag_name) ./

train:
	docker run $(runtime) -w /data --rm -it -v `pwd`:/data -t $(docker_image_tag_name) python ./train.py

shell:
	docker run $(runtime) -w /data --rm -it -v `pwd`:/data -t $(docker_image_tag_name) /bin/bash

lint:
	docker run $(runtime) -w /data --rm -it -v `pwd`:/data -t $(docker_image_tag_name) pylint ./unet/unet.py

jupyter:
	docker run $(runtime) -p 8888:8888 -w /data --rm -it -v `pwd`:/data -t $(docker_image_tag_name) jupyter notebook --allow-root \

log:
	docker run -p 6006:6006 -w /data --rm -v `pwd`:/data -t $(docker_image_tag_name) tensorboard --logdir ./logs &

test-unet:
	docker run $(runtime) -w /data --rm -it -v `pwd`:/data -t $(docker_image_tag_name) python ./unet/unet.py