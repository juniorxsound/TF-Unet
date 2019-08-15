    
dockerfile := "Dockerfile"
docker_image_tag_name := "juniorxsound/tf-unet:latest"

build:
	docker build -f ./$(dockerfile) -t $(docker_image_tag_name) ./

build-clean:
	docker build --no-cache -f ./$(dockerfile) -t $(docker_image_tag_name) ./

train:
	docker run -w /data --rm -it -v `pwd`:/data -t $(docker_image_tag_name) python ./train.py

shell:
	docker run -w /data --rm -it -v `pwd`:/data -t $(docker_image_tag_name) /bin/bash

jupyter:
	docker run -p 8888:8888 -w /data --rm -it -v `pwd`:/data -t $(docker_image_tag_name) jupyter notebook --allow-root \

test-unet:
	docker run -w /data --rm -it -v `pwd`:/data -t $(docker_image_tag_name) python ./src/unet.py
