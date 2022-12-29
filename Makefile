default: build

help:
	@echo 'Management commands for igmc:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the airflow_pipeline project project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t igmc 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name igmc -v `pwd`:/workspace/IGMC igmc:latest /bin/bash

up: build run

rm: 
	@docker rm igmc

stop:
	@docker stop igmc

reset: stop rm
