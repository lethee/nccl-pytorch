CUSTOM_IMG = lethee/nccl-pytorch:23.07.dev03
IMG = nvcr.io/nvidia/pytorch:23.07-py3


NNODES ?= 2
NPROCS ?= 2

MASTER_ADDR ?= localhost
MASTER_PORT = 29500

# Example
#
# node1$ MASTER_ADDR=node1 NNODES=2 NPROCS=2 make master
# node2$ MASTER_ADDR=node1 NNODES=2 NPROCS=2 make worker

all: run

master:
	docker run -it --rm \
	--net=host --gpus=all --shm-size=40G --ipc=host --ulimit memlock=-1 \
	-e MASTER_ADDR=$(MASTER_ADDR) -e MASTER_PORT=$(MASTER_PORT) \
	-e NNODES=$(NNODES) -e NPROCS=$(NPROCS) -e NODE_RANK=0 \
	--name master -v `pwd`:/host $(IMG) /host/run.sh

worker:
	docker run -it --rm \
	--net=host --gpus=all --shm-size=40G --ipc=host --ulimit memlock=-1 \
	-e MASTER_ADDR=$(MASTER_ADDR) -e MASTER_PORT=$(MASTER_PORT) \
	-e NNODES=$(NNODES) -e NPROCS=$(NPROCS) -e NODE_RANK=1 \
	--name worker -v `pwd`:/host $(IMG) /host/run.sh

remote:
	docker run -it --rm \
	--net=host --gpus=all --shm-size=40G --ipc=host --ulimit memlock=-1 \
	-e MASTER_ADDR=$(MASTER_ADDR) -e MASTER_PORT=$(MASTER_PORT) \
	-e NNODES=$(NNODES) -e NPROCS=$(NPROCS) -e NODE_RANK=0 \
	--name master -v `pwd`:/host $(IMG) bash -c 'curl -so- https://raw.githubusercontent.com/lethee/nccl-pytorch/refs/heads/main/run_remote.sh | bash'

push: Dockerfile
	docker build -t $(CUSTOM_IMG) .
	docker push $(CUSTOM_IMG)
	echo "Pushed $(CUSTOM_IMG)"