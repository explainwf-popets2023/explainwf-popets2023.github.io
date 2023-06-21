IMAGE_NAME=wfport-img
CONTAINER_NAME=wfport
BUILD_DIR=$(CURDIR)/build

PWD := $(realpath $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))

CODE_DIR=$(PWD)/code
DATA_DIR=$(PWD)/sample_data

all: docker_build
.PHONY: all

$(DATA_DIR):
	mkdir -p $@
	mkdir -p $@/section3/
	mkdir -p $@/section4and5/
	xzcat $(PWD)/../data/fidelity/tornet-net_0.25-load_2.0/run1/wget2-traces.log.xz | head -n 3000 1> $@/section3/wget.log
	xzcat $(PWD)/../data/sensitivity_robustness/tornet-net_0.25-load_1.5-a/run1/wget2-traces.log.xz | head -n 10000 1> $@/section4and5/1.5-a.log
	cp $(PWD)/../data/urls.txt $@/

$(BUILD_DIR): Dockerfile ./requirements.txt
	mkdir $@
	cp Dockerfile $@
	cp ./requirements.txt $@
	touch $@

# Docker
docker_build: $(BUILD_DIR)
	cd $(BUILD_DIR) && docker build -t $(IMAGE_NAME) .
.PHONY: docker_build

DOCKER_CODE_DIR=/mnt/bind/code
DOCKER_DATA_DIR=/mnt/bind/data

docker_run: docker_build $(CODE_DIR) $(DATA_DIR)
	docker run --name $(CONTAINER_NAME) -it \
		--mount type=bind,source=$(CODE_DIR),target=$(DOCKER_CODE_DIR) \
		--mount type=bind,source=$(DATA_DIR),target=$(DOCKER_DATA_DIR) \
		$(IMAGE_NAME) /bin/bash
.PHONY: run_docker

docker_clean:
	docker image rm $(IMAGE_NAME)
	docker rm -f $(CONTAINER_NAME)
.PHONY: clean_docker

# Clean up

clean:
	rm -rf $(BUILD_DIR)
.PHONY: clean

pristine: clean docker_clean
.PHONY: pristine
