# Use Debian Bullseye Slim as the base image
FROM debian:bullseye-slim

ARG DEBIAN_FRONTEND=noninteractive

# Update the package list and install the basic packages needed for development
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    parallel \
    curl

RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip

COPY requirements.txt .
RUN export MAKEFLAGS="-j$(nproc)" && python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt && \
  rm requirements.txt
