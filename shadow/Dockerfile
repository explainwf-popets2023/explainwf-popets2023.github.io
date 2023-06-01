FROM debian:11-slim as base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
# Define software versions
ENV SHADOW_COMMIT=127ad81a9370aa221838e088967d809fe932165f
ENV TGEN_COMMIT=fa98ebdcd309e1e5a3d9596e58dda05efd99c5e7
ENV ONIONTRACE_COMMIT=f271ead90526b29b3dd7218ce6e56813e3b4dce3
ENV TORNETTOOLS_COMMIT=75e59fab4f4e95fc7636df44202e287d76bfefea
# Get basic packages
RUN apt-get update && apt-get install -y apt-utils git sudo
RUN apt-get install -y locales
RUN dpkg-reconfigure locales
RUN dpkg-reconfigure tzdata

# Build shadow
FROM base as build-shadow
WORKDIR /root/src
RUN git clone --depth=1 -b main https://github.com/shadow/shadow.git
WORKDIR /root/src/shadow
RUN git fetch origin $SHADOW_COMMIT
RUN git checkout FETCH_HEAD
RUN test $(git rev-parse HEAD) = $SHADOW_COMMIT
ENV CC gcc
ENV CXX g++
ENV CONTAINER debian:11-slim
ENV BUILDTYPE release
ENV RUSTPROFILE minimal
RUN ci/container_scripts/install_deps.sh
RUN ci/container_scripts/install_extra_deps.sh
ENV PATH "/root/.cargo/bin:${PATH}"
RUN ./setup build --jobs `nproc` --prefix /opt --clean
RUN ./setup install

# Build tgen
FROM base as build-tgen
RUN apt-get update && apt-get install -y \
  cmake \
  libglib2.0-0 \
  libglib2.0-dev \
  libigraph-dev
WORKDIR /root/src
RUN git clone --depth=1 -b main https://github.com/shadow/tgen.git
WORKDIR /root/src/tgen
RUN git fetch origin $TGEN_COMMIT
RUN git checkout FETCH_HEAD
RUN test $(git rev-parse HEAD) = $TGEN_COMMIT
WORKDIR /root/src/tgen/build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/opt
RUN make -j`nproc`
RUN make install

# Build oniontrace
FROM base as build-oniontrace
RUN apt-get update && apt-get install -y \
  cmake \
  libglib2.0-0 \
  libglib2.0-dev
WORKDIR /root/src
RUN git clone --depth=1 -b main https://github.com/shadow/oniontrace.git
WORKDIR /root/src/oniontrace
RUN git fetch origin $ONIONTRACE_COMMIT
RUN git checkout FETCH_HEAD
RUN test $(git rev-parse HEAD) = $ONIONTRACE_COMMIT
WORKDIR /root/src/oniontrace/build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/opt
RUN make -j`nproc`
RUN make install

# Build tor
FROM base as build-tor
#COPY tor-gwf-0.4.7.10.tar /root/src/tor-gwf-0.4.7.10.tar
COPY tor-gwf-0.4.7.10-1st_hop_signal_only.tar /root/src/tor-gwf-0.4.7.10-1st_hop_signal_only.tar
RUN apt-get update && apt-get install -y \
  autoconf \
  automake \
  gcc \
  make \
  libevent-dev \
  openssl \
  libssl-dev \
  zlib1g-dev \
  libpsl-dev \
  libpsl5
WORKDIR /root/src
# ARG tor_repo=https://gitlab.torproject.org/tpo/core/tor.git
# ARG tor_branch=tor-0.4.7.10
# RUN GIT_SSL_NO_VERIFY=1 git clone --depth=1 -b $tor_branch $tor_repo
RUN tar xaf tor-gwf-0.4.7.10-1st_hop_signal_only.tar
WORKDIR /root/src/tor-gwf-0.4.7.10-1st_hop_signal_only
RUN bash autogen.sh
ENV LIBS=-lpsl
RUN ./configure \
  --prefix=/opt \
  --disable-unittests \
  --disable-asciidoc \
  --disable-manpage \
  --disable-html-manual \
  --disable-gcc-hardening \
  --disable-linker-hardening \
  --disable-fragile-hardening \
  --disable-expensive-hardening \
  --disable-seccomp \
  --disable-system-torrc \
  --disable-dependency-tracking \
  --disable-lzma \
  --disable-zstd \
  --enable-pic
RUN make -j`nproc`
RUN make install

FROM base as build-tornettools
RUN apt-get update && apt-get install -y \
  git \
  python3 \
  python3-pip \
  python3-venv
# Set up our python venv
ENV PYVENV=/opt
RUN python3 -m venv $PYVENV
ENV PATH=$PYVENV/bin:$PATH
# tornettools
WORKDIR /opt/src
RUN git clone --depth=1 -b main https://github.com/shadow/tornettools.git
WORKDIR /opt/src/tornettools
RUN git fetch origin $TORNETTOOLS_COMMIT
RUN git checkout FETCH_HEAD
RUN test $(git rev-parse HEAD) = $TORNETTOOLS_COMMIT
RUN pip3 install -r requirements.txt
RUN pip3 install -I .
# tgentools (needed for parsing)
WORKDIR /opt/src
RUN git clone --depth=1 -b main https://github.com/shadow/tgen.git
WORKDIR /opt/src/tgen
RUN git fetch origin $TGEN_COMMIT
RUN git checkout FETCH_HEAD
RUN test $(git rev-parse HEAD) = $TGEN_COMMIT
WORKDIR /opt/src/tgen/tools
RUN pip3 install -r requirements.txt
RUN pip3 install -I .
# oniontracetools (needed for parsing)
WORKDIR /opt/src
RUN git clone --depth=1 -b main https://github.com/shadow/oniontrace.git
WORKDIR /opt/src/oniontrace
RUN git fetch origin $ONIONTRACE_COMMIT
RUN git checkout FETCH_HEAD
RUN test $(git rev-parse HEAD) = $ONIONTRACE_COMMIT
WORKDIR /opt/src/oniontrace/tools
RUN pip3 install -r requirements.txt
RUN pip3 install -I .
# we also need the zimply python module for the wikipedia server
RUN pip3 install zimply
# # get the network graph files
# WORKDIR /root/src
# ARG tmodel_repo=https://github.com/tmodel-ccs2018/tmodel-ccs2018.github.io.git
# ARG tmodel_branch=master
# RUN git clone --depth=1 -b $tmodel_branch $tmodel_repo
# WORKDIR /root/src/tmodel-ccs2018.github.io/data/shadow/network
# RUN mkdir -p /root/.local/share
# RUN mv atlas_v201801.shadow_v2.gml.xz /root/.local/share/atlas_v201801.shadow_v2.gml.xz

# build special wget2 that supports socks proxy
FROM base as build-wget2
COPY wget2-socks.tar /root/src/wget2-socks.tar
WORKDIR /root/src
RUN tar xaf wget2-socks.tar
WORKDIR /root/src/wget2-socks
RUN apt-get update && apt-get install -y \
  autoconf \
  automake \
  autogen \
  autopoint \
  libtool \
  gettext \
  gcc \
  make \
  rsync \
  wget \
  tar \
  texinfo \
  pkg-config \
  flex \
  lzip \
  gnupg1 \
  libfl-dev \
  libidn2-dev \
  libpsl-dev \
  libgnutls28-dev \
  libiconv-hook-dev \
  zlib1g-dev \
  libpcre2-dev \
  libgpgme-dev \
  libassuan-dev \
  libnghttp2-dev
RUN git config --global --add safe.directory /root/src/wget2-socks
RUN git config --global --add safe.directory /root/src/wget2-socks/gnulib
RUN ./bootstrap
RUN ./configure \
  --prefix=/opt \
  --enable-threads=posix \
  --disable-doc \
  --without-lzip \
  --without-libhsts
# ENV CPPFLAGS=-DDEBUG
# ENV CFLAGS="-ggdb -O0"
RUN make
RUN make install

FROM base as simulate
WORKDIR /mnt
# python deps for tornettools etc.
RUN apt-get update && apt-get install -y \
  dstat \
  procps \
  xz-utils
# tor runtime deps
RUN apt-get update && apt-get install -y \
  zlib1g-dev \
  libevent-dev \
  openssl \
  libssl-dev \
  libpsl-dev \
  libpsl5
# shadow runtime deps
RUN apt-get update && apt-get install -y \
  libglib2.0-0 \
  libigraph-dev \
  libprocps-dev
# wget2 runtime deps
RUN apt-get update && apt-get install -y \
  gnupg1 \
  libfl-dev \
  libidn2-dev \
  libpsl-dev \
  libgnutls28-dev \
  libiconv-hook-dev \
  zlib1g-dev \
  libpcre2-dev \
  libgpgme-dev \
  libassuan-dev \
  libnghttp2-dev
# perf
#RUN apt-get update && apt-get install -y linux-tools-generic
#RUN mkdir -p /root/.local/bin
#RUN ln -s /usr/lib/linux-tools/5.4.0-81-generic/perf /root/.local/bin/perf
#COPY --from=tornettools-generate /root/data/tornet-* /root/data/tornet
RUN echo fs.nr_open = 104857600 >> /etc/sysctl.conf
RUN echo fs.file-max = 104857600 >> /etc/sysctl.conf
RUN echo vm.max_map_count = 1073741824 >> /etc/sysctl.conf
RUN echo kernel.pid_max = 4194300 >> /etc/sysctl.conf
RUN echo kernel.threads-max = 4194300 >> /etc/sysctl.conf
#RUN sysctl -p
# Get the binaries for our sim tools
COPY --from=build-shadow /opt /opt/
COPY --from=build-tgen /opt /opt/
COPY --from=build-tor /opt /opt/
COPY --from=build-oniontrace /opt /opt/
COPY --from=build-tornettools /opt /opt/
COPY --from=build-wget2 /opt /opt/
# Setup run env
ENV PYVENV=/opt
ENV PATH=/opt/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/lib
# source for testing
# COPY --from=build-shadow /root/src/shadow /opt/src/shadow

# Default command runs the simulation
#CMD tornettools simulate -s /opt/bin/shadow tornet
CMD bash run.sh
