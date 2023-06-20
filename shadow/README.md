[[back to homepage](/)]

# Shadow

Here we provide some Shadow simulation files for informational purposes---to
give a sense of how our Shadow simulations were run. The simulations are
resource-intensive and we do not expect them to be re-run. The simulations
take between 500 GiB and 1 Tib of RAM to run, and require 1-2 days of runtime
on our 36 core (72 HT) server machines.

The most useful Shadow outputs are the cell traces that are collected during the
simulations, and those are described in more detail on [the data page](/data)
and analyzed on the [ML page](/ml).

Before running the commands below, make sure you have a local copy of the artifact
(`git clone git@github.com:explainwf-popets2023/explainwf-popets2023.github.io.git`).

## Building an image using the Dockerfile

We ran our simulations in a container initially built using the
[Dockerfile](Dockerfile). This file encodes all of the commands used to build
the various software components needed for running the simulations, and it
contains the commit hashes of the versions of the software used in our
simulations. Building an image from the Dockerfile is straightforward. However,
the following two resources need to be dropped in place in order to successfully
build the image.
  - `wget2-socks.tar`: this is a tarball of wget2 after applying our patch (see
    [our wget2 page](/wget2))
  - `tor-gwf-0.4.7.10-1st_hop_signal_only.tar`: this is the Tor patch used to
    collect the cell traces from the entry relays. This component is not
    released publicly due to its sensitive nature.

The image can be built with

    docker build \
        --no-cache \
        --network=host \
        -f Dockerfile \
        -t shadowsim:wfport \
        .

## Running the image

The `shadowsim:wfport` image built above can be run with

    docker run \
        --privileged \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --log-driver none \
        --shm-size=1t \
        --pids-limit -1 \
        --ulimit nofile=10485760:10485760 \
        --ulimit nproc=-1:-1 \
        --ulimit sigpending=-1 \
        --ulimit rtprio=99:99 \
        -v /storage:/mnt:z \
        -it \
        shadowsim:wfport bash

The above bind-mounts the `/storage` directory containing a `wikidata`
subdirectory with a [wikipedia mirror
file](https://dumps.wikimedia.org/other/kiwix/zim/wikipedia/) inside the image
at `/mnt`. From there, we run a sequence of commands depending on which Shadow
config we run.

### Fidelity

The Shadow config for the fidelity simulations in Section 3 of the paper is located
at [fidelity/tornet-net_0.25-load_2.0.tar.xz](fidelity/tornet-net_0.25-load_2.0.tar.xz).

Untar this in `/storage` to make it available inside the docker image. Then
inside the docker image we want to run a sequence of commands to run the Shadow
simulation and parse out the results we want.

    SIM_DIR=/storage/tornet-net_0.25-load_2.0

    tornettools simulate \
        --shadow /opt/bin/shadow \
        --args "--parallelism=36 --template-directory=shadow.data.template" \
        --filename "shadow.config.crawl.yaml" \
        ${SIM_DIR}

    tornettools parse ${SIM_DIR}

    tornettools plot \
        ${SIM_DIR} \
        --prefix ${SIM_DIR}/pdfs

    cat ${SIM_DIR}/shadow.data/hosts/relay970guard/relay970guard.oniontrace.1001.stdout | \
        grep '650 GWF' | \
        cut -d' ' -f7- \
        > ${SIM_DIR}/wget2-traces.log

    tornettools archive ${SIM_DIR}

We repeated this process by running six simulations, the results are further
described on [the data page](/data).

### Sensitivity and Robustness

The Shadow configs for the sensitivity and robustness simulations in Sections 4
and 5 of the paper are located inside of this subdirectory:
[sensitivity_robustness](https://github.com/explainwf-popets2023/explainwf-popets2023.github.io/tree/main/shadow/sensitivity_robustness).

The process of running the simulations is nearly identical to that shown above, but needs to
be done for each of the nine Shadow configuration sim dir tarballs.

The only change from above is the command to extract the `wget2-traces.log` files. The `cat`
command in the above should be replaced with something like:

    for d in ${SIM_DIR}/shadow.data/hosts/relay*[0-9]guard ; do cat ${d}/*oniontrace*stdout ; done | grep '650 GWF' | cut -d' ' -f7- > ${SIM_DIR}/wget2-traces.log

We run each of the nine Shadow configurations twice, the results are further
described on [the data page](/data).
