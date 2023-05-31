#!/bin/bash

LOG=wget2-traces.log.xz
xzcat \
    run1/${LOG} \
    run2/${LOG} \
    run3/${LOG} \
    run4/${LOG} \
    run5/${LOG} \
    run6/${LOG} \
    | xz -T 0 > wget2-traces-tornet-net_0.25-load_2.0.log.xz
