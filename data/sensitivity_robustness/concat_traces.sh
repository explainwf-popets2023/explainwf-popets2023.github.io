#!/bin/bash

combine() {
    dir=$1
    xzcat ${dir}/run1/wget2-traces.log.xz ${dir}/run2/wget2-traces.log.xz | \
        xz -T 0 > ${dir}/wget2-traces-combined.log.xz
    xzcat ${dir}/run1/wget2-traces.log.xz | \
        grep "\"port\":80," | \
        xz -T 0 > ${dir}/wget2-traces-single-fetch.log.xz
}

combine tornet-net_0.25-load_1.5-a
combine tornet-net_0.25-load_1.5-b
combine tornet-net_0.25-load_1.5-c
combine tornet-net_0.25-load_2.0-a
combine tornet-net_0.25-load_2.0-b
combine tornet-net_0.25-load_2.0-c
combine tornet-net_0.25-load_2.5-a
combine tornet-net_0.25-load_2.5-b
combine tornet-net_0.25-load_2.5-c
