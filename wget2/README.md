# wget2 with SOCKS

This directory contains a custom patch we wrote for wget2 that enables SOCKS
protocol support.

Run `$ make wget2` to clone the wget2 tool repository and apply our patch.

You can run `$ make build` to build the patched version of wget2 if you have all
the necessary prerequisite files installed, or you can run `$ make build_nix` to
build the tool if you have nix installed on the building machine, which will
download the required prerequisites.

*NB* This patch is brittle and should not be used in any production code.
