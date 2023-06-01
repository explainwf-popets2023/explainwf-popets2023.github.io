# wget2 with SOCKS

This directory contains a custom patch we wrote for wget2 that enables SOCKS
protocol support.

Run `$ make wget2` to clone the wget2 tool repository and apply our patch.

You can run `$ make build` to build the patched version of wget2 if you have all
the necessary prerequisite files installed, or you can run `$ make build_nix` to
build the tool if you have nix installed on the building machine, which will
download the required prerequisites.

To use a SOCKS5 proxy, use the http-proxy swtich: the http-proxy option has been
overwritten. For example, to fetch pages using a SOCKS proxy running on
localhost (127.0.0.1) running on port 9050, provide the switches:

```
--http-proxy=127.0.0.1:9050 --https-proxy=127.0.0.1:9050
```

*NB* This patch is brittle and should not be used in any production code.
