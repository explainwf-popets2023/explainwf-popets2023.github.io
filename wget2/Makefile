all: wget2
.PHONY: all

wget2:
	git clone https://gitlab.com/gnuwget/wget2.git
	cd wget2 && git checkout edfd08 && git am ../socks.patch

build: wget2
	cd wget2 && ./bootstrap && ./configure && make -j

build_nix: wget2
	cd wget2 && nix-shell env.nix --run "./bootstrap && ./configure && make -j"

clean:
	rm -rf wget2
.PHONY: clean
