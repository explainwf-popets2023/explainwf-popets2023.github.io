# env.nix
with (import <nixpkgs> {});
mkShell {
  buildInputs = [
    flex
    lzip
    libidn2
    libpsl
    gnutls
    libiconv
    zlib
    pcre2
    gnupg1
    libassuan
    gpgme
    libevent
    openssl
    libtool
    autoconf
    automake
    pkg-config
    texinfo
  ];
}
