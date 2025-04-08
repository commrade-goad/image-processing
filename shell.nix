{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
    packages = [
        (pkgs.python312.withPackages(p: with p; [
            numpy
            opencv4
        ]))
        pkgs.pyright
    ];
}
