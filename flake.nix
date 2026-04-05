{
  description = "pydantic-ai-subagent-mcp development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python314;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.git
            pkgs.jq
            pkgs.tree-sitter
          ];

          shellHook = ''
            export PYTHONDONTWRITEBYTECODE=1
            export UV_PYTHON="${python}/bin/python3"
          '';
        };
      }
    );
}
