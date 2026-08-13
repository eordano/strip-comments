{
  lib,
  stdenvNoCC,
  makeWrapper,
  python3,
  shfmt,
}:

let
  python = python3.withPackages (ps: [ ps.pyyaml ]);
in
stdenvNoCC.mkDerivation {
  pname = "comments";
  version = "1.0.0";

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./comments.py
      ./lang
    ];
  };

  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall
    mkdir -p $out/share/comments $out/bin
    cp comments.py $out/share/comments/
    cp -r lang $out/share/comments/lang
    makeWrapper ${python.interpreter} $out/bin/comments \
      --prefix PATH : ${lib.makeBinPath [ shfmt ]} \
      --add-flags "$out/share/comments/comments.py"
    for sub in list strip; do
      makeWrapper ${python.interpreter} $out/bin/comments-$sub \
        --prefix PATH : ${lib.makeBinPath [ shfmt ]} \
        --add-flags "$out/share/comments/comments.py $sub"
    done
    runHook postInstall
  '';

  meta = {
    description = "Inventory and strip code comments under a semantics-equality gate";
    homepage = "https://github.com/eordano/strip-comments";
    license = lib.licenses.mit;
    mainProgram = "comments";
    platforms = lib.platforms.all;
  };
}
