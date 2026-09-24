#!/bin/sh
# lmk installer. Everything lands under ~/.lmk (or $LMK_HOME); nothing is installed
# system-wide, and removing that directory removes lmk. Run it again to upgrade.
#
#   git clone https://github.com/seabit-ai/lmk && cd lmk && ./install.sh
#   curl -fsSL https://raw.githubusercontent.com/seabit-ai/lmk/main/install.sh | sh
set -eu

LMK_HOME="${LMK_HOME:-$HOME/.lmk}"
LMK_REF="${LMK_REF:-}"              # a tag, branch or commit to fetch when not run from a checkout;
                                    # empty = the newest release tag (v*)
LMK_SRC="${LMK_SRC:-}"              # a checkout to install from; found by itself when run from one
LMK_BUILD="${LMK_BUILD:-}"          # the build id to record when LMK_SRC has no git history (Homebrew: the formula's version)
LMK_NO_PATH_LINK="${LMK_NO_PATH_LINK:-}"  # non-empty: do not link ~/.local/bin/lmk (Homebrew puts its own lmk on the PATH)
if [ -z "$LMK_SRC" ]; then
  HERE="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"
  [ -n "$HERE" ] && [ -d "$HERE/lmk" ] && [ -f "$HERE/ENGINE_COMMIT" ] && LMK_SRC="$HERE"
fi
APP="$LMK_HOME/app"
BIN="$LMK_HOME/bin"

say()  { printf '%s\n' "$*"; }
fail() { printf '✗ %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] || fail "lmk runs on Apple Silicon Macs only."
# Three of the dependencies are pinned git commits.
command -v git >/dev/null 2>&1 || fail "git is needed (run: xcode-select --install), then run this again."

mkdir -p "$APP" "$BIN"

# --- uv: one static binary; it brings its own Python 3.11, so none is needed on this Mac
if [ ! -x "$BIN/uv" ]; then
  say "· fetching uv (the Python installer lmk uses) into $BIN"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$BIN" UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1 \
    || fail "could not download uv"
fi
export UV_PYTHON_INSTALL_DIR="$LMK_HOME/python" UV_CACHE_DIR="$LMK_HOME/uv-cache"

# --- lmk's own code
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
if [ -n "$LMK_SRC" ]; then
  say "· installing lmk from $LMK_SRC"
  cp -R "$LMK_SRC/lmk" "$LMK_SRC/requirements.txt" "$LMK_SRC/ENGINE_COMMIT" "$STAGE/"
  if [ -n "$LMK_BUILD" ]; then
    BUILD="$LMK_BUILD"
  else
    # the tag when on one, else `v0.1.0-3-gabc1234` (3 commits past v0.1.0), else the short commit
    BUILD="$(git -C "$LMK_SRC" describe --tags --always --match 'v*' 2>/dev/null || echo unknown)"
    [ -z "$(git -C "$LMK_SRC" status --porcelain 2>/dev/null)" ] || BUILD="$BUILD-dirty-$(date +%m%d%H%M%S)"
  fi
else
  if [ -z "$LMK_REF" ]; then
    LMK_REF="$(git ls-remote --tags --refs https://github.com/seabit-ai/lmk 'v*' 2>/dev/null \
               | sed 's|.*refs/tags/||' | sort -V | tail -1)"
    [ -n "$LMK_REF" ] || fail "could not find a release of lmk on GitHub (to install a branch: LMK_REF=main sh install.sh)"
  fi
  say "· fetching lmk ($LMK_REF)"
  curl -fsSL "https://github.com/seabit-ai/lmk/archive/$LMK_REF.tar.gz" | tar -xz -C "$STAGE" --strip-components 1 \
    || fail "could not download lmk ($LMK_REF)"
  BUILD="$LMK_REF"
fi
find "$STAGE/lmk" -name __pycache__ -type d -prune -exec rm -rf {} +
rm -rf "$APP/lmk"
cp -R "$STAGE/lmk" "$APP/lmk"
cp "$STAGE/requirements.txt" "$STAGE/ENGINE_COMMIT" "$APP/"
printf '%s\n' "$BUILD" > "$APP/BUILD"

# --- the model runtime (mlx-engine), at the exact commit lmk was tested with.
# It ships no packaging metadata, so it is fetched as source, not pip-installed.
ENGINE_COMMIT="$(cat "$APP/ENGINE_COMMIT")"
# The marker alone is not proof: an installer before 2026-09-24 could write it next to an
# empty directory (failed download in a curl | tar pipe). The runtime's entry file must be there too.
if [ "$(cat "$APP/.engine/COMMIT" 2>/dev/null || true)" != "$ENGINE_COMMIT" ] \
   || [ ! -f "$APP/.engine/mlx-engine/mlx_engine/generate.py" ]; then
  say "· fetching the model runtime (mlx-engine ${ENGINE_COMMIT%"${ENGINE_COMMIT#???????}"})"
  rm -rf "$APP/.engine"
  mkdir -p "$APP/.engine/mlx-engine"
  if [ -n "$LMK_SRC" ] && git -C "$LMK_SRC/.engine/mlx-engine" cat-file -e "$ENGINE_COMMIT^{commit}" 2>/dev/null; then
    # From a checkout: the engine clone next to it has this commit — install what is here,
    # pushed or not. Nothing about a `make install` should depend on the network.
    say "  (from the checkout's own engine clone)"
    git -C "$LMK_SRC/.engine/mlx-engine" archive "$ENGINE_COMMIT" | tar -x -C "$APP/.engine/mlx-engine" \
      || fail "could not export mlx-engine from $LMK_SRC/.engine/mlx-engine"
  else
    # Download to a file first: in a curl | tar pipe a failed download (a commit not on GitHub)
    # leaves tar happy with an empty stream, and an empty runtime directory marked as installed.
    TARBALL="$APP/.engine/mlx-engine.tar.gz"
    curl -fsSL -o "$TARBALL" "https://github.com/seabit-ai/mlx-engine/archive/$ENGINE_COMMIT.tar.gz" \
      || fail "could not download mlx-engine at commit ${ENGINE_COMMIT%"${ENGINE_COMMIT#???????}"} — is it pushed to github.com/seabit-ai/mlx-engine?"
    tar -xz -C "$APP/.engine/mlx-engine" --strip-components 1 -f "$TARBALL" || fail "could not unpack mlx-engine"
    rm -f "$TARBALL"
  fi
  [ -f "$APP/.engine/mlx-engine/mlx_engine/generate.py" ] || fail "the mlx-engine download is not the runtime (mlx_engine/generate.py missing)"
  printf '%s\n' "$ENGINE_COMMIT" > "$APP/.engine/COMMIT"
fi

# --- python environment
if [ ! -x "$APP/.venv/bin/python" ]; then
  say "· creating a Python 3.11 environment"
  "$BIN/uv" venv --quiet --managed-python --python 3.11 "$APP/.venv"
fi
if ! cmp -s "$APP/requirements.txt" "$APP/.venv/.requirements-installed"; then
  say "· installing dependencies (about 1 GB)"
  "$BIN/uv" pip install --quiet --python "$APP/.venv/bin/python" -r "$APP/requirements.txt"
  cp "$APP/requirements.txt" "$APP/.venv/.requirements-installed"
fi

# --- the `lmk` command
cat > "$BIN/lmk" <<EOF
#!/bin/sh
export LMK_HOME="$LMK_HOME"
export PYTHONPATH="$APP/.engine/mlx-engine:$APP"
# -P: do not put the current directory on the import path — typed inside a clone of this
# repo, lmk would otherwise run the clone's code instead of the installed one
exec "$APP/.venv/bin/python" -P -m lmk "\$@"
EOF
chmod +x "$BIN/lmk"

# Only the default home puts `lmk` on the PATH: a second install elsewhere (a test, a
# trial) must not hijack the command.
LINKED=""
if [ "$LMK_HOME" = "$HOME/.lmk" ] && [ -z "$LMK_NO_PATH_LINK" ]; then
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) mkdir -p "$HOME/.local/bin" && ln -sf "$BIN/lmk" "$HOME/.local/bin/lmk" && LINKED=1 ;;
  esac
fi

say ""
say "✓ lmk $BUILD is installed in $LMK_HOME"
if [ -n "$LMK_NO_PATH_LINK" ]; then
  :   # Homebrew's lmk is the command
elif [ -n "$LINKED" ]; then
  say "  the lmk command is on your PATH (~/.local/bin/lmk)"
else
  say "  add it to your PATH:   export PATH=\"$BIN:\$PATH\"     (put that line in ~/.zshrc)"
fi
say ""
say "  next:"
say "    lmk up       download the model (16 GB, once) and start it, now and at every login"
