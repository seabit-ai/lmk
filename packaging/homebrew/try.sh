#!/bin/bash
# Try the Homebrew formula from the working tree without touching the real lmk on this Mac:
# a local tarball, the formula renamed lmk-try, a temporary LMK_HOME, then uninstall.
set -euo pipefail
cd "$(dirname "$0")/../.."
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
VERSION="0.0.0"
# the working tree, not HEAD: what is being tried is usually not committed yet (tracked + untracked, ignored left out)
git ls-files -co --exclude-standard -z | tar -czf "$WORK/lmk-$VERSION.tar.gz" --null -T - -s "|^|lmk-$VERSION/|"
SHA="$(shasum -a 256 "$WORK/lmk-$VERSION.tar.gz" | cut -d' ' -f1)"
# Homebrew only installs formulae from a tap: a throwaway local tap holds the renamed formula.
export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_ENV_HINTS=1
TAP="lmk-local/try"
brew untap "$TAP" >/dev/null 2>&1 || true
brew tap-new --no-git "$TAP" >/dev/null
TAP_DIR="$(brew --repository "$TAP")"
sed -e "s|^class Lmk |class LmkTry |" -e "s|url \".*\"|url \"file://$WORK/lmk-$VERSION.tar.gz\"|" \
    -e "s|sha256 \".*\"|sha256 \"$SHA\"|" -e "s|bin/\"lmk\"|bin/\"lmk-try\"|g" \
    packaging/homebrew/lmk.rb > "$TAP_DIR/Formula/lmk-try.rb"
echo "· brew install $TAP/lmk-try"
brew install --quiet "$TAP/lmk-try"
export LMK_HOME="$WORK/home"
mkdir -p "$LMK_HOME/app"
ENGINE_COMMIT="$(cat ENGINE_COMMIT)"
if ! curl -fsSLI "https://github.com/seabit-ai/mlx-engine/archive/$ENGINE_COMMIT.tar.gz" >/dev/null 2>&1 && [ -d "$HOME/.lmk/app/.engine" ]; then
  echo "· engine $ENGINE_COMMIT is not on GitHub yet: copying ~/.lmk/app/.engine"
  cp -R "$HOME/.lmk/app/.engine" "$LMK_HOME/app/.engine"
fi
echo "· first run through the shim (installs into $LMK_HOME)"
time lmk-try status || true
echo "· second run (no reinstall expected)"
lmk-try status || true
echo "· build recorded: $(cat "$LMK_HOME/app/BUILD")"
brew uninstall --quiet lmk-try
brew untap "$TAP" >/dev/null
echo "✓ formula works; lmk-try and the local tap removed"
