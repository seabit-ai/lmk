# Homebrew formula for lmk. Lives in the tap repository github.com/seabit-ai/homebrew-tap as
# Formula/lmk.rb; this copy is the source of truth, copied there at each release (see
# packaging/homebrew/README.md). `url` and `sha256` change with every release tag.
#
# Shape (docs/design/2026-09-20-lmk-oobe.md, D2): the formula installs lmk's source into
# libexec and one shell shim into bin. The shim installs, or upgrades, lmk into ~/.lmk on first
# use with lmk's own installer (uv, Python 3.11, the pinned engine, ~1 GB of dependencies) —
# nothing else lives in the Cellar. Models, cache, config and logs stay under ~/.lmk as always;
# `brew uninstall lmk` plus removing that directory removes everything.
class Lmk < Formula
  desc "One local model, always on, for your agent — Apple Silicon"
  homepage "https://github.com/seabit-ai/lmk"
  url "https://github.com/seabit-ai/lmk/archive/refs/tags/v0.7.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"

  depends_on :macos
  depends_on arch: :arm64
  # git is needed (three dependencies are pinned git commits) and comes with Xcode's command line
  # tools on every Mac that has Homebrew; the installer checks and says so. Not a formula dependency:
  # Homebrew's own git is 60 MB the user does not need.

  def install
    libexec.install Dir["*"]
    (bin/"lmk").write <<~EOS
      #!/bin/sh
      # lmk lives in ~/.lmk (its own Python, the engine, models, cache, config). This shim installs
      # or upgrades it there from #{libexec} when the installed build is not #{version}, then runs it.
      set -e
      LMK_HOME="${LMK_HOME:-$HOME/.lmk}"
      if [ "$(cat "$LMK_HOME/app/BUILD" 2>/dev/null)" != "v#{version}" ]; then
        LMK_SRC="#{libexec}" LMK_BUILD="v#{version}" LMK_NO_PATH_LINK=1 sh "#{libexec}/install.sh"
      fi
      exec "$LMK_HOME/bin/lmk" "$@"
    EOS
  end

  def caveats
    <<~EOS
      The first `lmk` command installs lmk's runtime into ~/.lmk (about 1 GB, a minute).
      Then:  lmk up   — downloads the model (16 GB, once) and starts it, now and at every login.
    EOS
  end

  test do
    # A real run would download a gigabyte of dependencies into the test HOME; the mechanics
    # that can be checked cheaply are checked: the installer parses, the shim points at it.
    system "sh", "-n", libexec/"install.sh"
    assert_match "install.sh", (bin/"lmk").read
    assert_predicate libexec/"ENGINE_COMMIT", :exist?
  end
end
