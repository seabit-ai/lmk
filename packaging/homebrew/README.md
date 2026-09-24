# Homebrew

`brew install seabit-ai/tap/lmk` — the tap is the repository `seabit-ai/homebrew-tap`, whose
`Formula/lmk.rb` is a copy of [`lmk.rb`](lmk.rb) here.

## Releasing

After tagging `vX.Y.Z` on main and pushing the tag:

```sh
curl -fsSL https://github.com/seabit-ai/lmk/archive/refs/tags/vX.Y.Z.tar.gz | shasum -a 256
```

Put the tag in `url` and the checksum in `sha256` of `lmk.rb`, commit here, copy the file to
`homebrew-tap/Formula/lmk.rb`, commit and push there. `brew upgrade lmk` on a user's Mac then
installs the new build into `~/.lmk` on the next `lmk` command (the shim compares the build id).

The engine commit in `ENGINE_COMMIT` must be pushed to `github.com/seabit-ai/mlx-engine` before
the tag: the shim's install fetches it from there (the same rule as the `curl | sh` installer).

## Trying a formula locally

`packaging/homebrew/try.sh` builds a tarball of the working tree, installs it as the formula
`lmk-try` (a different name, so the real `lmk` on this Mac is untouched) into a temporary
`LMK_HOME`, runs a command through the shim and uninstalls again. The engine is copied from the
existing `~/.lmk/app/.engine` when the commit is not on GitHub yet.
