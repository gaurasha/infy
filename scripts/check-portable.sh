#!/usr/bin/env bash
# The drag-and-drop test from GUIDELINES.md, run literally: copy one domain
# crate plus the two leaves into an empty workspace and run its tests there.
# If they pass, the folder can be lifted into another project as-is. If they
# do not, something in the crate reaches outside itself -- a path dependency,
# a test fixture in another crate, a workspace-inherited setting -- and that
# is exactly what this check exists to catch.
#
# The real target directory is reused so third-party crates are not rebuilt;
# only the copied crates compile, which keeps this to seconds after the first
# run. --offline: the dependency set is already resolved in Cargo.lock.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(pwd)
target="${CARGO_TARGET_DIR:-$root/target}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

status=0
for dir in crates/*/; do
  name=$(basename "$dir")
  case "$name" in kernel|wire|infy) continue ;; esac

  w="$tmp/$name"
  mkdir -p "$w/crates"
  cp -r crates/kernel crates/wire "crates/$name" "$w/crates/"
  cp Cargo.lock "$w/"
  cat > "$w/Cargo.toml" <<TOML
[workspace]
resolver = "2"
members = ["crates/*"]

[profile.dev.package."*"]
opt-level = 3
TOML

  echo "check-portable: infy-$name with only kernel + wire beside it"
  if ! (cd "$w" && CARGO_TARGET_DIR="$target" cargo test -p "infy-$name" --offline -q 2>&1 | tail -20); then
    echo "PORTABLE: infy-$name does not work on its own. See output above."
    status=1
  fi
done
exit $status
