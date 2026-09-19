#!/usr/bin/env bash
# Enforces the import rule in GUIDELINES.md ("Dependencies: the rule that makes
# it real") mechanically, because a rule nobody can verify is a rule that is
# already being broken. A single new line in a Cargo.toml is enough to couple
# two domains permanently and is invisible in review.
#
# Rule 1: a domain crate may depend on third-party crates, infy-kernel and
#         infy-wire. Never on a sibling domain -- directly or transitively.
# Rule 2: the leaves stay leaves. infy-kernel may depend on serde only;
#         infy-wire on infy-kernel and serde only. If a leaf grows a
#         dependency, every domain inherits it, and the isolation the rest of
#         the architecture buys is gone.
#
# `cargo tree -e normal` is used rather than parsing Cargo.toml by hand because
# it sees exactly what cargo resolves, including features and transitive edges,
# and excludes dev-dependencies (a test helper is not a coupling).
set -euo pipefail
cd "$(dirname "$0")/.."

LEAVES="infy-kernel infy-wire"
ROOT="infy"
fail=0

domains=()
for dir in crates/*/; do
  name=$(basename "$dir")
  case "$name" in
    kernel|wire|"$ROOT") ;;
    *) domains+=("infy-$name") ;;
  esac
done

is_allowed() { # is_allowed <name> <allowlist...>
  local needle=$1; shift
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

# Rule 1: transitive closure of every domain contains no sibling.
for crate in "${domains[@]}"; do
  deps=$(cargo tree -p "$crate" -e normal --prefix none 2>/dev/null \
         | awk '{print $1}' | sort -u | grep '^infy-' | grep -vx "$crate" || true)
  for dep in $deps; do
    if ! is_allowed "$dep" $LEAVES; then
      echo "ARCH: $crate depends on sibling domain $dep."
      echo "      Declare a narrow trait in $crate/src/deps.rs and let crates/$ROOT supply the concrete type."
      fail=1
    fi
  done
done

# Rule 2: leaves have an explicit allowlist of DIRECT dependencies.
check_leaf() { # check_leaf <crate> <allowed...>
  local crate=$1; shift
  local deps
  deps=$(cargo tree -p "$crate" -e normal --depth 1 --prefix none 2>/dev/null \
         | awk '{print $1}' | sort -u | grep -vx "$crate" || true)
  for dep in $deps; do
    if ! is_allowed "$dep" "$@"; then
      echo "ARCH: leaf $crate depends on $dep; it may depend only on: $*"
      fail=1
    fi
  done
}
check_leaf infy-kernel serde
check_leaf infy-wire infy-kernel serde

if [[ $fail -eq 0 ]]; then
  echo "check-arch: ok (${#domains[@]} domains, leaves: $LEAVES)"
fi
exit $fail
