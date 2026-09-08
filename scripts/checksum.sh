#!/usr/bin/env bash
# Print a checksum of every source file that must match between machines.
# Run it on both the source machine and the workstation, then compare the
# final SUMMARY line — manual SFTP has no way of telling you a file was missed.
#
#   bash scripts/checksum.sh            # per-file list + summary
#   bash scripts/checksum.sh --quiet    # summary only
set -euo pipefail

# Byte ordering must not depend on the machine's locale: macOS and glibc collate
# '_' differently, so `sort` would order ./ui/__init__.py differently on each
# side and the summary hash would differ even when every file matches.
export LC_ALL=C LANG=C
cd "$(dirname "$0")/.."

# git ls-files is preferred, but it exits 0 with no output when the tree is
# untracked — so test for content, not just for success.
files=$(git ls-files 2>/dev/null || true)
if [ -z "$files" ]; then
    files=$(find . -type f \
        \( -name '*.py' -o -name '*.yml' -o -name '*.toml' -o -name '*.txt' \
           -o -name '*.md' -o -name 'Dockerfile' -o -name 'Makefile' \
           -o -name 'Caddyfile' -o -name '*.sh' -o -name '.dockerignore' \) \
        -not -path './.git/*' -not -path '*/__pycache__/*' | LC_ALL=C sort)
fi

# .env is machine-local and .env.example is a template; both may legitimately
# differ, so they are excluded from the comparison.
files=$(printf '%s\n' "$files" | grep -Ev '^\./?\.env$|(^|/)\.env$')

hash_of() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi; }

tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
while IFS= read -r f; do
    [ -f "$f" ] && hash_of "$f" >> "$tmp"
done <<< "$files"

LC_ALL=C sort -k2 "$tmp" > "$tmp.sorted"
n=$(wc -l < "$tmp.sorted" | tr -d ' ')
sum=$(hash_of "$tmp.sorted" | cut -c1-16)

# `|| :` throughout: with `set -o pipefail`, piping this into `head` closes
# stdout early and a bare echo would make the script exit non-zero on a
# perfectly successful run.
if [ "${1:-}" != "--quiet" ]; then
    awk '{printf "%s  %s\n", substr($1,1,12), $2}' "$tmp.sorted" || :
    echo || :
fi
echo "SUMMARY  $n files  $sum" || :
