#!/usr/bin/env bash
# Print compiler warnings from .lane/build.log that land on lines this lane changed (git blame after d583c220).
ROOT=/d/AI/worktrees/sl1-spec-ckpt
cd "$ROOT" || exit 2
BASE=d583c220
n=0
grep -E "warning C[0-9]+" .lane/build.log | sort -u | while IFS= read -r w; do
    f=$(echo "$w" | sed -E 's/^([A-Za-z]:[^()]+)\(([0-9]+)\).*/\1/' | sed 's#\\#/#g; s#^D:/AI/worktrees/sl1-spec-ckpt/##')
    l=$(echo "$w" | sed -E 's/^[A-Za-z]:[^()]+\(([0-9]+)\).*/\1/')
    case "$l" in ''|*[!0-9]*) continue ;; esac
    [ -f "$f" ] || continue
    c=$(git blame -L "$l,$l" --porcelain "$f" 2>/dev/null | head -1 | cut -c1-40)
    [ -n "$c" ] || continue
    if ! git merge-base --is-ancestor "$c" "$BASE" 2>/dev/null; then
        echo "OWN: $w"
    fi
done
echo "own-warnings scan done"
