#!/bin/sh

set -eu

current_branch=$(git branch --show-current)
if [ "$current_branch" != "dev" ]; then
    echo "Refusing to sync: current branch is '$current_branch', expected 'dev'." >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "Refusing to sync: working tree has uncommitted changes." >&2
    git status --short
    exit 1
fi

git fetch origin dev
git merge --ff-only origin/dev
echo "Local dev is synchronized with origin/dev."