#!/usr/bin/env bash
# Publish this repository (all branches + tags, full history) to GitHub.
#
# Usage (from the repository root):
#   bash scripts/push_to_github.sh ribeirojr-gh
#
# Prerequisites: an EMPTY repository (default name "projeto03-26092026") created on
# GitHub (no README/license, so histories do not conflict), and git
# authentication configured (SSH key or HTTPS token).
set -euo pipefail
user="${1:?usage: push_to_github.sh <github-username> [repository]}"
repo="${2:-projeto03-26092026}"
remote="git@github.com:${user}/${repo}.git"
if [[ "${USE_HTTPS:-0}" == "1" ]]; then remote="https://github.com/${user}/${repo}.git"; fi

git remote remove origin 2>/dev/null || true
git remote add origin "$remote"
git push -u origin main
git push origin --all          # develop + every feature/* branch
git push origin --tags         # every release tag
echo "Done: https://github.com/${user}/${repo}"
