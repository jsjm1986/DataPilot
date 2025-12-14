#!/usr/bin/env bash
# Usage: set GITHUB_TOKEN env var, then run this script from opensource_export/
# Example:
#   export GITHUB_TOKEN=ghp_xxx
#   ./push_to_github.sh https://github.com/jsjm1986/DataPilot.git main

REMOTE_URL=${1:-"https://github.com/jsjm1986/DataPilot.git"}
BRANCH=${2:-main}

if ! command -v git >/dev/null 2>&1; then
  echo "git not found, please install git"
  exit 1
fi

if [ -z "$GITHUB_TOKEN" ]; then
  echo "Please set GITHUB_TOKEN environment variable with a token that can push to the target repo."
  exit 1
fi

echo "Adding remote: $REMOTE_URL"
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
git branch -M "$BRANCH"

# Push using token in URL (token will be in process args briefly)
SECURE_URL=${REMOTE_URL/https:\/\//https:\/\/$GITHUB_TOKEN@}
git push --set-upstream "$SECURE_URL" "$BRANCH"

echo "Push attempted. If it failed, verify token permissions and remote URL."
