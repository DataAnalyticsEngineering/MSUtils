#!/bin/bash

time=$(date)
REPO_URL="https://DAE:${GITHUB_TOKEN}@github.tik.uni-stuttgart.de/DAE/MSUtils.git"

# Get commit info of main
commit=$(git log -1 --pretty=%B)
commit_sha=$(git log -1 --pretty=%H)
commit_link="[\`$commit_sha\`](https://github.tik.uni-stuttgart.de/DAE/MSUtils/commit/$commit_sha)"

# Set up gh-pages directory
mkdir -p temp_gh_pages
cd temp_gh_pages || { printf "Failed to enter temp_gh_pages directory\n" >&2; exit 1; }
git clone "$REPO_URL" . || { printf "Failed to clone repository\n" >&2; exit 1; }
git switch gh-pages || { printf "Failed to switch to gh-pages branch\n" >&2; exit 1; }
cd .. || { printf "Failed to return to parent directory\n" >&2; exit 1; }

# Copying files - docs have been created before!
printf "Copying files...\n"
cp -r docs/_build/html/* temp_gh_pages/ || { printf "Failed to copy files\n" >&2; exit 1; }

# Add README
printf "Add README\n"
readmetext="# Documentation Branch

This Branch is only meant to host the documentation via github-pages. Nothing to see here. The contents of this branch are essentially a cache that's not intended to be viewed on github.com.

You can find it [here](https://pages.github.tik.uni-stuttgart.de/DAE/molten/).

**It is not to be merged with main!**

For the actual repository, go to branch \`main\`.

## Current State reflected:

### Last update: $time

### Commit SHA:
$commit_link

### Title:
$commit
"

cd temp_gh_pages || { printf "Failed to enter temp_gh_pages directory\n" >&2; exit 1; }
printf "%s" "$readmetext" > README.md || { printf "Failed to write README.md\n" >&2; exit 1; }
git add -A || { printf "Failed to add files to git\n" >&2; exit 1; }
git commit -m "Deploy for commit '$commit_sha', '$commit' on $time" || { printf "Failed to commit changes\n" >&2; exit 1; }

# Pushing files
printf "Pushing files\n"
printf "%s\n" "$commit"
git push --force-with-lease origin gh-pages || { printf "Failed to push changes\n" >&2; exit 1; }

# Cleanup
cd .. || { printf "Failed to return to parent directory\n" >&2; exit 1; }
rm -rf temp_gh_pages || { printf "Failed to remove temp_gh_pages directory\n" >&2; exit 1; }
