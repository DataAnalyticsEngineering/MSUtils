$time = Get-Date
# Set up gh-pages directory
mkdir temp_gh_pages
Set-Location temp_gh_pages
git clone https://github.tik.uni-stuttgart.de/DAE/MSUtils.git .
git switch gh-pages
Set-Location ..
# Make New main directory
mkdir temp_main
Set-Location temp_main
git clone https://github.tik.uni-stuttgart.de/DAE/MSUtils.git .
$commit = git log -1 --pretty=%B
# Build new website
# pixi shell -e docs -v
pixi run -v -e docs sphinx-build -M html docs docs/_build
Write-Output "compiling done!"
# Copying
Write-Output "Copying files..."
Copy-Item -r -force -v docs/_build/html/* ../temp_gh_pages/
Write-Output "Add README"
$readmetext = "# Documentation Branch

This Branch is only meant to host the documentation via github-pages. Nothing to see here. The contents of this branch are essentially a cache that's not intended to be viewed on github.com.

You can find it [here](https://pages.github.tik.uni-stuttgart.de/DAE/MSUtils/).

**It is not to be merged with main!**

For the actual repository, go to branch `main`.

## Current State reflected:
$commit
"
Set-Location ../temp_gh_pages
New-Item -Path README.md -Force
$readmetext | Add-Content -Path .\README.md
git add -A
git commit -a -m "Deploy for commit '$commit' on $time"
Write-Output "Pushing files"
Write-Output $commit
git push --force-with-lease origin gh-pages

Set-Location ..
Remove-Item .\temp_gh_pages\ -r -Force
Remove-Item .\temp_main\ -r -Force