# 拉取官方 SAM 3D Body 代码到 third_party/sam-3d-body-main
# 用法: powershell -ExecutionPolicy Bypass -File scripts/fetch_third_party.ps1 [-Commit <sha>]
param(
    [string]$Commit = "b5c765a0d89d789985e186d396315e7590887b94"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tmp = Join-Path $root ".tmp"
$target = Join-Path $root "third_party\sam-3d-body-main"
$archive = Join-Path $tmp "sam3dbody-$Commit.tar.gz"

New-Item -ItemType Directory -Force -Path $tmp | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root "third_party") | Out-Null

# GitHub 直连不可用时回退到公共加速代理
$urls = @(
    "https://codeload.github.com/facebookresearch/sam-3d-body/tar.gz/$Commit",
    "https://gh-proxy.com/https://codeload.github.com/facebookresearch/sam-3d-body/tar.gz/$Commit"
)

foreach ($url in $urls) {
    Write-Host "下载: $url"
    & curl.exe -L --retry 3 --retry-all-errors -C - --max-time 900 -o $archive $url
    if ($LASTEXITCODE -eq 0 -and (Test-Path $archive) -and (Get-Item $archive).Length -gt 1MB) {
        break
    }
}

if (-not (Test-Path $archive)) {
    throw "下载失败，请检查网络或在浏览器手动下载后放到 $archive"
}

if (Test-Path $target) { Remove-Item -Recurse -Force $target }
$extract = Join-Path $tmp "extract"
if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
New-Item -ItemType Directory -Force -Path $extract | Out-Null
& tar -xzf $archive -C $extract
$inner = Get-ChildItem $extract -Directory | Select-Object -First 1
Move-Item $inner.FullName $target

Write-Host "完成: $target"
Write-Host "下一步: 按 third_party/sam-3d-body-main/INSTALL.md 安装依赖，并准备 HuggingFace 门控权重。"
