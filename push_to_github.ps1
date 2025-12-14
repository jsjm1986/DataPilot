Param(
    [string]$RemoteUrl = "https://github.com/jsjm1986/DataPilot.git",
    [string]$Branch = "main"
)

# 使用方法：先在 PowerShell 中设置环境变量 GITHUB_TOKEN，然后运行此脚本
# Windows PowerShell (临时)： $Env:GITHUB_TOKEN = 'ghp_xxx'
# 然后在 opensource_export 目录中运行： .\push_to_github.ps1

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git 未安装或不可用，请先安装 git。"
    exit 1
}

if (-not $Env:GITHUB_TOKEN) {
    Write-Error "请先设置环境变量 GITHUB_TOKEN（含权限），例如：`$Env:GITHUB_TOKEN = 'ghp_xxx'`"
    exit 1
}

Write-Output "Adding remote: $RemoteUrl"
git remote remove origin 2>$null | Out-Null
git remote add origin $RemoteUrl

Write-Output "Creating branch $Branch and pushing using token..."
git branch -M $Branch

# 使用带 token 的临时 URL 推送（注意：token 可能出现在命令历史中，请妥善处理）
$token = $Env:GITHUB_TOKEN
$secureUrl = $RemoteUrl -replace '^https://', "https://$($token)@"

git push --set-upstream $secureUrl $Branch

Write-Output "Push attempted. If it failed, check token permissions and network."
