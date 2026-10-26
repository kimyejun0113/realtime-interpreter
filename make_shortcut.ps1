# Creates a "동시통역" shortcut on the desktop that starts the app without a console window.
# Usage: powershell -ExecutionPolicy Bypass -File make_shortcut.ps1
$project = $PSScriptRoot
$pythonw = Join-Path $project ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) {
    Write-Error "가상환경이 없습니다. README의 설치 단계를 먼저 진행하세요."
    exit 1
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut((Join-Path $desktop "동시통역.lnk"))
$link.TargetPath = $pythonw
$link.Arguments = "`"$(Join-Path $project 'main.py')`""
$link.WorkingDirectory = $project
$link.IconLocation = "$env:SystemRoot\System32\SndVol.exe,0"
$link.Description = "실시간 동시통역 (영어 -> 한국어 자막)"
$link.Save()
Write-Output "바로가기 생성: $(Join-Path $desktop '동시통역.lnk')"
