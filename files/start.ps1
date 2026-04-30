# DayTrade Terminal v3 - 起動スクリプト
# 実行方法: PowerShellで右クリック → "PowerShellで実行"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   DayTrade Terminal v3 - Real Data Edition" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Python確認
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Pythonが見つかりません。Python 3.8以上をインストールしてください。" -ForegroundColor Red
    pause; exit 1
}

# pip依存ライブラリインストール
Write-Host "[1/3] 依存ライブラリを確認中..." -ForegroundColor Yellow
python -m pip install flask flask-cors yfinance pandas numpy --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] ライブラリのインストールに失敗しました。" -ForegroundColor Red
    pause; exit 1
}
Write-Host "      OK" -ForegroundColor Green

# Flaskサーバー起動（バックグラウンド）
Write-Host "[2/3] Flaskサーバーを起動中 (port 5000)..." -ForegroundColor Yellow
$serverJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    python server.py
} -ArgumentList $scriptDir

Start-Sleep -Seconds 3

# ヘルスチェック
$healthy = $false
for ($i=0; $i -lt 5; $i++) {
    try {
        $res = Invoke-WebRequest -Uri "http://localhost:5000/api/health" -TimeoutSec 3 -ErrorAction Stop
        if ($res.StatusCode -eq 200) { $healthy = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}

if (-not $healthy) {
    Write-Host "[WARN] サーバーの起動確認ができませんでした。ブラウザで手動確認してください。" -ForegroundColor Yellow
} else {
    Write-Host "      サーバー起動確認OK" -ForegroundColor Green
}

# ブラウザで開く
Write-Host "[3/3] ブラウザを起動中..." -ForegroundColor Yellow
$indexPath = Join-Path $scriptDir "index.html"
Start-Process $indexPath

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   起動完了！" -ForegroundColor Green
Write-Host "   API:      http://localhost:5000" -ForegroundColor White
Write-Host "   チャート: index.html をブラウザで開く" -ForegroundColor White
Write-Host "   停止:     このウィンドウを閉じてください" -ForegroundColor White
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Ctrl+C で終了" -ForegroundColor Gray

# サーバーログ表示しながら待機
try {
    while ($true) {
        $output = Receive-Job -Job $serverJob
        if ($output) { Write-Host $output -ForegroundColor DarkGray }
        Start-Sleep -Seconds 2
    }
} finally {
    Stop-Job -Job $serverJob
    Remove-Job -Job $serverJob
    Write-Host "サーバーを停止しました。" -ForegroundColor Yellow
}
