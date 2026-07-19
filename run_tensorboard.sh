#!/usr/bin/env bash
# TensorBoard를 Cloudflare Quick Tunnel로 외부 공개 실행하고 접속 URL을 출력한다.
# 사용법: ./run_tensorboard.sh
# - 재실행하면 기존 tensorboard/cloudflared를 정리하고 새로 띄운다.
# - Quick Tunnel URL은 실행할 때마다 새로 발급된다 (고정 도메인 아님).
set -euo pipefail
cd "$(dirname "$0")"

PORT=6006
CLOUDFLARED="$HOME/.local/bin/cloudflared"
command -v cloudflared >/dev/null 2>&1 && CLOUDFLARED="$(command -v cloudflared)"

mkdir -p logs
pkill -f "tensorboard --logdir" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 1

nohup venv/bin/tensorboard --logdir logs --port "$PORT" --host 127.0.0.1 \
    > logs/tensorboard.out 2>&1 &
nohup "$CLOUDFLARED" tunnel --url "http://localhost:$PORT" --no-autoupdate \
    > logs/cloudflared.out 2>&1 &

echo "Cloudflare 터널 URL 발급 대기 중..."
URL=""
for _ in $(seq 1 30); do
    URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" logs/cloudflared.out | head -1 || true)
    [ -n "$URL" ] && break
    sleep 1
done

if [ -n "$URL" ]; then
    echo ""
    echo "=========================================="
    echo "  TensorBoard: $URL"
    echo "=========================================="
else
    echo "URL 발급 실패 — logs/cloudflared.out 로그를 확인하세요." >&2
    exit 1
fi
