#!/usr/bin/env bash
# Run on the VM, from any directory:  ~/stormlattice/deploy/deploy.sh
# Pulls main, rebuilds and restarts the API container, then waits for /health.
# It does not touch nginx or web/replay/; see "Deploy" in README.md for those.
set -euo pipefail

cd "$(dirname "$0")/.."
compose="docker compose -f docker/compose.yaml"

git pull --ff-only
$compose up -d --build

# The container needs a second or two before uvicorn listens.
for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8000/health; then
        echo
        exit 0
    fi
    sleep 1
done

echo "deploy: /health did not answer within 30 s; last container log lines:" >&2
$compose logs --tail 40 api >&2
exit 1
