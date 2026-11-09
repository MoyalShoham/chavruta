#!/usr/bin/env bash
# Boots the stack with the demo library and checks it end to end through nginx.
# Used by CI; runs locally too. Leaves nothing behind (down -v at the end).
set -euo pipefail
cd "$(dirname "$0")/.."
export COMPOSE_PROJECT_NAME=chavruta-smoke CHAVRUTA_CORPUS=demo CHAVRUTA_WEB_PORT=3299
trap 'docker compose logs --tail 40 api ingest || true; docker compose down -v >/dev/null 2>&1 || true; rm -rf secrets-smoke' EXIT
export CHAVRUTA_SECRETS_DIR=./secrets-smoke
docker compose up -d --build
base=http://localhost:$CHAVRUTA_WEB_PORT
for _ in $(seq 1 180); do
  if curl -fsS "$base/api/health" >/dev/null 2>&1; then break; fi
  sleep 5
done
curl -fsS "$base/api/health"
curl -fsS "$base/api/status" | grep -q '"closed"'
pw=$(cat secrets-smoke/learner_password)
jar=$(mktemp)
curl -fsS -c "$jar" -H 'X-Chavruta-Client: smoke' -H 'Content-Type: application/json' \
  -d "{\"username\":\"learner\",\"password\":\"$pw\"}" "$base/api/session" | grep -q LEARNER
curl -fsS -b "$jar" "$base/api/sugya?ref=Berakhot%202a" | grep -q '"Berakhot 2a:1"'
curl -fsS -b "$jar" "$base/api/works" | grep -q Deuteronomy
# Writes need the CSRF header.
code=$(curl -s -o /dev/null -w '%{http_code}' -b "$jar" -H 'Content-Type: application/json' \
  -d '{"ref":"Berakhot 2a:1","body":"x"}' "$base/api/notes")
[ "$code" = 403 ]
echo "smoke test passed"
