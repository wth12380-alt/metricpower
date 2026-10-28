#!/usr/bin/env bash
# One command to check everything: unit tests, Monte Carlo validation, app smoke test.
set -e
cd "$(dirname "$0")/.."
echo "== 1. unit tests (reproduce the R package's numbers) =="
python -m pytest -q
echo
echo "== 2. Monte Carlo validation of every implemented variance =="
python validation/validate_all.py --reps "${REPS:-10000}" --reps-cov "${REPS_COV:-5000}" | tail -20
echo
echo "== 3. app smoke test (starts the server, fetches the page, stops it) =="
python -c "import shiny" 2>/dev/null || { echo "shiny not installed: pip install -e '.[app]'"; exit 1; }
shiny run app.py --port 8765 > /tmp/metricpower_app.log 2>&1 &
PID=$!
sleep 4
CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/ || true)
kill $PID 2>/dev/null || true
if [ "$CODE" = "200" ]; then echo "app serves: HTTP 200"; else echo "app did not start (HTTP $CODE); see /tmp/metricpower_app.log"; exit 1; fi
echo
echo "ALL CHECKS PASSED"
