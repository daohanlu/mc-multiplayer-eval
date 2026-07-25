#!/usr/bin/env bash
#
# Deploy the human-eval bundle to the annotation host and keep it running.
#
#   ./deploy.sh              sync files, then start the server if it is not up
#   ./deploy.sh fetch        pull collected answers back into ./responses/
#   ./deploy.sh status       is the session alive, is the port answering
#   ./deploy.sh restart      restart the server (files are left alone)
#   ./deploy.sh stop         kill the tmux session
#   ./deploy.sh logs         tail the remote server log
#
# Two things this script will never do:
#   * delete responses/ on the remote (collected answers are the one thing
#     here that cannot be regenerated), and
#   * upload data/*_key.json, so the answer key never sits on a public host.
#
# Scoring stays local: ./deploy.sh fetch && python3 score_human_eval.py
#
set -euo pipefail

REMOTE="${REMOTE:-fred@69.30.0.74}"
REMOTE_DIR="${REMOTE_DIR:-/nas2/fred/solaris-human-eval}"
PORT="${PORT:-9001}"
SESSION="${SESSION:-solaris-human-eval}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

ssh_() { ssh -o ConnectTimeout=15 "$REMOTE" "$@"; }

require_build() {
  if [[ ! -f data/consistency_items.json || ! -d frames || ! -d videos ]]; then
    echo "error: bundle is not built. Run:  python3 build_human_eval.py" >&2
    exit 1
  fi
}

do_sync() {
  require_build
  echo "==> syncing $HERE -> $REMOTE:$REMOTE_DIR"
  ssh_ "mkdir -p '$REMOTE_DIR/responses'"

  # --delete keeps the remote clean, but excluded paths are protected from it
  # by default, so responses/ survives. The answer key and the bulky
  # supplementary source folder are never uploaded.
  # Live progress is useful at a terminal but turns a piped log into noise.
  local info="stats1"
  [[ -t 1 ]] && info="stats1,progress2"

  rsync -az --delete --human-readable --info="$info" \
    --exclude 'responses/' \
    --exclude 'data/*_key.json' \
    --exclude 'Model Generations on Eval/' \
    --exclude '__pycache__/' \
    --exclude '.DS_Store' \
    --exclude '*.pyc' \
    --exclude 'serve.log' \
    ./ "$REMOTE:$REMOTE_DIR/"
  echo
}

server_up() {
  ssh_ "tmux has-session -t '$SESSION' 2>/dev/null" && return 0 || return 1
}

do_start() {
  if server_up; then
    echo "==> tmux session '$SESSION' already running — leaving it alone"
  else
    echo "==> starting server in tmux session '$SESSION' on port $PORT"
    # If serve.py ever exits, keep the pane alive so the traceback is readable
    # rather than losing it with the session.
    ssh_ "tmux new-session -d -s '$SESSION' -c '$REMOTE_DIR' \
      \"python3 serve.py --host 0.0.0.0 --port $PORT 2>&1 | tee -a serve.log; \
        echo; echo '[serve.py exited — press enter for a shell]'; read _; exec bash\""
    sleep 2
  fi
  do_status
}

do_status() {
  echo "==> status"
  if server_up; then
    echo "    tmux session : up ($SESSION)"
  else
    echo "    tmux session : DOWN"
  fi

  local code
  code=$(ssh_ "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$PORT/ || echo 000")
  echo "    from remote  : HTTP $code  (http://127.0.0.1:$PORT/)"

  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "http://${REMOTE#*@}:$PORT/" || echo 000)
  if [[ "$code" == "200" ]]; then
    echo "    from outside : HTTP 200  ->  http://${REMOTE#*@}:$PORT/"
  else
    echo "    from outside : HTTP $code  (blocked by a firewall, or not yet up)"
  fi

  local n
  n=$(ssh_ "ls '$REMOTE_DIR/responses' 2>/dev/null | wc -l" || echo 0)
  echo "    response files on remote: $n"
}

do_fetch() {
  echo "==> pulling answers from $REMOTE:$REMOTE_DIR/responses/"
  mkdir -p responses
  rsync -az --human-readable --info=stats1 \
    "$REMOTE:$REMOTE_DIR/responses/" ./responses/
  echo
  ls -la responses/ | tail -n +2
  echo
  echo "score with:  python3 score_human_eval.py"
}

do_restart() {
  echo "==> restarting"
  ssh_ "tmux kill-session -t '$SESSION' 2>/dev/null || true"
  sleep 1
  do_start
}

do_stop() {
  echo "==> stopping tmux session '$SESSION'"
  ssh_ "tmux kill-session -t '$SESSION' 2>/dev/null || true"
  echo "    stopped (collected answers in $REMOTE_DIR/responses/ are untouched)"
}

do_logs() {
  ssh_ "tail -n 60 '$REMOTE_DIR/serve.log' 2>/dev/null || echo '(no log yet)'"
}

case "${1:-deploy}" in
  deploy|"") do_sync; do_start ;;
  sync)      do_sync ;;
  start)     do_start ;;
  status)    do_status ;;
  restart)   do_restart ;;
  stop)      do_stop ;;
  fetch)     do_fetch ;;
  logs)      do_logs ;;
  *)
    echo "usage: $0 [deploy|sync|start|status|restart|stop|fetch|logs]" >&2
    exit 2 ;;
esac
