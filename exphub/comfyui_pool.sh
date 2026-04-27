#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${COMFY_ROOT:-/home/hx/ComfyUI}"
CONDA_SH="${CONDA_SH:-/home/hx/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-comfyui}"

HOST="${HOST:-127.0.0.1}"
RUN_DIR="${COMFY_ROOT}/run"
LOG_DIR="${COMFY_ROOT}/logs"

mkdir -p "$RUN_DIR" "$LOG_DIR"

INSTANCES=(
  "gpu0:0:8188"
  "gpu1:1:8189"
  "gpu2:2:8190"
)

ensure_dirs() {
  for item in "${INSTANCES[@]}"; do
    IFS=":" read -r name gpu port <<< "$item"
    mkdir -p \
      "${COMFY_ROOT}/input/${name}" \
      "${COMFY_ROOT}/output/${name}" \
      "${COMFY_ROOT}/user/${name}" \
      "${COMFY_ROOT}/temp/${name}"
  done
}

is_port_used() {
  local port="$1"
  ss -ltn | awk '{print $4}' | grep -q ":${port}$"
}

pid_file() {
  local name="$1"
  echo "${RUN_DIR}/comfyui_${name}.pid"
}

start_one() {
  local name="$1"
  local gpu="$2"
  local port="$3"
  local pidf
  pidf="$(pid_file "$name")"

  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "[SKIP] ${name} already running, pid=$(cat "$pidf"), port=${port}"
    return
  fi

  if is_port_used "$port"; then
    echo "[ERROR] port ${port} already in use, cannot start ${name}"
    echo "        check with: ss -ltnp | grep ${port}"
    exit 1
  fi

  echo "[START] ${name}: gpu=${gpu}, port=${port}"

  (
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    cd "$COMFY_ROOT"

    CUDA_VISIBLE_DEVICES="$gpu" nohup python main.py \
      --listen "$HOST" \
      --port "$port" \
      --input-directory "${COMFY_ROOT}/input/${name}" \
      --output-directory "${COMFY_ROOT}/output/${name}" \
      --temp-directory "${COMFY_ROOT}/temp/${name}" \
      --user-directory "${COMFY_ROOT}/user/${name}" \
      --database-url "sqlite:///${COMFY_ROOT}/user/${name}/comfyui.db" \
      > "${LOG_DIR}/comfyui_${name}.log" 2>&1 &

    echo $! > "$pidf"
  )

  sleep 2

  if kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "[OK] ${name} started, pid=$(cat "$pidf"), url=http://${HOST}:${port}"
  else
    echo "[ERROR] ${name} failed to start. Log:"
    tail -n 80 "${LOG_DIR}/comfyui_${name}.log" || true
    exit 1
  fi
}

start_all() {
  ensure_dirs
  for item in "${INSTANCES[@]}"; do
    IFS=":" read -r name gpu port <<< "$item"
    start_one "$name" "$gpu" "$port"
  done

  echo
  echo "[INFO] waiting for services..."
  sleep 5
  status_all
}

stop_one() {
  local name="$1"
  local pidf
  pidf="$(pid_file "$name")"

  if [[ ! -f "$pidf" ]]; then
    echo "[SKIP] ${name}: no pid file"
    return
  fi

  local pid
  pid="$(cat "$pidf")"

  if kill -0 "$pid" 2>/dev/null; then
    echo "[STOP] ${name}: pid=${pid}"
    kill "$pid" || true

    for _ in {1..20}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done

    if kill -0 "$pid" 2>/dev/null; then
      echo "[KILL] ${name}: pid=${pid}"
      kill -9 "$pid" || true
    fi
  else
    echo "[SKIP] ${name}: pid not running"
  fi

  rm -f "$pidf"
}

stop_all() {
  for item in "${INSTANCES[@]}"; do
    IFS=":" read -r name gpu port <<< "$item"
    stop_one "$name"
  done
}

status_all() {
  for item in "${INSTANCES[@]}"; do
    IFS=":" read -r name gpu port <<< "$item"
    local pidf
    pidf="$(pid_file "$name")"

    if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
      if curl -fsS "http://${HOST}:${port}/queue" >/dev/null 2>&1; then
        echo "[OK] ${name}: pid=$(cat "$pidf"), gpu=${gpu}, url=http://${HOST}:${port}"
      else
        echo "[WARN] ${name}: pid=$(cat "$pidf") running, but HTTP not ready, url=http://${HOST}:${port}"
      fi
    else
      echo "[DOWN] ${name}: gpu=${gpu}, port=${port}"
    fi
  done
}

logs_one() {
  local name="${1:-gpu0}"
  tail -f "${LOG_DIR}/comfyui_${name}.log"
}

case "${1:-start}" in
  start)
    start_all
    ;;
  stop)
    stop_all
    ;;
  restart)
    stop_all
    sleep 2
    start_all
    ;;
  status)
    status_all
    ;;
  logs)
    logs_one "${2:-gpu0}"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs [gpu0|gpu1|gpu2]}"
    exit 1
    ;;
esac