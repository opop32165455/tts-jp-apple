#!/usr/bin/env bash
set -euo pipefail

PLIST_NAME="com.local.speech-server.plist"
PLIST_LABEL="com.local.speech-server"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
TARGET_PLIST="$LAUNCH_AGENTS_DIR/$PLIST_NAME"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="$PROJECT_DIR/.venv/bin/python"
SERVER_SCRIPT="$PROJECT_DIR/speech_server.py"
LOGS_DIR="$PROJECT_DIR/logs"
# launchd 的 stdout/stderr 固定放在内置盘，且不设置 WorkingDirectory（服务内部全部使用绝对路径），
# 这样即使项目放在外接 USB 盘上，launchd 也不会因无法 chdir / 打开日志而以 EX_CONFIG=78 失败。
LAUNCHD_LOGS_DIR="$HOME/Library/Logs/speech-server"

mkdir -p "$LOGS_DIR" "$LAUNCHD_LOGS_DIR"

generate_plist() {
  cat <<EOF > "$TARGET_PLIST"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_EXEC}</string>
        <string>${SERVER_SCRIPT}</string>
        <string>--port</string>
        <string>7790</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LAUNCHD_LOGS_DIR}/launchd_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LAUNCHD_LOGS_DIR}/launchd_stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${HOME}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
EOF
}

cmd_install() {
  echo "Installing LaunchAgent to $TARGET_PLIST..."
  mkdir -p "$LAUNCH_AGENTS_DIR"
  generate_plist
  launchctl unload "$TARGET_PLIST" 2>/dev/null || true
  launchctl load -w "$TARGET_PLIST"
  echo "Speech Server has been installed and configured for autostart on login."
  cmd_status
}

cmd_uninstall() {
  echo "Uninstalling LaunchAgent..."
  if [[ -f "$TARGET_PLIST" ]]; then
    launchctl unload "$TARGET_PLIST" 2>/dev/null || true
    rm -f "$TARGET_PLIST"
    echo "Removed $TARGET_PLIST."
  else
    echo "Plist not found. Nothing to remove."
  fi
}

cmd_start() {
  if [[ ! -f "$TARGET_PLIST" ]]; then
    echo "LaunchAgent not installed. Run './manage_service.sh install' first."
    exit 1
  fi
  launchctl load -w "$TARGET_PLIST" 2>/dev/null || launchctl start "$PLIST_LABEL"
  echo "Started service $PLIST_LABEL."
}

cmd_stop() {
  if [[ -f "$TARGET_PLIST" ]]; then
    launchctl unload "$TARGET_PLIST" 2>/dev/null || true
    echo "Stopped service $PLIST_LABEL."
  else
    echo "Service is not installed."
  fi
}

cmd_restart() {
  cmd_stop
  sleep 1
  cmd_start
}

cmd_status() {
  echo "=== Service Status ==="
  if launchctl list | grep -q "$PLIST_LABEL"; then
    echo "Status: RUNNING (launchd managed)"
    launchctl list | grep "$PLIST_LABEL" || true
  else
    echo "Status: STOPPED / NOT LOADED"
  fi
  echo ""
  echo "=== Health Check (HTTP 7790) ==="
  curl -s --connect-timeout 2 http://127.0.0.1:7790/health || echo "HTTP Server not responding yet"
  echo ""
}

cmd_logs() {
  local today_log="$LOGS_DIR/speech_server.log"
  echo "Tailing log: $today_log (Press CTRL+C to exit)..."
  if [[ -f "$today_log" ]]; then
    tail -f -n 50 "$today_log"
  else
    echo "No log file yet at $today_log"
  fi
}

case "${1:-status}" in
  install)
    cmd_install
    ;;
  uninstall)
    cmd_uninstall
    ;;
  start)
    cmd_start
    ;;
  stop)
    cmd_stop
    ;;
  restart)
    cmd_restart
    ;;
  status)
    cmd_status
    ;;
  logs)
    cmd_logs
    ;;
  *)
    echo "Usage: $0 {install|uninstall|start|stop|restart|status|logs}"
    exit 1
    ;;
esac
