#!/usr/bin/env bash
# <xbar.title>Speech Server</xbar.title>
# <xbar.desc>Status, start/stop/restart and TTS/STT engine switches for the local speech launchd service.</xbar.desc>
# <xbar.dependencies>bash,curl</xbar.dependencies>
# <swiftbar.hideAbout>true</swiftbar.hideAbout>
# <swiftbar.hideRunInTerminal>true</swiftbar.hideRunInTerminal>
# <swiftbar.hideLastUpdated>false</swiftbar.hideLastUpdated>
# <swiftbar.hideDisablePlugin>true</swiftbar.hideDisablePlugin>
# <swiftbar.hideSwiftBar>false</swiftbar.hideSwiftBar>
#
# SwiftBar plugin. Install by symlinking into the SwiftBar plugin folder (see README);
# the file name's ".10s." makes SwiftBar refresh every 10 seconds.

PORT="${SPEECH_SERVER_PORT:-7790}"
LABEL="com.local.speech-server"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

# Resolve the repo from this script's real location (the plugin folder holds a symlink).
PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
MANAGE="$PROJECT_DIR/manage_service.sh"
LOG_FILE="$PROJECT_DIR/logs/speech_server.log"
BASE_URL="http://127.0.0.1:$PORT"

# launchctl list line: "<pid|-> <last exit status> <label>"
launchd_line="$(/bin/launchctl list 2>/dev/null | /usr/bin/awk -v l="$LABEL" '$3 == l')"
pid="$(echo "$launchd_line" | /usr/bin/awk '{print $1}')"
last_exit="$(echo "$launchd_line" | /usr/bin/awk '{print $2}')"

health_code=""
health_body=""
if [[ -n "$launchd_line" ]]; then
  health_body="$(/usr/bin/curl -s -m 2 -w '\n%{http_code}' "$BASE_URL/health" 2>/dev/null)"
  health_code="$(echo "$health_body" | /usr/bin/tail -n 1)"
  health_body="$(echo "$health_body" | /usr/bin/sed '$d')"
fi

json_field() { # crude extractor for flat JSON from /health
  echo "$health_body" | /usr/bin/sed -n "s/.*\"$1\":\"\{0,1\}\([^\",}]*\).*/\1/p"
}

if [[ ! -f "$PLIST" ]]; then
  state="not_installed"; color="#8E8E93"; text="未安装"
elif [[ -z "$launchd_line" ]]; then
  state="stopped"; color="#8E8E93"; text="已停止"
elif [[ "$health_code" == "200" ]]; then
  state="running"; text="运行中"
  # Icon follows TTS (the engine in daily use): green on, blue = process up but TTS off.
  [[ "$(json_field tts)" == "on" ]] && color="#34C759" || color="#0A84FF"
elif [[ "$pid" == "-" ]]; then
  state="crashed"; color="#FF3B30"; text="异常退出（exit $last_exit），launchd 正在重启"
else
  state="loading"; color="#FF9F0A"; text="启动中 / 模型加载中"
fi

action() { # label, manage_service.sh subcommand
  echo "$1 | bash=$MANAGE param1=$2 terminal=false refresh=true"
}

switch_cmd() { # engine, on|off
  echo "bash=/usr/bin/curl param1=-s param2=-m param3=300 param4=-X param5=POST param6=$BASE_URL/v1/services/$1/$2 terminal=false refresh=true"
}

engine_item() { # engine (tts|stt), display name
  local st detail unload_in idle_hint
  st="$(json_field "$1")"
  unload_in="$(json_field "$1_unload_in_seconds")"
  [[ "$1" == "tts" ]] && idle_hint="无心跳" || idle_hint="空闲"
  case "$st" in
    on) [[ "$unload_in" =~ ^[0-9]+$ ]] \
          && detail="已加载，${idle_hint} $(( (unload_in + 59) / 60 )) 分钟后自动卸载，点击立即卸载" \
          || detail="已加载，点击卸载" ;;
    *)  detail="未加载，心跳或调用时自动加载，点击预加载" ;;
  esac
  case "$st" in
    on)      echo "$2（$detail） | checked=true $(switch_cmd "$1" off)" ;;
    loading) echo "$2（加载中…） | color=#FF9F0A" ;;
    *)       echo "$2（$detail） | $(switch_cmd "$1" on)" ;;
  esac
}

# Palette rendering so the menu bar icon keeps its status color (plain sfcolor is drawn as a template).
sfconfig="$(printf '{"renderingMode":"Palette","colors":["#FFFFFF","%s"]}' "$color" | /usr/bin/base64)"
echo "| sfimage=waveform.circle.fill sfconfig=$sfconfig"
echo "---"
echo "Speech Server：$text | color=$color"
if [[ "$state" == "running" ]]; then
  echo "PID $pid · 端口 $PORT · 语言 $(json_field lang_code) | size=12"
  echo "---"
  engine_item tts "TTS 语音合成"
  engine_item stt "STT 语音识别"
  echo "TTS 随 App 心跳加载，STT 按需加载 | size=11 color=#8E8E93"
elif [[ -n "$pid" && "$pid" != "-" ]]; then
  echo "PID $pid · 端口 $PORT | size=12"
fi
echo "---"
case "$state" in
  not_installed) action "安装并启动（开机自启）" install ;;
  stopped)       action "启动" start ;;
  *)             action "停止" stop; action "重启" restart ;;
esac
echo "---"
echo "实时日志（终端） | bash=/usr/bin/tail param1=-f param2=-n param3=100 param4=$LOG_FILE terminal=true"
echo "在控制台打开日志 | bash=/usr/bin/open param1=-a param2=Console param3=$LOG_FILE terminal=false"
echo "查看当前配置 /v1/config | href=$BASE_URL/v1/config"
echo "打开项目目录 | bash=/usr/bin/open param1=$PROJECT_DIR terminal=false"
