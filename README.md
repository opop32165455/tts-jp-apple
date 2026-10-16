# tts-jp-apple

**在 Apple Silicon Mac 上本地运行的语音合成（TTS）+ 语音识别（STT）HTTP 服务。** 以日语为主，同时支持中文和英文。

- **TTS**：[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（通过 [mlx-audio](https://github.com/Blaizzy/mlx-audio)），只需几秒参考音频即可克隆声音
- **STT**：[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)（`whisper-large-v3-turbo`），返回带时间戳的分段结果
- 完全离线运行，不需要 API Key，推理全部在 Metal GPU 上完成，统一内存占用约 2~3 GB
- 自带 macOS launchd 管理脚本，可以作为开机自启的常驻服务

> 本项目改造自 [kapi2800/qwen3-tts-apple-silicon](https://github.com/kapi2800/qwen3-tts-apple-silicon)，交互式终端应用 `main.py` 来自上游。在此基础上新增了 FastAPI 服务、Whisper STT、launchd 托管、批量生成脚本和音频后处理。上游同步方式见 [docs/UPSTREAM.md](docs/UPSTREAM.md)。

---

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [HTTP API](#http-api)
- [配置（环境变量）](#配置环境变量)
- [作为系统服务运行](#作为系统服务运行)
- [其他工具](#其他工具)
- [目录结构](#目录结构)
- [实现要点与常见问题](#实现要点与常见问题)
- [致谢](#致谢)

---

## 环境要求

- Apple Silicon Mac（M1 及以上）和 macOS。依赖 MLX / Metal，**不支持 Intel Mac 和 CPU 推理**
- Python 3.13（依赖中包含只支持 3.13 的 `audioop-lts`）
- 建议 16 GB 及以上统一内存

## 快速开始

### 1. 安装

```bash
git clone https://github.com/opop32165455/tts-jp-apple.git
cd tts-jp-apple

# 推荐使用 uv；也可以用 python3.13 -m venv .venv
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

STT 需要 `ffmpeg` 来解码音频。系统里没有 `ffmpeg` 时，服务会自动使用 `imageio-ffmpeg` 自带的版本，也可以用 `brew install ffmpeg` 安装。

### 2. 下载 TTS 模型

从 HuggingFace 下载 MLX 权重，放到 `models/` 下：

```bash
.venv/bin/hf download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base-8bit
```

| 模型 | 用途 |
|------|------|
| [Qwen3-TTS-12Hz-1.7B-Base-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) | **服务默认**，声音克隆 |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` / `-VoiceDesign-8bit`，以及对应的 0.6B 版本 | `main.py` 的预设音色 / 声音设计模式（可选） |

Whisper 模型会在第一次调用 STT 时自动下载到 HuggingFace 缓存。

### 3. 准备参考音频

仓库**不附带**参考音频。请准备一段 3~10 秒、干净无背景音的单人语音（WAV），用下面两种方式之一指定：

- 放到默认路径 `ref-audio/yuna_short_4s.wav`
- 用环境变量指定：`export SPEECH_SERVER_REF_AUDIO=/path/to/your_voice.wav`

再提供这段音频对应的文字（`SPEECH_SERVER_REF_TEXT`），克隆出来的声音会更像。

> 请只使用你有权使用的声音。

### 4. 启动并测试

```bash
.venv/bin/python speech_server.py --port 7790

# 另开一个终端
curl -s http://127.0.0.1:7790/health
curl -X POST http://127.0.0.1:7790/v1/tts -H "Content-Type: application/json" \
  -d '{"text": "こんにちは、今日もいい天気ですね。"}' --output outputs/test.wav
curl -X POST "http://127.0.0.1:7790/v1/transcribe?language=ja" \
  --data-binary @outputs/test.wav -H "Content-Type: application/octet-stream"
```

---

## HTTP API

默认地址：`http://127.0.0.1:7790`

### `GET /health`

TTS 模型加载期间返回 `503`，其他时候返回各引擎的状态（`on` / `loading` / `off`）：

```json
{ "status": "ok", "service": "speech_server", "lang_code": "japanese",
  "tts": "on", "stt": "off", "model_load_seconds": 3.42, "stt_load_seconds": 0.0,
  "stt_idle_unload_seconds": 600, "stt_unload_in_seconds": null }
```

### `GET /v1/services`、`POST /v1/services/{tts|stt}/{on|off}` — 引擎开关

TTS 和 STT 可以在运行中单独加载或卸载，不用重启进程，卸载后模型占用的内存会被释放。返回值和 `/health` 里的引擎字段一致。

- **TTS**：常驻内存。手动关闭后，`/v1/tts` 会返回 `503`，直到重新打开。
- **STT**：按需加载。卸载状态下收到识别请求时会自动加载模型（这次请求要多等约 2 秒），连续空闲 10 分钟（`SPEECH_SERVER_STT_IDLE_UNLOAD_SECONDS`）后自动卸载，可以省下约 1.5～2 GB 内存。`on` 用来提前预加载，`off` 用来立即卸载。

```bash
curl -X POST http://127.0.0.1:7790/v1/services/stt/on    # 预加载 Whisper（第一次会下载模型）
curl -X POST http://127.0.0.1:7790/v1/services/stt/off   # 立即卸载，释放内存
```

开关状态不会保存：服务重启后恢复成环境变量设定的值（默认 TTS 开，STT 不预加载）。

### `GET /v1/config`

返回当前生效的配置，包括模型路径、参考音频、采样参数、Whisper 模型等。

### `POST /v1/tts` — 语音合成 / 声音克隆

请求体（JSON）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | 是 | 要合成的文本 |
| `ref_audio` | string | 否 | 参考音频路径（服务所在机器上的路径），不传时使用默认值 |
| `ref_text` | string | 否 | 参考音频对应的文字 |
| `lang_code` | string | 否 | `japanese` / `chinese` / `english` |
| `temperature` | float | 否 | 默认 0.65 |
| `top_p` | float | 否 | 默认 0.95 |
| `max_tokens` | int | 否 | 不传时，短文本会自动收缩（见下文） |

返回 `audio/wav`。响应头 `X-TTS-Ref-Audio` 是实际使用的参考音频路径。

### `POST /v1/transcribe`（别名 `/v1/stt`）— 语音识别

STT 没有加载时，第一次请求会自动加载模型，所以会慢约 2 秒；之后的请求直接使用已加载的模型。

- Query 参数：`language`（例如 `ja`、`zh`、`en`）
- 请求体：音频二进制（WAV / MP3 / M4A / OGG / FLAC / WebM）

```json
{ "segments": [ { "start": 0.12, "end": 2.45, "text": "こんにちは、今日もいい天気ですね。" } ] }
```

---

## 配置（环境变量）

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `SPEECH_SERVER_HOST` | `127.0.0.1` | 监听地址 |
| `SPEECH_SERVER_PORT` | `7790` | 监听端口 |
| `SPEECH_SERVER_MODEL` | `models/Qwen3-TTS-12Hz-1.7B-Base-8bit` | TTS 模型目录 |
| `SPEECH_SERVER_REF_AUDIO` | `ref-audio/yuna_short_4s.wav` | 默认参考音频 |
| `SPEECH_SERVER_REF_TEXT` | 空 | 默认参考音频对应的文字 |
| `SPEECH_SERVER_LANG` | `japanese` | 默认语言 |
| `SPEECH_SERVER_TEMPERATURE` | `0.65` | 采样温度 |
| `SPEECH_SERVER_TOP_P` | `0.95` | Top-P |
| `SPEECH_SERVER_TOP_K` | `50` | Top-K |
| `SPEECH_SERVER_REPETITION_PENALTY` | `1.1` | 重复惩罚 |
| `SPEECH_SERVER_MAX_TOKENS` | `512` | 默认最大 token 数 |
| `SPEECH_SERVER_MLX_WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | STT 模型 |
| `SPEECH_SERVER_TTS_ENABLED` | `1` | 启动时是否加载 TTS |
| `SPEECH_SERVER_STT_ENABLED` | `0` | 启动时是否预加载 STT（不预加载时，调用时会自动加载） |
| `SPEECH_SERVER_STT_IDLE_UNLOAD_SECONDS` | `600` | STT 空闲多少秒后自动卸载，`0` 表示不卸载 |
| `SPEECH_SERVER_WIRED_LIMIT_MB` | `3072` | 把模型权重锁定在内存里，防止被系统换出（见下文），`0` 表示不锁定 |

旧的 `YUNA_TTS_*` 同名变量仍然兼容。

---

## 作为系统服务运行

`manage_service.sh` 会生成一个 launchd LaunchAgent（`com.local.speech-server`），实现开机自启和崩溃后自动重启：

```bash
./manage_service.sh install     # 生成 plist、注册并启动
./manage_service.sh status      # 进程状态 + /health 探测
./manage_service.sh logs        # 跟踪服务日志
./manage_service.sh restart     # 修改 speech_server.py 后需要重启才能生效
./manage_service.sh stop | start | uninstall
```

- 服务日志：`logs/speech_server.log`，每天零点切分，保留 30 天
- launchd 自身的 stdout/stderr：`~/Library/Logs/speech-server/`
- 服务运行时会占用 7790 端口和模型内存。要在前台调试，先执行 `./manage_service.sh stop`

### 菜单栏管理（SwiftBar）

`tools/swiftbar/speech-server.10s.sh` 是一个 [SwiftBar](https://github.com/swiftbar/SwiftBar) 插件，每 10 秒刷新一次。它在菜单栏显示服务状态，并提供启动 / 停止 / 重启、查看日志等操作：

- 🟢 运行中且 TTS 已开启，🔵 进程在运行但 TTS 已关闭，🟠 启动中或模型加载中，🔴 异常退出（launchd 正在重启），⚪ 已停止或未安装
- 下拉菜单：
  - PID / 端口 / 语言
  - **TTS / STT 开关**：已加载的显示 ✓，点击即可加载或卸载对应模型（调用 `/v1/services`）。STT 加载后还会显示多久后自动卸载
  - 启动 / 停止 / 重启（调用 `manage_service.sh`）
  - 实时日志、在控制台打开日志、`/v1/config`、打开项目目录

```bash
brew install --cask swiftbar
PLUG="$HOME/Library/Application Support/SwiftBar/Plugins"
mkdir -p "$PLUG" && ln -sf "$PWD/tools/swiftbar/speech-server.10s.sh" "$PLUG/"
defaults write com.ameba.SwiftBar PluginDirectory "$PLUG"   # 第一次启动 SwiftBar 前设置；如果已经在用 SwiftBar，就把插件链接到现有的插件目录
open -a SwiftBar
```

插件用的是软链接，所以修改仓库里的脚本会直接生效。端口不是 7790 时，在插件里修改 `PORT`。想开机自动显示，在 SwiftBar 设置里打开 Launch at Login。

---

## 其他工具

### `main.py` — 交互式终端应用（来自上游）

```bash
.venv/bin/python main.py    # 必须在仓库根目录运行
```

支持预设音色（CustomVoice）、声音设计（VoiceDesign）和声音克隆三种模式，可以切换 1.7B / 0.6B 模型。录入的声音保存在 `voices/`。

### `scripts/run_yuna_demo.py` — 离线批量克隆

直接在进程内加载模型，不经过 HTTP 服务。从 CSV（`id,content` 两列）批量生成 WAV：

```bash
.venv/bin/python scripts/run_yuna_demo.py \
  --csv data/your-input.csv \
  --ref-audio ref-audio/your_voice.wav \
  --out-dir outputs/batch-demo \
  --limit 10 --batch-size 5
```

- 按 `--batch-size` 和 `--max-batch-chars` 动态分批，失败时自动降到 `--fallback-batch-size` 重试
- `--start-row` 用于断点续跑，`--overwrite` 覆盖已有文件
- 后处理：开头淡入（`--fade-seconds`），结尾补静音（`--tail-silence-seconds`），并在静音里加入极低的底噪（`--tail-floor-amplitude`），防止导入工具把纯数字静音切掉。`--no-postprocess` 关闭后处理

---

## 目录结构

```text
tts-jp-apple/
├── speech_server.py      # FastAPI 服务（TTS + STT）
├── manage_service.sh     # launchd 服务管理
├── main.py               # 交互式终端应用（上游）
├── scripts/
│   └── run_yuna_demo.py  # 离线批量克隆
├── tools/swiftbar/       # 菜单栏状态与启停插件
├── AGENTS.md             # AI 编码助手的规则与文档路由（CLAUDE.md 引用它）
├── docs/
│   ├── architecture.md   # 内部实现细节
│   ├── agent-facts.md    # 环境踩坑记录
│   └── UPSTREAM.md       # 上游版本跟进说明
├── requirements.txt
├── models/               # MLX 模型权重（不纳入 git）
├── ref-audio/            # 参考音频（不纳入 git，自行放置）
├── data/                 # 批量输入 CSV（不纳入 git，自行放置）
├── voices/               # main.py 录入的声音（不纳入 git）
├── outputs/              # 生成的音频（不纳入 git）
└── logs/                 # 服务日志（不纳入 git）
```

---

## 实现要点与常见问题

- **开头的“噗”声**：声码器第一帧会产生直流冲击。服务端对输出加了 20 ms 线性淡入，客户端不需要再处理。
- **久未调用后第一次很慢**：内存紧张时，macOS 会把闲置进程的内存换出到交换区。MLX 默认不锁定模型内存，所以闲置的 TTS 权重（约 2.7 GB）会被换出，下一次调用要先把它读回来。实测这种情况要 6.5 秒，日志里空闲 1 小时以上后的第一次调用中位数是 13 秒（平时约 2 秒）。服务启动时会调用 `mx.set_wired_limit` 把模型锁定在内存里，从而避免这个问题。代价是这部分内存（TTS 约 2.7 GB；STT 加载期间另外约 2 GB）一直占着，不能被系统挪给其他程序。
- **短文本重复发音**：Qwen3-TTS 合成极短文本时，如果 `max_tokens` 过大，容易反复读同一个音。调用方没有传 `max_tokens` 且文本 ≤ 20 字时，服务会把它限制为 `max(32, min(默认值, 字数 × 8))`。生成的音频短于 0.3 秒会被当作截断，直接报错。
- **并发**：`MODEL.generate()` 不是线程安全的，所有合成请求都会串行执行。每次请求结束后会清理 MLX 缓存（上限 2 GB），并执行 GC。
- **STT 按需加载**：Whisper 很少用，所以启动时不加载，收到第一个识别请求时才加载，空闲 10 分钟后自动卸载。调用方不需要做任何改动，只是加载后的第一个请求会慢约 2 秒。
- **STT 只用 MLX**：`mlx-whisper` 在 Apple Silicon 上比 CPU 版快 4~5 倍（4.5 分钟音频约 11 秒），日语识别也更准，时间戳不容易漂移，所以没有设计 CPU 回退。MLX 出错时会直接报错，不会悄悄降级。第一次调用要加载模型，大约需要 20 秒。
- **STT 后处理**：用词级时间戳收紧每段的起止时间，丢掉短于 0.2 秒的段，并对重叠的段去重、重新排序。
- **端口冲突**：先 `./manage_service.sh stop`，或者用 `SPEECH_SERVER_PORT` / `--port` 换一个端口。

---

## 致谢

- [kapi2800/qwen3-tts-apple-silicon](https://github.com/kapi2800/qwen3-tts-apple-silicon) — 本项目的上游
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Alibaba Qwen 团队
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX 音频模型推理
- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Apple MLX Whisper 实现
- [mlx-community](https://huggingface.co/mlx-community) — MLX 量化模型
