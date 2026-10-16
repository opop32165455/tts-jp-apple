# 上游版本管理

本项目（[opop32165455/tts-jp-apple](https://github.com/opop32165455/tts-jp-apple)）改造自开源项目 [kapi2800/qwen3-tts-apple-silicon](https://github.com/kapi2800/qwen3-tts-apple-silicon)。本文说明如何保留上游原版，以及以后如何跟进上游的更新。

---

## 1. 仓库结构

| 项目 | 值 | 说明 |
|---|---|---|
| `origin` | `https://github.com/opop32165455/tts-jp-apple.git` | 自己的仓库，日常 push / pull 都用它 |
| `upstream` | `https://github.com/kapi2800/qwen3-tts-apple-silicon.git` | 上游仓库，只用来 fetch（push 地址已禁用，防止误推） |
| `main` | 改造后的代码 | 日常开发分支 |
| `upstream-main` | 跟踪 `upstream/main` | **只存放上游原版，不要在这个分支上改代码** |
| tag `upstream-2026-05-02` | `f74e52e` | 第一次拉取的上游版本（2026-05-02，"Add Star History section to README"） |

`upstream-main` 分支和 `upstream-*` tag 也推送到了 `origin`，在 GitHub 上可以直接看到原版。

查看原版文件（不用切换分支）：

```bash
git show upstream-2026-05-02:main.py            # 查看原版某个文件
git ls-tree -r --name-only upstream-2026-05-02  # 列出原版所有文件
```

把原版完整检出到一个独立目录，作为只读参考：

```bash
git worktree add ../tts-jp-apple-upstream upstream-2026-05-02
```

---

## 2. 跟进上游更新

```bash
# 1) 拉取上游最新提交（不会改动本地代码）
git fetch upstream

# 2) 查看上游新增了哪些提交、改了哪些文件
git log --oneline upstream-main..upstream/main
git diff upstream-main upstream/main --stat

# 3) 有更新时，记录新的版本快照（日期改成当天）
git branch -f upstream-main upstream/main
git tag upstream-YYYY-MM-DD upstream/main

# 4) 切回 main，把上游更新合并进来（有冲突就手动解决后再提交）
git checkout main
git merge upstream-main

# 5) 推送到自己的仓库
git push origin main upstream-main --tags
```

查看自己相对原版做了哪些改造：

```bash
git diff upstream-2026-05-02 main --stat
```

> 合并前先把 `main` 上的改动提交掉。工作区有未提交的改动时遇到冲突，很难恢复。

---

## 3. 不公开的本地文件

仓库是公开的，下面这些文件只保留在本地，已经加入 `.gitignore`：

- `ref-audio/*`：参考音频（声音样本）
- `data/*`：批量生成的输入数据
- `legacy/`：早期原型脚本
- 和具体业务绑定的脚本
- `CLAUDE.local.md`：本机相关的说明

这些文件**不会备份到 GitHub**，换机器时需要自己拷贝。不要用 `git add -f` 把它们强行加进仓库。
