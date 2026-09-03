# VoxCPM2 Voice QC 项目部署交接摘要

更新时间：2026-09-03

## 一句话状态

项目的本地开发、保护机制、单元测试、单段真实生成、26 段完整真实流程和 Codex 插件结构验证均已完成；本地 Git 仓库已提交且工作区干净。现在只差创建一个空的 GitHub 私有仓库、绑定远程地址并推送。

## 项目位置与 Git 状态

- 本地目录：`E:\Prefect\voice_qc`
- 当前分支：`codex/voice-qc-deployment`
- 当前提交：`e3c445d Build GPU-safe VoxCPM2 voice QC workflow`
- 写本摘要前状态：工作区干净，没有未提交修改
- 当前限制：尚未配置 GitHub remote；本机没有可用的 `gh` CLI，因此还没有推送到 GitHub

## 已实现的完整流程

```text
当前执行任务的 AI 按句意分段
→ Python 验证分段前后原文完全一致
→ 检查片段长度、GPU、显存、重复进程和输出目录
→ VoxCPM2 串行生成
→ ASR 与音频规则质检
→ 只重试未通过片段
→ 冻结原始候选音频
→ 在派生副本中把超过 200ms 的静音压缩到 200ms
→ 拼接整条音频
→ 最终整条 ASR 与音频质检
→ 只有最终质检通过才生成 final.wav
```

## 已固定的关键规则

### 语义分段

- 分段由当前工作的 AI 完成，不绑定 Ollama、OpenAI API、Claude API 或其他固定提供商。
- 句意、语法、因果、转折和自然朗读优先。
- `120～180` 字只是参考范围，不是机械切分条件。
- 单段 `240` 字为硬上限，超过后拒绝生成。
- 只允许改变分段边界，禁止增删、润色、纠错或改写原文。
- Python 会验证合并后的文本与原文完全一致。

配置：

```text
preferred_min_chars = 120
preferred_max_chars = 180
hard_max_chars = 240
```

### GPU 与进程保护

```text
device = cuda
asr_device = cuda:0
min_free_vram_mb = 4096
max_parallel = 1
```

生成前会读取 `nvidia-smi`，检查总显存、已用显存、空闲显存和利用率。以下情况直接拒绝生成：

- `nvidia-smi` 不可用或无法读取显存；
- 空闲显存低于 4096 MB；
- 任意片段超过 240 字；
- 检测到同一流程已有生成任务；
- 新运行可能覆盖已有输出目录。

VoxCPM2 生成严格串行，一次只生成一个片段，不使用线程池或进程池并行占用显存。

### 生成与重试参数

```text
cfg = 2.0
steps = 20
normalize = false
denoise = false
reference_mode = combined
max_quality_retries = 2
base_seed = 20235791
retry_seed_stride = 104729
```

每个片段最多生成 3 次：首次 1 次，加最多 2 次确定性重试。每轮只处理未通过质检的片段。

### 静音处理

- FFmpeg 检测参数：`silencedetect=noise=-50dB:d=0.05`。
- `<=200ms` 的静音保持不变。
- `>200ms` 的静音只在派生副本中压缩到 `200ms`。
- 原始候选音频不会被覆盖。

## 已完成验证

### 自动测试

- 使用解释器：`E:\VoxCPM2\.venv\Scripts\python.exe`
- 最新结果：29 个测试全部通过。
- 覆盖范围包括：原文一致性、240 字上限、GPU 阈值、Windows 进程锁、串行运行、重试、静音压缩、WAV 样本保留、最终报告和 `final.wav` 生成门禁。

### 单段真实生成

- VoxCPM2 加载、单段生成、ASR 读取、质检、派生静音处理和最终输出均通过。
- 未发现并行生成或遗留生成进程。

### 26 段完整真实流程

- 运行目录：`E:\Prefect\voice_qc\outputs\20260902-170506`
- 首轮片段：26
- 首轮未通过：7
- 总重新生成次数：9
- 最终未通过片段：0
- 最终整条质检：通过
- 最终文件：`E:\Prefect\voice_qc\outputs\20260902-170506\final.wav`
- 最终文件大小：46,035,996 字节
- 报告：`E:\Prefect\voice_qc\outputs\20260902-170506\report.txt`
- 静音记录：`E:\Prefect\voice_qc\outputs\20260902-170506\silence_edits.csv`

`silence_edits.csv` 已确认目标时长记录为 `200.0ms`。失败时流程会保留诊断文件，但不会伪造 `final.wav`。

## Codex 插件状态

插件目录：

```text
plugin/voxcpm2-voice-qc/
├─ .codex-plugin/plugin.json
├─ skills/voice-qc/SKILL.md
└─ scripts/run_voice_qc.ps1
```

插件结构验证已通过，`-CheckOnly` 启动路径已通过。插件只是本地流程入口，不包含模型、参考音频、私人文本、密钥或本机配置。插件目前只是项目内的可发布结构，没有安装到全局 Codex 环境。

## GitHub 应上传的文件

```text
voice_qc_flow.py
asr_worker.py
SEGMENTATION_PROMPT.md
README.md
config.example.json
tests/
plugin/
LICENSE
THIRD_PARTY_NOTICES.md
.gitignore
DEPLOYMENT_HANDOFF.md
```

## 严禁上传的内容

```text
config.json
input.txt
reference.wav
reference_text.txt
outputs/
.prefect/
__pycache__/
模型权重
ASR 模型
私人文案和私人音频
API 密钥、凭据、日志和缓存
```

这些内容已经由 `.gitignore` 排除。推送前仍应执行一次提交树和敏感信息检查。

## 最短部署操作

1. 在 GitHub 创建一个空的 **Private** 仓库，不要自动添加 README、LICENSE 或 `.gitignore`。
2. 在本机执行：

```powershell
git -C "E:\Prefect\voice_qc" remote add origin "https://github.com/<用户名>/<仓库名>.git"
git -C "E:\Prefect\voice_qc" push -u origin codex/voice-qc-deployment
```

3. 推送后在 GitHub 页面确认仓库仍为 Private，并确认没有 `config.json`、私人文本、音频、模型或 `outputs/`。

如果远程仓库已经存在，先用 `git -C "E:\Prefect\voice_qc" remote -v` 检查，不要重复添加 `origin`。

## 后续 AI 接手时不要重复做的事

- 不要重新跑 26 段真实生成来证明流程可用；完整验证已经通过。
- 不要改回 153ms；当前正式规则是超过 200ms 压缩到 200ms。
- 不要把 120～180 字做成硬切分；240 字才是硬上限。
- 不要绑定某一家 AI API；当前执行任务的 AI 负责语义分段。
- 不要并行调用 VoxCPM2。
- 不要安装或打包模型到插件和 GitHub 仓库。
- 没有新的代码修改时，接手工作的重点只是敏感信息复查、创建私有远程仓库和推送。

## 对外展示建议

可把项目描述为：一个面向长文案配音的、可复现且带 GPU 保护的 VoxCPM2 生成与质量控制流水线，包含 AI 语义分段、文本完整性校验、确定性重试、ASR/声学质检、非破坏性静音处理和最终成品门禁。

在改为公开仓库前，需要再次核对自有代码许可证、VoxCPM2、FunASR、FFmpeg 及模型权重各自的许可和第三方声明；私有仓库部署不需要等待这一步。
