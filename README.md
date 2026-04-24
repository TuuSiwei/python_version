# si-dataset-process（会话级 WAV + RTTM 版）

本目录用于把 Seamless Interaction 的原始参与者音频与 jsonl 标注处理成说话人分离训练数据：

1. `filelist.csv` 转为 `conversations.json`。
2. 按 conversation 下载所有参与者 `.wav` 与 `.jsonl`。
3. 混合多参与者音频，输出一个保留 48 kHz 的会话级 wav。
4. 合并每个参与者 jsonl 的句子级 `start/end`，生成同名 RTTM。
5. 按每 5000 个 input conversations 构建一个 Hugging Face Dataset config，上传到 `humanify/real_dia_dataset`，config 名为 `si_1`、`si_2` ...
6. 上传成功后删除对应 staging 目录，降低本地磁盘占用。

旧的 Qwen ASR/parquet 主链路已不再通过 CLI 暴露。

## 安装

```bash
cd python_version
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

上传需要配置 `HF_TOKEN`，可放在当前工作目录的 `.env` 中。

## 命令

### 生成 conversations.json

```bash
python main.py convert-session
```

输入为当前目录的 `filelist.csv`，输出为 `conversations.json`。

### 构建并上传 SI RTTM 数据集

```bash
python main.py download-transform
```

默认参数：

- `--repo-id humanify/real_dia_dataset`
- `--config-prefix si`
- `--chunk-size 5000`
- `--download-concurrency 64`
- `--process-concurrency 16`
- `--max-shards 128`
- `--window-sec 90`
- `--shift-sec 8`
- `--max-spks 4`
- `--feat-per-sec 25`

本地试跑不上传：

```bash
python main.py download-transform --no-upload --stop-after-conversation-count 10
```

从指定 conversation 下标继续：

```bash
python main.py download-transform --start-from-conversation 5000
```

## 产物与进度

| 路径 | 含义 |
| --- | --- |
| `final/si_rttm_staging/chunk_*_{config}/audio/` | 当前批次会话级 wav |
| `final/si_rttm_staging/chunk_*_{config}/rttm/` | 当前批次 RTTM |
| `final/si_rttm_logs/failed.jsonl` | 下载、解析、混音失败的 conversations |
| `final/si_rttm_logs/skipped.jsonl` | 短于 90 秒或没有有效窗口的 conversations |
| `final/si_rttm_progress.json` | 已上传并清理的批次进度 |

上传成功的批次会自动删除 staging 目录；上传失败或使用 `--no-upload` 时会保留，方便检查或重试。

运行日志会显示当前 5000-conversation 批次的进度，例如 `Chunk 2/13`，并在组内每完成一个 conversation 后输出 `1234/5000 24.68%`。

## RTTM 规则

每个原始 jsonl 非空行生成一行 RTTM，使用顶层 `start/end`，speaker 字段为 `participant_id`：

```text
SPEAKER {conversation_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {participant_id} <NA> <NA>
```

多参与者标注会按开始时间和 speaker id 排序。
