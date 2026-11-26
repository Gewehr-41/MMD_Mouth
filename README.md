# 节奏 VMD —— 歌词自动对口型（MMD / Blender）

把一首歌的音频 + 歌词，自动对齐每一句的起止时间，再生成 MMD 的嘴部形态键动作（`.vmd`）。

不需要打开 Blender：对齐、口型合成、VMD 写出全部独立完成，产物直接拖进 MMD 或 Blender 的
MMD 工具即可使用。

```
音频 + 歌词 ──► CTC 强制对齐 ──► 音素/拼音 ──► 口型曲线 ──► VMD
                  (mms_fa)                      (5 通道)
```

## 功能

- **自动对齐**：CTC 强制对齐（Meta MMS-FA，ONNX），整首一次计算 + 逐句精对齐
- **可编辑的歌词表**：双击单元格改时间/歌词，右键拆分、合并、插入、删除，空格试听
- **合成口型**：`ai` → A+I 混合权重，而不是单一口型跳变（日文 5 元音可组合出中文韵母）
- **两种动作风格**：`经典`（段落式，硬朗）/ `自然`（钟形曲线 + 平滑 + 协同发音）
- **形态键变体**：模型有 `あ２/あ３` 这类变体时可按通道选择 1/2/3
- **开口缩放 + 上下限**：整体缩放不改变曲线形状，上下限可逐通道设
- **混合语言**：中英、日英混唱可自动按字符集判定；中文走拼音，日文走 OpenJTalk，英文走拼写
- **逐帧关键帧**：VMD 形态键没有插值数据，默认每帧写一个关键帧，任何导入器都不会变成台阶
- **播放歌词**：整曲播放 + 当前行高亮滚动，单句试听

## 快速开始

### 方式一：下载发布包（推荐）

从 Releases 下载 `节奏VMD工具.zip`，解压后双击 `MMDRhythmVMD.exe`。

首次启动会提示下载对齐模型（约 1.2 GB，保存到程序目录 `data/models/`），之后完全离线可用。

### 方式二：源码运行

```powershell
py -3.10 -m venv .venv-worker
.venv-worker\Scripts\python.exe -m pip install -r requirements.txt
.venv-worker\Scripts\python.exe tools\rhythm_vmd_gui.py
```

或者双击仓库根目录的 `运行节奏VMD工具.cmd`。

模型放到 `data/models/mms_fa.onnx`（或设置环境变量 `MMD_MODEL`）：

```
https://hf-mirror.com/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx
```

## 图形界面

| 操作 | 说明 |
|---|---|
| 载入歌词并分割 | 按行（或按标点）切句 |
| 自动对齐（CTC） | 对齐整首，勾选"逐句精对齐"更准 |
| 生成 VMD | 写出 `.vmd` + `.lrc` + `.align.json` |
| 双击单元格 | 改开始 / 结束 / 歌词 |
| 空格 | 试听本行 |
| 右键 | 插入 / 拆分 / 合并 / 删除 / 跳到本行 |
| Ctrl+S / Ctrl+O / Ctrl+Enter | 保存 LRC / 载入 / 生成 |

## 命令行

```powershell
python tools\rhythm_vmd.py --audio 歌.m4a --lyrics 歌词.txt --out vmd_out --name 歌名
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--style classic\|natural` | 动作风格（默认 classic） |
| `--variants "U=2"` | 每通道形态键变体 1/2/3（默认全 1） |
| `--scale 0.85` | 开口整体缩放 |
| `--min 0 --max 0.9` | 开口上下限 |
| `--limits limits.json` | 逐通道上下限 |
| `--no-align` | 不做对齐，按 LRC 时间直接生成 |
| `--no-blend` | 关掉合成口型（只用 5 个基础元音） |
| `--sparse` | 稀疏关键帧（文件更小，部分导入器会变台阶） |
| `--language auto` | 按字符集判定语言 |

## 口型模型

输出 5 个通道：`あ い う え お`（以及闭口，通过压低全部元音实现）。

- **合成口型**：中文韵母被拆成日语 5 元音的组合，例如 `wo → U0.4+O0.6`、`ai → A0.6+I0.4`
- **自然风格**：每个音节是一条钟形窗，向两侧延伸约 25%，相邻音节自然交叠；再对输出做高斯平滑
- **经典风格**：原来的"平台 + 短过渡"行为，保留给喜欢硬朗节奏的用户

> 不同模型的 `あ２/あ３` 语义不同（常见是"更大/更小的同一口型"），使用前请先确认模型里确实存在该键。

## 构建发布包

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_tool.ps1
# 调试版（带控制台）
powershell -ExecutionPolicy Bypass -File tools\build_tool.ps1 -Console -EntryName MMDRhythmDebug -OutName VMDdebug
```

产物：`build-tool/dist/节奏VMD工具/` 和一个约 78 MB 的 zip（模型不打包，首次运行下载）。

## 目录结构

```
mmd_mouth/            与 Blender 无关的核心
  core/               事件模型、口型时间线、音素特征
  language/           G2P、韵母混合、罗马化（对齐用）
  recognition/lyrics.py  歌词解析 → 词 → 事件
tools/
  rhythm_vmd.py       核心流程 + 命令行
  rhythm_vmd_gui.py   Tkinter 界面
  ctc_align.py        CTC 强制对齐（numpy Viterbi）
  vmd_writer.py       VMD 读写
  audio_decode.py     ffmpeg 解码到 16k 单声道
  tool_support.py     路径/模型下载/设置
  build_tool.ps1      PyInstaller 打包
tests/                单元测试
samples/              示例歌词
```

## 依赖

- Python 3.10
- numpy、onnxruntime、pypinyin、cmudict、pyopenjtalk-plus
- ffmpeg（源码运行时可用 `imageio-ffmpeg` 自带的二进制）

## 许可

- 本仓库代码：MIT（见 [LICENSE](LICENSE)）
- 对齐模型 MMS-FA：**CC-BY-NC 4.0，禁止商用**，使用时需自行下载
- 发布包内的 ffmpeg：LGPL/GPL 组件，遵循其原始许可

## 致谢

- Meta [MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) 强制对齐模型
- [OpenJTalk](https://open-jtalk.sourceforge.net/) / pyopenjtalk-plus 日文 G2P
- [pypinyin](https://github.com/mozillazg/python-pinyin)、[cmudict](https://github.com/prosegrinder/python-cmudict)
