节奏 VMD 工具 —— 使用说明
==========================

启动
----
双击 RhythmVMD.exe 即可。不需要安装 Python、Blender 或任何运行库。

首次运行
--------
工具需要一个语音对齐模型（约 1.2 GB，仅首次使用需要联网下载）。
启动后会弹窗询问，点"是"即可，下载完成后自动保存到：

    data\models\mms_fa.onnx

如果自动下载失败，可以手动下载后放到上面这个位置：
https://hf-mirror.com/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx

日常使用
--------
1. 选择音频（支持 m4a / mp3 / wav / flac / ogg）
2. 选择歌词（txt 每行一句；也支持 lrc / json）
3. "载入歌词并分割" → "自动对齐（CTC）"
4. 双击表格单元格可直接修改开始/结束/歌词
5. "生成 VMD" → 得到 .vmd + .lrc + .align.json

表格操作：空格试听本行，Del 删除行，右键有菜单，
Ctrl+S 保存 LRC，Ctrl+O 载入，Ctrl+Enter 生成。

混合语言
--------
中英、日英混唱的歌词可以直接用；语言下拉选"自动"，
需要强制时在行内加 [zh] / [ja] / [en] 标记。

许可
----
对齐模型为 Meta MMS-FA（CC-BY-NC 4.0），禁止商用。
ffmpeg 为 LGPL/GPL 组件，随附其许可证。

数据位置
--------
所有数据都在本文件夹内（data 子目录），删除文件夹即彻底卸载。
