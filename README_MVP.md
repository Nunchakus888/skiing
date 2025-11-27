# 水印去除 MVP

基于 PaddleOCR + Stable Diffusion Inpaint 的最小可执行方案。

## 快速开始

```bash
# 安装依赖
uv sync

# 运行
uv run python remove_watermark_mvp.py images/test.jpg
uv run python remove_watermark_mvp.py images/test.jpg -o clean.jpg
```

## 工作原理

1. **PaddleOCR** - 检测图片中的文字（中英文支持）
2. **关键词匹配** - 识别水印关键词（滑呗、雪友等）
3. **Mask 生成** - 自动膨胀确保完全覆盖
4. **SD Inpaint** - 智能修复水印区域

## 性能

- 第一次运行会下载模型（~2GB）
- macOS 上 CPU 推理约 2 分钟/张
- GPU 推理约 10 秒/张

## 文件说明

- `remove_watermark_mvp.py` - 核心脚本（117行）
- `pyproject.toml` - uv 依赖配置

