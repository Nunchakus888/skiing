#!/bin/bash
# SD 去水印 - macOS M4 快速启动脚本 (使用 UV 包管理)

set -e

cd "$(dirname "$0")"

echo "🎨 Stable Diffusion 去水印工具 (macOS M4 优化版)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查 uv
if ! command -v uv &> /dev/null; then
    echo "❌ 未找到 UV 包管理工具"
    echo "   安装方法: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 检查并安装依赖
echo "📦 检查依赖..."

if ! uv run python -c "import diffusers, transformers, torch, PIL, cv2" &> /dev/null 2>&1; then
    echo "🔧 首次运行，安装依赖（约需 1-2 分钟）..."
    echo ""
    uv pip install diffusers transformers torch torchvision pillow opencv-python accelerate
    echo ""
    echo "✓ 依赖安装完成"
    echo ""
fi

# 运行脚本
echo "🚀 启动中..."
echo ""
uv run python remove_watermark_sd.py "$@"

