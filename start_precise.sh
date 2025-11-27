#!/bin/bash
# 精确去水印 - 保护原图版本

source skiing/bin/activate
echo "✓ 虚拟环境已激活: skiing"

cd "$(dirname "$0")"

echo "🎯 精确水印去除工具"
echo "   策略: OCR识别文字 + 只修复水印 + 完全保护原图"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 确保安装 EasyOCR（用于精确文字检测）
if ! python -c "import easyocr" >/dev/null 2>&1; then
    echo "📦 检测到未安装 EasyOCR，正在安装: uv pip install easyocr"
    if ! uv pip install easyocr; then
        echo "❌ EasyOCR 安装失败，请手动运行:"
        echo "   uv pip install easyocr"
        exit 1
    fi
    echo "✓ EasyOCR 安装完成"
fi

# 检查参数
if [ -z "$1" ]; then
    echo "用法: ./start_precise.sh <图片路径> [模式]"
    echo ""
    echo "模式选项:"
    echo "  fast      - 快速模式 (30步, ~1分钟)"
    echo "  standard  - 标准模式 (50步, ~2分钟) [默认]"
    echo "  high      - 高质量 (70步, ~3分钟) ⭐推荐"
    echo "  ultra     - 极致质量 (100步, ~5分钟)"
    echo ""
    echo "或直接指定步数: ./start_precise.sh image.jpg 80"
    echo ""
    echo "示例:"
    echo "  ./start_precise.sh images/11-22/02.JPG"
    echo "  ./start_precise.sh images/11-22/02.JPG high"
    echo "  ./start_precise.sh images/11-22/02.JPG 80"
    exit 1
fi

IMAGE_PATH="$1"
MODE="${2:-standard}"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ 文件不存在: $IMAGE_PATH"
    exit 1
fi

# 根据模式设置步数
case "$MODE" in
    fast)
        STEPS=30
        ;;
    standard)
        STEPS=50
        ;;
    high)
        STEPS=70
        ;;
    ultra)
        STEPS=100
        ;;
    [0-9]*)
        STEPS=$MODE
        ;;
    *)
        echo "❌ 未知模式: $MODE"
        echo "   使用: fast, standard, high, ultra 或数字"
        exit 1
        ;;
esac

echo "📷 输入图片: $IMAGE_PATH"
echo "⚙️  推理步数: $STEPS"
echo ""

# 运行
uv run python remove_watermark_precise.py "$IMAGE_PATH" "$STEPS"

