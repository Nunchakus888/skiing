#!/bin/bash
# 精准水印去除 - 保护原图

cd "$(dirname "$0")"

echo "🎯 精准水印去除工具"
echo "   核心原则: 精准检测 + 温和修复 + 保护原图"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查参数
if [ -z "$1" ]; then
    echo "用法: ./start_final.sh <图片路径> [模式]"
    echo ""
    echo "模式选项:"
    echo "  gentle    - 超保守 (50步, 0.90强度) - 最大程度保护原图"
    echo "  standard  - 保守修复 (50步, 0.93强度) [默认] ⭐推荐"
    echo "  balanced  - 平衡修复 (70步, 0.93强度) - 更好的去除效果"
    echo "  firm      - 坚定修复 (70步, 0.95强度) - 更彻底但仍保护原图"
    echo ""
    echo "示例:"
    echo "  ./start_final.sh images/11-22/02.JPG"
    echo "  ./start_final.sh images/11-22/02.JPG balanced"
    exit 1
fi

IMAGE_PATH="$1"
MODE="${2:-standard}"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ 文件不存在: $IMAGE_PATH"
    exit 1
fi

# 根据模式设置参数
case "$MODE" in
    gentle)
        STEPS=50
        STRENGTH=0.90
        ;;
    standard)
        STEPS=50
        STRENGTH=0.93
        ;;
    balanced)
        STEPS=70
        STRENGTH=0.93
        ;;
    firm)
        STEPS=70
        STRENGTH=0.95
        ;;
    *)
        echo "❌ 未知模式: $MODE"
        exit 1
        ;;
esac

echo "📷 输入图片: $IMAGE_PATH"
echo "⚙️  参数: $STEPS步 | 强度$STRENGTH | 精准检测"
echo ""

# 运行
uv run python remove_watermark_final.py "$IMAGE_PATH" "$STEPS" "$STRENGTH"

