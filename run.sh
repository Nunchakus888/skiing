#!/bin/bash
# 一键启动脚本 - 自动激活虚拟环境并运行

cd "$(dirname "$0")"

# 激活虚拟环境
source skiing/bin/activate

echo "✓ 虚拟环境已激活: skiing"
echo ""

# 运行去水印脚本
python remove_watermark_opencv.py

