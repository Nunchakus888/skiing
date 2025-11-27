#!/usr/bin/env python3
"""
使用 IOPaint (lama-cleaner) 去除图片水印
简单、高效、效果好的开源方案
"""

import subprocess
import sys
import os

def install_iopaint():
    """安装 IOPaint"""
    print("正在安装 IOPaint...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "iopaint"])

def remove_watermark_simple(image_path, output_path):
    """
    使用 IOPaint 去除水印（方案1：命令行方式）
    这是最简单的方式，自动检测并去除重复文字水印
    """
    print(f"处理图片: {image_path}")
    
    # 使用 IOPaint CLI 命令
    # --model lama: 使用 LaMa 模型（效果最好）
    # --device cpu/cuda: 根据你的设备选择
    cmd = [
        "iopaint", "run",
        "--model", "lama",
        "--device", "cpu",  # 如果有GPU可以改为 cuda
        "--port", "8080"
    ]
    
    print("\n将启动 IOPaint Web UI...")
    print("步骤：")
    print("1. 打开浏览器访问 http://localhost:8080")
    print("2. 上传图片")
    print("3. 使用画笔工具涂抹水印区域")
    print("4. 点击'Run'按钮自动去除")
    print("5. 下载处理后的图片")
    print("\n按 Ctrl+C 停止服务器\n")
    
    subprocess.run(cmd)

def remove_watermark_auto():
    """
    方案2：使用 Python API 自动化处理
    需要先手动创建一个水印mask
    """
    try:
        from PIL import Image
        import numpy as np
        from iopaint.model_manager import ModelManager
        from iopaint.schema import Config, HDStrategy, LDMSampler
        
        print("使用 IOPaint API 自动处理...")
        
        # 配置
        config = Config(
            ldm_steps=25,
            ldm_sampler=LDMSampler.ddim,
            hd_strategy=HDStrategy.ORIGINAL,
        )
        
        # 加载模型
        model = ModelManager(name="lama", device="cpu")
        
        # 读取图片
        image = Image.open("/Users/george/Documents/me/skiing/2025-11-20 BDH/images/382474_x700.JPG")
        image_array = np.array(image)
        
        # 创建水印mask（白色=需要去除的区域）
        # 这里需要检测文字区域，可以使用 OCR 或简单的阈值处理
        mask = detect_watermark(image_array)
        
        # 执行修复
        result = model(image_array, mask, config)
        
        # 保存结果
        result_image = Image.fromarray(result)
        result_image.save("/Users/george/Documents/me/skiing/2025-11-20 BDH/images/382474_cleaned.jpg")
        print("完成！保存到: 382474_cleaned.jpg")
        
    except ImportError:
        print("API 方式需要额外配置，建议使用 Web UI 方式")
        return False
    
    return True

def detect_watermark(image_array):
    """
    简单的水印检测（检测半透明重复文字）
    返回mask图像
    """
    import cv2
    
    # 转换为灰度图
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # 使用阈值检测水印（水印通常是半透明的）
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 膨胀操作，扩大水印区域
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask

if __name__ == "__main__":
    print("=" * 60)
    print("图片水印去除工具 - 使用 IOPaint")
    print("=" * 60)
    
    # 检查是否安装了 iopaint
    try:
        import iopaint
        print("✓ IOPaint 已安装")
    except ImportError:
        print("× IOPaint 未安装")
        choice = input("是否现在安装? (y/n): ")
        if choice.lower() == 'y':
            install_iopaint()
        else:
            print("请先安装: pip install iopaint")
            sys.exit(1)
    
    print("\n选择处理方式：")
    print("1. Web UI 方式（推荐，简单直观）")
    print("2. 批量处理所有图片")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        image_path = "/Users/george/Documents/me/skiing/2025-11-20 BDH/images/382474_x700.JPG"
        remove_watermark_simple(image_path, None)
    elif choice == "2":
        print("\n批量处理模式:")
        print("将启动 Web UI，你可以逐个处理或使用脚本批量处理")
        remove_watermark_simple(None, None)
    else:
        print("无效选择")

