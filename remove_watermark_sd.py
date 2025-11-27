#!/usr/bin/env python3
"""
使用 Stable Diffusion Inpainting 彻底去除水印
最强大的 AI 图像修复方案 - 一次性解决所有水印问题

优势：
1. 深度理解图像内容，生成自然纹理
2. 完美处理多层叠加、半透明水印
3. 保护人物细节，不会模糊脸部
4. 效果接近专业修图师水平
"""

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
import os
from pathlib import Path

class SDWatermarkRemover:
    """基于 Stable Diffusion 的智能水印去除器"""
    
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting", device=None):
        """
        初始化 SD Inpainting 模型
        
        Args:
            model_id: Hugging Face 模型 ID
            device: 'cuda', 'mps' (Mac M1/M2), 或 'cpu'
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float32  # MPS 对 float16 支持不完整
            else:
                device = "cpu"
                dtype = torch.float32
        
        print(f"🚀 初始化 Stable Diffusion Inpainting 模型...")
        print(f"   设备: {device}")
        print(f"   精度: {dtype}")
        print(f"   模型: {model_id}")
        
        self.device = device
        self.dtype = dtype
        
        # 加载模型
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,  # 关闭安全检查加速
        )
        self.pipe = self.pipe.to(device)
        
        # 优化设置
        if device == "cuda":
            self.pipe.enable_attention_slicing()  # 减少显存使用
            # self.pipe.enable_xformers_memory_efficient_attention()  # 可选：需要安装 xformers
        
        print("✓ 模型加载完成！\n")
    
    def detect_watermark_auto(self, image_path):
        """
        自动检测水印区域
        结合多种方法：颜色检测 + 边缘检测 + 位置推断
        """
        print(f"📷 分析图片: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 方法1: 检测灰蓝色水印（天空背景）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 灰蓝色范围（水印常用颜色）
        lower_gray_blue = np.array([90, 20, 80])
        upper_gray_blue = np.array([130, 150, 200])
        mask_color = cv2.inRange(hsv, lower_gray_blue, upper_gray_blue)
        
        # 方法2: 检测白色雪地上的深色文字
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bright_areas = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(gray, 50, 150)
        mask_dark_text = cv2.bitwise_and(edges, bright_areas)
        
        # 方法3: 边缘区域增强检测（水印通常在四周）
        border_size = int(min(h, w) * 0.15)
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        edge_mask[0:border_size, :] = 255  # 上
        edge_mask[h-border_size:h, :] = 255  # 下
        edge_mask[:, 0:border_size] = 255  # 左
        edge_mask[:, w-border_size:w] = 255  # 右
        
        # 合并检测结果
        mask = cv2.bitwise_or(mask_color, mask_dark_text)
        
        # 在边缘区域加强检测
        mask_edge_enhanced = cv2.bitwise_and(mask, edge_mask)
        mask = cv2.bitwise_or(mask, mask_edge_enhanced)
        
        # 形态学处理：连接文字笔画
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=3)
        
        # 保护中心人物区域（避免误检）
        center_x, center_y = w // 2, h // 2
        person_w, person_h = int(w * 0.3), int(h * 0.4)
        x1 = max(0, center_x - person_w // 2)
        x2 = min(w, center_x + person_w // 2)
        y1 = max(0, center_y - person_h // 2)
        y2 = min(h, center_y + person_h // 2)
        mask[y1:y2, x1:x2] = 0  # 清除人物区域
        
        # 统计
        watermark_pixels = np.sum(mask > 0)
        percentage = (watermark_pixels / (h * w)) * 100
        print(f"   检测到水印区域: {watermark_pixels:,} 像素 ({percentage:.1f}%)")
        
        return mask
    
    def remove_watermark(self, image_path, mask=None, output_path=None, 
                        prompt="high quality photo, natural, no watermark, clean",
                        negative_prompt="watermark, text, logo, blurry, low quality",
                        num_inference_steps=50,
                        guidance_scale=7.5):
        """
        使用 SD Inpainting 去除水印
        
        Args:
            image_path: 输入图片路径
            mask: 水印 mask（None=自动检测）
            output_path: 输出路径（None=自动生成）
            prompt: 正向提示词
            negative_prompt: 负向提示词
            num_inference_steps: 推理步数（越大越慢但效果越好，推荐 30-50）
            guidance_scale: 引导强度（推荐 7-8）
        """
        # 读取图片
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # 自动检测水印
        if mask is None:
            mask = self.detect_watermark_auto(image_path)
        
        # 转换 mask 为 PIL Image
        if isinstance(mask, np.ndarray):
            mask_pil = Image.fromarray(mask).convert("L")
        else:
            mask_pil = mask
        
        # 确保尺寸一致
        mask_pil = mask_pil.resize(original_size, Image.LANCZOS)
        
        # 调整图片大小（SD 对尺寸有要求，必须是 8 的倍数）
        def resize_to_multiple_of_8(img):
            w, h = img.size
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
            return img.resize((new_w, new_h), Image.LANCZOS)
        
        image_resized = resize_to_multiple_of_8(image)
        mask_resized = resize_to_multiple_of_8(mask_pil)
        
        print(f"\n🎨 开始 AI 修复...")
        print(f"   原始尺寸: {original_size}")
        print(f"   处理尺寸: {image_resized.size}")
        print(f"   推理步数: {num_inference_steps}")
        print(f"   提示词: {prompt}")
        
        # SD Inpainting
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=1.0,  # 完全重绘 mask 区域
        ).images[0]
        
        # 恢复原始尺寸
        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)
        
        # 保存结果
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_sd_cleaned{ext}"
        
        result.save(output_path, quality=95)
        print(f"✓ 修复完成！保存到: {output_path}\n")
        
        return result, output_path
    
    def batch_process(self, image_dir, output_dir=None, **kwargs):
        """批量处理文件夹中的所有图片"""
        image_dir = Path(image_dir)
        
        if output_dir is None:
            output_dir = image_dir / "sd_cleaned"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # 支持的图片格式
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = [f for f in image_dir.iterdir() 
                      if f.is_file() and f.suffix in extensions]
        
        print(f"\n📁 批量处理模式")
        print(f"   输入目录: {image_dir}")
        print(f"   输出目录: {output_dir}")
        print(f"   找到 {len(image_files)} 张图片\n")
        print("=" * 70)
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 处理: {image_file.name}")
            print("-" * 70)
            
            try:
                output_path = output_dir / image_file.name
                _, output = self.remove_watermark(
                    str(image_file),
                    output_path=str(output_path),
                    **kwargs
                )
                results.append((str(image_file), output, "成功"))
            except Exception as e:
                print(f"❌ 处理失败: {e}")
                results.append((str(image_file), None, f"失败: {e}"))
        
        print("\n" + "=" * 70)
        print("批量处理完成！\n")
        
        # 统计
        success = sum(1 for _, _, status in results if status == "成功")
        print(f"✓ 成功: {success}/{len(results)}")
        print(f"✗ 失败: {len(results) - success}/{len(results)}")
        
        return results


def main():
    """主程序 - 交互式界面"""
    print("=" * 70)
    print("🎨 Stable Diffusion 水印去除工具")
    print("   最强大的 AI 图像修复方案")
    print("=" * 70)
    print()
    
    # 初始化模型
    try:
        remover = SDWatermarkRemover()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n请确保已安装依赖:")
        print("   uv pip install diffusers transformers torch pillow opencv-python")
        return
    
    print("\n选择模式:")
    print("  1. 单张图片处理")
    print("  2. 批量处理文件夹")
    print()
    
    choice = input("请选择 (1-2): ").strip()
    
    if choice == "1":
        # 单张图片
        image_path = input("\n输入图片路径: ").strip().strip('"').strip("'")
        if not os.path.exists(image_path):
            print(f"❌ 文件不存在: {image_path}")
            return
        
        # 询问参数
        print("\n高级选项（直接回车使用默认值）:")
        steps = input("  推理步数 [30-50，默认 40]: ").strip()
        steps = int(steps) if steps else 40
        
        guidance = input("  引导强度 [5-10，默认 7.5]: ").strip()
        guidance = float(guidance) if guidance else 7.5
        
        # 处理
        print()
        remover.remove_watermark(
            image_path,
            num_inference_steps=steps,
            guidance_scale=guidance
        )
        
    elif choice == "2":
        # 批量处理
        image_dir = input("\n输入图片文件夹路径: ").strip().strip('"').strip("'")
        if not os.path.exists(image_dir):
            print(f"❌ 文件夹不存在: {image_dir}")
            return
        
        # 询问参数
        print("\n高级选项（直接回车使用默认值）:")
        steps = input("  推理步数 [30-50，默认 40]: ").strip()
        steps = int(steps) if steps else 40
        
        remover.batch_process(
            image_dir,
            num_inference_steps=steps
        )
    
    else:
        print("❌ 无效选择")
        return
    
    print("\n" + "=" * 70)
    print("✓ 全部完成！")
    print("=" * 70)


if __name__ == "__main__":
    # 快速测试用法
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image = os.path.join(script_dir, "images/11-22/02.JPG")
    
    if len(os.sys.argv) > 1:
        # 命令行模式
        image_path = os.sys.argv[1]
        remover = SDWatermarkRemover()
        remover.remove_watermark(image_path)
    elif os.path.exists(test_image):
        # 快速测试模式
        print("🧪 检测到测试图片，直接处理...")
        remover = SDWatermarkRemover()
        remover.remove_watermark(test_image, num_inference_steps=30)
    else:
        # 交互模式
        main()

