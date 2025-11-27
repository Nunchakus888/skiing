#!/usr/bin/env python3
"""
精确去除水印 - 保护原图版本
策略：
1. 使用 OCR 精确识别水印文字位置
2. 只修复水印区域，完全不改变其他部分
3. 使用保守的 SD 参数，保护人物和背景
"""

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
import os

class PreciseWatermarkRemover:
    """精确水印去除器 - 只处理文字，保护原图"""
    
    def __init__(self, device=None):
        """初始化模型"""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float32
            else:
                device = "cpu"
                dtype = torch.float32
        
        print(f"🚀 初始化 SD Inpainting 模型（设备: {device}）...")
        
        self.device = device
        self.dtype = dtype
        
        # 加载模型
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=dtype,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(device)
        
        if device == "cuda":
            self.pipe.enable_attention_slicing()
        
        print("✓ 模型加载完成\n")
    
    def detect_text_with_ocr(self, image_path):
        """
        使用 OCR 精确识别水印文字位置
        返回精确的文字 mask
        """
        try:
            import easyocr
        except ImportError:
            print("❌ 需要安装 EasyOCR")
            print("   uv pip install easyocr")
            return None
        
        print("📝 使用 OCR 识别水印文字...")
        
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 初始化 OCR
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        results = reader.readtext(image_path)
        
        # 水印关键词
        watermark_keywords = [
            '滑呗', 'app', '1000', '万', '雪友', '选择',
            '酒店', '教练', '摄影师', '约玩', '雪票', 'BDH'
        ]
        
        # 核心人物保护区域（只保护脸部核心 - 缩小到 20% x 20%）
        center_x, center_y = w // 2, h // 2
        face_w, face_h = int(w * 0.10), int(h * 0.10)  # 核心脸部区域
        face_x1 = center_x - face_w
        face_x2 = center_x + face_w
        face_y1 = center_y - face_h
        face_y2 = center_y + face_h
        
        detected_count = 0
        person_area_count = 0  # 人物区域的水印（需要特殊处理）
        
        for (bbox, text, prob) in results:
            # 检查是否是水印
            is_watermark = any(keyword in text for keyword in watermark_keywords)
            
            if not is_watermark or prob < 0.2:
                continue
            
            # 获取边界框
            pts = np.array(bbox, dtype=np.int32)
            center_x_text = int(np.mean(pts[:, 0]))
            center_y_text = int(np.mean(pts[:, 1]))
            
            # 检查是否在核心脸部区域（真正需要保护的）
            in_face_area = (face_x1 <= center_x_text <= face_x2 and 
                           face_y1 <= center_y_text <= face_y2)
            
            if in_face_area:
                print(f"   🛡️  跳过核心脸部: '{text}'")
                continue
            
            # 精确标记文字区域（适度扩大）
            width = int(np.max(pts[:, 0]) - np.min(pts[:, 0]))
            height = int(np.max(pts[:, 1]) - np.min(pts[:, 1]))
            
            # 检查是否在人物区域（身体部分）
            person_x1 = center_x - int(w * 0.25)
            person_x2 = center_x + int(w * 0.25)
            person_y1 = center_y - int(h * 0.35)
            person_y2 = center_y + int(h * 0.35)
            
            in_person_area = (person_x1 <= center_x_text <= person_x2 and 
                             person_y1 <= center_y_text <= person_y2)
            
            # 根据位置调整扩展比例
            if in_person_area:
                # 人物区域：保守扩展，避免破坏衣服纹理
                expand_ratio = 1.2
                person_area_count += 1
                location_tag = "👤人物"
            else:
                # 背景区域：正常扩展
                expand_ratio = 1.3
                location_tag = "🌄背景"
            
            new_width = int(width * expand_ratio)
            new_height = int(height * expand_ratio)
            
            x1 = max(0, center_x_text - new_width // 2)
            y1 = max(0, center_y_text - new_height // 2)
            x2 = min(w, center_x_text + new_width // 2)
            y2 = min(h, center_y_text + new_height // 2)
            
            # 标记区域
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            detected_count += 1
            print(f"   ✓ {location_tag} 水印: '{text}' (位置: {center_x_text}, {center_y_text})")
        
        # 温和的膨胀（连接相邻文字）
        if detected_count > 0:
            kernel = np.ones((8, 8), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        print(f"\n   检测结果: 总共 {detected_count} 处水印")
        print(f"   - 人物区域: {person_area_count} 处（保守处理）")
        print(f"   - 背景区域: {detected_count - person_area_count} 处")
        
        # 统计
        watermark_pixels = np.sum(mask > 0)
        percentage = (watermark_pixels / (h * w)) * 100
        print(f"   水印区域: {watermark_pixels:,} 像素 ({percentage:.1f}%)\n")
        
        return mask
    
    def remove_watermark(self, image_path, mask=None, output_path=None,
                        num_inference_steps=50, strength=0.95):
        """
        精确去除水印
        
        Args:
            strength: 修复强度 (0.8-1.0)
                     0.8-0.9: 保守，更好地保持原图
                     0.95-1.0: 激进，完全重绘
        """
        # 读取图片
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # 自动检测水印
        if mask is None:
            mask = self.detect_text_with_ocr(image_path)
            if mask is None:
                print("❌ OCR 检测失败")
                return None, None
        
        # 转换 mask
        if isinstance(mask, np.ndarray):
            mask_pil = Image.fromarray(mask).convert("L")
        else:
            mask_pil = mask
        
        mask_pil = mask_pil.resize(original_size, Image.LANCZOS)
        
        # 调整尺寸（8 的倍数）
        def resize_to_multiple_of_8(img):
            w, h = img.size
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
            return img.resize((new_w, new_h), Image.LANCZOS)
        
        image_resized = resize_to_multiple_of_8(image)
        mask_resized = resize_to_multiple_of_8(mask_pil)
        
        print(f"🎨 开始 AI 修复...")
        print(f"   原始尺寸: {original_size}")
        print(f"   处理尺寸: {image_resized.size}")
        print(f"   推理步数: {num_inference_steps}")
        print(f"   修复强度: {strength} (保护原图)")
        
        # 优化的提示词（专门针对滑雪照片）
        prompt = """professional skiing photo, natural snow mountain landscape, 
                    clear blue sky, natural lighting, high quality, photorealistic, 
                    clean image without any text or watermark"""
        
        negative_prompt = """text, watermark, logo, words, letters, numbers, 
                            chinese characters, overlay text, blurry, low quality, 
                            distorted, artificial, fake"""
        
        # SD Inpainting
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            strength=strength,  # 保守的强度
        ).images[0]
        
        # 恢复原始尺寸
        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)
        
        # 可选：混合原图和修复结果（进一步保护原图）
        result_array = np.array(result)
        original_array = np.array(image)
        mask_array = np.array(mask_pil)
        
        # 只在 mask 区域应用修复，其他地方保持原样
        mask_3d = np.stack([mask_array] * 3, axis=2) / 255.0
        blended = (result_array * mask_3d + original_array * (1 - mask_3d)).astype(np.uint8)
        result_blended = Image.fromarray(blended)
        
        # 保存结果
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_precise{ext}"
        
        result_blended.save(output_path, quality=95)
        print(f"✓ 修复完成！保存到: {output_path}\n")
        
        return result_blended, output_path


def main():
    """快速测试"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: uv run python remove_watermark_precise.py <图片路径> [推理步数]")
        print("\n示例:")
        print("  快速模式: uv run python remove_watermark_precise.py images/11-22/02.JPG")
        print("  高质量:   uv run python remove_watermark_precise.py images/11-22/02.JPG 70")
        print("  极致质量: uv run python remove_watermark_precise.py images/11-22/02.JPG 100")
        print("\n推荐步数:")
        print("  30-40: 快速（1-2分钟）")
        print("  50-70: 标准质量（2-3分钟）⭐推荐")
        print("  80-100: 最高质量（3-5分钟）")
        return
    
    image_path = sys.argv[1]
    
    # 获取推理步数（默认70）
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 70
    
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在: {image_path}")
        return
    
    # 根据步数选择模式
    if num_steps >= 80:
        mode = "🌟 极致质量模式"
    elif num_steps >= 60:
        mode = "⭐ 高质量模式"
    elif num_steps >= 40:
        mode = "✓ 标准模式"
    else:
        mode = "⚡ 快速模式"
    
    print("=" * 70)
    print("🎯 精确水印去除工具")
    print(f"   模式: {mode} ({num_steps} 步)")
    print("   策略: 只去除文字，完全保护原图")
    print("=" * 70)
    print()
    
    # 初始化
    remover = PreciseWatermarkRemover()
    
    # 处理
    remover.remove_watermark(
        image_path,
        num_inference_steps=num_steps,
        strength=0.95  # 保守的修复强度
    )
    
    print("=" * 70)
    print("✓ 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

