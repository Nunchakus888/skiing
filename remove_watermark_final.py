#!/usr/bin/env python3
"""
最终正确方案 - 精准检测 + 温和修复
核心原则：
1. 只检测水印文字（不是整个区域）
2. Mask应该<20%（而不是88.9%）
3. 适中的strength（0.96，而不是0.99）
4. 保护所有非水印元素
"""

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
import os
import easyocr

class PreciseWatermarkRemover:
    """精准水印去除 - 保护原图"""
    
    def __init__(self, device=None):
        """初始化"""
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
        
        print(f"🚀 初始化模型（设备: {device}）...")
        
        self.device = device
        self.dtype = dtype
        
        # 加载OCR
        print("   [1/2] 加载 EasyOCR...")
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=(device == 'cuda'))
        
        # 加载SD
        print("   [2/2] 加载 SD Inpainting...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=dtype,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(device)
        
        if device == "cuda":
            self.pipe.enable_attention_slicing()
        
        print("✓ 模型加载完成\n")
    
    def detect_watermark_precise(self, image_path):
        """
        精准检测水印 - 只检测文字，不检测其他元素
        
        策略：
        1. OCR检测文字位置（基础）
        2. 颜色检测灰蓝色文字（补充）
        3. 严格过滤：只保留小面积、高长宽比的区域（文字特征）
        """
        print("🔍 精准检测水印文字...")
        
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        h, w = img.shape[:2]
        mask_final = np.zeros((h, w), dtype=np.uint8)
        
        # ============== 方法1: OCR检测（主要方法）==============
        print("   [1/3] OCR检测文字位置...")
        
        ocr_results = self.ocr_reader.readtext(image_path)
        
        text_count = 0
        for bbox, text, conf in ocr_results:
            if conf < 0.3:  # 低置信度跳过
                continue
            
            # 获取边界框
            points = np.array(bbox, dtype=np.int32)
            x_min = max(0, points[:, 0].min() - 10)
            x_max = min(w, points[:, 0].max() + 10)
            y_min = max(0, points[:, 1].min() - 10)
            y_max = min(h, points[:, 1].max() + 10)
            
            # 绘制矩形
            cv2.rectangle(mask_final, (x_min, y_min), (x_max, y_max), 255, -1)
            text_count += 1
            print(f"      检测到: '{text}' (置信度: {conf:.2f})")
        
        print(f"      OCR检测: {text_count} 个文字区域")
        
        # ============== 方法2: 增强颜色检测（多种灰色调）==============
        print("   [2/3] 颜色检测多种灰色调水印...")
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mask_color_all = np.zeros((h, w), dtype=np.uint8)
        
        # 检测多种颜色的水印
        color_ranges = [
            # (名称, HSV下限, HSV上限)
            ("灰蓝色", [90, 15, 70], [130, 180, 200]),
            ("浅灰色", [0, 0, 120], [180, 60, 200]),
            ("深灰色", [0, 0, 60], [180, 100, 140]),
        ]
        
        color_total = 0
        for color_name, lower, upper in color_ranges:
            mask_range = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # 额外的亮度过滤（避免误检纯白和纯黑）
            mask_range = cv2.bitwise_and(mask_range, cv2.inRange(gray, 80, 220))
            
            # 形态学处理
            kernel_small = np.ones((3, 3), np.uint8)
            mask_range = cv2.morphologyEx(mask_range, cv2.MORPH_CLOSE, kernel_small)
            
            # 过滤：只保留文字形状的区域
            contours, _ = cv2.findContours(mask_range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 文字特征：面积在80-8000之间（放宽范围）
                if area < 80 or area > 8000:
                    continue
                
                # 计算长宽比
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = max(w_box, h_box) / max(min(w_box, h_box), 1)
                
                # 文字特征：长宽比 > 1.3（放宽）
                if aspect_ratio < 1.3:
                    continue
                
                # 保留此区域
                cv2.drawContours(mask_color_all, [contour], -1, 255, -1)
                count += 1
            
            if count > 0:
                print(f"      {color_name}: {count} 个区域")
                color_total += count
        
        # 合并到总mask
        mask_final = cv2.bitwise_or(mask_final, mask_color_all)
        
        print(f"      颜色检测总计: {color_total} 个水印区域")
        
        # ============== 方法3: 扩展Mask（连接附近的文字）==============
        print("   [3/3] 扩展并连接水印区域...")
        
        # 适度膨胀（覆盖文字边缘）
        kernel_dilate = np.ones((7, 7), np.uint8)
        mask_final = cv2.dilate(mask_final, kernel_dilate, iterations=2)
        
        # 闭运算（填充文字内部的空洞）
        kernel_close = np.ones((10, 10), np.uint8)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel_close)
        
        # 保护核心人物区域（扩大保护范围）
        center_x, center_y = w // 2, h // 2
        
        # 保护人脸核心（8% x 10%）
        face_w, face_h = int(w * 0.08), int(h * 0.10)
        mask_final[
            max(0, center_y-face_h):min(h, center_y+face_h),
            max(0, center_x-face_w):min(w, center_x+face_w)
        ] = 0
        
        # 保护身体核心（12% x 20%）
        body_w, body_h = int(w * 0.12), int(h * 0.20)
        mask_final[
            max(0, center_y):min(h, center_y+body_h),
            max(0, center_x-body_w):min(w, center_x+body_w)
        ] = 0
        
        mask_filtered = mask_final
        
        # 统计
        total_pixels = np.sum(mask_filtered > 0)
        percentage = (total_pixels / (h * w)) * 100
        
        print(f"\n   ✓ 最终检测结果:")
        print(f"      水印区域: {total_pixels:,} 像素 ({percentage:.1f}%)")
        
        if percentage > 30:
            print(f"      ⚠️  警告: 水印区域过大（>{30}%），可能误检！")
        elif percentage < 2:
            print(f"      ⚠️  警告: 水印区域过小（<{2}%），可能漏检！")
        else:
            print(f"      ✓ 水印区域合理")
        
        return mask_filtered
    
    def remove_watermark_tiled(self, image_path, mask=None, output_path=None,
                               num_inference_steps=50, strength=0.94, tile_size=300):
        """
        分块修复策略 - 解决密集水印问题
        
        将图片分成多个tile，每个tile单独修复，避免单次修复范围过大
        
        Args:
            strength: 0.94 = 平衡修复
            tile_size: 每个tile的大小（像素）
        """
        # 读取图片
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        w, h = original_size
        
        # 自动检测水印
        if mask is None:
            mask_np = self.detect_watermark_precise(image_path)
            if mask_np is None:
                print("❌ 水印检测失败")
                return None, None
        else:
            if isinstance(mask, np.ndarray):
                mask_np = mask
            else:
                mask_np = np.array(mask)
        
        # 检查mask大小
        mask_ratio = np.sum(mask_np > 0) / (mask_np.shape[0] * mask_np.shape[1])
        
        # 生成默认output_path
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_final{ext}"
        
        # 如果mask太大，使用分块策略
        if mask_ratio > 0.4:
            print(f"\n💡 检测到密集水印（{mask_ratio*100:.1f}%），启用分块修复策略...")
            return self._remove_watermark_tiled_impl(
                image, mask_np, output_path, num_inference_steps, strength, tile_size
            )
        else:
            # mask不大，直接修复
            print(f"\n✓ 水印覆盖率{mask_ratio*100:.1f}%，使用标准修复...")
            return self._remove_watermark_standard(
                image, mask_np, output_path, num_inference_steps, strength
            )
    
    def _remove_watermark_tiled_impl(self, image_pil, mask_np, output_path, 
                                     num_inference_steps, strength, tile_size):
        """分块修复实现"""
        import numpy as np
        from PIL import Image
        
        w, h = image_pil.size
        image_np = np.array(image_pil)
        result_np = image_np.copy()
        
        # 计算tile数量
        tiles_x = (w + tile_size - 1) // tile_size
        tiles_y = (h + tile_size - 1) // tile_size
        
        print(f"   分块策略: {tiles_x}x{tiles_y} = {tiles_x*tiles_y}个区块")
        print(f"   每个区块: {tile_size}x{tile_size}像素\n")
        
        processed_count = 0
        total_tiles = tiles_x * tiles_y
        
        # 计算实际需要处理的tile数量（用于显示进度）
        tiles_to_process = []
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x1 = tx * tile_size
                y1 = ty * tile_size
                x2 = min(x1 + tile_size, w)
                y2 = min(y1 + tile_size, h)
                tile_mask = mask_np[y1:y2, x1:x2]
                if np.sum(tile_mask > 0) / (tile_mask.size) >= 0.01:
                    tiles_to_process.append((tx, ty))
        
        total_active_tiles = len(tiles_to_process)
        print(f"   预计处理: {total_active_tiles} 个含有水印的区块")
        
        for i, (tx, ty) in enumerate(tiles_to_process):
            processed_count += 1
            
            # 计算当前tile的范围
            x1 = tx * tile_size
            y1 = ty * tile_size
            x2 = min(x1 + tile_size, w)
            y2 = min(y1 + tile_size, h)
            
            # 提取当前tile的mask
            tile_mask = mask_np[y1:y2, x1:x2]
            tile_mask_ratio = np.sum(tile_mask > 0) / (tile_mask.size)
            
            print(f"   [{processed_count}/{total_active_tiles}] 处理区块 ({tx},{ty}) 水印占比{tile_mask_ratio*100:.1f}% ... ", end="", flush=True)
            
            # 提取tile图像
            tile_image = image_np[y1:y2, x1:x2]
            tile_image_pil = Image.fromarray(tile_image)
            tile_mask_pil = Image.fromarray(tile_mask)
            
            # 调整尺寸到8的倍数
            tile_w, tile_h = tile_image_pil.size
            new_w = (tile_w // 8) * 8
            new_h = (tile_h // 8) * 8
            
            if new_w < 64 or new_h < 64:
                print("跳过（区块太小）")
                continue
            
            tile_image_resized = tile_image_pil.resize((new_w, new_h), Image.LANCZOS)
            tile_mask_resized = tile_mask_pil.resize((new_w, new_h), Image.LANCZOS)
            
            # SD修复
            prompt = """exact same content, preserve original, only remove text,
                        match surrounding colors and textures perfectly"""
            
            negative_prompt = """text, watermark, any changes, new elements, blurry"""
            
            try:
                # 启用内部进度条，让用户看到每一块的进度
                self.pipe.set_progress_bar_config(disable=False)
                print(f"\n   🚀 正在修复区块 [{processed_count}/{total_active_tiles}]...")
                
                tile_result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=tile_image_resized,
                    mask_image=tile_mask_resized,
                    num_inference_steps=20,  # 降低到20步，大幅提速
                    guidance_scale=5.5,
                    strength=min(0.95, strength + 0.02),
                ).images[0]
                
                # 恢复尺寸
                if tile_result.size != (tile_w, tile_h):
                    tile_result = tile_result.resize((tile_w, tile_h), Image.LANCZOS)
                
                # 写回result
                result_np[y1:y2, x1:x2] = np.array(tile_result)
                print("✅ 完成")
                
            except Exception as e:
                print(f"❌ 失败: {e}")
                continue
        
        print(f"\n✓ 分块修复完成！共处理{processed_count}个区块\n")
        
        # 保存结果
        result_pil = Image.fromarray(result_np)
        
        if output_path is None:
            base, ext = os.path.splitext(str(image_pil.filename) if hasattr(image_pil, 'filename') else "output.jpg")
            output_path = f"{base}_final{ext}"
        
        result_pil.save(output_path, quality=95)
        print(f"✓ 保存到: {output_path}\n")
        
        return result_pil, output_path
    
    def _remove_watermark_standard(self, image_pil, mask_np, output_path,
                                   num_inference_steps, strength):
        """标准修复实现"""
        original_size = image_pil.size
        
        # 转换 mask
        mask_pil = Image.fromarray(mask_np).convert("L")
        mask_pil = mask_pil.resize(original_size, Image.LANCZOS)
        
        # 调整尺寸
        def resize_to_multiple_of_8(img):
            w, h = img.size
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
            return img.resize((new_w, new_h), Image.LANCZOS)
        
        image_resized = resize_to_multiple_of_8(image_pil)
        mask_resized = resize_to_multiple_of_8(mask_pil)
        
        print(f"\n🎨 开始 AI 修复...")
        print(f"   原始尺寸: {original_size}")
        print(f"   处理尺寸: {image_resized.size}")
        print(f"   推理步数: {num_inference_steps}")
        print(f"   修复强度: {strength} （保守模式 - 严格保护原图）")
        
        # 优化的提示词 - 极度强调保持原样
        prompt = """exact same person, same face, same clothes, same pose, same everything,
                    preserve all original elements, only remove text overlay,
                    keep original background colors and textures,
                    inpaint only watermarked areas with matching background,
                    photorealistic, high quality"""
        
        negative_prompt = """text, watermark, logo, words, letters, chinese characters, stamps,
                            any changes to person, different face, different clothes, new elements,
                            different pose, different background, added objects, goggles change,
                            helmet change, cloth color change, face modification,
                            blurry, artifacts, distorted, unrealistic"""
        
        # SD Inpainting - 保守模式
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=6.0,  # 降低引导强度，减少创造性修改
            strength=strength,  # 保守修复
        ).images[0]
        
        # 恢复尺寸
        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)
        
        # 保存结果
        if output_path is None:
            output_path = "output_final.jpg"
        
        result.save(output_path, quality=95)
        print(f"\n✓ 修复完成！保存到: {output_path}\n")
        
        return result, output_path


def main():
    """主程序"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: uv run python remove_watermark_final.py <图片路径> [步数] [强度]")
        print("\n示例:")
        print("  标准: uv run python remove_watermark_final.py image.jpg")
        print("  高质量: uv run python remove_watermark_final.py image.jpg 70")
        print("  自定义: uv run python remove_watermark_final.py image.jpg 50 0.96")
        print("\n参数说明:")
        print("  步数: 30-100 (默认50)")
        print("  强度: 0.90-0.98 (默认0.96，保护原图)")
        return
    
    image_path = sys.argv[1]
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    strength = float(sys.argv[3]) if len(sys.argv) > 3 else 0.96
    
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在: {image_path}")
        return
    
    print("=" * 70)
    print("🎯 精准水印去除工具")
    print(f"   模式: 温和修复 (strength={strength})")
    print(f"   步数: {num_steps}")
    print("   策略: 精准检测 + 保护原图")
    print("=" * 70)
    print()
    
    # 初始化
    remover = PreciseWatermarkRemover()
    
    # 处理
    remover.remove_watermark_tiled(
        image_path,
        num_inference_steps=num_steps,
        strength=strength
    )
    
    print("=" * 70)
    print("✓ 全部完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

