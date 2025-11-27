#!/usr/bin/env python3
"""
Smart Watermark Remover using Stable Diffusion
智能水印去除工具 - 结合 OCR 学习与 SD 修复

Usage:
    uv run python remove_watermark_smart.py images/11-22/02.JPG
"""

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline
import easyocr
import os
import sys

class SmartRemover:
    def __init__(self):
        self.device = self._get_device()
        print(f"🚀 初始化智能去除引擎 (Device: {self.device})")
        
        # 1. Initialize OCR (for watermark learning)
        print("   [1/2] 加载文字识别模型...")
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=(self.device == 'cuda'))
        
        # 2. Initialize SD (for inpainting)
        print("   [2/2] 加载图像修复模型...")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=self.dtype,
            safety_checker=None
        ).to(self.device)
        
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            
        print("✓ 系统就绪\n")

    def _get_device(self):
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def generate_smart_mask(self, img_path):
        """
        Generate a comprehensive watermark mask using:
        1. OCR Detection (High precision anchors)
        2. Color Learning (Propagate to missed text)
        3. Pattern Matching (Repetitive elements)
        """
        print(f"🧠 正在分析图像结构: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # 1. OCR Pass - Get Anchors
        results = self.reader.readtext(img_path)
        ocr_mask = np.zeros((h, w), dtype=np.uint8)
        
        watermark_colors = []
        detected_texts = []
        
        for bbox, text, conf in results:
            # Filter out likely non-watermark text (low confidence or too large)
            if conf < 0.1: continue
            
            pts = np.array(bbox, dtype=np.int32)
            # Draw on OCR mask
            cv2.fillPoly(ocr_mask, [pts], 255)
            detected_texts.append(text)
            
            # Sample color from this region (center 50%)
            mask_roi = np.zeros((h,w), dtype=np.uint8)
            cv2.fillPoly(mask_roi, [pts], 255)
            # Erode to get core text color
            mask_roi = cv2.erode(mask_roi, np.ones((3,3)), iterations=1)
            
            mean_color = cv2.mean(img, mask=mask_roi)[:3]
            watermark_colors.append(mean_color)

        print(f"   识别到 {len(detected_texts)} 处文字锚点")
        
        # 2. Color Analysis & Expansion
        # If we found text, use its color profile to find missed text
        full_mask = ocr_mask.copy()
        
        # Default watermark colors (Blue-ish Grey & White) if OCR fails or as supplement
        # HSV Ranges for common watermarks
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Range 1: The typical "Huabei" Blue-Grey
        # H: 100-125 (Blue), S: 10-50 (Low Sat), V: 100-200 (Med Bright)
        mask_blue_grey = cv2.inRange(hsv, np.array([90, 10, 80]), np.array([130, 60, 220]))
        
        # Range 2: White/Light Grey text
        # Low Saturation, High Value
        mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        
        # Combine color priors
        color_mask = cv2.bitwise_or(mask_blue_grey, mask_white)
        
        # 3. Edge/Texture Filter
        # Watermarks are high-frequency text. Filter out flat areas (like sky) that matched color.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # Threshold gradient to find edges
        _, edge_mask = cv2.threshold(grad, 20, 255, cv2.THRESH_BINARY)
        
        # Refine Color Mask: Must be Color Match AND Edge Match
        refined_color_mask = cv2.bitwise_and(color_mask, edge_mask)
        
        # 4. Merge
        # Combine OCR (High confidence) with Refined Color (High recall)
        final_mask = cv2.bitwise_or(full_mask, refined_color_mask)
        
        # 5. Morphology Cleanup
        # Connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        final_mask = cv2.dilate(final_mask, kernel, iterations=2)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        # 6. Safety Mechanism: Face/Subject Protection
        # Simple heuristic: Don't touch the center-mass of the image if it looks like a person (skiing)
        # Assuming photographer focuses on subject.
        # Let's use a "soft" protection mask? No, hard mask is better for inpainting.
        # We will mask out the CENTER 20% of the image ONLY IF it's not explicitly OCR'd text.
        # Actually, OCR text on the face SHOULD be removed.
        # But random noise on the face should NOT.
        # So: Keep OCR parts. For the "Color/Edge" parts, exclude the central face region.
        
        center_h, center_w = h//2, w//2
        face_box_size = int(min(h,w) * 0.15)
        safe_zone_mask = np.zeros_like(final_mask)
        cv2.rectangle(safe_zone_mask, 
                     (center_w - face_box_size, center_h - face_box_size),
                     (center_w + face_box_size, center_h + face_box_size), 
                     255, -1)
        
        # Remove "Color/Edge" detections from Safe Zone, but KEEP OCR detections
        # Logic: Final = OCR + (RefinedColor - SafeZone)
        safe_refined = cv2.bitwise_and(refined_color_mask, cv2.bitwise_not(safe_zone_mask))
        final_mask = cv2.bitwise_or(full_mask, safe_refined)

        pixel_coverage = np.sum(final_mask > 0) / (h*w)
        print(f"   水印覆盖率: {pixel_coverage:.1%}")
        
        return final_mask

    def process(self, image_path, output_path=None):
        if not output_path:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_smart{ext}"
            
        # 1. Get Mask
        mask_np = self.generate_smart_mask(image_path)
        
        # 2. Prepare for SD
        image = Image.open(image_path).convert("RGB")
        mask = Image.fromarray(mask_np).convert("L")
        
        orig_w, orig_h = image.size
        
        # Resize to SD friendly (nearest multiple of 8, max 1024 for speed/quality balance)
        # SD1.5 is best at 512x512, but can do higher.
        scale_factor = 1.0
        if max(orig_w, orig_h) > 1280:
            scale_factor = 1280 / max(orig_w, orig_h)
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            mask = mask.resize((new_w, new_h), Image.NEAREST)
        
        # Ensure div by 8
        w, h = image.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        image = image.resize((w, h))
        mask = mask.resize((w, h))
        
        print(f"🎨 开始生成 (尺寸: {w}x{h})...")
        
        # 3. Inpaint
        prompt = "clean photography, high quality, 8k, snow mountain, skiing, clear details, photorealistic"
        negative_prompt = "watermark, text, logo, signature, blurry, artifacts, bad quality, distorted, ugly"
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=40,
            guidance_scale=7.5,
            strength=1.0
        ).images[0]
        
        # 4. Post-process & Save
        # Always resize result back to original dimensions to match orig_img
        if result.size != (orig_w, orig_h):
            result = result.resize((orig_w, orig_h), Image.LANCZOS)
            
        # Blend back original non-masked pixels to ensure perfect fidelity
        # (SD sometimes hallucinates slightly on non-masked areas due to VAE)
        # Reload original to be sure
        orig_img = Image.open(image_path).convert("RGB")
        orig_mask = Image.fromarray(mask_np).convert("L")
        
        # Ensure mask matches exactly
        if orig_mask.size != (orig_w, orig_h):
             orig_mask = orig_mask.resize((orig_w, orig_h), Image.NEAREST)
        
        # Composite
        final_result = Image.composite(result, orig_img, orig_mask)
        
        final_result.save(output_path, quality=95)
        print(f"✨ 处理完成: {output_path}")
        return output_path

def main():
    if len(sys.argv) < 2:
        print("Use: python remove_watermark_smart.py <image_path>")
        return
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"❌ Not found: {img_path}")
        return
        
    remover = SmartRemover()
    remover.process(img_path)

if __name__ == "__main__":
    main()

