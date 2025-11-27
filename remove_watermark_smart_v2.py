#!/usr/bin/env python3
"""
Smart Watermark Remover V6 - Multi-Threshold Enhanced
Strategy: Grounded-SAM + Multi-Scale Detection + Human Protection

Optimization Features:
1. 多阈值融合检测 (0.30/0.20/0.15) - 捕获模糊/半透明水印
2. 改进 Prompt (中英文 + 具体描述) - 提升语义理解
3. 增强形态学处理 (iterations=3) - 覆盖边缘残留
4. 人物保护机制 - 避免误伤主体

Further Optimization Options:
- 升级 SAM_TYPE 为 "vit_l" 或 "vit_h" (更精细分割)
- 降低阈值到 0.05 (极限模式，但可能误检)
- 添加图像增强预处理 (CLAHE)
"""

import os
import sys
import cv2
import torch
import numpy as np
import datetime
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor

# Configuration
GDINO_MODEL = "IDEA-Research/grounding-dino-base"

# SAM Model Selection (升级到更大模型以提高精度)
# vit_b: 90M params, 375MB  (当前)
# vit_l: 300M params, 1.2GB (推荐)
# vit_h: 600M params, 2.4GB (最强)
SAM_TYPE = "vit_b"  # 可改为 "vit_l" 或 "vit_h"
SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}
SAM_CHECKPOINT = os.path.expanduser(f"~/.cache/sam_{SAM_TYPE}.pth")
SAM_URL = SAM_URLS[SAM_TYPE]

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class SmartRemover:
    def __init__(self):
        self.device = self._get_device()
        print(f"🚀 初始化人物保护型去水印系统 (Device: {self.device})")
        
        # 1. Load GroundingDINO
        print("   [1/3] 加载 GroundingDINO (视觉定位)...")
        try:
            self.processor = AutoProcessor.from_pretrained(GDINO_MODEL)
            self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(self.device)
        except Exception as e:
            print(f"❌ GroundingDINO 加载失败: {e}")
            sys.exit(1)

        # 2. Load SAM
        print("   [2/3] 加载 SAM (精细分割)...")
        self._ensure_sam_model()
        try:
            import functools
            original_load = torch.load
            torch.load = functools.partial(original_load, weights_only=False)
            sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
            torch.load = original_load
            sam.to(self.device)
            self.sam_predictor = SamPredictor(sam)
        except Exception as e:
            print(f"❌ SAM 加载失败: {e}\n   rm {SAM_CHECKPOINT}")
            sys.exit(1)

        # 3. Load SD Inpainting
        print("   [3/3] 加载图像修复模型...")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=self.dtype,
            safety_checker=None
        ).to(self.device)
        if self.device == "cuda": self.pipe.enable_attention_slicing()
        print("✓ 系统就绪\n")

    def _get_device(self):
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def _ensure_sam_model(self):
        if os.path.exists(SAM_CHECKPOINT):
            # Header check (PK..)
            with open(SAM_CHECKPOINT, "rb") as f: header = f.read(4)
            if os.path.getsize(SAM_CHECKPOINT) > 100*1024*1024 and (header.startswith(b'PK') or header.startswith(b'\x80')):
                return
            os.remove(SAM_CHECKPOINT)
        
        print(f"   📥 下载 SAM 模型...")
        os.makedirs(os.path.dirname(SAM_CHECKPOINT), exist_ok=True)
        os.system(f"curl -L -k -o {SAM_CHECKPOINT} {SAM_URL}")
        if not os.path.exists(SAM_CHECKPOINT):
            os.system(f"wget --no-check-certificate {SAM_URL} -O {SAM_CHECKPOINT}")

    def detect(self, image_pil, prompt, box_threshold=0.3):
        """Generic detection wrapper"""
        inputs = self.processor(images=image_pil, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)
        
        target_sizes = torch.tensor([image_pil.size[::-1]], device=self.device)
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=box_threshold, target_sizes=target_sizes
        )[0]
        return results["boxes"].cpu().numpy(), results["scores"].cpu().numpy()

    def get_sam_mask(self, image_cv, boxes):
        """Get binary mask from boxes using SAM"""
        if len(boxes) == 0: return np.zeros(image_cv.shape[:2], dtype=np.uint8)
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            torch.tensor(boxes, device=self.device), image_cv.shape[:2]
        )
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=transformed_boxes, multimask_output=False,
        )
        # Combine all masks
        if len(masks) == 0: return np.zeros(image_cv.shape[:2], dtype=np.uint8)
        
        # masks: (N, 1, H, W) -> (N, H, W) -> (H, W)
        final = torch.any(masks.squeeze(1), dim=0).cpu().numpy().astype(np.uint8) * 255
        return np.ascontiguousarray(final)

    def detect_multiscale_enhanced(self, image_pil, image_cv, prompt):
        """
        增强版多尺度检测:
        1. 多阈值 (0.25, 0.15, 0.08)
        2. 多尺度图像 (1x, 1.5x, 2x) - 放大后小水印更易检测
        3. 图像增强 (CLAHE)
        """
        all_boxes, all_scores = [], []
        orig_w, orig_h = image_pil.size
        
        # 策略1: 原图多阈值 (极低阈值捕获模糊水印)
        for thresh in [0.25, 0.15, 0.08, 0.05]:
            boxes, scores = self.detect(image_pil, prompt, thresh)
            all_boxes.extend(boxes)
            all_scores.extend(scores)
        
        # 策略2: 放大图像检测（捕获小水印）
        for scale in [1.5, 2.0, 2.5]:  # 增加 2.5x 超级放大
            scaled_w = int(orig_w * scale)
            scaled_h = int(orig_h * scale)
            scaled_img = image_pil.resize((scaled_w, scaled_h), Image.BICUBIC)
            
            boxes, scores = self.detect(scaled_img, prompt, 0.10)  # 降低阈值
            if len(boxes) > 0:
                # 映射回原图坐标
                boxes_scaled = boxes / scale
                all_boxes.extend(boxes_scaled)
                all_scores.extend(scores)
        
        # 策略3: CLAHE 增强后检测（低对比度水印）
        lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        
        # 多阈值检测增强图
        for thresh in [0.15, 0.08, 0.05]:
            boxes, scores = self.detect(enhanced_pil, prompt, thresh)
            all_boxes.extend(boxes)
            all_scores.extend(scores)
        
        if len(all_boxes) == 0:
            return np.array([]), np.array([])
        
        # NMS 去重
        boxes_np = np.array(all_boxes)
        scores_np = np.array(all_scores)
        keep = []
        indices = np.argsort(scores_np)[::-1]
        
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            if len(indices) == 1: break
            
            box1 = boxes_np[i]
            rest_boxes = boxes_np[indices[1:]]
            
            x1 = np.maximum(box1[0], rest_boxes[:, 0])
            y1 = np.maximum(box1[1], rest_boxes[:, 1])
            x2 = np.minimum(box1[2], rest_boxes[:, 2])
            y2 = np.minimum(box1[3], rest_boxes[:, 3])
            
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
            iou = inter / (area1 + area2 - inter + 1e-6)
            
            indices = indices[1:][iou < 0.5]
        
        return boxes_np[keep], scores_np[keep]
    
    def detect_edge_fallback(self, image_cv):
        """边缘检测回退方案 - 捕获遗漏的长条形水印"""
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        
        # 水平方向闭运算（连接水印字符）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = image_cv.shape[:2]
        boxes = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)
            area = cw * ch
            # 水印特征: 长条形 (aspect > 3), 面积适中
            if aspect > 3 and 1000 < area < h * w * 0.03:
                boxes.append([x, y, x + cw, y + ch])
        
        return np.array(boxes) if boxes else np.array([])

    def generate_mask(self, image_path):
        print(f"🧠 分析图像: {os.path.basename(image_path)}")
        image_pil = Image.open(image_path).convert("RGB")
        image_cv = cv2.imread(image_path)
        self.sam_predictor.set_image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        h, w = image_cv.shape[:2]

        # 1. 增强多尺度检测
        print("   [1/4] 扫描水印 (多尺度增强)...")
        wm_prompt = "text overlay. watermark. translucent text. copyright. timestamp. username. 水印. 文字."
        wm_boxes, wm_scores = self.detect_multiscale_enhanced(image_pil, image_cv, wm_prompt)
        
        print(f"       GroundingDINO: {len(wm_boxes)} 处")
        
        # 2. 回退机制：边缘检测
        if len(wm_boxes) < 5:
            print("   [1.5/4] 启用边缘检测辅助...")
            edge_boxes = self.detect_edge_fallback(image_cv)
            if len(edge_boxes) > 0:
                print(f"       边缘检测: +{len(edge_boxes)} 处")
                wm_boxes = np.vstack([wm_boxes, edge_boxes]) if len(wm_boxes) > 0 else edge_boxes
                wm_scores = np.concatenate([wm_scores, np.ones(len(edge_boxes)) * 0.5]) if len(wm_scores) > 0 else np.ones(len(edge_boxes)) * 0.5
        
        if len(wm_boxes) == 0:
            print("   ⚠️ 未发现水印")
            return np.zeros((h, w), dtype=np.uint8)
        
        print(f"       总计检测: {len(wm_boxes)} 处水印")

        # 2. 扫描人物 (保护)
        print("   [2/4] 扫描人物 (保护)...")
        p_boxes, p_scores = self.detect(image_pil, "person.", 0.40)

        # 3. SAM 精细分割
        print("   [3/4] SAM 精细分割...")
        wm_mask = self.get_sam_mask(image_cv, wm_boxes)
        
        if len(p_boxes) > 0:
            p_mask = self.get_sam_mask(image_cv, p_boxes)
            print(f"   [4/4] 智能融合 (水印:{len(wm_boxes)} - 人物:{len(p_boxes)})...")
            
            # 人物保护区膨胀
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            p_mask_safe = cv2.dilate(p_mask, kernel, iterations=3)
            
            # 逻辑运算: Watermark - Person
            final_mask = cv2.bitwise_and(wm_mask, cv2.bitwise_not(p_mask_safe))
        else:
            print(f"   [4/4] 未检测到人物，直接生成 Mask...")
            final_mask = wm_mask
        
        # 后处理：连通域合并 + 强力膨胀覆盖残留
        # 合并附近的小块（水印往往是分散的多个字符）
        kernel_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_merge)
        
        # 强力膨胀覆盖边缘残留 (彻底消除文字边缘)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7))
        final_mask = cv2.dilate(final_mask, kernel_dilate, iterations=6)

        coverage = np.sum(final_mask > 0) / (h * w)
        print(f"   ✓ Mask完成 (覆盖率: {coverage:.1%})")
        
        return final_mask

    def process(self, image_path):
        base, ext = os.path.splitext(image_path)
        hms = datetime.datetime.now().strftime("%m-%d%H%M%S")
        output_path = f"{base}_{hms}{ext}"
        
        # 1. Mask
        mask_np = self.generate_mask(image_path)
        if np.sum(mask_np) == 0: return None
        
        cv2.imwrite(f"{base}_mask.png", mask_np)
        
        # 2. Prepare
        image_cv = cv2.imread(image_path)
        orig_h, orig_w = image_cv.shape[:2]
        
        # Resize for SD
        w, h = (orig_w // 8) * 8, (orig_h // 8) * 8
        
        # 3. Two-Stage Inpainting (混合策略)
        print(f"🎨 修复中 ({w}x{h})...")
        
        # Stage 1: OpenCV Inpaint (保守纹理填充，绝不添加新元素)
        print("   [Stage 1/2] OpenCV 纹理填充...")
        image_cv_resized = cv2.resize(image_cv, (w, h), interpolation=cv2.INTER_LANCZOS4)
        mask_cv_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Telea 算法：基于快速行进法，纯粹复制周围纹理
        opencv_result = cv2.inpaint(image_cv_resized, mask_cv_resized, 5, cv2.INPAINT_TELEA)
        opencv_result_pil = Image.fromarray(cv2.cvtColor(opencv_result, cv2.COLOR_BGR2RGB))
        
        # Stage 2: SD 轻度 Refine (仅平滑边缘，不改变内容)
        print("   [Stage 2/2] SD 边缘平滑...")
        mask_pil = Image.fromarray(mask_cv_resized).convert("L")
        
        result = self.pipe(
            # 极简 prompt：只平滑，不生成
            prompt="smooth edges, blend seamlessly, no changes",
            # 超强 negative：禁止一切生成行为
            negative_prompt=(
                "new content, generated content, created content, synthetic content, "
                "person, people, human, face, body, character, figure, skier, "
                "man, woman, child, athlete, "
                "object, objects, element, elements, "
                "watermark, text, logo, letters, "
                "change, modification, alteration, addition, "
                "distorted, artifacts, blurry"
            ),
            image=opencv_result_pil,  # ⚠️ 输入是 OpenCV 结果，不是原图
            mask_image=mask_pil,
            num_inference_steps=30,   # 极少步数，只平滑边缘
            guidance_scale=15.0,      # 最强引导
            strength=0.35             # ⭐⭐⭐ 极低：只允许 35% 修改（主要用于平滑）
        ).images[0]
        
        result = result.resize((orig_w, orig_h), Image.LANCZOS)
            
        # Composite
        orig_img = Image.open(image_path).convert("RGB")
        final = Image.composite(result, orig_img, Image.fromarray(mask_np).convert("L"))
        
        final.save(output_path, quality=95)
        print(f"✨ 完成: {output_path}")
        return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_watermark_smart_v2.py <img_path>")
    else:
        if os.path.exists(sys.argv[1]):
            SmartRemover().process(sys.argv[1])
        else:
            print("File not found")
