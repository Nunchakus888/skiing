# AI 模型去水印 - 安装指南

## 🎨 为什么使用 AI 模型？

传统 OCR + OpenCV 方法的局限：
- ❌ 水印是多层叠加，OCR 识别不完整
- ❌ `cv2.inpaint` 修复质量有限
- ❌ 机械的边界框难以处理复杂水印

**AI 模型的优势：**
- ✅ 智能理解图像内容
- ✅ 自然的纹理填充
- ✅ 完美保留图片细节
- ✅ 处理复杂、多层叠加的水印

---

## 📦 方案一：LaMa（推荐 - 最强效果）

LaMa 是目前最先进的开源图像修复模型，效果接近商业级别。

### 安装步骤：

```bash
cd /Users/george/Documents/me/skiing/showme

# 安装 lama-cleaner（包含 LaMa 模型）
uv pip install lama-cleaner torch torchvision

# 或者使用 IOPaint（lama-cleaner 的新名字）
uv pip install iopaint
```

### 使用：

```bash
# 脚本会自动检测 LaMa 是否可用
uv run python remove_watermark_opencv.py

# 选择模式 4（OCR 模式）或 7（混合模式）
# AI 修复会自动启用
```

---

## 📦 方案二：IOPaint（简单易用）

IOPaint 是一个用户友好的 AI 图像修复工具。

### 安装：

```bash
uv pip install iopaint
```

### 独立使用：

```bash
# 启动 Web UI
iopaint start --model lama --port 8080

# 浏览器打开 http://localhost:8080
# 上传图片，手动标记水印区域，点击修复
```

---

## 📦 方案三：Stable Diffusion Inpainting

使用 Hugging Face 的 SD Inpainting 模型（需要 GPU）。

### 安装：

```bash
uv pip install diffusers transformers accelerate
```

### 代码示例：

```python
from diffusers import StableDiffusionInpaintPipeline
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

result = pipe(
    prompt="natural ski photo, no watermark",
    image=image,
    mask_image=mask
).images[0]
```

---

## ⚙️ 修改代码使用 AI 修复

当前代码已支持 AI 修复！只需：

1. **安装 lama-cleaner 或 iopaint**
2. **运行脚本**（自动检测并使用 AI）

```bash
uv run python remove_watermark_opencv.py
```

---

## 🎯 推荐方案

| 方案 | 效果 | 速度 | 难度 | GPU |
|------|------|------|------|-----|
| **LaMa** ⭐ | 最好 | 中等 | 简单 | 可选 |
| IOPaint | 很好 | 快 | 最简单 | 否 |
| SD Inpainting | 极好 | 慢 | 中等 | 需要 |
| OpenCV | 一般 | 最快 | 最简单 | 否 |

**推荐：先试 LaMa/IOPaint（CPU 可用），效果不满意再考虑 SD（需要 GPU）**

---

## 📝 安装命令总结

```bash
cd /Users/george/Documents/me/skiing/showme

# 方案1: LaMa（推荐）
uv pip install iopaint

# 或者方案2: Stable Diffusion（需要强大GPU）
uv pip install diffusers transformers accelerate

# 然后运行
uv run python remove_watermark_opencv.py
```

完成！✨

