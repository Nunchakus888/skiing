#!/usr/bin/env python3
"""
使用 AI 模型去除水印 - 智能修复方案
支持多种修复后端：
1. LaMa (推荐) - 最先进的图像修复模型
2. cv2.inpaint - 传统快速修复

基于图像特征检测水印区域，使用AI模型智能修复
"""

import cv2
import numpy as np
from PIL import Image
import os
import re

# AI修复模型标志
LAMA_AVAILABLE = False
try:
    from lama_cleaner.model_manager import ModelManager
    from lama_cleaner.schema import Config, HDStrategy, LDMSampler
    LAMA_AVAILABLE = True
    print("✓ LaMa AI 修复模型已加载")
except ImportError:
    print("⚠️  LaMa 模型未安装，使用传统 cv2.inpaint")
    print("   安装方法: uv pip install lama-cleaner")

def detect_watermark_mask(image_path):
    """
    精确检测图片中的水印位置（改进版）
    只标记明显的水印区域，避免误伤人物和背景
    返回水印的 mask（白色=水印区域）
    """
    # 读取图片
    img = cv2.imread(image_path)
    
    # 检查图片是否读取成功
    if img is None:
        print(f"❌ 错误: 无法读取图片文件: {image_path}")
        return None, None
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 创建空白 mask
    mask = np.zeros(gray.shape, dtype=np.uint8)
    
    # 方法1: 检测非常浅的文字（水印通常是半透明的浅色）
    # 只检测非常亮的区域 (阈值提高到 240，更保守)
    _, light_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # 方法2: 使用自适应阈值检测局部异常（文字边缘）
    # 但只保留小块区域（文字特征）
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 2
    )
    
    # 反转（让文字变白）
    adaptive_inv = cv2.bitwise_not(adaptive)
    
    # 只保留小的连通区域（可能是文字）
    # 使用形态学操作去除大的区域（可能是人物、背景）
    kernel_small = np.ones((2, 2), np.uint8)
    adaptive_filtered = cv2.morphologyEx(adaptive_inv, cv2.MORPH_OPEN, kernel_small)
    
    # 再使用闭运算连接文字笔画
    kernel_close = np.ones((3, 15), np.uint8)  # 横向连接（中文字符特征）
    adaptive_filtered = cv2.morphologyEx(adaptive_filtered, cv2.MORPH_CLOSE, kernel_close)
    
    # 结合两种方法：只保留既浅色又有文字特征的区域
    mask = cv2.bitwise_and(light_mask, adaptive_filtered)
    
    # 过滤掉太小的区域（噪点）和太大的区域（人物）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_mask = np.zeros(gray.shape, dtype=np.uint8)
    for contour in contours:
        area = cv2.contourArea(contour)
        # 只保留合理大小的区域（文字通常是这个范围）
        if 50 < area < 5000:
            cv2.drawContours(final_mask, [contour], -1, 255, -1)
    
    # 轻微膨胀，确保覆盖文字边缘
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    
    return img, final_mask

def detect_person_region(img, conservative=True):
    """
    检测图片中的人物区域，用于排除人物避免误伤
    返回人物区域的 mask
    
    策略：保守估计，宁可多排除也不要误伤人物
    
    Args:
        conservative: 是否使用保守策略（更大的保护区域）
    """
    h, w = img.shape[:2]
    person_mask = np.zeros((h, w), dtype=np.uint8)
    
    center_x, center_y = w // 2, h // 2
    
    if conservative:
        # 保守策略：用于颜色检测等可能误伤的场景
        width_range = int(w * 0.45)  # 左右各45%
        height_range = int(h * 0.55)  # 上下各55%
    else:
        # 精确策略：用于模式匹配等精确度高的场景
        width_range = int(w * 0.30)  # 左右各30%
        height_range = int(h * 0.40)  # 上下各40%
    
    x1 = max(0, center_x - width_range)
    x2 = min(w, center_x + width_range)
    y1 = max(0, center_y - height_range)
    y2 = min(h, center_y + height_range)
    
    # 标记中心人物区域
    person_mask[y1:y2, x1:x2] = 255
    
    return person_mask

def create_edge_only_mask(img, border_size=0.15):
    """
    创建只包含图片边缘区域的 mask
    只处理边缘的水印，避免中心人物区域
    
    Args:
        border_size: 边缘宽度占图片的比例（默认15%）
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    border_h = int(h * border_size)
    border_w = int(w * border_size)
    
    # 上边缘
    mask[0:border_h, :] = 255
    # 下边缘
    mask[h-border_h:h, :] = 255
    # 左边缘
    mask[:, 0:border_w] = 255
    # 右边缘
    mask[:, w-border_w:w] = 255
    
    return mask

def detect_repeating_text_pattern(image_path, pattern_texts=None, show_debug=False):
    """
    检测重复出现的水印文本模式
    
    策略：
    1. 使用 OCR 识别所有文本
    2. 找到重复出现的文本片段（水印模式）
    3. 标记所有匹配该模式的区域
    4. 排除人物中心区域
    
    Args:
        pattern_texts: 预定义的水印文本模式列表，如果为 None 则自动检测
        例如：["滑呗app", "1000万", "雪友", "选择", "酒店", "教练", "摄影师", "约玩"]
    """
    try:
        import easyocr
    except ImportError:
        print("    ⚠️  需要 EasyOCR 进行模式检测")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 预定义的水印文本模式（常见的滑雪类水印）
    if pattern_texts is None:
        pattern_texts = [
            "滑呗", "app", "1000", "万", "雪友", "选择",
            "酒店", "教练", "摄影师", "约玩", "雪票",
            "BDH"  # 常见的水印缩写
        ]
    
    print(f"    识别水印模式: {', '.join(pattern_texts)}")
    
    # 初始化 OCR
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
    results = reader.readtext(image_path)
    
    # 获取人物区域（使用精确策略，只保护核心人物）
    person_mask = detect_person_region(img, conservative=False)
    
    # 统计每个文本出现的位置
    matched_count = 0
    total_count = 0
    
    for (bbox, text, prob) in results:
        total_count += 1
        
        # 检查是否匹配任何水印模式
        is_match = False
        for pattern in pattern_texts:
            if pattern.lower() in text.lower() or text.lower() in pattern.lower():
                is_match = True
                break
        
        if not is_match:
            continue
        
        # 获取边界框坐标
        pts = np.array(bbox, dtype=np.int32)
        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))
        
        # 检查是否在人物区域
        if person_mask[center_y, center_x] > 0:
            # 人物区域，跳过
            continue
        
        # 扩大边界框（覆盖完整文字和阴影）
        width = int(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        height = int(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        
        # 扩大 50% 确保覆盖（从 80% 降低，避免过度扩展）
        expand_ratio = 1.5
        new_width = int(width * expand_ratio)
        new_height = int(height * expand_ratio)
        
        x1 = max(0, center_x - new_width // 2)
        y1 = max(0, center_y - new_height // 2)
        x2 = min(w, center_x + new_width // 2)
        y2 = min(h, center_y + new_height // 2)
        
        # 标记该区域
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        matched_count += 1
        
        if show_debug:
            print(f"      匹配到: '{text}' 位置:({center_x}, {center_y})")
    
    # 温和的膨胀操作，连接临近的水印文字（降低强度）
    if matched_count > 0:
        kernel = np.ones((10, 10), np.uint8)  # 从 20×20 降低到 10×10
        mask = cv2.dilate(mask, kernel, iterations=2)  # 从 3 次降低到 2 次
        
        # 重要：膨胀后再次排除人物区域，防止扩展到人物
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(person_mask))
    
    print(f"    模式匹配: {matched_count}/{total_count} 处 (跳过人物区域)")
    
    return mask

def detect_dark_text_on_white(img, show_debug=False):
    """
    检测白色背景（雪地）上的深色文字水印
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 检测高亮度区域（白色/浅色背景）
    _, bright_areas = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 在亮区域中检测深色文字（边缘检测）
    edges = cv2.Canny(gray, 50, 150)
    
    # 只保留在明亮区域的边缘
    dark_text_mask = cv2.bitwise_and(edges, edges, mask=bright_areas)
    
    # 形态学处理：连接文字笔画
    kernel_h = np.ones((2, 8), np.uint8)
    dark_text_mask = cv2.morphologyEx(dark_text_mask, cv2.MORPH_CLOSE, kernel_h)
    
    kernel_v = np.ones((8, 2), np.uint8)
    dark_text_mask = cv2.morphologyEx(dark_text_mask, cv2.MORPH_CLOSE, kernel_v)
    
    # 膨胀以覆盖完整文字
    kernel_dilate = np.ones((5, 5), np.uint8)
    dark_text_mask = cv2.dilate(dark_text_mask, kernel_dilate, iterations=2)
    
    # 过滤掉太小或太大的区域
    mask_filtered = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(dark_text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 8000:  # 文字大小范围
            cv2.drawContours(mask_filtered, [contour], -1, 255, -1)
            count += 1
    
    if count > 0:
        print(f"    检测到白色背景深色文字: {count} 处")
    
    return mask_filtered

def detect_specific_color_watermark(img, show_debug=False, exclude_person=True, edge_only=True):
    """
    检测特定颜色的水印
    目标颜色: #67789d (RGB: 103, 120, 157) 和 #5e7da8 (RGB: 94, 125, 168)
    以及各种灰蓝色变体（适应不同光照和背景）
    
    Args:
        exclude_person: 是否排除人物中心区域，避免误伤衣服上的内容
        edge_only: 是否只处理边缘区域的水印（推荐开启）
    """
    h, w = img.shape[:2]
    
    # 获取人物区域 mask（用于排除，使用保守策略）
    person_mask = detect_person_region(img, conservative=True) if exclude_person else np.zeros((h, w), dtype=np.uint8)
    
    # 获取边缘区域 mask（只处理边缘）
    edge_mask = create_edge_only_mask(img) if edge_only else np.ones((h, w), dtype=np.uint8) * 255
    
    if edge_only:
        print("    仅检测边缘区域水印，保护中心人物")
    
    # 定义目标颜色范围（BGR格式）- 增加更多灰蓝色变体
    target_colors = [
        {'name': '#67789d', 'bgr': np.array([157, 120, 103]), 'tolerance': 35},
        {'name': '#5e7da8', 'bgr': np.array([168, 125, 94]), 'tolerance': 35},
        {'name': 'gray-blue-1', 'bgr': np.array([140, 110, 95]), 'tolerance': 30},  # 深灰蓝
        {'name': 'gray-blue-2', 'bgr': np.array([180, 135, 110]), 'tolerance': 30},  # 浅灰蓝
        {'name': 'gray-blue-3', 'bgr': np.array([150, 115, 90]), 'tolerance': 30},  # 中灰蓝
    ]
    
    mask_combined = np.zeros((h, w), dtype=np.uint8)
    
    for color_info in target_colors:
        target_bgr = color_info['bgr']
        color_name = color_info['name']
        tolerance = color_info.get('tolerance', 30)
        
        # 设置颜色容差（允许一定范围的颜色偏差）
        lower = np.array([max(0, target_bgr[0] - tolerance), 
                         max(0, target_bgr[1] - tolerance), 
                         max(0, target_bgr[2] - tolerance)])
        upper = np.array([min(255, target_bgr[0] + tolerance), 
                         min(255, target_bgr[1] + tolerance), 
                         min(255, target_bgr[2] + tolerance)])
        
        # 创建颜色 mask
        mask_color = cv2.inRange(img, lower, upper)
        
        # 只保留边缘区域
        mask_color = cv2.bitwise_and(mask_color, edge_mask)
        
        # 排除人物区域
        if exclude_person:
            mask_color = cv2.bitwise_and(mask_color, cv2.bitwise_not(person_mask))
        
        # 统计检测到的像素
        color_pixels = np.sum(mask_color > 0)
        if color_pixels > 100:  # 只显示明显检测到的
            print(f"    检测到颜色 {color_name}: {color_pixels} 像素")
        
        # 形态学处理：连接文字笔画（加强处理）
        kernel_h = np.ones((3, 10), np.uint8)  # 横向连接（加大核）
        mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel_h)
        
        kernel_v = np.ones((10, 3), np.uint8)  # 纵向连接（加大核）
        mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel_v)
        
        # 膨胀操作，覆盖文字边缘
        kernel_dilate = np.ones((5, 5), np.uint8)
        mask_color = cv2.dilate(mask_color, kernel_dilate, iterations=2)
        
        # 过滤掉太小和太大的区域（只保留文字大小）
        contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 15000:  # 扩大文字大小范围
                cv2.drawContours(mask_combined, [contour], -1, 255, -1)
    
    # 最终膨胀，确保覆盖完整
    kernel = np.ones((3, 3), np.uint8)
    mask_combined = cv2.dilate(mask_combined, kernel, iterations=2)
    
    return mask_combined

def detect_watermark_by_text(image_path, show_debug=False, use_color_detection=True, use_pattern_match=True):
    """
    使用 OCR 识别特定水印文字并精确标记
    可选：结合特定颜色检测 + 白色背景深色文字检测 + 模式匹配
    
    水印关键词：
    - 滑呗、app、1000万、雪友、选择
    - 雪票、酒店、教练、摄影师、约玩
    
    Args:
        use_pattern_match: 是否使用重复模式匹配（推荐）⭐
    
    需要安装: uv pip install easyocr
    """
    img = cv2.imread(image_path)
    
    # 检查图片是否读取成功
    if img is None:
        print(f"❌ 错误: 无法读取图片文件: {image_path}")
        return None, None
    
    h, w = img.shape[:2]
    
    # ============== 1. 模式匹配层（新增 - 最精确）⭐ ==============
    mask_pattern = np.zeros((h, w), dtype=np.uint8)
    if use_pattern_match:
        print("  [1/4] 重复模式匹配检测中...")
        mask_pattern = detect_repeating_text_pattern(image_path, show_debug=show_debug)
        if mask_pattern is not None:
            pattern_pixels = np.sum(mask_pattern > 0)
            print(f"    模式匹配检测到 {pattern_pixels} 像素")
        else:
            mask_pattern = np.zeros((h, w), dtype=np.uint8)
    
    # ============== 2. 颜色检测层（边缘区域的灰蓝色水印）==============
    mask_color = np.zeros((h, w), dtype=np.uint8)
    if use_color_detection:
        print("  [2/4] 边缘区域颜色检测中 (#67789d, #5e7da8)...")
        mask_color = detect_specific_color_watermark(img, show_debug, exclude_person=True, edge_only=True)
        color_pixels = np.sum(mask_color > 0)
        print(f"    颜色检测到 {color_pixels} 像素")
    
    # ============== 3. 白色背景深色文字检测层（雪地水印）==============
    print("  [3/4] 白色背景深色文字检测中...")
    mask_dark_on_white = detect_dark_text_on_white(img, show_debug)
    dark_pixels = np.sum(mask_dark_on_white > 0)
    if dark_pixels > 0:
        print(f"    白色背景检测到 {dark_pixels} 像素")
    
    # ============== 4. OCR 文字识别层（精确识别）==============
    mask_ocr = np.zeros((h, w), dtype=np.uint8)
    try:
        import easyocr
    except ImportError:
        print("  ⚠️  OCR 模块不可用")
        if use_color_detection and np.sum(mask_color) > 0:
            print("  使用颜色检测结果")
            return img, mask_color
        print("  ❌ 需要安装 EasyOCR 或至少一种检测方法可用")
        print("     安装命令: source skiing/bin/activate && uv pip install easyocr")
        return None, None
    
    print("  [4/4] OCR 文字识别中（补充检测）...")
    # 初始化 OCR，支持中文和英文
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
    
    # 检测文字
    results = reader.readtext(image_path)
    
    # 获取边缘区域和人物区域 mask（保守策略）
    edge_mask = create_edge_only_mask(img, border_size=0.20)  # 边缘20%
    person_mask = detect_person_region(img, conservative=True)
    
    # 定义水印关键词（支持正则表达式匹配）
    watermark_patterns = [
        r'滑呗',
        r'app',
        r'1000.*万',
        r'雪友',
        r'选择',
        r'雪票',
        r'酒店',
        r'教练',
        r'摄影师',
        r'约玩',
        r'BDH',  # 可能的英文缩写
    ]
    
    detected_watermarks = []
    
    for (bbox, text, prob) in results:
        # 检查是否匹配水印关键词
        is_watermark = False
        matched_keyword = None
        
        for pattern in watermark_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                is_watermark = True
                matched_keyword = pattern
                break
        
        if is_watermark and prob > 0.3:  # 置信度阈值
            # 获取边界框坐标
            pts = np.array(bbox, dtype=np.int32)
            
            # 扩大边界框，确保完全覆盖文字（包括边缘和阴影）
            # 计算中心点
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))
            
            # 检查是否在边缘区域或人物区域之外
            # 如果在人物区域内，跳过
            if person_mask[center_y, center_x] > 0:
                print(f"    ⚠️  跳过人物区域文字: '{text}'")
                continue
            
            # 计算宽高
            width = int(np.max(pts[:, 0]) - np.min(pts[:, 0]))
            height = int(np.max(pts[:, 1]) - np.min(pts[:, 1]))
            
            # 扩大 50%（增加扩展比例以覆盖边缘残留）
            expand_ratio = 1.5
            new_width = int(width * expand_ratio)
            new_height = int(height * expand_ratio)
            
            # 创建扩大后的矩形
            x1 = max(0, center_x - new_width // 2)
            y1 = max(0, center_y - new_height // 2)
            x2 = min(w, center_x + new_width // 2)
            y2 = min(h, center_y + new_height // 2)
            
            # 填充 mask_ocr
            cv2.rectangle(mask_ocr, (x1, y1), (x2, y2), 255, -1)
            
            detected_watermarks.append({
                'text': text,
                'confidence': prob,
                'keyword': matched_keyword,
                'bbox': (x1, y1, x2, y2)
            })
            
            print(f"    ✓ 识别到: '{text}' (置信度: {prob:.2f}, 匹配: {matched_keyword})")
    
    print(f"    OCR 识别到 {len(detected_watermarks)} 处水印文字")
    
    # 对 OCR mask 进行膨胀操作，确保边缘完全覆盖
    if np.sum(mask_ocr) > 0:
        kernel = np.ones((15, 15), np.uint8)  # 使用较大的核来膨胀
        mask_ocr = cv2.dilate(mask_ocr, kernel, iterations=2)
        print(f"    应用边缘扩展，确保完全覆盖文字残留")
    
    # ============== 合并四种检测结果 ==============
    print("\n合并检测结果...")
    
    # 合并所有 mask （模式匹配优先级最高）
    final_mask = cv2.bitwise_or(mask_pattern, mask_color)
    final_mask = cv2.bitwise_or(final_mask, mask_ocr)
    final_mask = cv2.bitwise_or(final_mask, mask_dark_on_white)
    
    # 统计覆盖率
    watermark_pixels = np.sum(final_mask > 0)
    total_pixels = h * w
    percentage = (watermark_pixels / total_pixels) * 100
    
    print(f"\n最终检测结果:")
    print(f"  模式匹配: {np.sum(mask_pattern > 0)} 像素 ⭐")
    print(f"  边缘颜色检测: {np.sum(mask_color > 0)} 像素")
    print(f"  白色背景深色文字: {np.sum(mask_dark_on_white > 0)} 像素")
    print(f"  OCR 补充检测: {np.sum(mask_ocr > 0)} 像素")
    print(f"  合并后: {watermark_pixels} 像素 ({percentage:.2f}%)")
    
    if watermark_pixels == 0:
        print("⚠️  未检测到水印")
        return img, None
    
    if show_debug:
        # 显示检测结果
        debug_img = img.copy()
        
        # 用不同颜色标记不同来源
        debug_img[mask_color > 0] = [0, 255, 255]    # 黄色 = 颜色检测
        debug_img[mask_ocr > 0] = [0, 255, 0]        # 绿色 = OCR
        
        # 在图片上标记 OCR 检测到的文字框
        for wm in detected_watermarks:
            x1, y1, x2, y2 = wm['bbox']
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 显示
        cv2.imshow("Original", cv2.resize(img, (800, 600)))
        cv2.imshow("Detection: Yellow=Color, Green=OCR", cv2.resize(debug_img, (800, 600)))
        cv2.imshow("Final Mask", cv2.resize(final_mask, (800, 600)))
        
        print("\n调试信息:")
        print("  黄色区域 = 颜色检测 (#67789d, #5e7da8)")
        print("  绿色区域 = OCR 识别的文字")
        print("\n按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img, final_mask

def detect_watermark_hybrid(image_path, show_debug=False):
    """
    混合检测模式：结合 OCR + 图像特征 + 颜色检测
    最全面的水印检测方法，特别适合半透明、艺术字水印
    
    检测策略：
    1. OCR 识别文字（识别清晰的文字）
    2. 检测重复的浅色斜纹（半透明水印特征）
    3. 检测特定颜色的文字（灰白色水印）
    4. 结合三种方法，生成精确的水印 mask
    """
    print("使用混合检测模式...")
    img = cv2.imread(image_path)
    
    # 检查图片是否读取成功
    if img is None:
        print(f"❌ 错误: 无法读取图片文件: {image_path}")
        print("   请检查:")
        print("   1. 文件路径是否正确")
        print("   2. 文件是否存在")
        print("   3. 文件格式是否受支持")
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 创建三个检测层
    mask_ocr = np.zeros((h, w), dtype=np.uint8)
    mask_color = np.zeros((h, w), dtype=np.uint8)
    mask_pattern = np.zeros((h, w), dtype=np.uint8)
    
    # ============== 方法1: OCR 文字识别 ==============
    try:
        import easyocr
        print("  [1/3] OCR 文字识别中...")
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        results = reader.readtext(image_path)
        
        watermark_patterns = [
            r'滑呗', r'app', r'1000.*万', r'雪友', r'选择',
            r'雪票', r'酒店', r'教练', r'摄影师', r'约玩', r'BDH',
            r'友.*选择', r'万.*友'  # 模糊匹配
        ]
        
        ocr_count = 0
        for (bbox, text, prob) in results:
            if prob > 0.2:  # 降低阈值，识别更多可能的水印
                is_watermark = any(re.search(p, text, re.IGNORECASE) for p in watermark_patterns)
                if is_watermark:
                    pts = np.array(bbox, dtype=np.int32)
                    # 扩大区域（从 1.3 增加到 1.5）
                    center_x, center_y = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    width = int((np.max(pts[:, 0]) - np.min(pts[:, 0])) * 1.5)
                    height = int((np.max(pts[:, 1]) - np.min(pts[:, 1])) * 1.5)
                    x1 = max(0, center_x - width // 2)
                    y1 = max(0, center_y - height // 2)
                    x2 = min(w, center_x + width // 2)
                    y2 = min(h, center_y + height // 2)
                    cv2.rectangle(mask_ocr, (x1, y1), (x2, y2), 255, -1)
                    ocr_count += 1
                    print(f"    ✓ 识别到: '{text}' (置信度: {prob:.2f})")
        
        # 对 OCR mask 进行膨胀操作，确保边缘完全覆盖
        if ocr_count > 0:
            kernel = np.ones((15, 15), np.uint8)
            mask_ocr = cv2.dilate(mask_ocr, kernel, iterations=2)
        
        print(f"    OCR 识别到 {ocr_count} 处水印文字")
    except ImportError:
        print("    ⚠️  OCR 不可用，跳过")
    except Exception as e:
        print(f"    ⚠️  OCR 出错: {e}")
    
    # ============== 方法2: 颜色特征检测（浅色半透明水印） ==============
    print("  [2/3] 颜色特征检测中...")
    
    # 转换到 HSV 色彩空间，更容易检测特定颜色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 检测浅色/白色文字（通常是 V 通道高，S 通道低）
    _, s, v = cv2.split(hsv)
    
    # 高亮度、低饱和度 = 浅色/白色水印
    _, high_brightness = cv2.threshold(v, 230, 255, cv2.THRESH_BINARY)
    _, low_saturation = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 结合两者
    light_mask = cv2.bitwise_and(high_brightness, low_saturation)
    
    # 检测灰色文字（中等亮度）
    gray_range = cv2.inRange(v, 180, 230)
    
    # 结合浅色和灰色检测
    color_mask = cv2.bitwise_or(light_mask, gray_range)
    
    # 形态学处理：连接文字笔画
    kernel_h = np.ones((2, 10), np.uint8)  # 横向连接
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_h)
    
    kernel_v = np.ones((10, 2), np.uint8)  # 纵向连接
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_v)
    
    # 过滤掉太小和太大的区域
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 8000:  # 文字大小范围
            # 检查长宽比（文字通常是细长的）
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            aspect_ratio = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1)
            if aspect_ratio < 15:  # 不是极长的线条
                cv2.drawContours(mask_color, [contour], -1, 255, -1)
                color_count += 1
    
    print(f"    颜色检测到 {color_count} 处疑似水印区域")
    
    # ============== 方法3: 重复模式检测（检测重复的水印） ==============
    print("  [3/3] 重复模式检测中...")
    
    # 使用边缘检测找到文字轮廓
    edges = cv2.Canny(gray, 30, 100)
    
    # 膨胀边缘
    kernel = np.ones((2, 2), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # 只保留在浅色区域的边缘（水印特征）
    edges_in_light = cv2.bitwise_and(edges_dilated, light_mask)
    
    # 连接文字
    kernel_connect = np.ones((3, 8), np.uint8)
    pattern_mask = cv2.morphologyEx(edges_in_light, cv2.MORPH_CLOSE, kernel_connect)
    
    # 过滤大小
    contours, _ = cv2.findContours(pattern_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pattern_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 5000:
            cv2.drawContours(mask_pattern, [contour], -1, 255, -1)
            pattern_count += 1
    
    print(f"    模式检测到 {pattern_count} 处疑似水印区域")
    
    # ============== 合并三种检测结果 ==============
    print("\n合并检测结果...")
    
    # 优先使用 OCR 结果（最准确）
    final_mask = mask_ocr.copy()
    
    # 添加颜色检测结果（去除与 OCR 重叠的部分，避免过度修复）
    mask_color_filtered = cv2.bitwise_and(mask_color, cv2.bitwise_not(mask_ocr))
    final_mask = cv2.bitwise_or(final_mask, mask_color_filtered)
    
    # 添加模式检测结果（最保守）
    mask_pattern_filtered = cv2.bitwise_and(mask_pattern, cv2.bitwise_not(final_mask))
    final_mask = cv2.bitwise_or(final_mask, mask_pattern_filtered)
    
    # 最终形态学优化：轻微膨胀，确保覆盖完整
    kernel_final = np.ones((3, 3), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel_final, iterations=1)
    
    # 统计
    watermark_pixels = np.sum(final_mask > 0)
    total_pixels = h * w
    percentage = (watermark_pixels / total_pixels) * 100
    
    print(f"\n最终检测结果:")
    print(f"  水印区域: {watermark_pixels} 像素 ({percentage:.2f}%)")
    
    if show_debug:
        # 显示各层检测结果
        debug_img = img.copy()
        
        # 用不同颜色标记不同来源的检测
        debug_img[mask_ocr > 0] = [0, 255, 0]      # 绿色 = OCR
        debug_img[mask_color_filtered > 0] = [255, 255, 0]  # 青色 = 颜色
        debug_img[mask_pattern_filtered > 0] = [0, 165, 255]  # 橙色 = 模式
        
        cv2.imshow("Original", cv2.resize(img, (800, 600)))
        cv2.imshow("Detection: Green=OCR, Cyan=Color, Orange=Pattern", 
                   cv2.resize(debug_img, (800, 600)))
        cv2.imshow("Final Mask", cv2.resize(final_mask, (800, 600)))
        
        # 显示各层
        cv2.imshow("Layer 1: OCR", cv2.resize(mask_ocr, (400, 300)))
        cv2.imshow("Layer 2: Color", cv2.resize(mask_color, (400, 300)))
        cv2.imshow("Layer 3: Pattern", cv2.resize(mask_pattern, (400, 300)))
        
        print("\n调试信息:")
        print("  绿色区域 = OCR 识别的文字")
        print("  青色区域 = 颜色检测的水印")
        print("  橙色区域 = 重复模式检测")
        print("\n按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img, final_mask

def inpaint_with_ai(img, mask, method='lama'):
    """
    使用AI模型进行智能图像修复
    
    Args:
        img: 输入图片 (BGR格式)
        mask: 水印mask (灰度图，255=需要修复)
        method: 修复方法 ('lama', 'cv2')
    
    Returns:
        修复后的图片
    """
    if method == 'lama' and LAMA_AVAILABLE:
        print("  使用 LaMa AI 模型修复...")
        try:
            # 转换格式：BGR -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            mask_pil = Image.fromarray(mask)
            
            # 初始化LaMa模型
            model = ModelManager(
                name="lama",
                device="cpu",  # 或 "cuda" 如果有GPU
            )
            
            # 配置参数
            config = Config(
                ldm_steps=50,
                ldm_sampler=LDMSampler.ddim,
                hd_strategy=HDStrategy.ORIGINAL,
                hd_strategy_crop_margin=128,
                hd_strategy_crop_trigger_size=1280,
                hd_strategy_resize_limit=2048,
            )
            
            # 执行修复
            result = model(img_pil, mask_pil, config)
            
            # 转换回OpenCV格式
            result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            print("  ✓ AI 修复完成")
            return result_bgr
            
        except Exception as e:
            print(f"  ⚠️  AI 修复失败: {e}")
            print("  回退到传统修复方法")
            method = 'cv2'
    
    # 使用传统cv2.inpaint方法
    if method == 'cv2' or not LAMA_AVAILABLE:
        # 智能选择修复半径
        watermark_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        percentage = (watermark_pixels / total_pixels) * 100
        
        if percentage < 1:
            inpaint_radius = 10
            print(f"  使用传统修复（精细模式，半径 {inpaint_radius}）")
        elif percentage < 5:
            inpaint_radius = 8
            print(f"  使用传统修复（标准模式，半径 {inpaint_radius}）")
        else:
            inpaint_radius = 5
            print(f"  使用传统修复（保守模式，半径 {inpaint_radius}）")
        
        # 两步修复
        result = cv2.inpaint(img, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
        
        if percentage < 5:
            kernel = np.ones((3, 3), np.uint8)
            mask_refined = cv2.erode(mask, kernel, iterations=1)
            if np.sum(mask_refined > 0) > 0:
                result = cv2.inpaint(result, mask_refined, inpaintRadius=inpaint_radius-2, flags=cv2.INPAINT_NS)
        
        return result

def remove_watermark(image_path, output_path=None, show_mask=False, use_ocr=False, use_hybrid=False, use_ai=True):
    """
    精确去除水印，不影响人物和背景
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径（默认自动生成）
        show_mask: 是否显示检测到的水印区域
        use_ocr: 是否使用 OCR 文字识别模式
        use_hybrid: 是否使用混合检测模式（最强大）⭐推荐⭐
        use_ai: 是否使用 AI 模型修复（LaMa）⭐推荐⭐
    """
    print(f"处理图片: {image_path}")
    
    # 显示修复方法
    if use_ai and LAMA_AVAILABLE:
        print("🎨 修复方法: LaMa AI 模型 (智能修复)")
    else:
        print("🔧 修复方法: OpenCV 传统修复")
    
    if use_hybrid:
        print("⭐ 使用混合检测模式（OCR + 图像特征 + 颜色）")
        img, mask = detect_watermark_hybrid(image_path, show_mask)
        mode_suffix = "_hybrid"
    elif use_ocr:
        print("使用 OCR 文字识别模式...")
        img, mask = detect_watermark_by_text(image_path, show_mask)
        mode_suffix = "_ocr"
        if mask is None:
            print("OCR 检测失败，切换到基础模式")
            use_ocr = False
    
    if not use_hybrid and not use_ocr:
        print("使用基础检测模式...")
        img, mask = detect_watermark_mask(image_path)
        mode_suffix = "_cleaned"
        
        # 统计检测到的水印区域
        watermark_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        percentage = (watermark_pixels / total_pixels) * 100
        
        print(f"检测到水印区域: {watermark_pixels} 像素 ({percentage:.2f}% 的图片)")
        
        if watermark_pixels == 0:
            print("⚠️  未检测到水印区域，图片不需要处理")
            return image_path
        
        if percentage > 30:
            print("⚠️  警告：检测到的区域过大，可能会误伤图片内容")
            print("建议使用混合模式（选项 7）或预览模式检查")
        
        if show_mask:
            # 显示检测到的水印区域（用绿色标记）
            debug_img = img.copy()
            debug_img[mask > 0] = [0, 255, 0]
            
            cv2.imshow("Original", cv2.resize(img, (800, 600)))
            cv2.imshow("Watermark Detection (Green)", cv2.resize(debug_img, (800, 600)))
            cv2.imshow("Mask", cv2.resize(mask, (800, 600)))
            print("\n检测到的水印区域已用绿色标记")
            print("按任意键继续...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    if mask is None or np.sum(mask > 0) == 0:
        print("⚠️  未检测到水印，跳过处理")
        return image_path
    
    # ============== 智能修复：使用AI模型或传统方法 ==============
    watermark_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    percentage = (watermark_pixels / total_pixels) * 100
    
    print(f"\n开始修复...")
    print(f"  水印覆盖率: {percentage:.2f}%")
    
    # 使用AI修复或传统方法
    if use_ai:
        result = inpaint_with_ai(img, mask, method='lama')
    else:
        result = inpaint_with_ai(img, mask, method='cv2')
    
    # 生成输出文件名
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}{mode_suffix}{ext}"
    
    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"✓ 完成！保存到: {output_path}")
    
    return output_path

def batch_remove_watermarks(image_dir, output_dir=None, use_ocr=False, use_hybrid=False):
    """
    批量处理文件夹中的所有图片
    
    Args:
        image_dir: 图片目录
        output_dir: 输出目录
        use_ocr: 是否使用 OCR 模式
        use_hybrid: 是否使用混合模式
    """
    if output_dir is None:
        if use_hybrid:
            suffix = "cleaned_hybrid"
        elif use_ocr:
            suffix = "cleaned_ocr"
        else:
            suffix = "cleaned"
        output_dir = os.path.join(image_dir, suffix)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.endswith(extensions) and os.path.isfile(os.path.join(image_dir, f))]
    
    print(f"\n找到 {len(image_files)} 张图片")
    if use_hybrid:
        mode_name = "混合检测（OCR + 图像特征 + 颜色）⭐推荐⭐"
    elif use_ocr:
        mode_name = "OCR 文字识别"
    else:
        mode_name = "基础检测"
    print(f"使用模式: {mode_name}")
    
    for i, filename in enumerate(image_files, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(image_files)}] {filename}")
        print('='*70)
        
        input_path = os.path.join(image_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            remove_watermark(input_path, output_path, show_mask=False, 
                           use_ocr=use_ocr, use_hybrid=use_hybrid)
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"✓ 全部完成！结果保存在: {output_dir}")
    print('='*70)

if __name__ == "__main__":
    print("=" * 70)
    print("OpenCV 水印去除工具 - 智能检测，保护主体元素")
    print("=" * 70)
    
    print("\n选择模式：")
    print("━" * 70)
    print("【基础模式】- 快速处理")
    print("  1. 单张图片")
    print("  2. 批量处理")
    print("  3. 预览检测效果")
    print()
    print("【OCR 模式】- 文字识别")
    print("  4. 单张图片")
    print("  5. 批量处理")
    print("  6. 预览识别效果")
    print()
    print("【混合模式】⭐最强⭐ - OCR + 图像特征 + 颜色检测")
    print("  7. 单张图片 (推荐首选)")
    print("  8. 批量处理 (最全面的去水印)")
    print("  9. 预览检测效果 (查看三层检测)")
    print("━" * 70)
    
    choice = input("\n请选择 (1-9): ").strip()
    
    # 使用当前脚本所在目录作为基础路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "images/11-22/02.JPG")
    image_dir = os.path.join(script_dir, "images/11-22")
    
    # 基础模式
    if choice == "1":
        print("\n【基础模式】单张图片")
        remove_watermark(image_path, use_ocr=False, use_hybrid=False)
        
    elif choice == "2":
        print("\n【基础模式】批量处理")
        batch_remove_watermarks(image_dir, use_ocr=False, use_hybrid=False)
        
    elif choice == "3":
        print("\n【基础模式】预览")
        print("绿色标记 = 检测到的水印区域")
        remove_watermark(image_path, show_mask=True, use_ocr=False, use_hybrid=False)
    
    # OCR 模式
    elif choice == "4":
        print("\n【OCR 模式】单张图片")
        print("识别关键词: 滑呗、app、雪友、雪票、教练等\n")
        remove_watermark(image_path, use_ocr=True, use_hybrid=False)
        
    elif choice == "5":
        print("\n【OCR 模式】批量处理")
        batch_remove_watermarks(image_dir, use_ocr=True, use_hybrid=False)
        
    elif choice == "6":
        print("\n【OCR 模式】预览")
        print("显示识别到的文字和边界框\n")
        remove_watermark(image_path, show_mask=True, use_ocr=True, use_hybrid=False)
    
    # 混合模式 ⭐推荐⭐
    elif choice == "7":
        print("\n⭐【混合模式】单张图片 - 最全面的水印检测")
        print("结合:")
        print("  • OCR 文字识别")
        print("  • 颜色特征检测（浅色/灰色水印）")
        print("  • 重复模式检测（艺术字/半透明）")
        print()
        remove_watermark(image_path, use_ocr=False, use_hybrid=True)
        
    elif choice == "8":
        print("\n⭐【混合模式】批量处理")
        print("使用最强检测算法处理所有图片\n")
        batch_remove_watermarks(image_dir, use_ocr=False, use_hybrid=True)
        
    elif choice == "9":
        print("\n⭐【混合模式】预览 - 查看三层检测")
        print("颜色说明:")
        print("  • 绿色 = OCR 识别的文字")
        print("  • 青色 = 颜色检测的水印")
        print("  • 橙色 = 重复模式检测")
        print()
        remove_watermark(image_path, show_mask=True, use_ocr=False, use_hybrid=True)
    
    else:
        print("❌ 无效选择")
    
    print("\n" + "━" * 70)
    print("💡 使用建议:")
    print("  • 首次使用: 选择【9】预览混合模式检测效果")
    print("  • 日常处理: 选择【7】混合模式单张或【8】批量")
    print("  • 快速处理: 选择【1】基础模式")
    print("  • 精确识别: 选择【4】OCR 模式（需要网络下载模型）")
    print("━" * 70)

