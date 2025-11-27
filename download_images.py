import asyncio
import aiohttp
from pathlib import Path
from time import time
import json
import os
import datetime
import argparse

# 配置：需要扫描的目录列表
DETAILS_DIRS = ["assets/photoDetails", "assets/videoDetails"]

async def download(session, data, save_dir="./images/", idx=0, total=0):
    """下载图片或视频"""
    prefix_str = f"[{idx}/{total}] " if idx else ""
    
    # 处理不同的数据格式：photo, image, videos
    urls = []
    if "data" in data and data["data"]:
        d = data["data"]
        if "photo" in d and d["photo"] and d["photo"].get("image"):
            images = d["photo"]["image"]
            for size in ["x300", "x700"]:
                if images.get(size):
                    urls.append(images[size])
        elif "image" in d and d["image"]:
            images = d["image"]
            for size in ["x300", "x700"]:
                if images.get(size):
                    urls.append(images[size])
        elif "videos" in d and d["videos"]:
            # 视频格式：包含 cover 和 url
            videos = d["videos"]
            if videos.get("url"):
                urls.append(videos["url"])
    
    if not urls:
        print(f"{prefix_str}⚠️  数据格式不支持或无内容")
        return
    
    # datetime format
    datetime_format = "%Y-%m-%d"
    save_path = Path(save_dir + datetime.datetime.now().strftime(datetime_format) + "/")
    save_path.mkdir(parents=True, exist_ok=True)
    
    for url in urls:
        # 从URL提取原始文件名（包含后缀）
        original_name = url.split("/")[-1]
        ext = Path(original_name).suffix  # 如 .JPG, .mp4
        filename = f"{idx}_{Path(original_name).stem.split('_')[-1]}{ext}"
        
        filepath = save_path / filename
        
        # 如果文件已存在，跳过下载
        if filepath.exists():
            print(f"{prefix_str}⊘ 跳过已存在: {filepath}")
            continue
        
        # 下载（视频超时时间更长）
        timeout = 300 if ext.lower() in ['.mp4', '.mov', '.avi'] else 30
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()
                content = await response.read()
                with open(filepath, 'wb') as f:
                    f.write(content)
            print(f"{prefix_str}✓ {filepath}")
        except Exception as e:
            print(f"{prefix_str}✗ 下载失败 {url}: {e}")


def load_details(details_dir):
    """从指定目录加载所有配置文件"""
    details_path = Path(details_dir)
    
    if not details_path.exists():
        return []
    
    assets = []
    for file_path in details_path.iterdir():
        if file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    assets.append(data)
            except Exception as e:
                print(f"⚠️  读取文件失败 {file_path}: {e}")
    
    return assets


def load_all_details(dirs=None):
    """从多个目录加载所有配置文件"""
    dirs = dirs or DETAILS_DIRS
    assets = []
    for d in dirs:
        loaded = load_details(d)
        if loaded:
            print(f"从 {d} 加载了 {len(loaded)} 个文件")
            assets.extend(loaded)
    return assets


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载图片/视频")
    parser.add_argument("--dirs", "-d", nargs="+", default=DETAILS_DIRS,
                        help="指定要扫描的目录，默认: photoDetails videoDetails")
    args = parser.parse_args()
    
    assets = load_all_details(args.dirs)
    
    if not assets:
        print("没有找到任何文件")
        exit(0)
    
    print(f"\n开始下载 {len(assets)} 个文件...\n")
    
    async def download_all():
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://fenxuekeji.com/'
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [download(session, asset, idx=idx, total=len(assets)) 
                     for idx, asset in enumerate(assets, 1)]
            await asyncio.gather(*tasks)
    
    asyncio.run(download_all())
    
    print(f"\n✓ 全部完成！共处理 {len(assets)} 个文件")

