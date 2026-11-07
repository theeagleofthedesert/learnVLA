"""把失败视频抽成一张 6 帧拼图（时间从左到右），方便肉眼归类失败原因。
用法（libero venv，含 imageio[ffmpeg]）:
  python scripts/contact_sheet.py results/eval_official/libero_10/videos/*_fail.mp4 --out results/failure_gallery/libero_10
每张图：上行 agentview，下行 wrist（视频本身已是 224x448 左右拼接，这里按时间横向排 6 帧）。
"""
import argparse, pathlib, numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=6)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
for v in a.videos:
    frames = imageio.mimread(v, memtest=False)
    idx = np.linspace(0, len(frames) - 1, a.n).astype(int)
    tiles = []
    for i in idx:
        im = Image.fromarray(frames[i]); d = ImageDraw.Draw(im); d.text((4, 4), f"t={i}/{len(frames)}", fill=(255, 255, 0)); tiles.append(im)
    w, h = tiles[0].size; sheet = Image.new("RGB", (w * a.n, h))
    for k, t in enumerate(tiles): sheet.paste(t, (k * w, 0))
    p = out / (pathlib.Path(v).stem + ".png"); sheet.save(p); print(p, len(frames))
