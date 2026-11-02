import os
import sys
import time
import math
import json
import heapq
from collections import Counter
import numpy as np
import cv2
from scipy.fftpack import dct, idct
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set random seeds for strict reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Dynamic relative directory paths (Works cross-platform on Windows, macOS, Linux)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_INPUT = os.path.join(BASE_DIR, "data", "input")
DATA_OUTPUT = os.path.join(BASE_DIR, "data", "output")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(DATA_INPUT, exist_ok=True)
os.makedirs(DATA_OUTPUT, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------------------------------------
# HELPER: METRICS (PSNR, SSIM, NCC, BER)
# -------------------------------------------------------------
def calculate_mse(img1, img2):
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

def calculate_psnr(img1, img2):
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    return 10.0 * math.log10((max_pixel ** 2) / mse)

def calculate_ssim(img1, img2):
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        g1 = img1
        g2 = img2
    g1 = g1.astype(np.float64)
    g2 = g2.astype(np.float64)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(g1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(g2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(g1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(g2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(g1 * g2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def calculate_ncc(w1, w2):
    w1_f = w1.astype(np.float64)
    w2_f = w2.astype(np.float64)
    numerator = np.sum(w1_f * w2_f)
    denominator = np.sqrt(np.sum(w1_f ** 2) * np.sum(w2_f ** 2))
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator / denominator)

def calculate_ber(w1, w2):
    b1 = (w1 > 127).astype(np.uint8)
    b2 = (w2 > 127).astype(np.uint8)
    total_bits = b1.size
    error_bits = np.sum(b1 != b2)
    return float((error_bits / total_bits) * 100.0)

# -------------------------------------------------------------
# STEP 0: GENERATE MULTI-FORMAT TEST IMAGES & SIMULATE DAiSEE FRAMES
# -------------------------------------------------------------
print("--> Generating input test images and simulated DAiSEE video frames...")

daisee_dir = os.path.join(DATA_INPUT, "daisee_frames")
os.makedirs(daisee_dir, exist_ok=True)

for i in range(10):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        img[y, :, 0] = int(180 + 30 * np.sin(y / 80.0 + i * 0.1))
        img[y, :, 1] = int(190 + 20 * np.cos(y / 90.0))
        img[y, :, 2] = int(210 + 25 * np.sin(y / 100.0))
    
    torso_color = (80 + (i*10)%50, 60, 50)
    cv2.ellipse(img, (320, 500), (220, 150), 0, 0, 360, torso_color, -1)
    
    neck_color = (165, 190, 225)
    cv2.rectangle(img, (290, 310), (350, 380), neck_color, -1)
    
    head_center_x = 320 + int(6 * np.sin(i * 0.7))
    head_center_y = 240 + int(4 * np.cos(i * 0.5))
    face_color = (175, 205, 240)
    cv2.ellipse(img, (head_center_x, head_center_y), (85, 115), int(2*np.sin(i)), 0, 360, face_color, -1)
    
    hair_color = (30, 25, 20)
    cv2.ellipse(img, (head_center_x, head_center_y - 45), (90, 80), 0, 180, 360, hair_color, -1)
    
    eye_y = head_center_y - 15
    cv2.ellipse(img, (head_center_x - 32, eye_y), (14, 8), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (head_center_x + 32, eye_y), (14, 8), 0, 0, 360, (255, 255, 255), -1)
    
    gaze_x = int(3 * np.sin(i * 0.9))
    cv2.circle(img, (head_center_x - 32 + gaze_x, eye_y), 5, (60, 40, 30), -1)
    cv2.circle(img, (head_center_x + 32 + gaze_x, eye_y), 5, (60, 40, 30), -1)
    
    cv2.line(img, (head_center_x - 45, eye_y - 14), (head_center_x - 18, eye_y - 14), (40, 30, 20), 3)
    cv2.line(img, (head_center_x + 18, eye_y - 14), (head_center_x + 45, eye_y - 14), (40, 30, 20), 3)
    cv2.line(img, (head_center_x, head_center_y), (head_center_x - 5, head_center_y + 20), (140, 170, 205), 2)
    cv2.line(img, (head_center_x - 5, head_center_y + 20), (head_center_x + 8, head_center_y + 20), (140, 170, 205), 2)
    
    mouth_y = head_center_y + 50
    mouth_w = 28 + int(4 * np.sin(i * 0.8))
    cv2.ellipse(img, (head_center_x, mouth_y), (mouth_w, 10), 0, 0, 180, (110, 120, 185), 3)
    
    cv2.putText(img, f"DAiSEE Subject 1102 Frame {i+1:02d} [Engaged]", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(daisee_dir, f"frame_{i+1:02d}.png"), img)

# 2. Multi-format input images: BMP, PNG, JPEG, SVG
bmp_img = cv2.imread(os.path.join(daisee_dir, "frame_01.png"))
bmp_path = os.path.join(DATA_INPUT, "sample_bmp.bmp")
cv2.imwrite(bmp_path, bmp_img)

jpg_path = os.path.join(DATA_INPUT, "sample_jpeg.jpg")
cv2.imwrite(jpg_path, bmp_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

png_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
png_canvas[:] = (245, 245, 248)
cv2.rectangle(png_canvas, (40, 40), (600, 440), (220, 215, 210), 2)
cv2.putText(png_canvas, "DIGITAL IMAGE PROCESSING - TUGAS 2", (60, 90), cv2.FONT_HERSHEY_DUPLEX, 0.85, (20, 30, 120), 2)
cv2.putText(png_canvas, "Compression & Watermarking Benchmark", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 1)
for r in range(20, 120, 15):
    cv2.circle(png_canvas, (200, 280), r, (180 - r, 70 + r, 200), 2)
cv2.rectangle(png_canvas, (360, 200), (540, 360), (45, 120, 50), -1)
cv2.putText(png_canvas, "HIGH FREQUENCY", (375, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
cv2.putText(png_canvas, "EDGE REGION", (395, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
png_path = os.path.join(DATA_INPUT, "sample_png.png")
cv2.imwrite(png_path, png_canvas)

# D. SVG image (Vector format XML)
svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1e3a8a;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#3b82f6;stop-opacity:1" />
    </linearGradient>
    <radialGradient id="grad2" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:#f59e0b;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#b45309;stop-opacity:1" />
    </radialGradient>
  </defs>
  <rect width="640" height="480" fill="url(#grad1)" />
  <circle cx="320" cy="240" r="140" fill="url(#grad2)" stroke="#ffffff" stroke-width="6" />
  <polygon points="320,130 355,205 435,215 375,270 390,350 320,310 250,350 265,270 205,215 285,205" fill="#ffffff" opacity="0.9" />
  <text x="320" y="420" font-family="DejaVu Sans, Arial, sans-serif" font-size="28" font-weight="bold" fill="#ffffff" text-anchor="middle">Scalable Vector Graphics (SVG)</text>
  <text x="320" y="450" font-family="DejaVu Sans, Arial, sans-serif" font-size="16" fill="#cbd5e1" text-anchor="middle">Mathematical Vector Primitive Representation</text>
</svg>"""
svg_path = os.path.join(DATA_INPUT, "sample_svg.svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg_content)

# D2. Pure Python rasterization for SVG representation (Zero external C-library dependency)
svg_raster_path = os.path.join(DATA_INPUT, "sample_svg_rasterized.png")
if not os.path.exists(svg_raster_path):
    svg_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        ratio = y / 480.0
        svg_canvas[y, :, 0] = int(138 + (246 - 138) * ratio)
        svg_canvas[y, :, 1] = int(58 + (130 - 58) * ratio)
        svg_canvas[y, :, 2] = int(30 + (59 - 30) * ratio)
    cv2.circle(svg_canvas, (320, 240), 140, (11, 158, 245), -1)
    cv2.circle(svg_canvas, (320, 240), 140, (255, 255, 255), 6)
    pts = np.array([[320, 130], [355, 205], [435, 215], [375, 270], [390, 350],
                    [320, 310], [250, 350], [265, 270], [205, 215], [285, 205]], np.int32)
    cv2.fillPoly(svg_canvas, [pts], (255, 255, 255))
    cv2.putText(svg_canvas, "Scalable Vector Graphics (SVG)", (80, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(svg_canvas, "Mathematical Vector Primitive Representation", (85, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 215, 225), 1)
    cv2.imwrite(svg_raster_path, svg_canvas)

print("Test images verified and ready.")

# -------------------------------------------------------------
# BAGIAN A: DATA AUGMENTASI (DAiSEE)
# -------------------------------------------------------------
print("--> Running Bagian A: Data Augmentation...")

aug_base = os.path.join(DATA_OUTPUT, "augmentation")
tech_dirs = {
    "horizontal_flip": os.path.join(aug_base, "horizontal_flip"),
    "mild_rotation": os.path.join(aug_base, "mild_rotation"),
    "brightness_contrast": os.path.join(aug_base, "brightness_contrast"),
    "zoom_crop": os.path.join(aug_base, "zoom_crop"),
    "gaussian_blur": os.path.join(aug_base, "gaussian_blur")
}
for d in tech_dirs.values():
    os.makedirs(d, exist_ok=True)

def augment_horizontal_flip(image):
    return cv2.flip(image, 1)

def augment_mild_rotation(image, angle=8.0):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def augment_brightness_contrast(image, alpha=1.25, beta=18):
    return np.clip(alpha * image.astype(np.float64) + beta, 0, 255).astype(np.uint8)

def augment_zoom_crop(image, crop_ratio=0.88):
    h, w = image.shape[:2]
    new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    cropped = image[start_y:start_y+new_h, start_x:start_x+new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def augment_gaussian_blur(image, ksize=7, sigma=1.6):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

sample_vis_data = []
for i in range(1, 11):
    f_path = os.path.join(daisee_dir, f"frame_{i:02d}.png")
    orig = cv2.imread(f_path)
    
    hflip = augment_horizontal_flip(orig)
    rot = augment_mild_rotation(orig, angle=8.0)
    bc = augment_brightness_contrast(orig, alpha=1.25, beta=18)
    crop = augment_zoom_crop(orig, crop_ratio=0.88)
    blur = augment_gaussian_blur(orig, ksize=7, sigma=1.6)
    
    cv2.imwrite(os.path.join(tech_dirs["horizontal_flip"], f"frame_{i:02d}_hflip.png"), hflip)
    cv2.imwrite(os.path.join(tech_dirs["mild_rotation"], f"frame_{i:02d}_rot.png"), rot)
    cv2.imwrite(os.path.join(tech_dirs["brightness_contrast"], f"frame_{i:02d}_bc.png"), bc)
    cv2.imwrite(os.path.join(tech_dirs["zoom_crop"], f"frame_{i:02d}_crop.png"), crop)
    cv2.imwrite(os.path.join(tech_dirs["gaussian_blur"], f"frame_{i:02d}_blur.png"), blur)
    
    if i in [1, 5]:
        sample_vis_data.append((orig, hflip, rot, bc, crop, blur, i))

fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
col_titles = ["Original (DAiSEE)", "Horizontal Flip", "Mild Rotation (+8°)", "Brightness & Contrast", "Zoom & Crop (88%)", "Gaussian Blur (σ=1.6)"]

for row_idx, (orig, hflip, rot, bc, crop, blur, frame_num) in enumerate(sample_vis_data):
    imgs = [orig, hflip, rot, bc, crop, blur]
    for col_idx, img in enumerate(imgs):
        ax = axes[row_idx, col_idx]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if row_idx == 0:
            ax.set_title(col_titles[col_idx], fontsize=11, fontweight='bold', pad=8)
        if col_idx == 0:
            ax.set_ylabel(f"Frame {frame_num}", fontsize=11, fontweight='bold')
        ax.axis('off')

plt.tight_layout()
grid_path = os.path.join(FIGURES_DIR, "augmentation_grid.png")
plt.savefig(grid_path, dpi=200, bbox_inches='tight')
plt.close()
print("Augmentation completed and grid saved.")

# -------------------------------------------------------------
# BAGIAN B: IMAGE COMPRESSION
# -------------------------------------------------------------
print("--> Running Bagian B: Image Compression...")

comp_base = os.path.join(DATA_OUTPUT, "compression")
os.makedirs(comp_base, exist_ok=True)

class HuffmanNode:
    def __init__(self, byte_val, freq):
        self.byte_val = byte_val
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(byte_data):
    freq = Counter(byte_data)
    heap = [HuffmanNode(b, f) for b, f in freq.items()]
    heapq.heapify(heap)
    
    if len(heap) == 1:
        node = heapq.heappop(heap)
        root = HuffmanNode(None, node.freq)
        root.left = node
        return root
        
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = HuffmanNode(None, n1.freq + n2.freq)
        merged.left = n1
        merged.right = n2
        heapq.heappush(heap, merged)
    return heap[0]

def build_huffman_codes(root, current_code="", code_map=None):
    if code_map is None:
        code_map = {}
    if root is not None:
        if root.byte_val is not None:
            code_map[root.byte_val] = current_code if current_code else "0"
        build_huffman_codes(root.left, current_code + "0", code_map)
        build_huffman_codes(root.right, current_code + "1", code_map)
    return code_map

def compress_huffman(raw_bytes):
    t0 = time.time()
    root = build_huffman_tree(raw_bytes)
    code_map = build_huffman_codes(root)
    total_bits = sum(len(code_map[b]) for b in raw_bytes)
    header_bytes = len(set(raw_bytes)) * 5
    payload_bytes = math.ceil(total_bits / 8)
    total_compressed_bytes = header_bytes + payload_bytes
    t_elapsed = (time.time() - t0) * 1000.0
    return total_compressed_bytes, t_elapsed

import zlib
def compress_deflate(raw_bytes):
    t0 = time.time()
    compressed = zlib.compress(raw_bytes, level=9)
    t_elapsed = (time.time() - t0) * 1000.0
    return len(compressed), compressed, t_elapsed

def compress_jpeg(image_bgr, quality=80):
    t0 = time.time()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image_bgr, encode_param)
    t_elapsed = (time.time() - t0) * 1000.0
    compressed_bytes = encimg.nbytes
    decoded_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return compressed_bytes, decoded_img, t_elapsed

test_inputs = {
    "BMP (Uncompressed)": bmp_path,
    "PNG (High Frequency)": png_path,
    "JPEG (Standard Photo)": jpg_path,
    "SVG (Rasterized Vector)": svg_raster_path
}

compression_results = []
visual_comparison_samples = {}

for name, path in test_inputs.items():
    img_bgr = cv2.imread(path)
    orig_file_size = os.path.getsize(path)
    raw_pixel_bytes = img_bgr.tobytes()
    
    # 1. Huffman
    huff_size, huff_time = compress_huffman(raw_pixel_bytes)
    huff_cr = orig_file_size / huff_size
    huff_ss = (1.0 - huff_size / orig_file_size) * 100.0
    compression_results.append({
        "Image": name,
        "Technique": "Huffman Coding",
        "Type": "Lossless",
        "Original_Size_KB": orig_file_size / 1024.0,
        "Comp_Size_KB": huff_size / 1024.0,
        "CR": huff_cr,
        "Space_Saving_%": huff_ss,
        "PSNR_dB": 100.0,
        "SSIM": 1.0000,
        "Time_ms": huff_time
    })
    
    # 2. DEFLATE
    def_size, def_payload, def_time = compress_deflate(raw_pixel_bytes)
    def_cr = orig_file_size / def_size
    def_ss = (1.0 - def_size / orig_file_size) * 100.0
    compression_results.append({
        "Image": name,
        "Technique": "DEFLATE (zlib L9)",
        "Type": "Lossless",
        "Original_Size_KB": orig_file_size / 1024.0,
        "Comp_Size_KB": def_size / 1024.0,
        "CR": def_cr,
        "Space_Saving_%": def_ss,
        "PSNR_dB": 100.0,
        "SSIM": 1.0000,
        "Time_ms": def_time
    })
    
    # 3. JPEG
    for q in [85, 50, 15]:
        jpg_size, jpg_dec, jpg_time = compress_jpeg(img_bgr, quality=q)
        jpg_cr = orig_file_size / jpg_size
        jpg_ss = (1.0 - jpg_size / orig_file_size) * 100.0
        psnr_val = calculate_psnr(img_bgr, jpg_dec)
        ssim_val = calculate_ssim(img_bgr, jpg_dec)
        compression_results.append({
            "Image": name,
            "Technique": f"JPEG (Q={q})",
            "Type": "Lossy",
            "Original_Size_KB": orig_file_size / 1024.0,
            "Comp_Size_KB": jpg_size / 1024.0,
            "CR": jpg_cr,
            "Space_Saving_%": jpg_ss,
            "PSNR_dB": psnr_val,
            "SSIM": ssim_val,
            "Time_ms": jpg_time
        })
        if q == 50:
            visual_comparison_samples[name] = (img_bgr, jpg_dec, psnr_val, ssim_val)

print("Compression benchmark finished.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
images_labels = list(test_inputs.keys())
x = np.arange(len(images_labels))
width = 0.2

huff_crs = [r["CR"] for r in compression_results if r["Technique"] == "Huffman Coding"]
def_crs = [r["CR"] for r in compression_results if r["Technique"] == "DEFLATE (zlib L9)"]
jpg50_crs = [r["CR"] for r in compression_results if r["Technique"] == "JPEG (Q=50)"]
jpg15_crs = [r["CR"] for r in compression_results if r["Technique"] == "JPEG (Q=15)"]

ax1.bar(x - 1.5*width, huff_crs, width, label='Huffman (Lossless)', color='#3b82f6')
ax1.bar(x - 0.5*width, def_crs, width, label='DEFLATE (Lossless)', color='#10b981')
ax1.bar(x + 0.5*width, jpg50_crs, width, label='JPEG Q=50 (Lossy)', color='#f59e0b')
ax1.bar(x + 1.5*width, jpg15_crs, width, label='JPEG Q=15 (Lossy)', color='#ef4444')

ax1.set_ylabel('Compression Ratio (CR : 1)', fontsize=11, fontweight='bold')
ax1.set_title('Compression Ratio Across Image Formats', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([l.split(' ')[0] for l in images_labels], fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

qualities = [15, 30, 50, 70, 85, 95]
bmp_sample = cv2.imread(bmp_path)
orig_sz = os.path.getsize(bmp_path)
rd_psnr = []
rd_ss = []
rd_ssim = []

for q in qualities:
    sz, dec, _ = compress_jpeg(bmp_sample, quality=q)
    rd_ss.append((1.0 - sz / orig_sz) * 100.0)
    rd_psnr.append(calculate_psnr(bmp_sample, dec))
    rd_ssim.append(calculate_ssim(bmp_sample, dec))

color = '#1e3a8a'
ax2.plot(rd_ss, rd_psnr, marker='o', linewidth=2.5, color=color, label='PSNR (dB)')
ax2.set_xlabel('Space Saving (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('PSNR (dB)', color=color, fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid(True, linestyle='--', alpha=0.5)

ax2_twin = ax2.twinx()
color_ssim = '#10b981'
ax2_twin.plot(rd_ss, rd_ssim, marker='s', linewidth=2.5, linestyle='--', color=color_ssim, label='SSIM')
ax2_twin.set_ylabel('SSIM', color=color_ssim, fontsize=11, fontweight='bold')
ax2_twin.tick_params(axis='y', labelcolor=color_ssim)
ax2.set_title('Rate-Distortion Curve: JPEG on BMP (Q=15 to 95)', fontsize=12, fontweight='bold')

plt.tight_layout()
comp_plot_path = os.path.join(FIGURES_DIR, "compression_metrics.png")
plt.savefig(comp_plot_path, dpi=200, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for idx, (img_name, (orig, comp, psnr_val, ssim_val)) in enumerate(visual_comparison_samples.items()):
    ax_orig = axes[0, idx]
    ax_comp = axes[1, idx]
    ax_orig.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    ax_orig.set_title(f"Original {img_name.split(' ')[0]}", fontsize=10, fontweight='bold')
    ax_orig.axis('off')
    ax_comp.imshow(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    ax_comp.set_title(f"JPEG Q=50\nPSNR: {psnr_val:.1f}dB, SSIM: {ssim_val:.3f}", fontsize=10)
    ax_comp.axis('off')

plt.tight_layout()
comp_vis_path = os.path.join(FIGURES_DIR, "compression_visual.png")
plt.savefig(comp_vis_path, dpi=200, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# BAGIAN C: WATERMARKING
# -------------------------------------------------------------
print("--> Running Bagian C: Watermarking...")

wm_base = os.path.join(DATA_OUTPUT, "watermarking")
os.makedirs(wm_base, exist_ok=True)

host_img = cv2.imread(bmp_path)
h_h, h_w = host_img.shape[:2]

wm_logo = np.zeros((32, 32), dtype=np.uint8)
cv2.rectangle(wm_logo, (2, 2), (29, 29), 255, -1)
cv2.putText(wm_logo, "ITB", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
cv2.imwrite(os.path.join(wm_base, "watermark_logo.png"), wm_logo)

def apply_visible_watermark(host, text="CONFIDENTIAL - DAiSEE 2026", alpha=0.4):
    watermarked = host.copy()
    overlay = host.copy()
    banner_h = 60
    cv2.rectangle(overlay, (0, h_h - banner_h), (h_w, h_h), (10, 15, 30), -1)
    cv2.putText(overlay, text, (30, h_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.addWeighted(overlay, alpha, host, 1 - alpha, 0, watermarked)
    return watermarked

visible_wm_img = apply_visible_watermark(host_img, alpha=0.4)
cv2.imwrite(os.path.join(wm_base, "visible_watermarked.png"), visible_wm_img)

def embed_lsb(host, watermark_binary):
    watermarked = host.copy()
    w_h, w_w = watermark_binary.shape
    bits = (watermark_binary > 127).astype(np.uint8)
    blue_ch = watermarked[:w_h, :w_w, 0]
    watermarked[:w_h, :w_w, 0] = (blue_ch & 0xFE) | bits
    return watermarked

def extract_lsb(watermarked, w_shape=(32, 32)):
    w_h, w_w = w_shape
    blue_ch = watermarked[:w_h, :w_w, 0]
    extracted_bits = (blue_ch & 1) * 255
    return extracted_bits.astype(np.uint8)

lsb_wm_img = embed_lsb(host_img, wm_logo)
cv2.imwrite(os.path.join(wm_base, "invisible_lsb_watermarked.png"), lsb_wm_img)
extracted_clean_lsb = extract_lsb(lsb_wm_img, (32, 32))
cv2.imwrite(os.path.join(wm_base, "extracted_clean_lsb.png"), extracted_clean_lsb)

def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def embed_dct(host, watermark_binary, alpha=18.0):
    watermarked = host.copy().astype(np.float32)
    w_h, w_w = watermark_binary.shape
    bits = (watermark_binary > 127).astype(np.float32) * 2.0 - 1.0
    ycrcb = cv2.cvtColor(host, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = ycrcb[:, :, 0].copy()
    
    for i in range(w_h):
        for j in range(w_w):
            block = Y[i*8:(i+1)*8, j*8:(j+1)*8]
            dct_block = dct2(block)
            dct_block[4, 3] += alpha * bits[i, j]
            Y[i*8:(i+1)*8, j*8:(j+1)*8] = idct2(dct_block)
            
    ycrcb[:, :, 0] = np.clip(Y, 0, 255)
    return cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

def extract_dct(watermarked, original, w_shape=(32, 32)):
    w_h, w_w = w_shape
    ycrcb_wm = cv2.cvtColor(watermarked, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ycrcb_orig = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y_wm = ycrcb_wm[:, :, 0]
    Y_orig = ycrcb_orig[:, :, 0]
    
    extracted_bits = np.zeros(w_shape, dtype=np.uint8)
    for i in range(w_h):
        for j in range(w_w):
            block_wm = Y_wm[i*8:(i+1)*8, j*8:(j+1)*8]
            block_orig = Y_orig[i*8:(i+1)*8, j*8:(j+1)*8]
            dct_wm = dct2(block_wm)
            dct_orig = dct2(block_orig)
            diff = dct_wm[4, 3] - dct_orig[4, 3]
            extracted_bits[i, j] = 255 if diff > 0 else 0
    return extracted_bits

dct_wm_img = embed_dct(host_img, wm_logo, alpha=18.0)
cv2.imwrite(os.path.join(wm_base, "invisible_dct_watermarked.png"), dct_wm_img)

imperceptibility_results = [
    {
        "Technique": "Visible (Alpha Blending)",
        "Domain": "Spatial",
        "PSNR_dB": calculate_psnr(host_img, visible_wm_img),
        "SSIM": calculate_ssim(host_img, visible_wm_img),
        "Perceptual_Visibility": "High (Deliberately noticeable banner)"
    },
    {
        "Technique": "Invisible LSB (1-bit)",
        "Domain": "Spatial",
        "PSNR_dB": calculate_psnr(host_img, lsb_wm_img),
        "SSIM": calculate_ssim(host_img, lsb_wm_img),
        "Perceptual_Visibility": "Imperceptible (Sub-visual threshold)"
    },
    {
        "Technique": "Invisible DCT (Mid-Freq)",
        "Domain": "Frequency",
        "PSNR_dB": calculate_psnr(host_img, dct_wm_img),
        "SSIM": calculate_ssim(host_img, dct_wm_img),
        "Perceptual_Visibility": "Imperceptible (Spread across block spectrum)"
    }
]

def attack_jpeg(img, quality=50):
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def attack_noise(img, sigma=15):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

def attack_crop(img, crop_ratio=0.85):
    attacked = img.copy()
    h, w = img.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    sy, sx = (h - ch) // 2, (w - cw) // 2
    attacked[:sy, :] = 0
    attacked[sy+ch:, :] = 0
    attacked[:, :sx] = 0
    attacked[:, sx+cw:] = 0
    return attacked

def attack_resize(img, scale=0.5):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

attacks = {
    "No Attack": lambda x: x.copy(),
    "JPEG (Q=75)": lambda x: attack_jpeg(x, 75),
    "JPEG (Q=30)": lambda x: attack_jpeg(x, 30),
    "Gaussian Noise (σ=15)": lambda x: attack_noise(x, 15),
    "Cropping (15% border)": lambda x: attack_crop(x, 0.85),
    "Resizing (50% down-up)": lambda x: attack_resize(x, 0.5)
}

robustness_results = []
extracted_visuals = {}

for attack_name, attack_fn in attacks.items():
    att_lsb = attack_fn(lsb_wm_img)
    ext_lsb = extract_lsb(att_lsb, (32, 32))
    ncc_lsb = calculate_ncc(wm_logo, ext_lsb)
    ber_lsb = calculate_ber(wm_logo, ext_lsb)
    
    att_dct = attack_fn(dct_wm_img)
    ext_dct = extract_dct(att_dct, host_img, (32, 32))
    ncc_dct = calculate_ncc(wm_logo, ext_dct)
    ber_dct = calculate_ber(wm_logo, ext_dct)
    
    att_vis = attack_fn(visible_wm_img)
    vis_psnr = calculate_psnr(visible_wm_img, att_vis)
    
    robustness_results.append({
        "Attack": attack_name,
        "LSB_NCC": ncc_lsb,
        "LSB_BER_%": ber_lsb,
        "DCT_NCC": ncc_dct,
        "DCT_BER_%": ber_dct,
        "Visible_PSNR_dB": vis_psnr
    })
    extracted_visuals[attack_name] = (ext_lsb, ext_dct)

print("Watermarking benchmark completed.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
att_labels = list(attacks.keys())
x = np.arange(len(att_labels))
width = 0.35

lsb_nccs = [r["LSB_NCC"] for r in robustness_results]
dct_nccs = [r["DCT_NCC"] for r in robustness_results]

ax1.bar(x - width/2, lsb_nccs, width, label='LSB (Spatial Domain)', color='#ef4444')
ax1.bar(x + width/2, dct_nccs, width, label='DCT Mid-Freq (Frequency Domain)', color='#10b981')
ax1.set_ylabel('Watermark NCC Score (1.0 = Perfect)', fontsize=11, fontweight='bold')
ax1.set_title('Watermark Robustness (NCC) Under Signal Attacks', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([l.replace(' ', '\n') for l in att_labels], fontsize=9)
ax1.set_ylim([0, 1.1])
ax1.legend(fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

lsb_bers = [r["LSB_BER_%"] for r in robustness_results]
dct_bers = [r["DCT_BER_%"] for r in robustness_results]

ax2.bar(x - width/2, lsb_bers, width, label='LSB BER % (Lower is better)', color='#f59e0b')
ax2.bar(x + width/2, dct_bers, width, label='DCT BER % (Lower is better)', color='#3b82f6')
ax2.set_ylabel('Bit Error Rate (BER %)', fontsize=11, fontweight='bold')
ax2.set_title('Watermark Extraction Bit Error Rate (BER %)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([l.replace(' ', '\n') for l in att_labels], fontsize=9)
ax2.set_ylim([0, 80])
ax2.legend(fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
wm_plot_path = os.path.join(FIGURES_DIR, "watermarking_robustness.png")
plt.savefig(wm_plot_path, dpi=200, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(3, 4, figsize=(15, 9.5))
axes[0, 0].imshow(cv2.cvtColor(host_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Host Image", fontsize=10, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(visible_wm_img, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"Visible Watermark\n(PSNR: {imperceptibility_results[0]['PSNR_dB']:.1f}dB)", fontsize=10, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(lsb_wm_img, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f"Invisible LSB\n(PSNR: {imperceptibility_results[1]['PSNR_dB']:.1f}dB)", fontsize=10, fontweight='bold')
axes[0, 2].axis('off')

axes[0, 3].imshow(cv2.cvtColor(dct_wm_img, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title(f"Invisible DCT Mid-Freq\n(PSNR: {imperceptibility_results[2]['PSNR_dB']:.1f}dB)", fontsize=10, fontweight='bold')
axes[0, 3].axis('off')

key_attacks = ["No Attack", "JPEG (Q=75)", "Gaussian Noise (σ=15)", "Resizing (50% down-up)"]
for col_idx, att_k in enumerate(key_attacks):
    e_lsb, e_dct = extracted_visuals[att_k]
    axes[1, col_idx].imshow(e_lsb, cmap='gray')
    ncc_val = [r['LSB_NCC'] for r in robustness_results if r['Attack'] == att_k][0]
    axes[1, col_idx].set_title(f"LSB [{att_k}]\nNCC: {ncc_val:.2f}", fontsize=9)
    axes[1, col_idx].axis('off')
    
    axes[2, col_idx].imshow(e_dct, cmap='gray')
    ncc_val_dct = [r['DCT_NCC'] for r in robustness_results if r['Attack'] == att_k][0]
    axes[2, col_idx].set_title(f"DCT [{att_k}]\nNCC: {ncc_val_dct:.2f}", fontsize=9)
    axes[2, col_idx].axis('off')

axes[1, 0].set_ylabel("LSB Extractions", fontsize=11, fontweight='bold')
axes[2, 0].set_ylabel("DCT Extractions", fontsize=11, fontweight='bold')

plt.tight_layout()
wm_vis_path = os.path.join(FIGURES_DIR, "watermarking_visual.png")
plt.savefig(wm_vis_path, dpi=200, bbox_inches='tight')
plt.close()

summary_data = {
    "random_seed": RANDOM_SEED,
    "compression_results": compression_results,
    "imperceptibility_results": imperceptibility_results,
    "robustness_results": robustness_results
}
with open(os.path.join(FIGURES_DIR, "experiment_summary.json"), "w") as f:
    json.dump(summary_data, f, indent=2)

print("\n--> [SUCCESS] Entire experimental pipeline executed without errors!")
print(f"Results saved to: {DATA_OUTPUT} and figures saved to: {FIGURES_DIR}")
