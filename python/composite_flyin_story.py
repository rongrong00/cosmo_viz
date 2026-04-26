"""Composite halo-8 fly-in: 5 species × 192 frames = 960 total.

Each species owns a contiguous slice of the shared camera path. At each
boundary a 30-frame linear crossfade overlaps the two species (both read
the same camera frame). A species label is drawn in the bottom-left and
crossfades with the visuals.
"""
import os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SPECIES = [
    ('dm',    'output/snap158_halo8_flyin_dm_png',     'Dark matter density'),
    ('gas',   'output/snap158_halo8_flyin_gas_png',    'Gas density'),
    ('temp',  'output/snap158_halo8_flyin_temp_png',   'Gas temperature'),
    ('metal', 'output/snap158_halo8_flyin_metal_png',  'Gas metallicity'),
    ('star',  'output/snap158_halo8_stars_flyin',      'Stellar light'),
]
N_FRAMES   = 960
SEG        = N_FRAMES // len(SPECIES)   # 192
XFADE      = 30                          # frames centered on each boundary
OUT_DIR    = 'output/snap158_halo8_flyin_story_png'
OUT_MP4    = 'output/snap158_halo8_flyin_story.mp4'
FONT_PATH  = '/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf'


def segment_info(k):
    """For output frame k return (s_prev, s_next, alpha) where output =
    (1-alpha)*s_prev + alpha*s_next. If not in fade, s_prev==s_next and alpha=0."""
    # boundaries at k = SEG, 2*SEG, 3*SEG, 4*SEG
    for s in range(1, len(SPECIES)):
        b = s * SEG
        if b - XFADE // 2 <= k < b + XFADE // 2:
            alpha = (k - (b - XFADE // 2) + 0.5) / XFADE
            return s - 1, s, float(np.clip(alpha, 0.0, 1.0))
    s = min(k // SEG, len(SPECIES) - 1)
    return s, s, 0.0


def load_rgb(png):
    im = Image.open(png).convert('RGB')
    return np.asarray(im, dtype=np.float32) / 255.0


def draw_label(arr, text, alpha, font):
    if alpha <= 0.01:
        return arr
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    W, H = im.size
    margin = int(0.035 * H)
    # shadow + text
    a = int(255 * alpha)
    bbox = d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
    x = margin
    y = H - margin - th
    # translucent black panel behind text
    pad = int(0.25 * th)
    d.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad],
                fill=(0, 0, 0, int(0.45 * a)))
    d.text((x + 2, y + 2), text, fill=(0, 0, 0, a), font=font)
    d.text((x, y), text, fill=(255, 255, 255, a), font=font)
    im = Image.alpha_composite(im.convert('RGBA'), overlay).convert('RGB')
    return np.asarray(im, dtype=np.float32) / 255.0


def worker(k):
    s_prev, s_next, alpha = segment_info(k)
    _, dir_prev, label_prev = SPECIES[s_prev]
    _, dir_next, label_next = SPECIES[s_next]
    png_prev = f'{dir_prev}/frame_{k:04d}.png'
    png_next = f'{dir_next}/frame_{k:04d}.png'
    A = load_rgb(png_prev)
    if s_prev == s_next:
        out = A
    else:
        B = load_rgb(png_next)
        out = (1.0 - alpha) * A + alpha * B
    H = out.shape[0]
    font = ImageFont.truetype(FONT_PATH, max(28, H // 22))
    # label visibility: prev fades out (1-alpha), next fades in (alpha)
    if s_prev == s_next:
        out = draw_label(out, label_prev, 1.0, font)
    else:
        out = draw_label(out, label_prev, 1.0 - alpha, font)
        out = draw_label(out, label_next, alpha, font)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(
        f'{OUT_DIR}/frame_{k:04d}.png')
    return k


def main():
    for _, d, _ in SPECIES:
        if not os.path.isdir(d):
            print(f'missing dir: {d}', file=sys.stderr); sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = list(range(N_FRAMES))
    with Pool(16) as pool:
        for n, _ in enumerate(pool.imap_unordered(worker, frames, chunksize=4)):
            if n % 50 == 0:
                print(f'  [{n}/{N_FRAMES}]', flush=True)
    print('done compositing PNGs', flush=True)


if __name__ == '__main__':
    main()
