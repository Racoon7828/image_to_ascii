import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def image_to_ascii(image_path, new_width=None, invert=True, contrast=2.0,
                   canny_low=50, canny_high=150, font_ratio=0.45, output=None, color=False):
    ASCII_CHARS = '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,"^`\'. '

    if not os.path.isfile(image_path):
        print(f"오류: 파일을 찾을 수 없습니다 → {image_path}", file=sys.stderr)
        sys.exit(1)

    if new_width is None:
        try:
            new_width = os.get_terminal_size().columns - 1
        except OSError:
            new_width = 100

    img = Image.open(image_path).convert('L')
    img = ImageEnhance.Contrast(img).enhance(contrast)
    if invert:
        img = ImageOps.invert(img)

    width, height = img.size
    new_height = int(new_width * (height / width) * font_ratio)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    pixels = np.array(img, dtype=np.uint8)

    if color:
        img_rgb = Image.open(image_path).convert('RGB')
        img_rgb = img_rgb.resize((new_width, new_height), Image.LANCZOS)
        rgb = np.array(img_rgb, dtype=np.uint8)

    # 밝기 → ASCII 매핑
    base_idx = np.clip(
        (pixels / 255 * (len(ASCII_CHARS) - 1)).astype(int),
        0, len(ASCII_CHARS) - 1
    )
    ascii_grid = np.array(list(ASCII_CHARS))[base_idx]

    # Canny 엣지 감지
    blurred = cv2.GaussianBlur(pixels, (3, 3), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    Gx = cv2.Sobel(pixels.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(pixels.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    angle = np.degrees(np.arctan2(Gy, Gx)) % 180

    mid_brightness = (pixels > 50) & (pixels < 210)
    edge_mask = (edges == 255) & mid_brightness

    char_map = np.where(
        (angle < 22.5) | (angle >= 157.5), '-',
        np.where(angle < 67.5, '/',
        np.where(angle < 112.5, '|', '\\'))
    )
    ascii_grid[edge_mask] = char_map[edge_mask]

    # numpy view로 행 단위 문자열 빠르게 생성
    lines = ascii_grid.view(f'U{new_width}').ravel().tolist()

    if color:
        vfunc = np.vectorize(lambda r, g, b, ch: f'\033[38;2;{r};{g};{b}m{ch}')
        colored = vfunc(rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], ascii_grid)
        print('\n'.join(''.join(row) + '\033[0m' for row in colored))
    else:
        print('\n'.join(lines))

    if output:
        ext = os.path.splitext(output)[1].lower()
        if ext == '.html':
            _save_html(output, ascii_grid, rgb if color else None)
        else:
            with open(output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        print(f"저장 완료: {output}", file=sys.stderr)


def _save_html(path, ascii_grid, rgb=None):
    rows = []
    for r, row in enumerate(ascii_grid):
        parts = []
        for c, ch in enumerate(row):
            safe = ch.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            if rgb is not None:
                R, G, B = int(rgb[r, c, 0]), int(rgb[r, c, 1]), int(rgb[r, c, 2])
                parts.append(f'<span style="color:rgb({R},{G},{B})">{safe}</span>')
            else:
                parts.append(safe)
        rows.append(''.join(parts))

    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
        'body{background:#000;margin:0}'
        'pre{font-family:monospace;font-size:10px;line-height:1.1;color:#fff}'
        '</style></head><body><pre>'
        + '\n'.join(rows)
        + '</pre></body></html>'
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description='이미지를 ASCII 아트로 변환')
    parser.add_argument('image',          nargs='?', default='19c9c4348c23a338.png', help='입력 이미지 파일')
    parser.add_argument('-W', '--width',  type=int,   help='출력 너비 (기본: 터미널 너비)')
    parser.add_argument('-i', '--invert', action='store_true', default=True)
    parser.add_argument('--no-invert',    dest='invert', action='store_false')
    parser.add_argument('-c', '--contrast',  type=float, default=2.0,  help='대비 강도 (기본: 2.0)')
    parser.add_argument('--canny-low',       type=int,   default=50,   help='Canny 하한 임계값 (기본: 50)')
    parser.add_argument('--canny-high',      type=int,   default=150,  help='Canny 상한 임계값 (기본: 150)')
    parser.add_argument('--font-ratio',      type=float, default=0.45, help='폰트 세로/가로 비율 (기본: 0.45)')
    parser.add_argument('--color',           action='store_true',       help='컬러 ANSI 출력')
    parser.add_argument('-o', '--output',    help='저장 파일 경로 (.txt 또는 .html)')
    args = parser.parse_args()

    image_to_ascii(
        image_path = args.image,
        new_width  = args.width,
        invert     = args.invert,
        contrast   = args.contrast,
        canny_low  = args.canny_low,
        canny_high = args.canny_high,
        font_ratio = args.font_ratio,
        output     = args.output,
        color      = args.color,
    )


if __name__ == '__main__':
    main()
