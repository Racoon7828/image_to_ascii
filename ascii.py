import os
import sys
import time
import argparse
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ASCII_CHARS = '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,"^`\'. '

# 터미널 화면 지우기 + 커서를 좌상단으로 이동 (영상 재생용)
CLEAR = '\033[2J\033[H'


def _get_width(new_width):
    if new_width is not None:
        return new_width
    try:
        return os.get_terminal_size().columns - 1
    except OSError:
        return 100


def _frame_to_ascii(pixels, canny_low, canny_high):
    """그레이스케일 픽셀 배열 → ASCII 문자 격자 반환"""
    base_idx = np.clip(
        (pixels / 255 * (len(ASCII_CHARS) - 1)).astype(int),
        0, len(ASCII_CHARS) - 1
    )
    ascii_grid = np.array(list(ASCII_CHARS))[base_idx]

    # Canny 엣지 감지
    blurred = cv2.GaussianBlur(pixels, (3, 3), 0)
    edges   = cv2.Canny(blurred, canny_low, canny_high)

    Gx = cv2.Sobel(pixels.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(pixels.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    angle = np.degrees(np.arctan2(Gy, Gx)) % 180

    # 중간 밝기 픽셀의 강한 엣지만 방향 문자로 교체
    mid_brightness = (pixels > 50) & (pixels < 210)
    edge_mask = (edges == 255) & mid_brightness

    char_map = np.where(
        (angle < 22.5) | (angle >= 157.5), '-',
        np.where(angle < 67.5, '/',
        np.where(angle < 112.5, '|', '\\'))
    )
    ascii_grid[edge_mask] = char_map[edge_mask]

    return ascii_grid


def image_to_ascii(image_path, new_width=None, invert=True, contrast=2.0,
                   canny_low=50, canny_high=150, font_ratio=0.45, output=None, color=False):
    """정적 이미지 → ASCII 변환 후 터미널 출력 및 파일 저장"""
    if not os.path.isfile(image_path):
        print(f"오류: 파일을 찾을 수 없습니다 → {image_path}", file=sys.stderr)
        sys.exit(1)

    new_width = _get_width(new_width)

    # 흑백 이미지~
    img = Image.open(image_path).convert('L')
    img = ImageEnhance.Contrast(img).enhance(contrast)
    if invert:
        img = ImageOps.invert(img)

    width, height = img.size
    new_height = int(new_width * (height / width) * font_ratio)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    pixels = np.array(img, dtype=np.uint8)

    # 컬러 이미지~
    if color:
        img_rgb = Image.open(image_path).convert('RGB')
        img_rgb = img_rgb.resize((new_width, new_height), Image.LANCZOS)
        rgb = np.array(img_rgb, dtype=np.uint8)

    ascii_grid = _frame_to_ascii(pixels, canny_low, canny_high)
    lines = ascii_grid.view(f'U{new_width}').ravel().tolist()

    if color:
        vfunc   = np.vectorize(lambda r, g, b, ch: f'\033[38;2;{r};{g};{b}m{ch}')
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


def video_to_ascii(video_path, new_width=None, invert=True, contrast=2.0,
                   canny_low=50, canny_high=150, font_ratio=0.45, color=False,
                   target_fps=12, loop=False):
    """GIF·영상 → 프레임별 ASCII 변환 후 터미널 재생"""
    if not os.path.isfile(video_path):
        print(f"오류: 파일을 찾을 수 없습니다 → {video_path}", file=sys.stderr)
        sys.exit(1)

    new_width = _get_width(new_width)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 영상을 열 수 없습니다 → {video_path}", file=sys.stderr)
        sys.exit(1)

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    target_fps = min(src_fps, target_fps)
    skip       = max(1, round(src_fps / target_fps))
    delay      = 1.0 / target_fps

    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 루프 시 처음으로
            n = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                n += 1
                if n % skip != 0:
                    continue

                t0 = time.time()

                # BGR → 그레이스케일 변환 + 대비 + 반전
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                gray = (gray / 255.0 - 0.5) * contrast * 255 + 128
                if invert:
                    gray = 255 - gray
                gray = np.clip(gray, 0, 255).astype(np.uint8)

                h, w = gray.shape
                new_height = max(1, int(new_width * (h / w) * font_ratio))
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)

                ascii_grid = _frame_to_ascii(gray, canny_low, canny_high)
                lines = ascii_grid.view(f'U{new_width}').ravel().tolist()

                if color:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame = cv2.resize(rgb_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    vfunc     = np.vectorize(lambda r, g, b, ch: f'\033[38;2;{r};{g};{b}m{ch}')
                    colored   = vfunc(rgb_frame[:, :, 0], rgb_frame[:, :, 1], rgb_frame[:, :, 2], ascii_grid)
                    frame_str = '\n'.join(''.join(row) + '\033[0m' for row in colored)
                else:
                    frame_str = '\n'.join(lines)

                # 화면 지우고 프레임 출력 (커서 깜빡임 방지)
                sys.stdout.write(CLEAR + frame_str)
                sys.stdout.flush()

                # 처리 시간 제외한 나머지 대기
                elapsed = time.time() - t0
                if delay > elapsed:
                    time.sleep(delay - elapsed)

            if not loop:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print()  # 커서 복원


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
    parser = argparse.ArgumentParser(description='이미지 / GIF / 영상을 ASCII 아트로 변환')
    parser.add_argument('file',               help='입력 파일 (이미지·GIF·영상)')
    parser.add_argument('-W', '--width',      type=int,   help='출력 너비 (기본: 터미널 너비)')
    parser.add_argument('-i', '--invert',     action='store_true', default=True)
    parser.add_argument('--no-invert',        dest='invert', action='store_false')
    parser.add_argument('-c', '--contrast',   type=float, default=2.0,  help='대비 강도 (기본: 2.0)')
    parser.add_argument('--canny-low',        type=int,   default=50,   help='Canny 하한 임계값 (기본: 50)')
    parser.add_argument('--canny-high',       type=int,   default=150,  help='Canny 상한 임계값 (기본: 150)')
    parser.add_argument('--font-ratio',       type=float, default=0.45, help='폰트 세로/가로 비율 (기본: 0.45)')
    parser.add_argument('--color',            action='store_true',      help='컬러 ANSI 출력')
    parser.add_argument('--fps',              type=int,   default=12,   help='영상 재생 FPS (기본: 12)')
    parser.add_argument('--loop',             action='store_true',      help='영상·GIF 반복 재생')
    parser.add_argument('-o', '--output',     help='저장 파일 경로 (.txt 또는 .html, 이미지 전용)')
    args = parser.parse_args()

    # 확장자로 이미지/영상 구분
    ext = os.path.splitext(args.file)[1].lower()
    is_video = ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif')

    if is_video:
        video_to_ascii(
            video_path  = args.file,
            new_width   = args.width,
            invert      = args.invert,
            contrast    = args.contrast,
            canny_low   = args.canny_low,
            canny_high  = args.canny_high,
            font_ratio  = args.font_ratio,
            color       = args.color,
            target_fps  = args.fps,
            loop        = args.loop,
        )
    else:
        image_to_ascii(
            image_path  = args.file,
            new_width   = args.width,
            invert      = args.invert,
            contrast    = args.contrast,
            canny_low   = args.canny_low,
            canny_high  = args.canny_high,
            font_ratio  = args.font_ratio,
            output      = args.output,
            color       = args.color,
        )


if __name__ == '__main__':
    main()
