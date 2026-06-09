import base64
import io
import os
import sys
import tempfile

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file
from PIL import Image, ImageDraw, ImageFont
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ascii import ASCII_CHARS, _frame_to_ascii

app = Flask(__name__)

# ── 영상 저장용 폰트·문자 아틀라스 ───────────────────────
def _load_font(size=10):
    for path in [r'C:\Windows\Fonts\cour.ttf', r'C:\Windows\Fonts\consola.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf']:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

_FONT = _load_font(10)
try:
    _b  = _FONT.getbbox('M')
    _CW = _b[2] - _b[0]
    _CH = _b[3] - _b[1] + 2
except Exception:
    _CW, _CH = 6, 11
def _build_atlas():
    chars = list(set(ASCII_CHARS + '-/|\\'))
    bitmaps = []
    for ch in chars:
        img = Image.new('L', (_CW, _CH), 0)
        ImageDraw.Draw(img).text((0, 0), ch, fill=255, font=_FONT)
        bitmaps.append(np.array(img))
    return chars, np.stack(bitmaps)

_ATLAS_CHARS, _ATLAS = _build_atlas()
_CHAR_IDX = {ch: i for i, ch in enumerate(_ATLAS_CHARS)}

def _lines_to_image(lines, resized_bgr, color: bool) -> np.ndarray:
    """ASCII 라인 → numpy RGB 이미지 (아틀라스 인덱싱)"""
    rows = len(lines)
    cols = len(lines[0]) if lines else 0
    idx  = np.array([[_CHAR_IDX.get(ch, 0) for ch in line] for line in lines], dtype=np.int32)
    bitmaps = _ATLAS[idx]  # (rows, cols, CH, CW)

    if not color:
        gray = bitmaps.transpose(0, 2, 1, 3).reshape(rows * _CH, cols * _CW)
        return np.stack([gray] * 3, axis=2)

    rgb      = resized_bgr[:, :, ::-1].astype(np.float32)
    bitmaps_f = bitmaps.astype(np.float32) / 255.0
    colored  = (bitmaps_f[..., np.newaxis] * rgb[:, :, np.newaxis, np.newaxis, :]).astype(np.uint8)
    return colored.transpose(0, 2, 1, 3, 4).reshape(rows * _CH, cols * _CW, 3)


# ── 단일 프레임 변환 ─────────────────────────────────────
def process_frame(img_bytes, width, contrast, ratio, invert, canny_low, canny_high, color):
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("이미지를 디코딩할 수 없습니다.")

    h, w = frame.shape[:2]
    new_h = max(1, int(width * (h / w) * ratio))
    resized = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = (gray / 255.0 - 0.5) * contrast * 255 + 128
    if invert:
        gray = 255 - gray
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    ascii_grid = _frame_to_ascii(gray, canny_low, canny_high)
    lines = ascii_grid.view(f'U{width}').ravel().tolist()

    result = {'lines': lines}
    if color:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        result['colors'] = base64.b64encode(rgb.tobytes()).decode()
    return result


# ── 라우트 ───────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')


@app.route('/api/convert', methods=['POST'])
def convert():
    try:
        file       = request.files.get('file')
        width      = int(request.form.get('width', 100))
        contrast   = float(request.form.get('contrast', 2.5))
        ratio      = float(request.form.get('ratio', 0.45))
        invert     = request.form.get('invert', 'true') == 'true'
        canny_low  = int(request.form.get('canny_low', 50))
        canny_high = int(request.form.get('canny_high', 150))
        color      = request.form.get('color', 'false') == 'true'
        return jsonify(process_frame(file.read(), width, contrast, ratio,
                                     invert, canny_low, canny_high, color))
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/convert-video', methods=['POST'])
def convert_video():
    """영상 전체를 ASCII 변환 후 mp4로 반환"""
    vid_path = out_path = None
    try:
        file       = request.files.get('file')
        width      = int(request.form.get('width', 80))
        contrast   = float(request.form.get('contrast', 2.5))
        ratio      = float(request.form.get('ratio', 0.45))
        invert     = request.form.get('invert', 'true') == 'true'
        canny_low  = int(request.form.get('canny_low', 50))
        canny_high = int(request.form.get('canny_high', 150))
        color      = request.form.get('color', 'false') == 'true'
        fps_out    = int(request.form.get('fps', 12))

        # 업로드 영상 임시 저장
        ext = os.path.splitext(file.filename)[1] or '.mp4'
        fd, vid_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        file.save(vid_path)

        fd2, out_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd2)

        cap = cv2.VideoCapture(vid_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        skip = max(1, round(src_fps / fps_out))
        n = 0

        with imageio.get_writer(out_path, fps=fps_out, codec='libx264',
                                pixelformat='yuv420p', quality=7) as writer:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                n += 1
                if n % skip != 0:
                    continue

                h, w = frame.shape[:2]
                new_h = max(1, int(width * (h / w) * ratio))
                resized = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)

                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
                gray = (gray / 255.0 - 0.5) * contrast * 255 + 128
                if invert:
                    gray = 255 - gray
                gray = np.clip(gray, 0, 255).astype(np.uint8)

                ascii_grid = _frame_to_ascii(gray, canny_low, canny_high)
                lines = ascii_grid.view(f'U{width}').ravel().tolist()
                writer.append_data(_lines_to_image(lines, resized, color))

        cap.release()
        os.remove(vid_path)
        vid_path = None

        # 파일을 메모리에 읽고 삭제 후 전송
        with open(out_path, 'rb') as f:
            data = f.read()
        os.remove(out_path)
        out_path = None

        return send_file(io.BytesIO(data), mimetype='video/mp4',
                         as_attachment=True, download_name='ascii.mp4')

    except Exception as e:
        import traceback; traceback.print_exc()
        for p in [vid_path, out_path]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False, threaded=True, port=5000)
