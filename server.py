import base64
import io

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from ascii import ASCII_CHARS, _frame_to_ascii

app = Flask(__name__)


def process_frame(img_bytes: bytes, width: int, contrast: float, ratio: float,
                  invert: bool, canny_low: int, canny_high: int, color: bool) -> dict:
    """이미지 바이트 → ASCII 변환 결과 반환"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("이미지를 디코딩할 수 없습니다.")

    h, w = frame.shape[:2]
    new_h = max(1, int(width * (h / w) * ratio))
    resized = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)

    # BGR → 그레이스케일 + 대비 + 반전
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


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/convert', methods=['POST'])
def convert():
    """브라우저에서 보낸 이미지/프레임을 ASCII로 변환"""
    try:
        file      = request.files.get('file')
        width     = int(request.form.get('width', 100))
        contrast  = float(request.form.get('contrast', 2.5))
        ratio     = float(request.form.get('ratio', 0.45))
        invert    = request.form.get('invert', 'true') == 'true'
        canny_low = int(request.form.get('canny_low', 50))
        canny_high= int(request.form.get('canny_high', 150))
        color     = request.form.get('color', 'false') == 'true'

        result = process_frame(
            file.read(), width, contrast, ratio,
            invert, canny_low, canny_high, color
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False, threaded=True, port=5000)
