import numpy as np
import cv2
import gradio as gr

ASCII_CHARS = '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,"^`\'. '

def _frame_to_ascii(pixels, canny_low, canny_high):
    base_idx = np.clip(
        (pixels / 255 * (len(ASCII_CHARS) - 1)).astype(int),
        0, len(ASCII_CHARS) - 1
    )
    ascii_grid = np.array(list(ASCII_CHARS))[base_idx]

    blurred = cv2.GaussianBlur(pixels, (3, 3), 0)
    edges   = cv2.Canny(blurred, canny_low, canny_high)

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
    return ascii_grid


def convert(image: np.ndarray, width: int, contrast: float, ratio: float,
            canny_low: int, invert: bool, color: bool) -> str:
    """PIL Image(numpy) → ASCII 문자열 (HTML)"""
    if image is None:
        return ""

    canny_high = canny_low * 3
    h, w = image.shape[:2]
    new_h = max(1, int(width * (h / w) * ratio))

    # 리사이즈
    resized = cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)

    # 그레이스케일 + 대비 + 반전
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = (gray / 255.0 - 0.5) * contrast * 255 + 128
    if invert:
        gray = 255 - gray
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    ascii_grid = _frame_to_ascii(gray, canny_low, canny_high)
    lines = ascii_grid.view(f'U{width}').ravel().tolist()

    if not color:
        # 일반 모드: <pre> 텍스트 (HTML 특수문자 이스케이프)
        text = html.escape('\n'.join(lines))
        return f'<pre style="background:#000;color:#fff;font-size:10px;line-height:1.15;padding:8px">{text}</pre>'

    # 컬러 모드: 픽셀마다 <span> 색상 적용
    rows = []
    for r, line in enumerate(lines):
        parts = []
        for c, ch in enumerate(line):
            R, G, B = int(resized[r, c, 0]), int(resized[r, c, 1]), int(resized[r, c, 2])
            safe = ch.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            safe = '&nbsp;' if safe == ' ' else safe
            parts.append(f'<span style="color:rgb({R},{G},{B})">{safe}</span>')
        rows.append(''.join(parts))

    inner = '\n'.join(rows)
    return f'<pre style="background:#000;font-size:10px;line-height:1.15;padding:8px">{inner}</pre>'


with gr.Blocks(title="ASCII Art Converter", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# ASCII Art Converter")

    with gr.Row():
        with gr.Column(scale=1):
            inp_image  = gr.Image(label="이미지 업로드", type="numpy")
            width      = gr.Slider(40, 300, value=100, step=1,   label="너비")
            contrast   = gr.Slider(0.5, 5.0, value=2.5, step=0.1, label="대비")
            ratio      = gr.Slider(0.3, 0.7, value=0.45, step=0.01, label="폰트 비율")
            canny_low  = gr.Slider(0, 150, value=50, step=1,     label="엣지 민감도")
            invert     = gr.Checkbox(value=True,  label="반전")
            color      = gr.Checkbox(value=False, label="컬러")
            btn        = gr.Button("변환", variant="primary")

        with gr.Column(scale=2):
            out_html = gr.HTML(label="ASCII 결과")

    # 버튼 클릭 시 변환
    btn.click(
        fn=convert,
        inputs=[inp_image, width, contrast, ratio, canny_low, invert, color],
        outputs=out_html,
    )

    # 슬라이더 변경 시 자동 재변환
    for ctrl in [width, contrast, ratio, canny_low, invert, color]:
        ctrl.change(
            fn=convert,
            inputs=[inp_image, width, contrast, ratio, canny_low, invert, color],
            outputs=out_html,
        )


if __name__ == '__main__':
    demo.launch()
