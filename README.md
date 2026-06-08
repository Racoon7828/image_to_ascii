# ASCII Art Converter

이미지 / GIF / 영상을 ASCII 아트로 변환하는 도구입니다.

---

## 파일 구성

```
ascii.py          Python CLI - 터미널에 ASCII 아트 출력
web/
  server.py       Flask 백엔드 - 브라우저 GUI용 API 서버
  index.html      브라우저 GUI - 드래그앤드롭, 실시간 미리보기
requirements.txt  의존 라이브러리 목록
```

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 사용법

### 브라우저 GUI (`web/server.py` + `web/index.html`)

```bash
python web/server.py   # http://localhost:5000
```

Flask 서버 실행 후 브라우저에서 `http://localhost:5000` 접속.  
GIF·영상 실시간 변환을 지원하며, 변환은 모두 `ascii.py` 로직이 처리합니다.

**지원 파일:** JPG · PNG · GIF · MP4 · WEBM

| 컨트롤 | 설명 |
|--------|------|
| 너비 | ASCII 문자 열 수 (40~300) |
| 대비 | 명암 강도 (0.5~5.0) |
| 폰트비율 | 문자 셀 세로/가로 비율 보정 (0.3~0.7) |
| 엣지 | Canny 엣지 감지 민감도 (0~150) |
| 크기 | 화면 표시 폰트 크기 (4~20px) |
| 반전 | 밝기 반전 |
| 컬러 | 원본 이미지 색상 유지 |
| TXT / HTML | 결과물 파일 저장 |

---

### CLI (`ascii.py`)

```bash
# 기본 실행
python ascii.py image.png

# 옵션 지정
python ascii.py image.png -W 120 -c 3.0 --font-ratio 0.5

# 반전 끄기
python ascii.py image.png --no-invert

# 파일로 저장
python ascii.py image.png -o result.txt
python ascii.py image.png -o result.html

# 컬러 ANSI 출력
python ascii.py image.png --color

# GIF / 영상 재생
python ascii.py video.mp4
python ascii.py anim.gif --fps 15 --loop
```

#### CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-W`, `--width` | 터미널 너비 | ASCII 출력 열 수 |
| `-c`, `--contrast` | `2.0` | 대비 강도 |
| `--font-ratio` | `0.45` | 폰트 세로/가로 비율 보정 |
| `--canny-low` | `50` | Canny 엣지 하한 임계값 |
| `--canny-high` | `150` | Canny 엣지 상한 임계값 |
| `--color` | off | ANSI 트루컬러 출력 |
| `--no-invert` | — | 밝기 반전 끄기 |
| `--fps` | `12` | 영상·GIF 재생 FPS |
| `--loop` | off | 영상·GIF 반복 재생 |
| `-o`, `--output` | — | 결과 저장 경로 (`.txt` / `.html`, 이미지 전용) |

---

## 변환 알고리즘

1. **그레이스케일 변환** — BT.601 가중치 (`0.299R + 0.587G + 0.114B`)
2. **대비 강화** — 0.5 기준 배율 조정
3. **밝기 → ASCII 매핑** — 70자 문자셋, 짙음(`$`) → 옅음(` `)
4. **Canny 엣지 감지** — Gaussian 블러 후 이중 임계값으로 엣지 검출
5. **방향 문자 교체** — Sobel로 엣지 각도 계산 후 `- / | \` 로 대체 (중간 밝기 픽셀만)

---

## 의존 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| `numpy` | 픽셀 배열 연산, ASCII 매핑 벡터화 |
| `opencv-python` | Canny·Sobel 엣지 감지, 영상 처리, 리사이즈 |
| `Pillow` | 정적 이미지 로드, 대비 조정, 반전 |
| `flask` | 브라우저 GUI용 API 서버 |
