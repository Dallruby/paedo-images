# 정진연구소 사이트 (사주 겁나 잘보는 관상가 아저씨)

레포 루트에 있는 아래 3개 파일이 사이트 전부입니다. 빌드 스텝 없는 순수 정적 사이트.

| 파일 | 설명 |
| --- | --- |
| `index.html` | HTML + CSS + JS 전부 인라인된 단일 파일 |
| `title.png` | 손글씨 타이틀 (투명 배경) |
| `bgm.mp3` | 배경음악 |

`index.html`이 `title.png`, `bgm.mp3`를 **상대경로**로 참조합니다. 파일명·위치를 바꾸면 깨집니다.
두 미디어 파일은 압축·리사이즈 금지 (원본 화질/음질 유지).

외부 의존성은 CDN 2개뿐이라 설치할 것 없음:
- Pretendard 폰트 (jsdelivr)
- lunar-javascript 1.6.12 (jsdelivr) — 만세력 계산용

## Vercel 연결

1. vercel.com → **Add New… → Project** → 이 레포(`Dallruby/paedo-images`) Import
2. **Framework Preset**: `Other`
3. **Root Directory**: `./` (그대로 둠)
4. Build Command / Output Directory: 비워둠 (건드리지 않음)
5. Deploy

이후 이 레포에 푸시하면 자동으로 다시 배포됩니다.

> GitHub Pages로 해도 동작합니다. Settings → Pages → Source를 `main` 브랜치 / `/ (root)`로 두면
> `https://dallruby.github.io/paedo-images/` 에 뜹니다.

## 수정할 때 주의

만세력 계산 로직(`index.html` 하단 `$("#calc").onclick`)은 아래가 **의도된 동작**입니다. 버그로 보고 고치지 마세요.

- 절기 판정용으로 KST에서 60분을 뺌 (lunar-javascript가 절기를 베이징 기준으로 계산) → 연주·월주
- 일주·시주는 KST에서 32분을 뺀 진태양시로 계산 (서울 경도 보정)
- 진태양시 23시 이후면 다음 날로 넘김 (조자시)
- 지장간 테이블(`HD`)은 한국식으로 직접 넣어둔 것. 라이브러리 값과 다름
- 십성(`tg()`)도 직접 계산. 라이브러리가 간체자로 주기 때문
- 1961년 이전 출생은 막아둠 (한국 표준시 변경 이력)
- 출생 시각 모르면 정오 가정, 시주 자리에 `?` 표기

출력 한 줄 형식(`[사주]性:.../曆:.../生:...`)은 다른 시스템이 파싱하므로 **바꾸지 말 것**.
