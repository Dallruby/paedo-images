# 그놈들은 오늘도 멋있었다 · 미니홈피

2004년 단월사립고 학생들의 싸이월드 미니홈피를 모바일 세로형으로 재편한 정적 사이트.
크랙AI 연재작 `2000년대 귀여니 소설풍` 프롬프트의 홍보 · 세계관 소개용.

빌드 없음, 의존성 없음. `index.html` 하나로 동작한다.

## 구조

```
index.html   전체 (스타일 · 마크업 · 스크립트)
bgm/         인물별 BGM 5개 (you · junrae · wonho · seolhee · suhyun)
img/         미니룸 · 사진첩 이미지. 현재 비어 있음 — 아래 이름대로 넣으면 자동으로 붙는다
```

```
img/miniroom_user.png  miniroom_junrae.png  miniroom_wonho.png  miniroom_seolhee.png  miniroom_suhyun.png
img/photo_user_1~2.jpg  photo_wonho_1~3.jpg  photo_seolhee_1~6.jpg  photo_suhyun_1~2.jpg
```

전부 가로 4:3 권장. 마준래는 사진첩 0장이 정상.

## 로컬 실행

브라우저에서 `index.html` 을 열거나, 오디오 자동재생 확인용으로 간단한 서버를 띄운다.

```
python3 -m http.server 8000
```

## 배포 (GitHub Pages)

Settings → Pages → Source: **Deploy from a branch** → Branch: `main` / `(root)` 저장.
`.nojekyll` 이 있어 별도 빌드 없이 그대로 서빙된다.

자세한 콘텐츠 · 스킨 · 시대 고증 규칙은 [`인수인계.md`](./인수인계.md) 참고.
