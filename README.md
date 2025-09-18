# 📚 LIBanalysiscusor - 공공도서관 설문 시각화 대시보드

이 프로젝트는 공공도서관 이용자 설문조사 데이터를 분석하고 시각화하는 Streamlit 기반 웹 애플리케이션입니다.

## 📋 주요 기능

- 📊 **도서관 이용양태 분석**: 이용자 패턴 및 행동 분석
- 🧩 **이용자 세그먼트 조합 분석**: 다양한 이용자 그룹 분석
- 🗺️ **자치구 구성 문항 분석**: 지역별 도서관 이용 현황
- 🖼️ **도서관 이미지 분석**: 이용자 인식 및 만족도 분석
- 🏋️ **도서관 강약점 분석**: SWOT 분석 및 개선점 도출
- 🔍 **공통 심화 분석**: 전체 및 영역별 상세 분석
- 🧠 **전략 인사이트**: AI 기반 인사이트 제공

## 🚀 시작하기

### 로컬에서 실행

1. 저장소를 클론합니다:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

3. 앱을 실행합니다:
```bash
streamlit run main.py
```

4. 브라우저에서 `http://localhost:8501`로 접속합니다.

### Streamlit Cloud에서 배포

1. 이 저장소를 GitHub에 푸시합니다.
2. [Streamlit Cloud](https://share.streamlit.io/)에 접속합니다.
3. "New app"을 클릭하고 GitHub 저장소를 연결합니다.
4. 메인 파일 경로를 `main.py`로 설정합니다.
5. "Deploy"를 클릭합니다.

## 📁 프로젝트 구조

```
├── main.py              # 메인 스트림릿 앱 (도서관 분석 대시보드)
├── main_backup.py       # 백업된 기본 스트림릿 앱
├── requirements.txt     # Python 패키지 의존성
├── README.md           # 프로젝트 설명서
├── .gitignore          # Git 무시 파일
├── assets/             # 정적 자산
│   ├── logo.png        # 로고 이미지
│   └── requirements.txt # 원본 requirements.txt
├── .streamlit/         # 스트림릿 설정
│   └── config.toml     # 스트림릿 구성 파일
└── .github/
    └── workflows/
        └── deploy.yml  # GitHub Actions 워크플로우
```

## 🛠️ 개발

### 새로운 기능 추가

1. `main.py`에서 새로운 메뉴 항목을 추가합니다.
2. 해당 기능을 구현합니다.
3. 필요시 `requirements.txt`에 새로운 패키지를 추가합니다.

### 의존성 관리

새로운 패키지를 추가한 후:
```bash
pip freeze > requirements.txt
```

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여

1. 이 저장소를 포크합니다.
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.
