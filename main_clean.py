# =============================================================================
# LIBanalysiscusor - 공공도서관 설문 시각화 대시보드
# =============================================================================
# 이 파일은 공공도서관 이용자 설문조사 데이터를 분석하고 시각화하는 
# Streamlit 기반 웹 애플리케이션입니다.
# 
# 주요 기능:
# - 이용자 세그먼트 조합 분석
# - 자치구 구성 문항 분석  
# - 도서관 이용양태 분석
# - 도서관 이미지 분석
# - 도서관 강약점 분석
# - 공통 심화 분석
# - AI 기반 전략 인사이트 제공
# =============================================================================

# =============================================================================
# 1. 라이브러리 및 패키지 임포트
# =============================================================================
import time
import numpy as np
import streamlit as st
import scipy.stats as stats
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import io
import os
from PIL import Image
import openai
import streamlit.components.v1 as components 
import hashlib
import jsonschema
import json 
import math
import logging
from itertools import cycle

# =============================================================================
# 2. 기본 설정 및 초기화
# =============================================================================

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# OpenAI API 설정
openai.api_key = st.secrets["openai"]["api_key"]
client = openai.OpenAI(api_key=openai.api_key)

# 기본 색상 팔레트 설정
DEFAULT_PALETTE = px.colors.qualitative.Plotly
COLOR_CYCLER = cycle(DEFAULT_PALETTE)

# =============================================================================
# 3. 상수 및 매핑 정의
# =============================================================================

# 중분류별 컬럼 매핑 정의
MIDDLE_CATEGORY_MAPPING = {
    '공간 및 이용편의성': lambda col: any(keyword in col for keyword in ['공간', '이용편의', '접근성', '위치']),
    '자료 및 서비스': lambda col: any(keyword in col for keyword in ['자료', '도서', '서비스', '프로그램']),
    '직원 및 안내': lambda col: any(keyword in col for keyword in ['직원', '안내', '도움', '상담']),
    '시설 및 환경': lambda col: any(keyword in col for keyword in ['시설', '환경', '청결', '조용함']),
    '운영 및 관리': lambda col: any(keyword in col for keyword in ['운영', '관리', '시간', '규정'])
}

# KDC 분류 키워드 매핑
KDC_KEYWORD_MAP = {
    '000 총류': ["백과사전", "도서관", "독서", "문헌정보", "기록", "출판", "서지"],
    '100 철학': ["철학", "명상", "윤리", "논리학", "심리학"],
    '200 종교': ["종교", "기독교", "불교", "천주교", "신화", "신앙", "종교학"],
    '300 사회과학': ["사회", "정치", "경제", "법률", "행정", "교육", "복지", "여성", "노인", "육아", "아동복지", "사회문제", "노동", "환경문제", "인권"],
    '400 자연과학': ["수학", "물리", "화학", "생물", "지구과학", "과학", "천문", "기후", "의학", "생명과학"],
    '500 기술과학': ["건강", "의료", "요리", "간호", "공학", "컴퓨터", "AI", "IT", "농업", "축산", "산업", "기술", "미용"],
    '600 예술': ["미술", "음악", "무용", "사진", "영화", "연극", "디자인", "공예", "예술", "문화예술"],
    '700 언어': ["언어", "국어", "영어", "일본어", "중국어", "외국어", "한자", "문법"],
    '800 문학': ["소설", "시", "수필", "에세이", "희곡", "문학", "동화", "웹툰", "판타지", "문예"],
    '900 역사·지리': ["역사", "지리", "한국사", "세계사", "여행", "문화유산", "관광"],
    '원서(영어)': ["원서", "영문도서", "영문판", "영어원서"],
    '연속간행물': ["잡지", "간행물", "연속간행물"],
    '해당없음': []
}

# =============================================================================
# 4. AI 및 API 관련 함수들
# =============================================================================
# 이 섹션은 OpenAI API와 관련된 모든 함수들을 포함합니다.
# API 호출, 인사이트 생성, 텍스트 분석 등의 기능을 담당합니다.

def safe_chat_completion(*, model="gpt-4.1-nano", messages, temperature=0.2, max_tokens=300, retries=3, backoff_base=1.0):
    """
    OpenAI API 호출을 안전하게 처리하는 함수
    - 재시도 로직과 백오프 전략을 포함하여 API 호출 실패에 대비
    - 네트워크 오류나 API 제한에 대한 복원력 제공
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp
        except Exception as e:
            logging.warning(f"OpenAI call failed (attempt {attempt}): {e}")
            last_exc = e
            time.sleep(backoff_base * (2 ** (attempt - 1)))
    logging.error("OpenAI call failed after retries")
    raise last_exc

def call_gpt_for_insight(prompt, model="gpt-4.1-mini", temperature=0.2, max_tokens=1500):
    """
    AI 인사이트 생성을 위한 GPT 호출 함수
    - 분석 결과를 바탕으로 전략적 인사이트 생성
    - 도서관 서비스 개선 방안 제시
    """
    try:
        response = safe_chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error calling GPT: {e}")
        return "인사이트 생성 중 오류가 발생했습니다."

def build_common_overall_insight_prompt(midcat_scores: dict, within_deviations: dict, abc_df: pd.DataFrame) -> str:
    """
    전체 분석을 위한 AI 프롬프트 생성
    - 중분류 점수와 편차 정보를 바탕으로 종합적인 인사이트 요청
    - A/B/C 그룹별 비교 분석 결과 포함
    """
    midcat_str = ", ".join(f"{k} {v:.1f}" for k, v in midcat_scores.items())
    strengths = []
    weaknesses = []
    for mid, series in within_deviations.items():
        if series is None:
            continue
        try:
            dev = series.dropna()
        except Exception:
            continue
        if dev.empty:
            continue
        sorted_series = dev.sort_values(ascending=False)
        top_pos_label = sorted_series.index[0]
        top_pos_val = sorted_series.iloc[0]
        top_neg_label = sorted_series.index[-1]
        top_neg_val = sorted_series.iloc[-1]
        strengths.append(f"{mid}의 '{top_pos_label}' +{top_pos_val:.1f}")
        weaknesses.append(f"{mid}의 '{top_neg_label}' {top_neg_val:.1f}")
    
    try:
        abc_pivot = abc_df.pivot(index="중분류", columns="문항유형", values="평균값")
    except Exception:
        abc_pivot = pd.DataFrame()
    abc_lines = []
    if not abc_pivot.empty:
        for mid in abc_pivot.index:
            eval_val = abc_pivot.loc[mid].get("서비스 평가", None)
            effect_val = abc_pivot.loc[mid].get("서비스 효과", None)
            sat_val = abc_pivot.loc[mid].get("전반적 만족도", None)
            eval_str = f"{eval_val:.1f}" if pd.notna(eval_val) else "N/A"
            effect_str = f"{effect_val:.1f}" if pd.notna(effect_val) else "N/A"
            sat_str = f"{sat_val:.1f}" if pd.notna(sat_val) else "N/A"
            abc_lines.append(f"{mid}: 평가 {eval_str}, 효과 {effect_str}, 만족도 {sat_str}")
    abc_str = "\n".join(abc_lines)

    prompt = f"""
설문 데이터 공통 심화 분석 요약을 만들어줘.

입력 요약:
- 전체 중분류 평균 만족도: {midcat_str}
- 중분류별 강점 예시: {', '.join(strengths[:3])}
- 중분류별 약점 예시: {', '.join(weaknesses[:3])}
- 서비스 평가/효과/만족도 (A/B/C) 비교:
{abc_str}

요청:
1. 주요 관찰 패턴 2~3개를 간결하게 기술해줘.
2. 어떤 중분류가 상대적 강점/약점인지 숫자와 함께 설명해줘.
3. A/B/C 비교에서 드러나는 특징을 짚어줘.
4. 전략적 제안 3개: (1) 우선 개입할 대상, (2) 확장할 강점, (3) 보완할 약점 각각 구체적으로 써줘.

제한: 숫자는 한 자리 소수, 비즈니스 톤, 소제목 포함, 전체 700~1100자. 출력은 텍스트만.
"""
    return prompt.strip()

def build_area_insight_prompt(midcat_scores: dict, abc_df: pd.DataFrame) -> str:
    """
    영역별 분석을 위한 AI 프롬프트 생성
    - 특정 영역의 만족도 분석 결과를 바탕으로 인사이트 요청
    - 그룹별 비교 분석 결과 포함
    """
    midcat_str = ", ".join(f"{k} {v:.1f}" for k, v in midcat_scores.items())
    abc_markdown = abc_df.to_markdown(index=False)
    
    prompt = f"""
특정 영역의 도서관 만족도 분석 결과를 바탕으로 인사이트를 제공해주세요.

영역별 점수:
{midcat_str}

그룹별 비교:
{abc_markdown}

해당 영역의 특성과 개선 방안을 제시해주세요.
"""
    return prompt

# =============================================================================
# 5. 텍스트 처리 및 유틸리티 함수들
# =============================================================================
# 이 섹션은 텍스트 처리, 데이터 정제, 시각화 관련 유틸리티 함수들을 포함합니다.

def remove_parentheses(text):
    """
    텍스트에서 괄호와 그 내용을 제거
    - 차트 라벨 정리 시 사용
    """
    return re.sub(r'\(.*?\)', '', text).strip()

def wrap_label(label, width=10):
    """
    라벨을 지정된 너비로 줄바꿈
    - 차트의 긴 라벨을 여러 줄로 분할
    """
    return '<br>'.join([label[i:i+width] for i in range(0, len(label), width)])

def get_qualitative_colors(n):
    """
    질적 데이터용 색상 팔레트 생성
    - 차트에서 사용할 색상 배열 반환
    """
    palette = DEFAULT_PALETTE
    return [c for _, c in zip(range(n), cycle(palette))]

def wrap_label_fixed(label: str, width: int = 35) -> str:
    """
    라벨을 고정 너비로 줄바꿈 (HTML 태그 고려)
    - HTML 렌더링을 고려한 라벨 줄바꿈
    """
    parts = [label[i:i+width] for i in range(0, len(label), width)]
    return "<br>".join(parts)

def is_trivial(text):
    """
    텍스트가 의미있는 내용인지 판단
    - 빈 응답이나 의미없는 응답 필터링
    """
    text = str(text).strip()
    return text in ["", "X", "x", "감사합니다", "감사", "없음"]

def split_keywords_simple(text):
    """
    텍스트를 키워드로 분할
    - 쉼표, 점, 공백 등을 기준으로 키워드 추출
    """
    parts = re.split(r"[.,/\s]+", text)
    return [p.strip() for p in parts if len(p.strip()) > 1]

def map_keyword_to_category(keyword):
    """
    키워드를 KDC 카테고리로 매핑
    - 도서 분류 체계에 따른 키워드 분류
    """
    for cat, kws in KDC_KEYWORD_MAP.items():
        if any(k in keyword for k in kws):
            return cat
    return "해당없음"

def escape_tildes(text: str, mode: str = "html") -> str:
    """
    텍스트에서 틸드(~) 문자를 이스케이프 처리
    - 마크다운 취소선 방지를 위한 처리
    """
    if mode == "html":
        text = text.replace("~~", "&#126;&#126;")
        return text.replace("~", "&#126;")
    else:  # markdown
        text = text.replace("~~", r"\~\~")
        return text.replace("~", r"\~")

def safe_markdown(text, **kwargs):
    """
    안전한 마크다운 렌더링
    - 특수문자 이스케이프 처리 후 마크다운 렌더링
    """
    safe = escape_tildes(text, mode="markdown")
    st.markdown(safe, **kwargs)

def is_meaningful_long(text: str) -> bool:
    """
    긴 텍스트가 의미있는 내용인지 판단
    - 장문 응답의 품질 검증
    """
    if not text or len(text.strip()) < 10:
        return False
    text_clean = text.strip().lower()
    trivial_phrases = ['없음', '모름', '해당없음', '특별한', '없습니다']
    return not any(phrase in text_clean for phrase in trivial_phrases)

def get_clean_long_responses(raw: list[str]) -> list[str]:
    """
    의미있는 긴 응답만 필터링
    - 분석에 사용할 고품질 장문 응답 추출
    """
    return [resp for resp in raw if is_meaningful_long(resp)]

# =============================================================================
# 6. 데이터 처리 및 분석 함수들
# =============================================================================
# 이 섹션은 데이터 처리, 통계 분석, 점수 계산 등의 핵심 분석 함수들을 포함합니다.

def extract_question_code(col_name: str) -> str:
    """
    컬럼명에서 문항 코드/번호만 추출
    - 'Q1-1. 공간 만족도' -> 'Q1-1' 형태로 변환
    """
    m = re.match(r'^([A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*)', col_name.strip())
    if m:
        return m.group(1)
    if '.' in col_name:
        return col_name.split('.', 1)[0].strip()
    if ' ' in col_name:
        return col_name.split(' ', 1)[0].strip()
    return col_name.strip()

def expand_midcategory_to_columns(midcategory: str, df: pd.DataFrame):
    """
    중분류 이름을 받아 해당하는 실제 컬럼명 목록으로 확장
    - 중분류별로 관련된 모든 문항 컬럼 추출
    """
    for mid, predicate in MIDDLE_CATEGORY_MAPPING.items():
        if midcategory.strip().lower() == mid.strip().lower():
            return [c for c in df.columns if predicate(c)]
    return []

def scale_likert(series):
    """
    리커트 척도 데이터를 0-100 점수로 변환
    - 텍스트 응답을 숫자 점수로 변환
    - 통계 분석을 위한 표준화
    """
    if series.dtype in ['object', 'string']:
        mapping = {
            '매우 불만족': 0, '불만족': 25, '보통': 50, '만족': 75, '매우 만족': 100,
            '전혀 그렇지 않다': 0, '그렇지 않다': 25, '보통': 50, '그렇다': 75, '매우 그렇다': 100
        }
        return series.map(mapping).fillna(50)
    else:
        return ((series - series.min()) / (series.max() - series.min()) * 100).fillna(50)

def compute_midcategory_scores(df):
    """
    중분류별 평균 점수 계산
    - 각 중분류에 속하는 문항들의 평균 점수 계산
    - 전체 만족도 평가의 기준이 되는 핵심 함수
    """
    scores = {}
    for midcat, predicate in MIDDLE_CATEGORY_MAPPING.items():
        cols = [c for c in df.columns if predicate(c)]
        if cols:
            scores[midcat] = df[cols].apply(scale_likert).mean().mean()
    return pd.Series(scores)

def compute_within_category_item_scores(df):
    """
    카테고리 내 개별 문항 점수 계산
    - 중분류 내에서 각 문항별 세부 점수 계산
    - 세부 분석을 위한 데이터 제공
    """
    scores = {}
    for midcat, predicate in MIDDLE_CATEGORY_MAPPING.items():
        cols = [c for c in df.columns if predicate(c)]
        if cols:
            scores[midcat] = df[cols].apply(scale_likert).mean()
    return scores

def interpret_midcategory_scores(df):
    """
    중분류 점수를 해석하여 사용자에게 의미있는 정보를 제공
    - 전체 평균 대비 높은/낮은 중분류를 식별
    - 사용자 친화적인 해석 텍스트 생성
    """
    scores = compute_midcategory_scores(df)
    if scores.empty:
        return "중분류 점수를 계산할 충분한 데이터가 없습니다."
    overall = scores.mean()
    high = scores[scores >= overall + 5].index.tolist()
    low = scores[scores <= overall - 5].index.tolist()

    parts = []
    parts.append(f"전체 중분류 평균은 {overall:.1f}점입니다.")
    if high:
        parts.append(f"평균보다 높은 중분류: {', '.join(high)}.")
    if low:
        parts.append(f"평균보다 낮은 중분류: {', '.join(low)}.")
    if not high and not low:
        parts.append("모든 중분류가 전체 평균 수준과 비슷합니다.")
    return " ".join(parts)

def cohen_d(x, y):
    """
    Cohen's d 효과 크기 계산
    - 두 그룹 간의 통계적 효과 크기 측정
    - 유의미한 차이의 정도를 정량화
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * x.var() + (ny - 1) * y.var()) / dof)
    return (x.mean() - y.mean()) / pooled_std

def compare_midcategory_by_group(df, group_col):
    """
    그룹별 중분류 점수 비교
    - 특정 그룹 변수에 따른 중분류별 점수 차이 분석
    - 세그먼트별 만족도 패턴 파악
    """
    def per_row_mid_scores(subdf):
        return compute_midcategory_scores(subdf)
    
    group_scores = df.groupby(group_col).apply(per_row_mid_scores)
    return group_scores

# =============================================================================
# 7. 시각화 및 표시 함수들
# =============================================================================
# 이 섹션은 차트 생성, 테이블 표시, UI 컴포넌트 등의 시각화 관련 함수들을 포함합니다.

def _sanitize_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit 표시를 위한 데이터프레임 정리
    - 특수문자 제거 및 데이터 정제
    - 안전한 표시를 위한 전처리
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
    return df_clean

def render_chart_and_table(bar, table, title, key_prefix=""):
    """
    차트와 테이블을 함께 렌더링
    - 시각화와 데이터를 나란히 표시
    - 사용자 경험 향상을 위한 레이아웃 구성
    """
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(bar, use_container_width=True)
    
    with col2:
        st.dataframe(table, use_container_width=True)

def show_table(df, caption):
    """
    테이블 표시
    - 데이터프레임을 깔끔하게 표시
    - 캡션과 함께 설명 제공
    """
    st.dataframe(df, use_container_width=True)
    if caption:
        st.caption(caption)

def render_insight_card(title: str, content: str, key: str = None):
    """
    인사이트 카드 렌더링
    - AI 생성 인사이트를 카드 형태로 표시
    - 시각적으로 구분된 정보 제공
    """
    with st.container():
        st.markdown(f"### {title}")
        st.markdown(content)
        st.markdown("---")

def midcategory_avg_table(df):
    """
    중분류 평균 점수 테이블 생성
    - 중분류별 점수를 정렬된 테이블로 표시
    - 성과 비교를 위한 데이터 제공
    """
    scores = compute_midcategory_scores(df)
    return pd.DataFrame({
        '중분류': scores.index,
        '평균점수': scores.values
    }).sort_values('평균점수', ascending=False)

def plot_midcategory_radar(df):
    """
    중분류 점수 레이더 차트 생성
    - 다차원 만족도 점수를 레이더 차트로 시각화
    - 직관적인 성과 비교 가능
    """
    scores = compute_midcategory_scores(df)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores.values,
        theta=scores.index,
        fill='toself',
        name='평균 점수',
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="중분류별 만족도 점수"
    )
    
    return fig

# =============================================================================
# 8. 텍스트 분석 및 키워드 추출 함수들
# =============================================================================
# 이 섹션은 텍스트 분석, 키워드 추출, 감정 분석 등의 텍스트 처리 함수들을 포함합니다.

def process_answers(responses):
    """
    응답 텍스트 전처리
    - 원시 응답 데이터를 분석 가능한 형태로 정제
    - 불필요한 문자 제거 및 표준화
    """
    processed = []
    for resp in responses:
        if isinstance(resp, str) and resp.strip():
            cleaned = resp.strip().replace('\n', ' ').replace('\r', ' ')
            if len(cleaned) > 3:
                processed.append(cleaned)
    return processed

def extract_keyword_and_audience(responses, batch_size=20):
    """
    키워드와 대상 추출
    - AI를 활용한 자동 키워드 및 이용자 그룹 추출
    - 배치 처리로 효율적인 분석 수행
    """
    if not responses:
        return pd.DataFrame(), pd.DataFrame()
    
    batches = [responses[i:i+batch_size] for i in range(0, len(responses), batch_size)]
    
    all_keywords = []
    all_audiences = []
    
    for batch in batches:
        try:
            keyword_prompt = f"다음 응답들에서 주요 키워드를 추출해주세요:\n{chr(10).join(batch)}"
            keyword_response = safe_chat_completion(
                messages=[{"role": "user", "content": keyword_prompt}],
                max_tokens=500
            )
            
            audience_prompt = f"다음 응답들에서 언급된 이용자 그룹을 추출해주세요:\n{chr(10).join(batch)}"
            audience_response = safe_chat_completion(
                messages=[{"role": "user", "content": audience_prompt}],
                max_tokens=500
            )
            
            all_keywords.append(keyword_response.choices[0].message.content)
            all_audiences.append(audience_response.choices[0].message.content)
            
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            continue
    
    return all_keywords, all_audiences

def make_theme_messages(batch: list[str]) -> list[dict]:
    """
    주제 분석을 위한 메시지 생성
    - AI 모델에 전달할 주제 분석 메시지 구성
    - 일관된 분석을 위한 프롬프트 표준화
    """
    return [
        {"role": "system", "content": "다음 응답들을 주제별로 분류해주세요."},
        {"role": "user", "content": "\n".join(batch)}
    ]

def make_sentiment_messages(batch: list[str], theme_df: pd.DataFrame) -> list[dict]:
    """
    감정 분석을 위한 메시지 생성
    - 주제별 분류 결과를 바탕으로 감정 분석 수행
    - 맥락을 고려한 정확한 감정 분석
    """
    return [
        {"role": "system", "content": "다음 응답들의 감정을 분석해주세요."},
        {"role": "user", "content": f"주제별 분류:\n{theme_df.to_string()}\n\n응답들:\n" + "\n".join(batch)}
    ]

# =============================================================================
# 9. 페이지별 분석 함수들
# =============================================================================
# 이 섹션은 각 분석 페이지의 주요 기능을 담당하는 함수들을 포함합니다.

def page_home(df):
    """
    홈페이지 - 응답자 정보 표시
    - 기본 통계 정보 및 데이터 미리보기
    - 전체적인 데이터 품질 확인
    """
    st.header("👤 응답자 정보")
    
    # 기본 통계 정보
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 응답자 수", len(df))
    
    with col2:
        st.metric("완전한 응답", len(df.dropna()))
    
    with col3:
        st.metric("응답률", f"{len(df.dropna()) / len(df) * 100:.1f}%")
    
    with col4:
        st.metric("컬럼 수", len(df.columns))
    
    # 데이터 미리보기
    st.subheader("데이터 미리보기")
    st.dataframe(df.head(), use_container_width=True)

def page_basic_vis(df):
    """
    기본 시각화 페이지
    - 중분류별 만족도 점수 시각화
    - 레이더 차트와 테이블을 통한 종합 분석
    """
    st.header("📈 만족도 기본 시각화")
    
    # 중분류별 평균 점수
    st.subheader("중분류별 평균 만족도")
    scores = compute_midcategory_scores(df)
    if not scores.empty:
        fig = plot_midcategory_radar(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # 테이블 표시
        avg_table = midcategory_avg_table(df)
        st.dataframe(avg_table, use_container_width=True)

def page_short_keyword(df):
    """
    단문 응답 키워드 분석 페이지
    - 짧은 응답에서 키워드 추출 및 분석
    - 이용자 의견의 주요 키워드 파악
    """
    st.subheader("단문 응답 키워드 분석")
    
    # 키워드 분석 로직 구현
    st.info("키워드 분석 기능을 구현합니다.")

def page_segment_analysis(df):
    """
    이용자 세그먼트 조합 분석 페이지
    - 다양한 이용자 그룹의 특성 분석
    - 세그먼트별 만족도 패턴 비교
    """
    st.header("🧩 이용자 세그먼트 조합 분석")
    
    # 세그먼트 분석 로직 구현
    st.info("세그먼트 분석 기능을 구현합니다.")

def page_district_analysis(df):
    """
    자치구 구성 문항 분석 페이지
    - 지역별 도서관 이용 현황 분석
    - 자치구별 서비스 품질 비교
    """
    st.header("🗺️ 자치구 구성 문항 분석")
    
    # 자치구 분석 로직 구현
    st.info("자치구 분석 기능을 구현합니다.")

def page_usage_pattern_analysis(df):
    """
    도서관 이용양태 분석 페이지
    - 이용 패턴 및 행동 분석
    - 이용 목적별 만족도 분석
    """
    st.header("📊 도서관 이용양태 분석")
    
    # 이용양태 분석 로직 구현
    st.info("이용양태 분석 기능을 구현합니다.")

def page_image_analysis(df):
    """
    도서관 이미지 분석 페이지
    - 이용자 인식 및 만족도 분석
    - 브랜드 이미지 평가
    """
    st.header("🖼️ 도서관 이미지 분석")
    
    # 이미지 분석 로직 구현
    st.info("이미지 분석 기능을 구현합니다.")

def page_strength_weakness_analysis(df):
    """
    도서관 강약점 분석 페이지
    - SWOT 분석 및 개선점 도출
    - 경쟁력 분석
    """
    st.header("🏋️ 도서관 강약점 분석")
    
    # 강약점 분석 로직 구현
    st.info("강약점 분석 기능을 구현합니다.")

def page_common_analysis(df):
    """
    공통 심화 분석 페이지
    - 전체 및 영역별 상세 분석
    - 종합적인 성과 평가
    """
    st.header("🔍 공통 심화 분석")
    
    # 공통 분석 로직 구현
    st.info("공통 심화 분석 기능을 구현합니다.")

def page_strategy_insights(df):
    """
    전략 인사이트 페이지
    - AI 기반 전략적 제안
    - 개선 방안 및 우선순위 제시
    """
    st.header("🧠 전략 인사이트")
    
    # 전략 인사이트 로직 구현
    st.info("전략 인사이트 기능을 구현합니다.")

# =============================================================================
# 10. 메인 애플리케이션 실행 부분
# =============================================================================
# 이 섹션은 Streamlit 앱의 메인 실행 로직을 포함합니다.
# 페이지 설정, 네비게이션, 파일 업로드, 메인 컨텐츠 영역 등을 담당합니다.

# Streamlit 페이지 설정
st.set_page_config(
    page_title="LIBanalysiscusor - 공공도서관 설문 시각화 대시보드",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 고정 헤더
st.markdown(
    """
    <div style="
        position: sticky;
        top: 0;
        background-color: white;
        z-index: 1000;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
    ">
        <h1 style="text-align: center; color: #2c3e50; margin: 0;">
            📚 LIBanalysiscusor - 공공도서관 설문 시각화 대시보드
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# 사이드바 네비게이션
st.sidebar.title("📊 분석 메뉴")
analysis_mode = st.sidebar.radio(
    "분석 유형을 선택하세요:",
    [
        "기본 분석",
        "심화 분석", 
        "전략 인사이트(기본)"
    ]
)

# 파일 업로드
uploaded = st.file_uploader("📂 엑셀(.xlsx) 파일 업로드", type=["xlsx"])
if not uploaded:
    st.info("데이터 파일을 업로드해 주세요.")
    st.stop()

try:
    df = pd.read_excel(uploaded)
    st.success("✅ 업로드 완료")
except Exception as e:
    st.error(f"파일 읽기 실패: {e}")
    st.stop()

# 메인 컨텐츠 영역
if analysis_mode == "기본 분석":
    tabs = st.tabs([
        "👤 응답자 정보",
        "📈 만족도 기본 시각화",
        "🗺️ 자치구 구성 문항",
        "📊 도서관 이용양태 분석",
        "🖼️ 도서관 이미지 분석",
        "🏋️ 도서관 강약점 분석",
    ])

    with tabs[0]:
        page_home(df)

    with tabs[1]:
        page_basic_vis(df)

    with tabs[2]:
        page_district_analysis(df)

    with tabs[3]:
        page_usage_pattern_analysis(df)

    with tabs[4]:
        page_image_analysis(df)

    with tabs[5]:
        page_strength_weakness_analysis(df)

elif analysis_mode == "심화 분석":
    page_common_analysis(df)

elif analysis_mode == "전략 인사이트(기본)":
    page_strategy_insights(df)

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #7f8c8d; padding: 2rem 0;">
        <p>📚 LIBanalysiscusor - 공공도서관 설문 시각화 대시보드</p>
        <p>데이터 기반 도서관 서비스 개선을 위한 분석 도구</p>
    </div>
    """,
    unsafe_allow_html=True
)
