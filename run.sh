#!/bin/bash

echo "스트림릿 앱을 시작합니다..."
echo ""

echo "필요한 패키지를 설치합니다..."
pip install -r requirements.txt

echo ""
echo "스트림릿 앱을 실행합니다..."
streamlit run main.py
