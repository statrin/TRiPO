import streamlit as st
from streamlit_chat import message

# 벡터 DB 및 llm 라이브러리
import pinecone
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import openai
import streamlit as st

# 임베딩 모델 위한 라이브러리
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd

# 출력 시간 확인 위한 라이브러리
import time

# 출력 형태 변환 위한 라이브러리
import re
import json

import ast  # 문자열을 안전하게 리스트로 변환하기 위한 모듈

# api 로드에 필요
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 1. OpenAI API 키 설정 및 임베딩 모델 초기화   

# 배포용 api 설정
#openai_api_key = st.secrets["OPENAI_API_KEY"]
#pinecone_api_key = st.secrets["PINECONE_API_KEY"]

#if not openai_api_key or not pinecone_api_key:
#    raise ValueError("API 키가 Streamlit Secrets에 설정되지 않았습니다.")

# API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
# OpenAI 라이브러리에 API 키 설정
import openai
openai.api_key = openai_api_key

# embedding 모델 로드
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")


# 2. 파인콘 초기화     
from pinecone import Pinecone, ServerlessSpec

pinecone = Pinecone(api_key=pinecone_api_key)
index = pinecone.Index("trip-index")

# 3. 검색 함수 정의
# 선호 숙소 형태 및 동행인 기반
def search_places(city, companions, lodging_style):
    query = f"Best accommdations in {city} for {companions} with focus on {lodging_style}."
    query_embedding = model.encode(query).tolist()
    namespace = f"{city}_lodging"
    results_style = index.query(vector=query_embedding, top_k=20, namespace=namespace, include_metadata=True)
    return results_style


# 4. 숙소 리스트 생성을 위한 프롬프트 템플릿 - 영어 버전
persona = """
You are an AI expert in accommodation recommendations, acting as a travel agent to recommend a personalized accommodation list to your customers.

Do not hallucinate information. Only provide accurate and reliable details based on the given data or context.
"""

prompt_template = """
Create a list of the **top 5** accommodations for your customer's trip based on their <personal information> and the <requirements>.

<personal infomation>
1. Travel City: {city}
2. Companions: {companions}
3. Accommodation Preferences: {lodging_style}
</personal infomation>

<requirements>
1. Offer the proper option based on their accommodation preferences and travel companions among hotels, resorts, inns, hostels, B&B, and so on.
2. Ensure the accommodations are in safe areas of {city}.
3. Please make the list consist of the \n{recommendations}.
4. **Only for an approximate nightly price, please retrieve this information from external sources such as Agoda, Booking.com, and so on. The currency unit is Korean WON.**
</requirements>
"""

output = """
Ensure the output is valid JSON and strictly adheres to the structure and letter case below:
[
    {{"Name": "Paris Perfect", "Price": "1,200,000 ₩", "Location": "25 Pl. Dauphine, 75001 Paris, France"}},
    {{"Name": "Beau M Hostel", "Price": "200,000 ₩", "Location": "108 Rue Damrémont, 75018 Paris, France"}}
]

"""


# 5. 프롬프트와 LLMChain 설정
llm = ChatOpenAI(
    temperature=0.1,
    model_name="gpt-4o",  # 4-turbo보다 빠르고, 한국어도 더 잘함
    openai_api_key=openai_api_key,
)


# 숙소 추천 생성 함수
def generate_accommodation_recommendations(city, companions, lodging_style, recommendations):

    # 페르소나 주입
    filled_persona = persona  # .format() 제거
    print(f"Filled Persona:\n{filled_persona}\n")

    # 템플릿에 사용자 정보 삽입
    #a1 = time.time()
    try:
        formatted_prompt = prompt_template.format(
            city=city,
            companions=companions,
            lodging_style=lodging_style,
            recommendations=recommendations
        ) + "\n" + output
        print(f"Formatted Prompt:\n{formatted_prompt}\n")
    except KeyError as e:
        print(f"Error in formatting prompt: {e}")
        return None
    #a2 = time.time()
    #print(f"Prompt Time: {a2 - a1:.2f} seconds")

    # LangChain 프롬프트 구성
    #b2 = time.time()
    try:
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(filled_persona),
                HumanMessagePromptTemplate.from_template(formatted_prompt)
            ]
        )
    except Exception as e:
        print(f"Error in creating ChatPromptTemplate: {e}")
        return None
    #b3 = time.time()
    #print(f"Prompt Template Time: {b3 - b2:.2f} seconds")

    # LLMChain 설정 및 실행
    #c1 = time.time()
    try:
        conversation = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True
        )
        result = conversation.run({})
    except Exception as e:
        print(f"Error during LLMChain execution: {e}")
        return None
    #c2 = time.time()
    #print(f"Conversation Time: {c2 - c1:.2f} seconds")
    return result


# 6. 메인 함수: 사용자 입력 및 숙소 추천 실행

# 최종 추천 함수 생성
def final_recommendations(city, companions, lodging_style):

    # 출력 시간 확인
    #start_time = time.time()  # 시작 시간 기록
    #a1 = time.time()
    # 사용자 입력 예시
    accommodation_details = {
        "city": city,
        "companions": companions,
        "lodging_style": lodging_style
    }

    # 파인콘에서 숙소 검색 실행
    search_results = search_places(
        city=accommodation_details["city"],
        companions=accommodation_details["companions"],
        lodging_style=accommodation_details["lodging_style"]
    )
    #a2 = time.time()
    #print(f"Search Time: {a2 - a1:.2f} seconds")

    #b1 = time.time()

    # 파인콘에서 가져온 추천 숙소 리스트 구성 (최대 20개)
    # 각 추천 숙소 정보를 딕셔너리로 구성
    
    temps = [
        {
            "Name": match.metadata['1_이름'],
            "Rating": match.metadata['3_평점'],
            "Location": match.metadata['2_주소'],
            "Type": match.metadata['8_유형'],
            "Image": (
                ast.literal_eval(match.metadata['9_이미지'])[0] 
                if match.metadata['9_이미지'].startswith("[") and match.metadata['9_이미지'].endswith("]") 
                else match.metadata['9_이미지'].split(', ')[0]
            )  # 첫 번째 이미지 URL 처리
        }
        for match in search_results.matches
    ]


    # Name, Location, Rating만 리스트로 구성
    recommendations = [[temp['Name'], temp['Location'], temp['Rating']] for temp in temps]

    # temps 데이터를 데이터프레임으로 변환
    df_temps = pd.DataFrame(temps)

    # 숙소 추천 생성 호출
    accommodation_recommendations = generate_accommodation_recommendations(
        city=accommodation_details["city"],
        companions=accommodation_details["companions"],
        lodging_style=accommodation_details["lodging_style"],
        recommendations=recommendations
    )

    # 불필요한 설명 제거 및 JSON 변환
    start_index = accommodation_recommendations.find("[")
    end_index = accommodation_recommendations.rfind("]")
    json_text = accommodation_recommendations[start_index:end_index + 1].strip()

    # JSON 유효성 확인
    json_text = re.sub(r"\n\s*", "", json_text)
    accommodations = json.loads(json_text)

    # 데이터프레임으로 변환
    df_json = pd.DataFrame(accommodations)

    # 이름과 Location을 기준으로 병합
    df_result = df_json.merge(
        df_temps,
        on=["Name", "Location"],  # 병합 기준: Name과 Location
        how="left"   # df_json 기준 inner merge
    )

    #b2 = time.time()
    #print(f"Recommendation Time: {b2 - b1:.2f} seconds")


    # 출력 시간 확인
    #end_time = time.time()  # 종료 시간 기록

    # 실행 시간 출력
    #execution_time = end_time - start_time
    #print(f"\nExecution Time: {execution_time:.2f} seconds")

    # 결과 출력
    return df_result
