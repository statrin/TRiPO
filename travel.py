import pandas as pd
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import numpy as np
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
import getpass
import os
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
output_parser = StrOutputParser()

import pandas as pd
import re
import requests
import openai

# api 로드에 필요
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 1. OpenAI API 키 설정 및 임베딩 모델 초기화   

# 배포용 api 설정
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

if not openai_api_key or not pinecone_api_key:
    raise ValueError("API 키가 Streamlit Secrets에 설정되지 않았습니다.")

# API 키 가져오기
#openai_api_key = os.getenv("OPENAI_API_KEY")
#pinecone_api_key = os.getenv("PINECONE_API_KEY")
# OpenAI 라이브러리에 API 키 설정
import openai
openai.api_key = openai_api_key

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2") # embedding 모델 로드

# 2. 파인콘 초기화
pinecone = Pinecone(api_key=pinecone_api_key)
index = pinecone.Index("trip-index")

# 3. 검색 함수 정의

# 검색 함수 정의

# 여행 스타일 및 동행인 기반 관광지 검색
def search_places_style(city, companions, travel_style):
    query = f"Best places in {city} for {companions} with focus on {travel_style}."
    query_embedding = model.encode(query).tolist()
    namespace = f"{city}_tour"
    results_style= index.query(vector=query_embedding, top_k=60, namespace=namespace, include_metadata=True)
    return results_style

# 여행 스타일 및 동행인 기반 맛집 검색
def search_restaurants_style(city, companions, travel_style):
    query = f"Best restaurants in {city} for {companions} with focus on {travel_style}."
    query_embedding = model.encode(query).tolist()
    namespace = f"{city}_tour"
    results_restaurants= index.query(vector=query_embedding, top_k=60, namespace=namespace, include_metadata=True)
    return results_restaurants

# 기본 유명 관광지
def search_places(city):
    query = f"The most famous tour places in {city}"
    query_embedding = model.encode(query).tolist()
    namespace = f"{city}_tour"
    results_best= index.query(vector=query_embedding, top_k=60, namespace=namespace, include_metadata=True)
    return results_best

# 검색 결과 합치기 및 데이터프레임 변환
def merge_and_deduplicate_places_to_df(results_best, results_style, results_restaurants):
    # 세 결과 리스트를 결합
    combined_results = results_best['matches'] + results_style['matches'] + results_restaurants['matches']

    # 각 항목에서 메타데이터 추출하여 데이터프레임 생성
    places_data = []
    for item in combined_results:
        places_data.append({
            "name": item['metadata'].get('1_이름', 'N/A'),
            "address": item['metadata'].get('2_주소', 'N/A'),
            "rating": item['metadata'].get('3_평점', 0),
            "latitude": item['metadata'].get('4_위도', 0),
            "longitude": item['metadata'].get('5_경도', 0),
            "review": item['metadata'].get('6_리뷰', 'N/A'),
            "opening_hours": item['metadata'].get('7_영업시간', 'N/A'),
            "type": item['metadata'].get('8_유형', 'N/A'),
            "image_url": item['metadata'].get('9_이미지', 'N/A')  # 원본 이미지 데이터 그대로 추가
        })

    # 데이터프레임 생성
    df = pd.DataFrame(places_data)

    # 중복 제거 (장소 이름과 주소를 기준으로)
    df = df.drop_duplicates(subset=["name", "address"]).reset_index(drop=True)

    # 이미지 URL 처리: 대괄호 제거 및 첫 번째 URL 추출
    def process_image_url(image_data):
        if isinstance(image_data, str) and image_data.startswith("[") and image_data.endswith("]"):
            return image_data.strip("[]").replace("'", "").split(",")[0].strip()
        return image_data  # 예외 처리

    df['image_url'] = df['image_url'].apply(process_image_url)

    return df




## 여행일정 생성 프롬프트 (영어로)

persona = """
You are a travel itinerary AI expert with a specialization in creating highly optimized and user-friendly travel plans.

Objective:
Your primary goal is to generate travel itineraries tailored to user preferences, including companion types, travel styles, and selected travel dates.
The itineraries must include tourist attractions, restaurants, and cultural experiences, while optimizing for time and location proximity.

Guidelines and Limitations:
Only include verified factual data in itineraries (e.g., information from reputable travel guides or databases).
Do not include personal opinions, speculative information, or promotional content.
Ensure itineraries comply with user-provided constraints such as travel style, preferences, and group type."""



prompt_template = """You are a travel itinerary AI expert. Create a travel itinerary for {city} for a duration of {trip_duration}.
The travel information is as follows:

- Trip duration: {trip_duration}
- Companions: {companions}
- Travel style: {travel_style}
- Preferred itinerary style: {itinerary_style}
- user additional request : {user_request}

**Please create a travel itinerary using the following list of places based on the conditions**:
{places_list}

**Constraints**:
1. Ensure the itinerary includes all {trip_duration} days, with each day divided into morning, afternoon, and evening.
2. The same place or restaurant **should not appear more than once** in the itinerary. A place included on one day should not appear on any other day.
  2-1. No location or restaurant should be repeated across the entire itinerary to ensure a unique and diverse travel experience.
3. Include the opening hours and address of each place.
4. Add a brief one-sentence introduction for each place.
5. Based on the user additional request information, please create a travel itinerary reflecting the user's additional requests. If there are no additional requests, please proceed as planned.
6. Optimize routes using places_list with latitude and longitude data:
  6-1. Within the same time slot (Morning/Afternoon/Evening), ensure all places are within 1km or a 15-minute walking distance. Exceptions (e.g., must-visit landmarks) allow up to 2km.
  6-2. Between time slots, ensure travel time (e.g., Morning → Afternoon → Evening) is within 30 minutes via public transport, with a maximum distance of 20km. Prioritize staying within the same area to minimize travel time and complexity.
7. Consider **the operating hours** of each place when organizing the itinerary.
8. Adjust the itinerary based on the selected itinerary style.
   8-1. If the '빼곡한 일정' style is selected, **include 2 tourist attractions and 1 restaurant** in the morning, afternoon, and evening, totaling **9 activities per day**.
   8-2. If the '널널한 일정' style is selected, **include 1 tourist attraction and 1 restaurant** in the morning, afternoon, and evening, totaling **6 activities per day**.
9. Please provide the result in **JSON format**, using the example structure below.
10. **The result should be provided in Korean.**

**Output Structure**:
- Ensure the output contains the date, time period, place name, description, and operating hours for each entry.

**Example Output Structure**:

```json
{{
    "여행 일정": [
        {{
            "날짜": "Day 1",
            "시간대": "오전",
            "장소명": "Galerie Vivienne",
            "장소 소개": "19세기 파리의 매력을 간직한 아름다운 쇼핑 아케이드입니다.",
            "운영시간": "8:30 AM – 8:00 PM"}},
        {{
            "날짜": "Day 1",
            "시간대": "오전",
            "장소명": "Domaine National du Palais-Royal",
            "장소 소개": "역사적인 궁전과 정원이 있는 관광 명소입니다.",
            "운영시간": "8:30 AM – 10:30 PM"
        }},
        {{
            "날짜": "Day 1",
            "시간대": "오전",
            "장소명": "Boulangerie LIBERTÉ",
            "장소 소개": "신선한 빵과 페이스트리를 제공하는 인기 있는 빵집입니다.",
            "운영시간": "7:30 AM – 8:00 PM"
        }},
        {{
            "날짜": "Day 1",
            "시간대": "오후",
            "장소명": "Galeries Lafayette Champs-Élysées",
            "장소 소개": "파리의 대표적인 쇼핑 명소로 다양한 브랜드를 만날 수 있습니다.",
            "운영시간": "10:00 AM – 9:00 PM"
        }}
    ]
}}



"""

from langchain.memory import ConversationBufferMemory
# 객체 생성
llm = ChatOpenAI(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o",  # 모델명
)

# Memory for storing conversation history
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)



from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 여행 일정 생성 함수
def generate_itinerary_recommendations(city, trip_duration, companions, travel_style, itinerary_style, user_request, places_list):

    # 페르소나 주입
    filled_persona = persona.format()

    # 템플릿에 사용자 정보 삽입
    formatted_prompt = prompt_template.format(
        city=city,
        trip_duration=trip_duration,
        companions=companions,
        travel_style=travel_style,  # travel_style 리스트를 문자열로 변환
        itinerary_style=itinerary_style,
        user_request = user_request,
        places_list=places_list
    )

    # 프롬프트 구성
    prompt = ChatPromptTemplate(
        template=formatted_prompt,
        messages=[
            SystemMessagePromptTemplate.from_template(filled_persona),  # 페르소나 주입
            HumanMessagePromptTemplate.from_template("{input}")  # 질문 입력
        ]
    )
    # Memory for storing conversation history
    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # RunnableSequence로 구성
    conversation = prompt | llm

    # 여러 입력 변수를 하나의 딕셔너리로 전달 (invoke로 변경)
    result = conversation.invoke({
        "input": formatted_prompt
    })
    return result



import pandas as pd
import json

def process_and_merge_itinerary(itinerary, final_results):
    """
    프롬프트 결과물과 검색된 장소 데이터를 결합하여 최종 DataFrame을 생성하는 함수.
    
    Parameters:
        itinerary (str): 프롬프트 결과물(JSON 문자열 포함)
        places_df (pd.DataFrame): merge_and_deduplicate_places_to_df 결과 DataFrame

    Returns:
        pd.DataFrame: itinerary와 places_df를 장소명을 기준으로 결합한 DataFrame
    """
    # Step 1: itinerary에서 JSON 부분만 추출
    start_index = itinerary.find("{")  # JSON 시작 위치
    end_index = itinerary.rfind("}")   # JSON 끝 위치
    json_text = itinerary[start_index:end_index+1]
    data = json.loads(json_text)

    # Step 2: "여행 일정" 키 아래의 리스트를 DataFrame으로 변환
    itinerary_df = pd.DataFrame(data["여행 일정"])

    # Step 3: places_df에서 필요한 열 선택 및 컬럼 이름 통일
    final_results = final_results.rename(columns={
        'name': '장소명',  # 이름 열 통일
        'address': '주소',     # 주소 열 이름 통일
        'image_url': '이미지'  # 이미지 열 이름 통일
    })

    # Step 5: inner join 수행 (장소명을 기준으로)
    merged_df = pd.merge(itinerary_df, final_results[['장소명', '주소', '이미지']], on='장소명', how='inner')

    # Step 6: 최종 DataFrame 반환
    return merged_df


# 6. 메인 함수: 사용자 입력 및 여행일정 생성
def final_recommendations(city, trip_duration, companions, travel_style, itinerary_style, user_request = None):

    # 사용자 입력 예시
    itinerary_details = {
        "city": city,
        "trip_duration": trip_duration,
        #"travel_dates": "2024-11-15 ~ 2024-11-18",
        "companions": companions,
        "travel_style": travel_style,
        "itinerary_style" : itinerary_style,
        "user_request" : user_request
    }

    # 파인콘에서 장소 검색 실행
    results_style = search_places_style(
        city=itinerary_details["city"],
        companions=itinerary_details["companions"],
        travel_style=itinerary_details["travel_style"]
    )

    results_restaurants = search_restaurants_style(
        city=itinerary_details["city"],
        companions=itinerary_details["companions"],
        travel_style=itinerary_details["travel_style"]
    )

    results_best = search_places(
        city=itinerary_details["city"]
    )

    final_results = merge_and_deduplicate_places_to_df(results_style, results_best,results_restaurants)


    places_list = "\n".join([
        f"- {row['name']} (카테고리: {row['type']}, 위도: {row['latitude']}, 경도: {row['longitude']}, 운영시간: {row.get('opening_hours', 'N/A')})"
        for _, row in final_results.iterrows()
    ])
    # 여행일정 생성 호출
    itinerary = generate_itinerary_recommendations(
        city=itinerary_details["city"],
        trip_duration=itinerary_details["trip_duration"],
        #travel_dates=accommodation_details["travel_dates"],
        companions=itinerary_details["companions"],
        travel_style=itinerary_details["travel_style"],
        itinerary_style=itinerary_details["itinerary_style"],
        user_request=itinerary_details["user_request"],
        places_list=places_list,
        #itinerary=itinerary
    )
    df_itinerary = process_and_merge_itinerary(itinerary.content, final_results)

    # 결과 출력
    return df_itinerary
