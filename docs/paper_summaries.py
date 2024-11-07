# -*- coding: utf-8 -*
import os
import json 
from openai import OpenAI 
from dotenv import load_dotenv
from debugger import debug_shell 
from logging_util import log_chatgpt_call 

# .env 파일의 내용을 로드합니다.
load_dotenv()

# 환경 변수 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = openai_api_key)

@log_chatgpt_call
def extract_paper_info(
    text: str, 
    model: str = 'gpt-4o-mini-2024-07-18', 
) -> str:
    response = client.chat.completions.create(
        model = model, 
        messages = [
            {
                'role': 'system', 
                'content': 'You are a data extraction assistant that extracts key information from research papers. Your task is to identify and structure the paper title, authors, main content, and key points, among other details.',
            }, 
            {
                'role': 'user', 
                'content': text, 
            }
        ], 
        response_format={
            'type': 'json_schema',
            'json_schema': {
                'name': 'paper_info',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'title': {
                            'description': 'The title of the paper',
                            'type': 'string',
                        },
                        'authors': {
                            'description': 'List of authors of the paper',
                            'type': 'array',
                            'items': {'type': 'string'}
                        },
                        'main_content': {
                            'description': 'The main content or abstract of the paper',
                            'type': 'string',
                        },
                        'key_points': {
                            'description': 'Important points or findings in the paper',
                            'type': 'array',
                            'items': {'type': 'string'}
                        },
                        'used_models': {
                            'description': 'Models used in the research',
                            'type': 'array',
                            'items': {'type': 'string'}
                        },
                        'topic': {
                            'description': 'Research topic or area',
                            'type': 'string'
                        },
                        'classification': {
                            'description': 'Classification of the paper',
                            'type': 'string'
                        },
                        'publication_year': {
                            'description': 'Year of publication',
                            'type': 'integer'
                        },
                        'source': {
                            'description': 'Journal or conference where the paper was published',
                            'type': 'string'
                        },
                        'doi': {
                            'description': 'Digital Object Identifier of the paper',
                            'type': 'string'
                        },
                        'keywords': {
                            'description': 'Keywords related to the paper',
                            'type': 'array',
                            'items': {'type': 'string'}
                        },
                        'experimental_results': {
                            'description': 'Summary of experimental results',
                            'type': 'string'
                        },
                        'future_research_directions': {
                            'description': 'Suggested future research directions',
                            'type': 'string'
                        },
                        'references': {
                            'description': 'List of references cited in the paper',
                            'type': 'array',
                            'items': {'type': 'string'}
                        }
                    },
                    'required': ['title', 'authors', 'main_content', 'key_points', 'used_models', 
                                 'topic', 'classification', 'publication_year', 'source', 'doi', 
                                 'keywords', 'experimental_results', 'future_research_directions', 
                                 'references']
                }
            }
        }
    )
    
    return response

if __name__ == '__main__':
    data_path = '../data/paper_summaries/Automatic_Soccer_Video_Summarization_Using_Deep_Learning.txt'
    
    # 파일에서 텍스트를 읽어옵니다.
    with open(data_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # 논문 정보 추출
    completion = extract_paper_info(text)
    
    # 논문 제목을 추출
    paper_info = json.loads(completion.choices[0].message.content)
    full_title = paper_info.get('title', 'Unknown_Title')
    
    # 파일 이름에 사용할 제목 생성
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        if char in full_title:
            title = full_title.split(char)[0]  # 구분 기호 앞까지만 제목을 사용
            break
    else:
        title = full_title  # 구분 기호가 없으면 전체 제목 사용

    title = title.replace(" ", "_")  # 공백을 '_'로 대체하여 파일 이름으로 사용
    
    # 결과를 JSON 파일로 저장할 경로 지정
    output_path = f'../paper_summaries_results/{title}_summary.json'
    
    # 폴더가 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 결과를 JSON 파일로 저장
    with open(output_path, 'w+', encoding='utf-8') as json_file:
        json.dump(paper_info, json_file, ensure_ascii=False, indent=4)

    print("논문 정보가 JSON 파일로 저장되었습니다:", output_path)
