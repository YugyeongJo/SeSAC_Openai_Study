import base64
import os
import json

from dotenv import load_dotenv
from openai import OpenAI 

from debugger import debug_shell 
from logging_util import log_chatgpt_call 

# .env 파일의 내용을 로드합니다.
load_dotenv()

# 환경 변수 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = openai_api_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@log_chatgpt_call
def analyze_image(image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Please describe what might be in the image as if it were a traffic light photo (e.g., red, yellow, or green light)",
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                },
                },
            ],
            }
        ],
        response_format = {
            'type' : 'json_schema', 
            'json_schema': {
                'name': 'brinker_classification', 
                'schema': {
                    'type': 'object', 
                    'properties': { 
                        'color_classification': {
                            'description': '신호등의 색(빨간색, 초록색, 노란색, 기타 중 하나)', 
                            'type': 'string', 
                        }, 
                    }
                }
            }
        }
    )
    
    return response

if __name__ == '__main__':
    image_dir = '../data/images/blinker'
    results = []
    
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        
        response = analyze_image(image_path)
        
        # response에서 실제 데이터를 추출
        response_data = response.choices[0].message.content
        # image_name을 response_data에 추가
        result = {
            'image_name': image_file,
            'response': response_data
        }
        
        # 결과를 리스트에 추가
        results.append(result)
    
    # 결과를 JSON 파일로 저장
    output_path = '../analyze_image_results/blinker_classification.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w+', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"결과가 {output_path}에 저장되었습니다.")