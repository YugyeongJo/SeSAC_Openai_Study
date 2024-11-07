# -*- coding: utf-8 -*

import json 
from openai import OpenAI 

from debugger import debug_shell 
from logging_util import log_chatgpt_call 

openai_api_key = ''
client = OpenAI(api_key = openai_api_key)

@log_chatgpt_call
def extract_email(
    sentence: str, 
    model: str = 'gpt-4o-mini-2024-07-18', 
) -> str:
    response = client.chat.completions.create(
        model = model, 
        messages = [
            {
                'role': 'system', 
                'content': 'Read the sentence, and extract the email and domain contained in the sentence. The email is ****@***.** format, and the domain is the text in the email behind the @',
            }, 
            {
                'role': 'user', 
                'content': sentence, 
            }
        ], 
        response_format = {
            'type' : 'json_schema', 
            'json_schema': {
                'name': 'email_addrress', 
                'schema': {
                    'type': 'object', 
                    'properties': {
                        'email': {
                            'description': 'The email address in the sentence', 
                            'type': 'string', 
                        }, 
                        'domain': {
                            'description': 'Domain name of the email address in the sentence', 
                            'type': 'string', 
                        }, 
                    }
                }
            }
        }
    )

    return response 

@log_chatgpt_call
def extract_meeting_schedule(
    sentence: str, 
    model: str = 'gpt-4o-mini-2024-07-18', 
) -> str:
    response = client.chat.completions.create(
        model = model, 
        messages = [
            {
                'role': 'system', 
                'content': '한국어로 된 회의록을 읽고, 회의 이후 잡힌 미팅들에 대한 정보를 잘 정리해서 json 형식으로 바꿔줌.',
            }, 
            {
                'role': 'user', 
                'content': sentence, 
            }
        ], 
        response_format = {
            'type' : 'json_schema', 
            'json_schema': {
                'name': 'meeting_schedule', 
                'schema': {
                    'type': 'object', 
                    'properties': {
                        '회의 일정': {
                            'description': '회의록 안에서 알 수 있는 후속 회의 일정. [YYYY-MM-DD HH:MM]', 
                            'type': 'string', 
                        }, 
                        '회의 참가자': {
                            'description': '{{사람 이름}}({{사람 직급}}) / ... 형식으로 정리된 회의에 참여할 사람들.', 
                            'type': 'string', 
                        }, 
                        '회의 주제': {
                            'description': '회의의 주요 안건', 
                            'type': 'string', 
                        },
                    }
                }
            }
        }
    )

    return response 


if __name__ == '__main__':
    # completion = extract_email('Hello. My name is seungwoo schin, and my email is principia12@ncsoft.com.')
    # debug_shell()
    # print(completion.choices[0].message.content)

    with open('meeting.txt', 'r', encoding = 'utf-8') as f:
        completion = extract_meeting_schedule(f.read())
        print(completion.choices[0].message.content)
        debug_shell()