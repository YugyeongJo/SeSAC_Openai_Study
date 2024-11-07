import random
import os
import nltk
from pprint import pprint 
from debugger import debug_shell
from openai import OpenAI
from logging_util import log_chatgpt_call
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
from statistics import mean

# .env 파일의 내용을 로드합니다.
load_dotenv()

# 환경 변수 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = openai_api_key)

@log_chatgpt_call
def ask_chatgpt(
    query: str, 
    pre_quest: str,
    model:str = 'gpt-4o-mini-2024-07-18',
):
    completion = client.chat.completions.create(
        model=f"{model}",
        messages=[
            {"role": "system", "content": "You are an incredibly powerful translator that translates English to French very well. Use the provided example translations to help improve accuracy for similar sentences."},
            {
                "role": "user",
                "content": f"{pre_quest}\n\nTranslate the following sentence based on the provided context:\n{query}"
            }
        ]
    )

    return completion 

def parse_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = []

        for line in f.readlines()[::100]:
            line = line.strip()
            lst = line.split('\t')
            
            if len(lst) == 3:
                source, target, etc = lst
            elif len(lst) == 2:
                source, target = lst 


            source = source.split()
            target = target.split()
            data.append((source, target))
            
        return data
    
# BLEU 점수를 계산하는 함수
def calculate_bleu(reference, candidate):
    # 각 문장을 토큰화
    reference_tokens = [reference.split()]  # 참조 번역 (리스트 형태로 전달)
    candidate_tokens = candidate.split()    # ChatGPT 번역 결과

    # BLEU 점수 계산
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens)
    return bleu_score

if __name__ == '__main__':
    
    data_path = '../data/eng-fra.txt'
    
    data = parse_file(data_path)
    
    random.shuffle(data)
    split_index = int(len(data) * 0.2)
    pre_quest_data = data[:split_index]
    translate_data = data[split_index:]
    
    # `pre_quest` 데이터를 하나의 텍스트로 합칩니다.
    pre_quest = "\n".join(["\t".join([" ".join(src), " ".join(tgt)]) for src, tgt in pre_quest_data])
    
    avg_bleu_score = []
    
    for eng, fra in translate_data:
        # `eng` 리스트를 문자열로 변환하여 번역 요청
        eng_text = ' '.join(eng)
        completion = ask_chatgpt(eng_text, pre_quest=pre_quest)
        translated_text = completion.choices[0].message.content
        
        # BLEU 점수 계산
        reference_translation = ' '.join(fra)
        bleu_score = calculate_bleu(reference_translation, translated_text)
        avg_bleu_score.append(bleu_score)
        
    print("평균 BLEU 점수:", round(mean(avg_bleu_score), 3))
    
    # pprint(completion)
    # print(completion.choices[0].message.content)