import torch

def preprocess_text(text, tokenizer, max_len=32):
    # HuggingFace 토크나이저 방식 사용: 자동 패딩, 텐서 반환
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return inputs  # 딕셔너리 형태 {'input_ids': tensor, 'attention_mask': tensor}

def predict_topic(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred


from konlpy.tag import Komoran
import string
from tqdm import tqdm

# Komoran 객체 생성 (형태소 분석기)
komoran = Komoran()  # Komoran을 클래스처럼 사용하여 인스턴스를 생성

# 불용어 리스트 로드 (예시 경로)
STOPWORD_FILE = r'C:\Users\KDT-37\Desktop\KDT_7\11_NLP\D250409\koreanStopwords_unique.txt'

with open(STOPWORD_FILE, mode='r', encoding='utf-8') as f:
    stop_words = f.readlines()

# 불용어 리스트에서 줄바꿈 제거
stop_words = [word.replace('\n', '') for word in stop_words]

# 한국어 구두점 리스트
korean_punctuation = ['∼','%', '.', ',', '!', '?', '(', ')', '[', ']', '{', '}', '-', '...', '"', ':', ';', '·', '…']


# 한국어 자음 리스트
korean_consonants = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 
                     'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


# 토큰화 진행 기능 함수
def tokenize(sentences):
    tokenized_sentences = []

    # tqdm을 사용하여 진행 상태를 출력
    for sent in tqdm(sentences, total=len(sentences)):
        sent = str(sent)

        # 형태소 분석 (morphs는 형태소 단위로 나누는 메서드)
        morphs = komoran.morphs(sent)

        # 구두점, 불용어, 숫자, 조사 제거
        cleaned_tokens = []
        for token in morphs:
            # 한국어 구두점, 불용어, 숫자, 조사 제거
            if token not in korean_punctuation and token not in stop_words and not token.isdigit() and token not in korean_consonants:
                cleaned_tokens.append(token)

        tokenized_sentences.append(cleaned_tokens)

    return tokenized_sentences

from collections import Counter

# 정수 인코딩을 위한 함수
def build_vocab(tokenized_sentences):
    # 모든 단어를 모은 리스트
    all_tokens = [token for sentence in tokenized_sentences for token in sentence]
    
    # 단어 빈도수 계산
    word_counts = Counter(all_tokens)
    
    # 빈도수 높은 순서대로 단어 인덱스를 할당
    word2index = {word: index + 1 for index, (word, _) in enumerate(word_counts.most_common())}
    word2index["<PAD>"] = 0  # 패딩을 위한 <PAD> 토큰 추가
    word2index['<UNK>'] = 1  # 패딩을 위한 <UNK> 토큰 추가
    return word2index

# 정수 인코딩 함수
def texts_to_sequences(tokenized_X_data, word_to_index):
  encoded_X_data = []
  for sent in tokenized_X_data:
    index_sequences = []
    for word in sent:
      try:
          index_sequences.append(word_to_index[word])
      except KeyError:
          index_sequences.append(word_to_index['<UNK>'])
    encoded_X_data.append(index_sequences)
  return encoded_X_data

index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key


import numpy as np

# 패딩 함수
def pad_sequences(sentences, max_len):
  features = np.zeros((len(sentences), max_len), dtype=int)
  for index, sentence in enumerate(sentences):
    if len(sentence) != 0:
      features[index, :len(sentence)] = np.array(sentence)[:max_len]
  return features

