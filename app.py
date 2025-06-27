import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from konlpy.tag import Komoran
from tqdm import tqdm
import torch.nn.functional as F

# -------------------- 모델 정의 --------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        # 양방향이면 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]  # 마지막 레이어의 hidden state
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# -------------------- 사전 로딩 --------------------
# vocab.csv: 학습 시 사용된 단어 사전과 동일해야 함
vocab_df = pd.read_csv(r"C:\Users\KDT-37\Desktop\KDT_7\11_NLP\PROJECT\전체_vocab.csv")
vocab = vocab_df['0'].tolist() if '0' in vocab_df.columns else vocab_df.iloc[:, 0].tolist()

word_to_index = {'<PAD>': 0, '<UNK>': 1}
for idx, word in enumerate(vocab):
    word_to_index[word] = idx + 2

# 불용어
stopwords = pd.read_csv(r'C:\Users\KDT-37\Desktop\KDT_7\11_NLP\D250409\koreanStopwords_unique.txt', header=None)[0].tolist()

# -------------------- 설정 --------------------
vocab_size = 25825  # 고정 (모델과 일치)
embedding_dim = 200
hidden_dim = 128
output_dim = 7
max_len = 17

model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(r"C:\Users\KDT-37\Desktop\KDT_7\11_NLP\PROJECT\model\0415_LSTM_binary_best_model.pth", map_location=torch.device("cpu")))
model.eval()

# -------------------- 전처리 함수 --------------------
# Komoran 객체 생성 (형태소 분석기)
komoran = Komoran()

# 불용어 리스트 로드
STOPWORD_FILE = r'C:\Users\KDT-37\Desktop\KDT_7\11_NLP\D250409\koreanStopwords_unique.txt'

with open(STOPWORD_FILE, mode='r', encoding='utf-8') as f:
    stop_words = f.readlines()

# 불용어 리스트에서 줄바꿈 제거
stop_words = [word.replace('\n', '') for word in stop_words]

# 한국어 구두점 리스트
korean_punctuation = ['∼','%', '.', ',', '!', 
                      '?', '(', ')', '[', ']', 
                      '{', '}', '-', '...', '"', 
                      ':', ';', '·', '…', '⋯']

# 한국어 자음 리스트
korean_consonants = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 
                     'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 
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

# 텍스트를 숫자 인덱스 텐서로 변환
def text_to_tensor(text, word_to_index, max_len):
    indices = [word_to_index.get(word, word_to_index['<UNK>']) for word in text]
    # 패딩 처리
    indices = indices[:max_len] + [word_to_index['<PAD>']] * (max_len - len(indices))
    return torch.tensor(indices).unsqueeze(0)  # 배치 차원 추가

label_map = {
    0: "정치", 1: "경제", 2: "사회", 3: "생활/문화", 4: "세계", 5: "IT/과학", 6: "스포츠"
}

# -------------------- Streamlit 앱 --------------------
st.title("뉴스 제목 분류기")
user_input = st.text_input("뉴스 제목을 입력해주세요:")

if st.button("예측하기") and user_input:
    tokenized_input = tokenize([user_input])[0]  # 입력을 토큰화하고 첫 번째 문장만 사용
    input_tensor = text_to_tensor(tokenized_input, word_to_index, max_len)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    
    st.success(f"예측 결과: {label_map.get(pred, '알 수 없음')}")

