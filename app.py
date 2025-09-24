from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import openai
import pickle
import numpy as np
from tensorflow import keras
from dotenv import load_dotenv
import os

# Lambda 레이어 역직렬화 허용 (자체 학습 모델이므로 안전)
keras.config.enable_unsafe_deserialization()

load_dotenv('api.env')
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# FAQ 임베딩 로드
print("FAQ 데이터 로딩 중...")
with open('data/models/faq_embeddings.pkl', 'rb') as f:
    faq_data = pickle.load(f)
print(f"FAQ 데이터 로드 완료: {len(faq_data)}개")

# Siamese 재랭킹 모델 로드
print("Siamese 모델 로딩 중...")
try:
    siamese_model = keras.models.load_model('data/models/siamese_ranker.h5')
    print("Siamese 모델 로드 완료!")
except Exception as e:
    print(f"모델 로딩 오류: {e}")
    siamese_model = None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        user_question = request.json.get('question')
        
        if not user_question:
            return jsonify({'error': '질문을 입력해주세요'}), 400
        
        # 1. 질문 임베딩
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=user_question
        )
        user_embedding = np.array(response.data[0].embedding)
        
        # 2. Siamese 모델로 재랭킹
        if siamese_model is not None:
            scores = []
            for item in faq_data:
                doc_embedding = np.array(item['embedding'])
                score = siamese_model.predict(
                    [user_embedding.reshape(1, -1), doc_embedding.reshape(1, -1)],
                    verbose=0
                )[0][0]
                scores.append({'faq': item['faq'], 'score': float(score)})
            
            sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
            top_match = sorted_scores[0]
            max_score = top_match['score']
            use_model = 'siamese_ranker'
        else:
            def cosine_similarity(vec1, vec2):
                return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            scores = []
            for item in faq_data:
                sim = cosine_similarity(user_embedding, item['embedding'])
                scores.append({'faq': item['faq'], 'score': float(sim)})
            
            sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
            top_match = sorted_scores[0]
            max_score = top_match['score']
            use_model = 'cosine_similarity'
        
        # 3. 임계값 기반 분기
        if max_score >= 0.5:
            # FAQ 기반 답변 (구조화)
            top_3 = sorted_scores[:3]
            
            context = "\n\n".join([
                f"[{item['faq']['category']}] Q: {item['faq']['question']}\nA: {item['faq']['answer']}"
                for item in top_3
            ])
            
            # 구조화된 프롬프트
            gpt_response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """당신은 지니언스의 NAC 전문가 AI 'GuardX'입니다. 
답변은 반드시 다음 형식으로 제공하세요:

[문제 진단]
문제의 핵심 원인을 1-2문장으로 요약

[해결 방법]
1. 첫 번째 단계 설명
2. 두 번째 단계 설명
3. 세 번째 단계 설명
(필요시 4-5단계 추가 가능)

[참고사항]
추가 팁이나 주의사항 (1-2문장)

간결하고 명확하게 답변하세요. 각 단계는 구체적이고 실행 가능해야 합니다."""},
                    {"role": "user", "content": f"질문: {user_question}\n\n참고 FAQ:\n{context}"}
                ]
            )
            
            return jsonify({
                'answer': gpt_response.choices[0].message.content,
                'source': f'{use_model} + FAQ',
                'confidence': max_score,
                'model': use_model,
                'references': [item['faq'] for item in top_3]
            })
        else:
            # 일반 AI 답변 (간결하게)
            gpt_response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "친절한 AI 어시스턴트로 답변하세요. 답변은 3-5문장 이내로 간결하게 제공하세요."},
                    {"role": "user", "content": user_question}
                ]
            )
            
            return jsonify({
                'answer': gpt_response.choices[0].message.content,
                'source': 'OpenAI 일반 지식',
                'confidence': max_score,
                'model': 'gpt-4o-mini',
                'references': []
            })
            
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """서버 상태 확인"""
    return jsonify({
        'status': 'ok',
        'faq_count': len(faq_data),
        'model_loaded': siamese_model is not None,
        'model_type': 'Siamese Neural Network' if siamese_model else 'Cosine Similarity Fallback'
    })

if __name__ == '__main__':
    import webbrowser
    print("\n" + "="*50)
    print("GuardX AI 서버 시작")
    print(f"FAQ 데이터: {len(faq_data)}개")
    if siamese_model:
        print("Siamese 재랭킹 모델: ✓ 활성화")
    else:
        print("Siamese 재랭킹 모델: ✗ 미활성화 (코사인 유사도 사용)")
    print("서버 주소: http://localhost:5000")
    print("="*50 + "\n")
    
    webbrowser.open('http://localhost:5000')
    app.run(debug=True, port=5000)
