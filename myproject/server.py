import tensorflow as tf
from concurrent import futures
import grpc
from proto import message_pb2 
from proto import message_pb2_grpc
import numpy as np
import io
import os
import librosa
from tensorflow import keras
from scipy.signal import resample

# 모델 로딩
model_path = "/app/model" if os.getenv('DOCKER_ENV') == 'True' else 'C:/Users/82109/gRPC/myproject/SentimentAnalysis'
model = tf.saved_model.load(model_path)
print(dir(model))

class MyMLModelServicer(message_pb2_grpc.MyMLModelServicer):
    def Predict(self, request, context):
        # 바이트 배열을 파일로 해석
        audio_file = io.BytesIO(request.audio)
        
        # 오디오 파일 로드
        audio, sr = librosa.load(audio_file, sr=None)
        
        # 오디오 데이터를 모델에 입력할 수 있는 형태로 변환
        if len(audio) < 149:
            audio = np.pad(audio, (0, 149 - len(audio)))  # 패딩 추가
        else:
            audio = audio[:149]  # 초과하는 부분 잘라냄

        audio = np.expand_dims(audio, axis=0)  # 배치 차원 추가
        audio = np.expand_dims(audio, axis=-1)  # 채널 차원 추가

        y_pred = model(audio)[0]

        # 예측 지수를 감정 레이블과 짝지음
        labels = ['a', 'd', 'f', 'h', 's']  
        labels_probs = dict(zip(labels, y_pred))  # 감정 레이블과 확률을 짝지음

        # 가장 큰 지수 값을 가진 레이블과 그 지수 값 추출
        max_prob_label = max(labels_probs, key=labels_probs.get)
        max_prob = labels_probs[max_prob_label]
        max_emotion = message_pb2.Emotion(label=max_prob_label, probability=float(max_prob), emotion_index=f"{max_prob * 100:.2f}")

        # 감정이 스트레스에 미치는 영향
        stress_factors = {'a': 2.0, 'd': 0.8, 'f': 1.2, 'h': -0.2, 's': 1.0}

        # 각감정을 감정 지수로 계산하고, 스트레스 지수 계산
        stress_index = 0
        emotions = []
        for label, prob in labels_probs.items():
            emotion_index = prob * 100  # 감정 지수 계산
            emotions.append(message_pb2.Emotion(label=label, probability=float(prob), emotion_index=f"{emotion_index:.2f}"))
            
            stress_index += stress_factors[label] * prob  # 스트레스 지수 계산

        # 결과 반환
        stress_index_str = f"{stress_index * 100:.2f}"
        return message_pb2.PredictReply(emotions=emotions, max_emotion=max_emotion, stress_index=stress_index_str)

def serve():
    port = os.getenv('PORT', '8080')  # PORT 환경 변수를 사용하도록 수정
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    message_pb2_grpc.add_MyMLModelServicer_to_server(MyMLModelServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    print(f"Server starting on port {port}")  # 서버 시작 로그 메시지 추가
    server.start()
    print("Server started. Ready to receive requests.")  # 서버 시작 완료 로그 메시지 추가
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
