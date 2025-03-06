"""
(GestureRecognizer)

Este módulo implementa reconhecimento de gestos manuais com:
- MediaPipe Hands: Para deteção e rastreamento preciso dos pontos da mão
- TensorFlow: Para classificação dos gestos através de um modelo treinado
- Suavização temporal: Para reduzir falsos positivos e melhorar a estabilidade

O sistema suporta três modos de operação:
- gesture: Reconhecimento de gestos básicos
- letter: Reconhecimento de letras em Língua Gestual
- word: Reconhecimento de palavras em Língua Gestual
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import json
from collections import deque

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.model = None
        self.labels = []
        self.mode = "gesture"
        
        # suavização das previsões
        self.prediction_history = deque(maxlen=5)
        self.confidence_threshold = 0.25
        self.min_consecutive_predictions = 2
        
        self.mean = None
        self.std = None
        
        self._load_model()
        
    def _load_model(self):
        """
        Carrega o modelo neural treinado com as suas configurações

        Este método:
        1. Carrega o modelo específico para o modo atual (gesture/letter/word)
        2. Carrega as labels correspondentes para classificação
        3. Configura a normalização dos dados usando as estatísticas do treino
        """
        model_path = os.path.join('models', f'{self.mode}_model.h5')
        labels_path = os.path.join('models', f'{self.mode}_labels.json')
        data_path = os.path.join('models', f'{self.mode}_training_data.npz')
        
        try:
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
                
            self.model = tf.keras.models.load_model(model_path)
            
            if os.path.exists(data_path):
                data = np.load(data_path)
                X = data['data']
                self.mean = X.mean(axis=0)
                self.std = X.std(axis=0)
            
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
        except Exception as e:
            print(f"Error loading model: {e}")
            
    def process_frame(self, frame):
        """
        Processa um frame de vídeo para detetar e reconhecer gestos

        Args:
            frame: Imagem BGR do OpenCV

        Returns:
            tuple: (frame processado com visualizações, gesto reconhecido)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        translation = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = self._extract_landmarks(hand_landmarks)
                
                if self.model is not None:
                    prediction = self._predict(landmarks)
                    if prediction:
                        translation = prediction
                    
                self._draw_enhanced_hand(frame, hand_landmarks)
                
        return frame, translation
        
    def _extract_landmarks(self, hand_landmarks):
        """
        Extrai os pontos de referência da mão

        Realiza:
        1. Extração das coordenadas x, y, z de cada ponto
        2. Normalização para tornar o reconhecimento invariante à escala
        3. Proteção contra divisão por zero

        Args:
            hand_landmarks: Pontos da mão detetados pelo MediaPipe

        Returns:
            numpy.array: Vetor de características 
        """
        try:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            z_coords = [lm.z for lm in hand_landmarks.landmark]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            min_z, max_z = min(z_coords), max(z_coords)
            
            eps = 1e-6
            x_range = max(max_x - min_x, eps)
            y_range = max(max_y - min_y, eps)
            z_range = max(max_z - min_z, eps)
            
            landmarks = []
            for landmark in hand_landmarks.landmark:
                norm_x = (landmark.x - min_x) / x_range
                norm_y = (landmark.y - min_y) / y_range
                norm_z = (landmark.z - min_z) / z_range
                
                landmarks.extend([norm_x, norm_y, norm_z])
            
            return np.array(landmarks)
            
        except Exception as e:
            print(f"Error when extracting reference points: {str(e)}")
            raise
            
    def _predict(self, landmarks):
        """
        Realiza a previsão do gesto usando o modelo neural

        Implementa:
        1. Normalização dos dados de entrada
        2. Previsão com o modelo carregado
        3. Filtragem temporal para estabilidade
        4. Verificação de confiança mínima

        Args:
            landmarks: Vetor de características da mão

        Returns:
            str: Gesto reconhecido ou None se abaixo do limiar de confiança
        """
        try:
            landmarks = landmarks.reshape(1, -1)
            
            if self.mean is not None and self.std is not None:
                landmarks = (landmarks - self.mean) / self.std
            
            prediction = self.model.predict(landmarks, verbose=0)
            
            top_idx = np.argmax(prediction[0])
            confidence = prediction[0][top_idx]
            
            if confidence > self.confidence_threshold:
                predicted_label = self.labels[top_idx]
                self.prediction_history.append(predicted_label)
            
            if len(self.prediction_history) >= self.min_consecutive_predictions:
                recent_predictions = list(self.prediction_history)[-self.min_consecutive_predictions:]
                if all(x == recent_predictions[0] for x in recent_predictions):
                    result = recent_predictions[0]
                    return result
            
            return None
            
        except Exception as e:
            print(f"Forecast error: {str(e)}")
            return None
        
    def _draw_enhanced_hand(self, frame, landmarks):
        """
        Visualização melhorada da mão detetada

        Características:
        1. Conexões entre os pontos da mão
        2. Pontos de articulação com tamanho variável baseado na profundidade
        3. Cores distintas para melhor visualização

        Args:
            frame: Imagem onde desenhar
            landmarks: Pontos da mão a visualizar
        """
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(255, 255, 255), thickness=1)
        )
        
        height, width = frame.shape[:2]
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            radius = max(1, min(5, int(5 * (1 + landmark.z))))
            cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)
            
    def set_mode(self, mode):
        """
        Altera o modo de reconhecimento do sistema

        Args:
            mode: Novo modo ('gesture', 'letter' ou 'word')
            
        Efeitos:
            - Recarrega o modelo apropriado
            - Limpa o histórico de previsões
        """
        if mode in ["gesture", "letter", "word"]:
            self.mode = mode
            self._load_model()
            self.prediction_history.clear()
