"""
(GestureRecognizer)

Este módulo implementa um sistema avançado de reconhecimento de gestos manuais utilizando:
- MediaPipe Hands: Para deteção e rastreamento preciso dos pontos anatómicos da mão
- TensorFlow: Para classificação dos gestos através de um modelo de deep learning treinado
- Suavização temporal: Para reduzir falsos positivos e melhorar a estabilidade das predições

O sistema suporta três modos de operação:
- gesture: Reconhecimento de gestos básicos (apontar, ok, punho, etc.)
- letter: Reconhecimento de letras em Língua Gestual Portuguesa
- word: Reconhecimento de palavras completas em Língua Gestual Portuguesa
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
        """
        Inicializa o reconhecedor de gestos com as configurações predefinidas.
        
        Configurações:
        - MediaPipe Hands: Para deteção e rastreamento das mãos em tempo real
        - Modelo neural TensorFlow: Para classificação dos gestos
        - Sistema de suavização temporal: Para estabilizar as previsões
        """
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
        
        # Sistema de suavização das previsões para reduzir instabilidade
        self.prediction_history = deque(maxlen=5)
        self.confidence_threshold = 0.25
        self.min_consecutive_predictions = 2
        
        self.mean = None
        self.std = None
        
        self._load_model()
        
    def _load_model(self):
        """
        Carrega o modelo neural treinado com as respetivas configurações.

        Este método realiza:
        1. Carregamento do modelo específico para o modo atual (gesture/letter/word)
        2. Carregamento das etiquetas correspondentes para classificação
        3. Configuração da normalização dos dados utilizando as estatísticas do treino
        4. Otimização para utilização de GPU se disponível
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
        Processa um frame de vídeo para detetar e reconhecer gestos manuais.

        Este método realiza:
        1. Conversão do formato de cor para processamento pelo MediaPipe
        2. Deteção das mãos no frame e extração dos seus pontos de referência
        3. Classificação dos gestos utilizando o modelo neural carregado
        4. Visualização das mãos e do resultado da classificação no frame

        Args:
            frame: Imagem no formato BGR (padrão do OpenCV)

        Returns:
            tuple: (frame processado com visualizações, gesto reconhecido)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        title = {
            "gesture": "Gestos Disponiveis",
            "letter": "Letras Disponieis",
            "word": "Palavras Disponiveis"
        }.get(self.mode, "Etiquetas Disponíveis")
        self._draw_labels_overlay(frame, title, self.labels)
        
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
        Extrai e normaliza os pontos de referência da mão para processamento.

        Processo detalhado:
        1. Extração das coordenadas x, y, z de cada um dos 21 pontos da mão
        2. Normalização para tornar o reconhecimento invariante à escala e posição
        3. Implementação de proteção contra divisão por zero
        4. Transformação em vetor de características adequado ao modelo

        Args:
            hand_landmarks: Objeto contendo os 21 pontos da mão detetados pelo MediaPipe

        Returns:
            numpy.array: Vetor de características normalizado (63 valores: 21 pontos × 3 coordenadas)
        """
        try:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            z_coords = [lm.z for lm in hand_landmarks.landmark]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            min_z, max_z = min(z_coords), max(z_coords)
            
            eps = 1e-6  # Epsilon para evitar divisão por zero
            x_range = max(max_x - min_x, eps)
            y_range = max(max_y - min_y, eps)
            z_range = max(max_z - min_z, eps)
            
            normalized_landmarks = []
            for i in range(len(hand_landmarks.landmark)):
                lm = hand_landmarks.landmark[i]
                
                # Normalização de cada coordenada para o intervalo [0,1]
                norm_x = (lm.x - min_x) / x_range
                norm_y = (lm.y - min_y) / y_range
                norm_z = (lm.z - min_z) / z_range
                
                normalized_landmarks.extend([norm_x, norm_y, norm_z])
                
            features = np.array(normalized_landmarks, dtype=np.float32)
            
            # Aplicação da normalização estatística se disponível
            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / (self.std + eps)
                
            return features
            
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None
            
    def _predict(self, landmarks):
        """
        Realiza a classificação de um gesto com base nos pontos da mão extraídos.
        
        Funcionalidades:
        1. Utiliza o modelo TensorFlow para classificar os pontos da mão
        2. Aplica suavização temporal para reduzir falsos positivos
        3. Implementa um sistema de threshold para filtrar predições de baixa confiança
        4. Exige múltiplas predições consecutivas para confirmar um gesto
        
        Args:
            landmarks: Vetor de características da mão normalizado
            
        Returns:
            str: Etiqueta do gesto reconhecido, ou None se nenhum gesto for reconhecido
        """
        if landmarks is None or self.model is None:
            return None
            
        try:
            # Expande dimensões para compatibilidade com o formato esperado pelo modelo
            input_data = np.expand_dims(landmarks, axis=0)
            
            # Predição com o modelo neural
            predictions = self.model.predict(input_data, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
            
            # Filtragem por confiança mínima
            if confidence < self.confidence_threshold:
                self.prediction_history.clear()
                return None
                
            predicted_label = self.labels[predicted_idx]
            self.prediction_history.append(predicted_label)
            
            # Verifica se há predições consecutivas consistentes
            if (len(self.prediction_history) >= self.min_consecutive_predictions and 
                all(p == predicted_label for p in list(self.prediction_history)[-self.min_consecutive_predictions:])):
                return predicted_label
                
            return None
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
            
    def _draw_enhanced_hand(self, frame, hand_landmarks):
        """
        Desenha uma visualização melhorada da mão detetada no frame.
        
        Implementa:
        1. Desenho das conexões entre pontos da mão (esqueleto)
        2. Visualização dos pontos-chave com cores diferentes baseadas na função anatómica
        3. Destaque especial para pontos importantes para o reconhecimento
        
        Args:
            frame: Imagem onde desenhar a visualização
            hand_landmarks: Pontos de referência da mão detetados pelo MediaPipe
        """
        # Desenha as conexões da mão (esqueleto)
        self.mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
        )
        
        # Desenha pontos-chave específicos com destaque
        height, width, _ = frame.shape
        landmarks = hand_landmarks.landmark
        
        # Pontos-chave para destaque especial
        key_points = [0, 4, 8, 12, 16, 20]  # Pulso, pontas dos dedos
        
        for idx, landmark in enumerate(landmarks):
            x, y = int(landmark.x * width), int(landmark.y * height)
            
            # Tamanho e cor baseados na importância do ponto
            if idx in key_points:
                radius = 6
                color = (0, 255, 255)  # Amarelo para pontos-chave
            else:
                radius = 3
                color = (0, 255, 0)    # Verde para outros pontos
                
            cv2.circle(frame, (x, y), radius, color, -1)
            
    def _draw_labels_overlay(self, frame, title, labels):
        """
        Desenha um painel lateral com as etiquetas disponíveis para reconhecimento.
        
        Funcionalidades:
        1. Cria um painel semitransparente no lado esquerdo do frame
        2. Lista todas as etiquetas disponíveis (gestos, letras ou palavras)
        3. Ajusta o tamanho do painel conforme o número de etiquetas
        
        Args:
            frame: Imagem onde desenhar o painel
            title: Título do painel ("Gestos Disponíveis", etc.)
            labels: Lista de etiquetas disponíveis para exibição
        """
        if labels:
            overlay = frame.copy()
            y_pos = 30
            height = 25 + (len(labels) * 25)
            cv2.rectangle(overlay, (5, 10), (200, y_pos + height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            cv2.putText(frame, title, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for label in labels:
                y_pos += 25
                cv2.putText(frame, f"- {label}", (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 10), (200, 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            cv2.putText(frame, f"Nenhum {title.lower()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def set_mode(self, mode):
        """
        Altera o modo de reconhecimento do sistema e carrega o modelo correspondente.
        
        Este método:
        1. Verifica se o modo solicitado é válido
        2. Atualiza o modo de operação atual
        3. Carrega o modelo específico para o novo modo
        4. Limpa o histórico de predições para evitar conflitos
        
        Args:
            mode: Novo modo de operação ('gesture', 'letter' ou 'word')
        """
        if mode in ["gesture", "letter", "word"]:
            self.mode = mode
            self._load_model()
            self.prediction_history.clear()
