"""
(FaceDetector)

Este módulo implementa a deteção e reconhecimento facial utilizando tecnologias OpenCV e MediaPipe.
Proporciona funcionalidades para identificar, visualizar e reconhecer características faciais.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

class FaceDetector:
    def __init__(self):
        """
        Inicializa o detetor facial com múltiplas capacidades.
        
        Configurações:
        - MediaPipe Face Mesh para deteção detalhada da malha facial
        - Cascade Classifier do OpenCV para deteção rápida de faces
        - Sistema de reconhecimento facial baseado em LBPH (Local Binary Patterns Histograms)
        - Estruturas de dados para armazenamento de faces conhecidas e respetivas etiquetas
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.known_faces = []
        self.face_labels = []
        self.label_names = {}
        self.next_label = 0
        self.mode = "mesh"
        self.model_file = os.path.join('models', 'face_model.h5')
        self.labels_file = os.path.join('models', 'face_labels.json')
        self.training_data_file = os.path.join('models', 'face_training_data.npz')
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('faces', exist_ok=True)
        
        self.load_recognizer()
        
    def load_recognizer(self):
        """
        Carrega o modelo de reconhecimento facial e dados associados.

        Processo sequencial:
        1. Inicializa o reconhecedor LBPH (Local Binary Patterns Histograms)
        2. Carrega nomes/etiquetas de faces reconhecidas do ficheiro JSON
        3. Carrega os dados de treino anteriores do ficheiro NPZ
        4. Treina o modelo com os dados carregados
        
        Em caso de erro, inicia um modelo vazio para evitar falhas na execução.
        """
        try:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            if os.path.exists(self.labels_file):
                with open(self.labels_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.label_names = data
                    elif isinstance(data, list):
                        self.label_names = {str(i): name for i, name in enumerate(data)}
                    self.next_label = max([int(label) for label in self.label_names.keys()]) + 1 if self.label_names else 0
                print(f"Loaded {len(self.label_names)} face labels: {self.label_names}")
                
                # Log detalhado das etiquetas carregadas
                for label_id, name in self.label_names.items():
                    print(f"  - Label ID {label_id}: {name}")
            else:
                print("No face labels found")
                self.label_names = {}
                self.next_label = 0
            
            if os.path.exists(self.training_data_file):
                data = np.load(self.training_data_file, allow_pickle=True)
                print(f"Keys available in the training data: {data.files}")
                
                if 'data' in data.files and 'labels' in data.files:
                    face_data = data['data']
                    face_labels = data['labels']
                    print(f"Loaded {len(face_data)} facial samples")
                    
                    self.known_faces = []
                    self.face_labels = []
                    
                    for i in range(len(face_data)):
                        face = face_data[i]
                        # Garante que a face está no formato correto para o LBPH
                        if len(face.shape) == 1:  # Se for um array 1D (flatten)
                            face = np.reshape(face, (100, 100))
                        if len(face.shape) == 3:  # Se for RGB
                            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                            
                        self.known_faces.append(face)
                        self.face_labels.append(int(face_labels[i]))
                    
                    self.known_faces = np.array(self.known_faces)
                    self.face_labels = np.array(self.face_labels, dtype=np.int32)
                    
                    print(f"Processed {len(self.known_faces)} training faces")
                    
                    if len(self.known_faces) > 0:
                        # Garantir que as faces e etiquetas têm o mesmo comprimento
                        print(f"Known faces shape: {self.known_faces.shape}, Labels shape: {self.face_labels.shape}")
                        
                        # Treinar o reconhecedor
                        self.face_recognizer.train(self.known_faces, self.face_labels)
                        print("Model successfully trained!")
                        
                        # Mostra as etiquetas reconhecidas
                        label_counts = {}
                        for label in self.face_labels:
                            if label in label_counts:
                                label_counts[label] += 1
                            else:
                                label_counts[label] = 1
                        print(f"Labels distribution: {label_counts}")
                        for label, count in label_counts.items():
                            name = self.label_names.get(str(label), "Unknown")
                            print(f"  - {name} (ID: {label}): {count} samples")
                    else:
                        print("No faces to train with")
                        self.face_recognizer.train([np.zeros((100, 100), dtype=np.uint8)], np.array([0]))
                else:
                    print("No facial data found in the training file")
                    self.known_faces = []
                    self.face_labels = []
                    self.face_recognizer.train([np.zeros((100, 100), dtype=np.uint8)], np.array([0]))
            else:
                print("No training data found")
                self.known_faces = []
                self.face_labels = []
                self.face_recognizer.train([np.zeros((100, 100), dtype=np.uint8)], np.array([0]))
                
        except Exception as e:
            print(f"Error loading recognizer: {e}")
            print("Initializing with an empty model")
            self.label_names = {}
            self.known_faces = []
            self.face_labels = []
            self.next_label = 0
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.face_recognizer.train([np.zeros((100, 100), dtype=np.uint8)], np.array([0]))
            
    def save_recognizer(self):
        """
        Guarda o estado atual do reconhecedor facial em ficheiros persistentes.

        Elementos guardados:
        1. Modelo treinado (.yml) - Para preservar os parâmetros do algoritmo LBPH
        2. Etiquetas e nomes (JSON) - Para manter o mapeamento entre IDs e nomes reais
        3. Dados de treino (NPZ) - Para potencial retreino ou análise posterior
            - Amostras faciais (matrizes 100x100)
            - Etiquetas correspondentes (valores inteiros)
        """
        try:
            with open(self.labels_file, 'w') as f:
                json.dump(self.label_names, f, indent=4)
            print(f"Saved {len(self.label_names)} face labels: {self.label_names}")
            
            if self.known_faces:
                np.savez(self.training_data_file, 
                        data=np.array(self.known_faces),
                        labels=np.array(self.face_labels))
                print(f"Saved {len(self.known_faces)} facial samples")
            
            self.face_recognizer.write(self.model_file)
            print("Face recognition model saved")
            
        except Exception as e:
            print(f"Error saving recognizer: {e}")
            
    def add_face(self, frame, name):
        """
        Adiciona uma nova face ao sistema de reconhecimento.
        
        Este método:
        1. Deteta a face no frame usando o Cascade Classifier
        2. Pré-processa a face (redimensionamento e equalização de histograma)
        3. Gera múltiplas amostras com variações para melhorar a robustez
        4. Actualiza o modelo de reconhecimento com a nova face
        5. Guarda os dados actualizados em ficheiros persistentes
        
        Args:
            frame: Imagem contendo a face a ser adicionada
            name: Nome/etiqueta para associar à face
            
        Returns:
            bool: True se a face foi adicionada com sucesso, False caso contrário
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) != 1:
            print(f"Expected 1 face, found {len(faces)}")
            return False
            
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))
        face_roi = cv2.equalizeHist(face_roi)  # Equalização para melhorar contraste
        
        os.makedirs('faces', exist_ok=True)
        
        samples = []
        # Aumentar a quantidade de amostras para melhorar reconhecimento
        for i in range(30):  # 30 amostras por face
            # Variações nos parâmetros de imagem para aumentar robustez
            alpha = np.random.uniform(0.7, 1.3)  # Contraste (ampliado)
            beta = np.random.uniform(-15, 15)    # Brilho (ampliado)
            
            # Aplicar variações de contraste e brilho
            adjusted = cv2.convertScaleAbs(face_roi, alpha=alpha, beta=beta)
            
            # Aplicar equalização e desfoque aleatório para simular movimento
            if i % 3 == 0:  # Para cada terço das amostras
                blur_size = np.random.choice([1, 3, 5])
                adjusted = cv2.GaussianBlur(adjusted, (blur_size, blur_size), 0)
            
            # Adicionar ruído gaussiano para simular variações de ambiente
            noise = np.random.normal(0, 3, adjusted.shape).astype(np.uint8)  # Ruído controlado
            augmented = cv2.add(adjusted, noise)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i}"
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join('faces', filename)
            cv2.imwrite(filepath, augmented)
            
            samples.append(augmented)
            self.known_faces.append(augmented)
            self.face_labels.append(self.next_label)
            
        self.label_names[str(self.next_label)] = name
        self.next_label += 1
        
        # Configurar reconhecedor com parâmetros otimizados para melhor performance
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=3,       # Raio aumentado para capturar mais textura (default=1)
            neighbors=12,   # Pontos vizinhos aumentados para maior discriminação (default=8)
            grid_x=10,      # Divisão horizontal aumentada para maior detalhe (default=8)
            grid_y=10,      # Divisão vertical aumentada para maior detalhe (default=8)
            threshold=100   # Limiar mais permissivo para melhorar taxa de reconhecimento
        )
        
        self.face_recognizer.train(np.array(self.known_faces), np.array(self.face_labels))
        self.save_recognizer()
        
        return True

    def process_frame(self, frame):
        """
        Processa um frame de vídeo aplicando o modo de deteção facial ativo.

        Modos disponíveis:
        - mesh: Deteção e visualização da malha facial completa (468 pontos)
        - iris: Deteção e rastreamento específico da íris ocular
        - recognition: Reconhecimento de faces previamente registadas

        Args:
            frame: Imagem a processar (formato BGR do OpenCV)

        Returns:
            numpy.ndarray: Frame processado com as visualizações correspondentes
        """
        if self.mode == "mesh":
            return self._process_mesh(frame)
        elif self.mode == "iris":
            return self._process_iris(frame)
        else:
            return self._process_recognition(frame)
            
    def _process_recognition(self, frame):
        """
        Implementa o modo de reconhecimento facial no frame atual.
        
        Funcionalidades:
        1. Deteta faces usando o Cascade Classifier
        2. Para cada face, aplica o reconhecedor LBPH
        3. Exibe um painel com todas as faces registadas
        4. Desenha retângulos e nomes/percentagens de confiança sobre as faces reconhecidas
        
        Args:
            frame: Imagem a processar (formato BGR)
            
        Returns:
            numpy.ndarray: Frame com visualização do reconhecimento facial
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Desenha um painel com as faces reconhecidas no canto superior direito
        if len(self.label_names) > 0:
            padding = 10
            panel_height = min(180, 40 * len(self.label_names))
            panel_width = 200
            
            # Fundo do painel - posicionado no canto superior direito
            cv2.rectangle(frame, 
                        (frame.shape[1] - panel_width - padding, padding),
                        (frame.shape[1] - padding, panel_height + padding),
                        (0, 0, 0), -1)
            
            # Título do painel
            cv2.putText(frame, "Recognized Faces:", 
                       (frame.shape[1] - panel_width - padding + 5, padding + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Lista de nomes das faces registadas
            y_pos = padding + 45
            for label_id, name in self.label_names.items():
                cv2.putText(frame, f"- {name}", 
                           (frame.shape[1] - panel_width - padding + 10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 25
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            
            try:
                # Configuração do threshold para melhorar a confiança de reconhecimento
                self.face_recognizer.setThreshold(100)  # Valores mais altos são mais permissivos
                
                label, distance = self.face_recognizer.predict(face_roi)
                # Fórmula para converter distância em percentagem de confiança
                confidence = max(0, min(100, 100 * (1 - distance/150)))
                
                label_str = str(label)
                name = self.label_names.get(label_str)
                
                # Informações detalhadas para depuração
                print(f"Face detected - Label ID: {label}, Distance: {distance}, Confidence: {confidence:.1f}%, Name: {name}")
                
                if confidence < 5 or not name:  # Limiar baixo para maximizar reconhecimentos
                    name = "Unknown"
                    color = (0, 0, 255)  # Vermelho para faces desconhecidas
                else:
                    color = (0, 255, 0)  # Verde para faces reconhecidas
                
                # Desenha retângulo ao redor da face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Fundo para melhorar visibilidade do texto
                text_size = cv2.getTextSize(f"{name} ({confidence:.1f}%)", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, 
                            (x, y - text_size[1] - 10),
                            (x + text_size[0], y),
                            (0, 0, 0), -1)
                
                # Nome e percentagem de confiança
                cv2.putText(frame, f"{name} ({confidence:.1f}%)",
                           (x, y - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                           
            except Exception as e:
                print(f"Error: {e}")
        
        return frame

    def _process_mesh(self, frame):
        """
        Processa o frame aplicando a deteção de malha facial completa.
        
        Utiliza o MediaPipe Face Mesh para identificar e visualizar os 
        468 pontos de referência da face, criando uma representação detalhada 
        da estrutura facial.

        Args:
            frame: Imagem a processar (formato BGR)

        Returns:
            numpy.ndarray: Frame com a malha facial visualizada
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._draw_face_mesh(frame, face_landmarks)
                
        return frame

    def _process_iris(self, frame):
        """
        Processa o frame com foco específico na deteção da íris ocular.
        
        Este modo utiliza pontos específicos do MediaPipe Face Mesh (468-478)
        que correspondem às íris dos olhos. Visualiza tanto os pontos individuais
        quanto círculos ao redor das íris.

        Args:
            frame: Imagem a processar (formato BGR)

        Returns:
            numpy.ndarray: Frame com as íris oculares destacadas
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            height, width, _ = frame.shape
            for face_landmarks in results.multi_face_landmarks:
                left_iris = []
                for idx in range(468, 473):  # Índices da íris esquerda
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    left_iris.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 69, 255), -1)
                
                right_iris = []
                for idx in range(473, 478):  # Índices da íris direita
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    right_iris.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 69, 255), -1)
                
                # Desenha círculos ao redor de cada íris
                if left_iris:
                    center = np.mean(left_iris, axis=0).astype(int)
                    radius = int(np.mean([np.linalg.norm(np.array(p) - center) for p in left_iris]))
                    cv2.circle(frame, tuple(center), radius, (0, 255, 0), 1)
                    cv2.circle(frame, tuple(center), 2, (0, 69, 255), -1)
                
                if right_iris:
                    center = np.mean(right_iris, axis=0).astype(int)
                    radius = int(np.mean([np.linalg.norm(np.array(p) - center) for p in right_iris]))
                    cv2.circle(frame, tuple(center), radius, (0, 255, 0), 1)
                    cv2.circle(frame, tuple(center), 2, (0, 69, 255), -1)
        
        return frame

    def _draw_face_mesh(self, frame, landmarks):
        """
        Renderiza os pontos de referência da malha facial no frame.
        
        Cada um dos 468 pontos é desenhado como um pequeno círculo verde,
        criando uma representação visual detalhada da estrutura facial.

        Args:
            frame: Imagem onde desenhar os pontos
            landmarks: Objeto contendo os pontos de referência da malha facial
        """
        height, width, _ = frame.shape
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
    def get_mode(self):
        """
        Obtém o modo atual de deteção facial.

        Returns:
            str: Modo atual ("mesh", "iris" ou "recognition")
        """
        return self.mode
        
    def toggle_mode(self):
        """
        Alterna entre os diferentes modos de deteção facial.

        Modos disponíveis em sequência cíclica:
        - mesh: Deteção de malha facial completa (468 pontos)
        - iris: Deteção e rastreamento específico das íris oculares
        - recognition: Reconhecimento de faces previamente registadas
        
        Quando muda para o modo de reconhecimento, recarrega o reconhecedor
        para garantir que os dados mais recentes são utilizados.
        """
        if self.mode == "mesh":
            self.mode = "iris"
        elif self.mode == "iris":
            self.mode = "recognition"
            self.load_recognizer()  # Recarrega o reconhecedor ao mudar para o modo de reconhecimento
        else:
            self.mode = "mesh"
