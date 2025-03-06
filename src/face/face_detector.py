"""
(FaceDetector)

Este módulo implementa a deteção e reconhecimento facial.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

class FaceDetector:
    def __init__(self):
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
        self.model_file = os.path.join('models', 'face_recognizer.yml')
        self.labels_file = os.path.join('models', 'face_labels.json')
        self.training_data_file = os.path.join('models', 'face_training_data.npz')
        
        os.makedirs('models', exist_ok=True)
        
        self.load_recognizer()
        
    def load_recognizer(self):
        """
        Carrega o modelo de reconhecimento facial e dados associados

        Processo:
        1. Inicia o reconhecedor LBPH
        2. Carrega nomes/labels de faces reconhecidas
        3. Carrega os dados de treinos anteriores
        4. Treina o modelo com os dados carregados
        
        Em caso de erro, inicia um modelo vazio
        """
        try:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            if os.path.exists(self.labels_file):
                with open(self.labels_file, 'r') as f:
                    names_list = json.load(f)
                    self.label_names = {str(i): name for i, name in enumerate(names_list)}
                print(f"Loaded {len(self.label_names)} face labels: {self.label_names}")
            else:
                print("No face tag found")
                self.label_names = {}
            
            if os.path.exists(self.training_data_file):
                data = np.load(self.training_data_file)
                print(f"Keys available in the training data: {data.files}")
                
                if 'data' in data.files:
                    face_data = data['data']
                    face_labels = data['labels']
                    print(f"Loaded {len(face_data)} facial samples")
                    print(f"Facial data format: {face_data.shape}")
                    print(f"Label format: {face_labels.shape}")
                    
                    self.known_faces = []
                    self.face_labels = []
                    for i in range(len(face_data)):
                        face = face_data[i]
                        if len(face.shape) == 3:
                            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                        face = cv2.resize(face, (100, 100))
                        self.known_faces.append(face)
                        self.face_labels.append(face_labels[i])
                    
                    self.known_faces = np.array(self.known_faces)
                    self.face_labels = np.array(self.face_labels, dtype=np.int32)
                    
                    print(f"Processed {len(self.known_faces)} training faces")
                    print(f"Final data format: {self.known_faces.shape}")
                    print(f"Final label format: {self.face_labels.shape}")
                    
                    self.face_recognizer.train(self.known_faces, self.face_labels)
                    print("Model successfully trained!")
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
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.face_recognizer.train([np.zeros((100, 100), dtype=np.uint8)], np.array([0]))
            
    def save_recognizer(self):
        """
        Guarda o estado atual do reconhecedor facial

        Guarda:
        1. Modelo treinado (.yml)
        2. Labels e nomes (JSON)
        3. Dados de treino (NPZ)
            - Amostras faciais
            - labels correspondentes
        """
        try:
            with open(self.labels_file, 'w') as f:
                json.dump(self.label_names, f, indent=4)
            print(f"Saved {len(self.label_names)} face tags")
            
            if self.known_faces:
                np.savez(self.training_data_file, 
                        data=np.array(self.known_faces),
                        labels=np.array(self.face_labels))
                print(f"Saved {len(self.known_faces)} facial samples")
            
            self.face_recognizer.write(self.model_file)
            print("Facial recognition model saved")
            
        except Exception as e:
            print(f"Error saving recognizer: {e}")
            
    def add_face(self, frame, name):
        """
        Adiciona uma nova face ao sistema de reconhecimento

        Processo:
        1. Remove a face na imagem com o Haar Cascade
        2. Pré-processa a face (escala e conversão para cinza)
        3. Adiciona aos dados de treino
        4. Atualiza o modelo
        5. Salva uma cópia da face para referência

        Args:
            frame: Imagem contendo a face
            name: Nome da pessoa

        Returns:
            bool: True se a face foi adicionada com sucesso
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                print("No face detected in the image")
                return False
                
            if len(faces) > 1:
                print("Multiple faces detected. Make sure only one face is visible.")
                return False
                
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            face_roi = cv2.resize(face_roi, (100, 100))
            
            label = str(self.next_label)
            self.known_faces.append(face_roi)
            self.face_labels.append(int(label))
            self.label_names[label] = name
            self.next_label += 1
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join('faces', filename)
            cv2.imwrite(filepath, face_roi)
            
            self.face_recognizer.train(self.known_faces, np.array(self.face_labels))
            self.save_recognizer()
            
            print(f"New face added: {name}")
            return True
            
        except Exception as e:
            print(f"Error adding face: {e}")
            return False
            
    def process_frame(self, frame):
        """
        Processa um frame de acordo com o modo atual

        Modos disponíveis:
        - mesh: Deteção de malha facial completa
        - iris: Deteção e rastreamento da íris ocular
        - recognition: Reconhecimento de faces reconhecidas

        Args:
            frame: Frame a processar

        Returns:
            numpy.ndarray: Frame processado com visualizações
        """
        if self.mode == "mesh":
            return self._process_mesh(frame)
        elif self.mode == "iris":
            return self._process_iris(frame)
        else:
            return self._process_recognition(frame)
            
    def _process_mesh(self, frame):
        """
        Processa o frame usando a deteção de mesh facial

        Args:
            frame: Frame a processar

        Returns:
            numpy.ndarray: Frame com mesh facial desenhada
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._draw_face_mesh(frame, face_landmarks)
                
        return frame

    def _process_iris(self, frame):
        """
        Processa o frame com foco na deteção da íris ocular

        Args:
            frame: Frame a processar

        Returns:
            numpy.ndarray: Frame com pontos da íris marcados
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            height, width, _ = frame.shape
            for face_landmarks in results.multi_face_landmarks:
                left_iris = []
                for idx in range(468, 473):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    left_iris.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 69, 255), -1)
                
                right_iris = []
                for idx in range(473, 478):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    right_iris.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 69, 255), -1)
                
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
        
    def _process_recognition(self, frame):
        """
        Processa o frame com reconhecimento facial
        Args:
            frame: Frame a processar

        Returns:
            numpy.ndarray: Frame com faces reconhecidas
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            
            try:
                label, distance = self.face_recognizer.predict(face_roi)
                confidence = max(0, min(100, 100 * (1 - distance/100)))
                
                label_str = str(label)
                name = self.label_names.get(label_str, "Unknown")
                
                if confidence < 30:
                    name = "Unknown"
                    print(f"Face detected but confidence very low: {confidence:.1f}% (distance: {distance:.1f})")
                else:
                    name = f"{name} ({confidence:.1f}%)"
                    print(f"Face recognized: {name} (distance: {distance:.1f})")
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, 
                            (x - 2, y - text_size[1] - 10),
                            (x + text_size[0] + 2, y),
                            (0, 0, 0), -1)
                
                cv2.putText(frame, name,
                           (x, y - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                           
            except Exception as e:
                print(f"Facial recognition error: {e}")
                print(f"Label: {label if 'label' in locals() else 'unknown'}")
                print(f"Label names: {self.label_names}")
        
        return frame
        
    def _draw_face_mesh(self, frame, landmarks):
        """
        Desenha os pontos de referência da mesh facial no frame

        Args:
            frame: Frame a desenhar
            landmarks: Pontos de referência da malha facial
        """
        height, width, _ = frame.shape
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
    def get_mode(self):
        """
        Obtém o modo atual de deteção facial

        Returns:
            str: Modo atual
        """
        return self.mode
        
    def toggle_mode(self):
        """
        Modos faciais

        Modos disponíveis:
        - mesh: Deteção de malha facial completa
        - iris: Deteção e rastreamento de íris
        - recognition: Reconhecimento de faces conhecidas
        """
        if self.mode == "mesh":
            self.mode = "iris"
        elif self.mode == "iris":
            self.mode = "recognition"
            if os.path.exists(self.model_file):
                try:
                    self.face_recognizer.read(self.model_file)
                    print("Facial recognition model loaded")
                except Exception as e:
                    print(f"Error loading face model: {e}")
        else:
            self.mode = "mesh"
