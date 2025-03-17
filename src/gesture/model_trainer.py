"""
(ModelTrainer)

Este módulo implementa um sistema de treino para modelos neurais:
- modos (gestos, letras, palavras, faces)
- treina em tempo real com webcam
- treino com conjuntos de dados/imagens (datasets)
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import mediapipe as mp
import json
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
import yaml
import argparse

class ModelTrainer:
    def __init__(self, mode=None, training_type=None):
        self.mode = mode
        self.training_type = training_type
        
        if not mode:
            print("\nWelcome to RTTSSL Training!")
            print("\nSelect training mode:")
            print("1. Gesture Mode")
            print("2. Letter Mode")
            print("3. Word Mode")
            print("4. Face Mode")
            
            while True:
                try:
                    choice = input("\nChoose mode (1-4): ").strip()
                    if choice == '1':
                        self.mode = "gesture"
                        break
                    elif choice == '2':
                        self.mode = "letter"
                        break
                    elif choice == '3':
                        self.mode = "word"
                        break
                    elif choice == '4':
                        self.mode = "face"
                        break
                    else:
                        print("Invalid choice. Please enter 1, 2, 3, or 4.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        if not training_type:
            print("\nSelect training type:")
            print("1. Real-Time Training (using webcam)")
            print("2. Dataset Training (using saved images)")
            
            while True:
                try:
                    choice = input("\nChoose training type (1-2): ").strip()
                    if choice == '1':
                        self.training_type = "realtime"
                        break
                    elif choice == '2':
                        self.training_type = "dataset"
                        break
                    else:
                        print("Invalid choice. Please enter 1 or 2.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        print(f"\nInitializing {self.mode.upper()} mode with {self.training_type.upper()} training...")
        
        self.config = self._load_config()
        self.keybinds = self.config['keybinds']
        self.training_config = self.config.get('training', {})
        
        if self.mode == "face":
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=self.training_config.get('confidence_threshold', 0.5),
                model_selection=1
            )
        else:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.training_type == "dataset",
                max_num_hands=2,
                min_detection_confidence=self.training_config.get('confidence_threshold', 0.5),
                min_tracking_confidence=self.training_config.get('confidence_threshold', 0.5),
                model_complexity=self.training_config.get('model_complexity', 1)
            )
            
        self.model = None
        self.labels = []
        self.samples_per_label = {}
        
        self._load_existing_model()

    def _load_config(self):
        """
        Carrega e valida configurações do sistema

        As configurações incluem:
        - KEYBINDS 
        - Parâmetros de treino
        - Configurações de captura
        - Limiares de confiança
        """
        config_path = os.path.join('config', 'configs.yaml')
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {
                'keybinds': {
                    'quit_training': 'q',
                    'start_stop_recording': 's',
                    'new_label': 'n',
                    'retrain_label': 'r',
                    'info_display': 'i',
                    'start_training': 't'
                },
                'training': {
                    'max_samples_per_label': 1000,
                    'window_size': {'width': 1280, 'height': 720},
                    'confidence_threshold': 0.5,
                    'model_complexity': 1
                }
            }
            
    def _load_existing_model(self):
        """
        Carrega modelo e dados de treino existentes

        Processo:
        1. Carrega modelo treinado (.h5)
        2. Carrega labels (JSON)
        3. Carrega dados de treino (NPZ)
        4. Organiza amostras por labels
        
        Cria um modelo novo se ainda não houver nenhum criado
        """
        try:
            model_path = os.path.join('models', f'{self.mode}_model.h5')
            labels_path = os.path.join('models', f'{self.mode}_labels.json')
            data_path = os.path.join('models', f'{self.mode}_training_data.npz')
            
            if os.path.exists(model_path) and os.path.exists(labels_path):
                self.model = keras.models.load_model(model_path)
                with open(labels_path, 'r') as f:
                    self.labels = json.load(f)
                
                if os.path.exists(data_path):
                    data = np.load(data_path)
                    self.samples_per_label = {label: [] for label in self.labels}
                    
                    for sample, label_idx in zip(data['data'], data['labels']):
                        label = self.labels[label_idx]
                        self.samples_per_label[label].append(sample)
                        
                    print(f"Existing model loaded with labels: {self.labels}")
                    for label, samples in self.samples_per_label.items():
                        print(f"  - {label}: {len(samples)} samples")
        except Exception as e:
            print(f"Note: No existing model found or error loading: {e}")
            
    def train_from_images(self):
        """
        Treina o modelo com imagens do conjunto de dados

        Processo:
        1. Verifica os datasets
        2. Processa as imagens por label
        3. Extrai as características com o MediaPipe
        4. Prepara os dados para o treino
        5. Treina e testa o modelo

        Os datasets devem seguir a seguinte estrutura:
        datasets/
          └── [mode]/
              ├── [label1]/
              │   ├── imagem1.jpg
              │   └── imagem2.jpg
              └── [label2]/
                  └── ...
        """
        print(f"\nFound {len(self.labels)} labels: {self.labels}")
        
        features = []
        labels = []
        label_indices = {label: idx for idx, label in enumerate(self.labels)}
        
        dataset_dir = os.path.join('datasets', self.mode)
        if not os.path.exists(dataset_dir):
            print(f"\nError: Dataset directory '{dataset_dir}' not found!")
            print("Please create the directory and add your image data.")
            return
        
        for label in self.labels:
            label_dir = os.path.join(dataset_dir, label)
            if not os.path.exists(label_dir):
                print(f"\nWarning: Directory for label '{label}' not found in {label_dir}")
                continue
            
            print(f"\nProcessing {label}...")
            image_files = [f for f in os.listdir(label_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Found {len(image_files)} images")
            
            valid_features = []
            for image_file in image_files:
                image_path = os.path.join(label_dir, image_file)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(image)
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # extrai as coordenadas x, y das landmarks
                            coords = []
                            for landmark in hand_landmarks.landmark:
                                coords.extend([landmark.x, landmark.y])
                            valid_features.append(coords)
                            labels.append(label_indices[label])
                            
                except Exception as e:
                    print(f"Error processing {image_file}: {str(e)}")
                    continue
            
            if valid_features:
                features.extend(valid_features)
                print(f"Features extracted successfully from {len(valid_features)} images")
            else:
                print(f"No features extracted for label {label}")
        
        if not features:
            print("\nError: No features extracted from any images!")
            return
            
        # conversor de arrays numpy
        X = np.array(features)
        y = np.array(labels)
        
        # divisão dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # criação e treino do model
        self.model = self._create_model(X_train.shape[1])
        
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # teste do modelo
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc:.4f}")
        
        # model save point
        self._save_model()
        print("\nModel saved successfully!")

    def _create_model(self, input_shape):
        """
        Cria e configura a arquitetura do modelo neural

        Arquitetura:
        - Camadas densas com dropout
        - Ativação ReLU e Softmax
        - Otimizador Adam
        - Métricas de precisão

        Args:
            input_shape (int): Número de características de entrada

        Returns:
            tensorflow.keras.Model: Modelo configurado
        """
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(self.labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self):
        # inicia o processo de treino com base no tipo de treino selecionado
        if self.training_type == "dataset":
            self.train_from_images()
        else:
            self.train_realtime()

    def train_realtime(self):
        """
        Treina o modelo com captura de frames em tempo real

        Teclas de controlo:
        - 's': Iniciar/parar gravação
        - 'n': Nova label
        - 'r': Retreinar gesto/palavra/letra
        - 't': Iniciar treino
        - 'q': Sair
        """
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            # resolução
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            recording = False
            frame_count = 0
            current_label = None
            
            # output dos comandos 
            print("\nTraining controls:")
            print(f"- Press '{self.keybinds['quit_training']}' to quit")
            print(f"- Press '{self.keybinds['start_stop_recording']}' to start/stop recording samples")
            print(f"- Press '{self.keybinds['new_label']}' to create a new label")
            print(f"- Press '{self.keybinds['retrain_label']}' to retrain a specific label from scratch")
            print(f"- Press '{self.keybinds['info_display']}' to display training information")
            print(f"- Press '{self.keybinds['start_training']}' to start training the model")
            print("\n")
            
            print(f"Current mode: {self.mode.upper()}")
            if self.labels:
                print("Current labels and samples:")
                for label in self.labels:
                    count = len(self.samples_per_label.get(label, []))
                    print(f"  - {label}: {count} samples")
            
            while cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # processa frame com base no modo de treino selecionado
                    if self.mode == "face":
                        results = self.face_detection.process(frame_rgb)
                        if results.detections:
                            for detection in results.detections:
                                self.mp_drawing.draw_detection(frame, detection)
                    else:
                        results = self.hands.process(frame_rgb)
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                self.mp_drawing.draw_landmarks(
                                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # infos do treino
                    status = f"Mode: {self.mode.upper()}"
                    if current_label:
                        status += f" | Label: {current_label}"
                        if current_label in self.samples_per_label:
                            status += f" ({len(self.samples_per_label[current_label])} samples)"
                    if recording:
                        status += " | RECORDING"
                        status += f" | Frame: {frame_count}"
                    
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    if recording and current_label:
                        features = self._extract_landmarks(results, frame if self.mode == "face" else None)
                        if features is not None:
                            if current_label not in self.samples_per_label:
                                self.samples_per_label[current_label] = []
                            self.samples_per_label[current_label].append(features)
                            frame_count += 1
                    
                    cv2.namedWindow('Training')
                    window_size = self.training_config.get('window_size', {'width': 1280, 'height': 720})
                    cv2.resizeWindow('Training', window_size['width'], window_size['height'])
                    cv2.imshow('Training', frame)
                    
                    if cv2.getWindowProperty('Training', cv2.WND_PROP_VISIBLE) < 1:
                        break
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(self.keybinds['quit_training']):
                        break
                    elif key == ord(self.keybinds['start_stop_recording']):
                        if current_label:
                            # verifica se o número máximo de amostras foi alcançado
                            if current_label in self.samples_per_label and len(self.samples_per_label[current_label]) >= self.training_config.get('max_samples_per_label', 1000):
                                print("\n[X] Maximum of 1000 samples reached for this label")
                                continue
                                
                            recording = not recording
                            if not recording:
                                print(f"\n[+] Recorded {frame_count} samples for {current_label}")
                                frame_count = 0
                        else:
                            print("\n[X] Please define a label first using 'n'")
                    elif key == ord(self.keybinds['new_label']):
                        # fecha a janela antes da entrada para evitar problemas de foco
                        cv2.destroyWindow('Training')
                        
                        prompt = "Enter "
                        if self.mode == "gesture":
                            prompt += "gesture label"
                        elif self.mode == "letter":
                            prompt += "letter"
                        elif self.mode == "face":
                            prompt += "face label"
                        else:
                            prompt += "word"
                        label = input(f"\n{prompt}: ").strip()
                        
                        # reabre a janela após a entrada
                        cv2.namedWindow('Training')
                        window_size = self.training_config.get('window_size', {'width': 1280, 'height': 720})
                        cv2.resizeWindow('Training', window_size['width'], window_size['height'])
                        
                        if label:
                            current_label = label
                            if label not in self.labels:
                                self.labels.append(label)
                                self.samples_per_label[label] = []
                                print(f"[+] Added new label: {label}")
                            print(f"[+] Current label set to: {label}")
                            print(f"[+] Current labels: {self.labels}")
                    elif key == ord(self.keybinds['retrain_label']):
                        if not self.labels:
                            print("\n[X] No labels available for retraining")
                            continue
                        
                        # fecha a janela antes da entrada
                        cv2.destroyWindow('Training')
                        
                        print("\nAvailable labels for retraining:")
                        for i, label in enumerate(self.labels):
                            count = len(self.samples_per_label.get(label, []))
                            print(f"{i + 1}. {label} ({count} samples)")
                        
                        try:
                            choice = input("\nEnter the number of the label to retrain (or 0 to cancel): ")
                            
                            # reabre a janela após entrada
                            cv2.namedWindow('Training')
                            window_size = self.training_config.get('window_size', {'width': 1280, 'height': 720})
                            cv2.resizeWindow('Training', window_size['width'], window_size['height'])
                            
                            if choice.isdigit():
                                choice = int(choice)
                                if choice == 0:
                                    print("Retraining cancelled")
                                    continue
                                if 1 <= choice <= len(self.labels):
                                    label = self.labels[choice - 1]
                                    print(f"\n[+] Clearing samples for {label}")
                                    self.samples_per_label[label] = []
                                    current_label = label
                                    recording = False
                                    frame_count = 0
                                    print(f"[+] Ready to record new samples for {label}")
                                    print("[+] Press 's' to start recording")
                                else:
                                    print("[X] Invalid choice")
                            else:
                                print("[X] Please enter a number")
                        except Exception as e:
                            print(f"[X] Error during retraining: {e}")
                    elif key == ord(self.keybinds['info_display']):
                        print("\nTraining information:")
                        print(f"Current mode: {self.mode.upper()}")
                        print("Current labels and samples:")
                        for label in self.labels:
                            count = len(self.samples_per_label.get(label, []))
                            print(f"  - {label}: {count} samples")
                    elif key == ord(self.keybinds['start_training']):
                        if not all(len(samples) >= 1 for samples in self.samples_per_label.values() if samples):
                            print("\n[X] Not enough samples for some labels. Need at least 20 samples per label.")
                            continue
                        
                        if self.mode == "face":
                            # Para o modo face, apenas guardamos os dados sem treinar um modelo TensorFlow
                            # O FaceDetector usará estes dados para treinar o reconhecedor LBPH
                            print("\n[+] Saving face data and labels...")
                            success = self._save_model()
                            if success:
                                print("[+] Face data saved successfully. Use the face recognition mode to test.")
                                print("[!] Note: To activate face recognition, press 'F' and then 'E' twice in the main application")
                            else:
                                print("[X] Error saving face data")
                        else:
                            # Para outros modos, treina normalmente o modelo TensorFlow
                            if self._train_model():
                                self._save_model()
                
                except KeyboardInterrupt:
                    print("\n[!] Training interrupted by user")
                    break
                except Exception as e:
                    print(f"\n[X] Error during training: {e}")
                    continue
            
        except KeyboardInterrupt:
            print("\n[!] Training interrupted by user")
        except Exception as e:
            print(f"\n[X] Error initializing training: {e}")
        finally:
            print("\n[*] Cleaning up...")
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            
            # guarda o progresso atual se houver amostras
            if any(len(samples) > 0 for samples in self.samples_per_label.values()):
                try:
                    print("[*] Saving current progress...")
                    self._save_model()
                    print("[+] Progress saved successfully")
                except Exception as e:
                    print(f"[X] Error saving progress: {e}")
        
    def _extract_landmarks(self, detection_result, frame=None):
        """
        Extrai e normaliza landmarks com base no modo

        Processo:
        1. Extrai pontos de referência
        2. Normaliza coordenadas
        3. Aplica transformações necessárias

        Args:
            detection_result: Resultado da detecção
            frame: Frame da webcam/imagem (opcional)

        Returns:
            list: Vetor de características normalizado
            bool: Sucesso da extração
        """
        try:
            if self.mode == "face":
                if not detection_result.detections:
                    return None
                    
                # deteção da face
                detection = detection_result.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                # converte coordenadas relativas para absolutas
                height, width = frame.shape[:2]
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # extrai região da face e redimensiona para tamanho padrão
                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0:
                    return None
                    
                face_img = cv2.resize(face_img, (100, 100))  # Mesmo tamanho usado no FaceDetector (100x100)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
                # Aplicar equalização de histograma como no FaceDetector
                face_img = cv2.equalizeHist(face_img)
                
                # Guarda a imagem diretamente, sem normalizar
                # O FaceDetector espera imagens sem normalização
                return face_img.flatten()
                
            else:
                if not detection_result.multi_hand_landmarks:
                    return None
                    
                hand_landmarks = detection_result.multi_hand_landmarks[0]
                
                # obtem todas as coordenadas primeiro
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                z_coords = [lm.z for lm in hand_landmarks.landmark]
                
                # calculo da caixa delimitadora
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                min_z, max_z = min(z_coords), max(z_coords)
                
                # calculo dos intervalos com epsilon para evitar divisão por zero
                eps = 1e-6
                x_range = max(max_x - min_x, eps)
                y_range = max(max_y - min_y, eps)
                z_range = max(max_z - min_z, eps)
                
                # normalização das coordenadas em relação à caixa delimitadora
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    # normalização de cada coordenada para o intervalo [0, 1]
                    norm_x = (landmark.x - min_x) / x_range
                    norm_y = (landmark.y - min_y) / y_range
                    norm_z = (landmark.z - min_z) / z_range
                    
                    landmarks.extend([norm_x, norm_y, norm_z])
                
                return np.array(landmarks)
                
        except Exception as e:
            print(f"Error extracting landmarks: {str(e)}")
            raise
        
    def _train_model(self):
        try:
            # prepara os dados do treino
            X = []
            y = []
            for label_idx, label in enumerate(self.labels):
                samples = self.samples_per_label[label]
                X.extend(samples)
                y.extend([label_idx] * len(samples))
            
            X = np.array(X)
            y = np.array(y)
            
            # normaliza os dados do treino
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X = (X - mean) / std
            
            # classificação multiclasse (sempre)
            print(f"Training multiclass model with labels: {self.labels}")
            output_units = len(self.labels)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)  # stratify
            
            # criação e compilação do modelo com arquitetura melhorada
            self.model = keras.Sequential([
                layers.Lambda(lambda x: (x - mean) / std, input_shape=(X.shape[1],)),
                
                # primeiro bloco
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # segundo bloco
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # terceiro bloco
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                # camada de saída
                layers.Dense(output_units, activation='softmax')
            ])
            
            self.model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=0.001,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07
                ),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # callback antecipado com parâmetros melhorados
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # alterado para verificar a precisão
                patience=30,
                restore_best_weights=True,
                min_delta=0.01  # limitador de melhoria de 1%
            )
            
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', 
                factor=0.5,
                patience=10,
                min_lr=0.00001,
                verbose=1
            )
            
            history = self.model.fit(
                X_train, y_train,
                epochs=150,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # avaliação/verificação
            test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
            print(f'\nTest accuracy: {test_acc:.4f}')
            
            self._save_model()
            self._save_history(history.history)
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise
        
    def _save_model(self):
        """
        Guarda o modelo e dados relacionados

        Ficheiros guardados:
        - Modelo treinado (.h5)
        - Labels (JSON)
        - Dados de normalização e treino (NPZ)
        """
        try:
            os.makedirs('models', exist_ok=True)
            
            if self.mode == "face":
                # Para faces, guardamos no formato compatível com FaceDetector
                face_data = []
                face_labels = []
                label_names = {}
                
                for idx, label in enumerate(self.labels):
                    label_names[str(idx)] = label
                    samples = self.samples_per_label.get(label, [])
                    for sample in samples:
                        # Converte o array 1D de volta para imagem 100x100
                        face_img = np.reshape(sample, (100, 100))
                        face_data.append(face_img)
                        face_labels.append(idx)
                
                # Guarda as labels num formato compatível com o FaceDetector
                with open(os.path.join('models', 'face_labels.json'), 'w') as f:
                    json.dump(label_names, f, indent=4)
                
                # Guarda os dados de treino
                if face_data:
                    np.savez(os.path.join('models', 'face_training_data.npz'),
                            data=np.array(face_data),
                            labels=np.array(face_labels))
                
                # Nota: Não guardamos um modelo TensorFlow para faces
                # O FaceDetector usa OpenCV LBPH Recognizer
                print(f"[+] Saved {len(face_data)} face samples and {len(label_names)} labels")
            else:
                # Para gestos/letras/palavras, mantém o comportamento original
                model_file = os.path.join('models', f'{self.mode}_model.h5')
                self.model.save(model_file)
                
                labels_file = os.path.join('models', f'{self.mode}_labels.json')
                with open(labels_file, 'w') as f:
                    json.dump(self.labels, f)
                
                # Guarda também os dados de treino para estatísticas e normalização
                all_data = np.vstack([np.array(self.samples_per_label[label]) for label in self.labels if self.samples_per_label[label]])
                all_labels = np.hstack([[i] * len(self.samples_per_label[label]) for i, label in enumerate(self.labels) if self.samples_per_label[label]])
                
                np.savez(os.path.join('models', f'{self.mode}_training_data.npz'),
                        data=all_data,
                        labels=all_labels,
                        label_names=np.array(self.labels))
                
                print(f"[+] Model saved to {model_file}")
                print(f"[+] Labels saved to {labels_file}")
            
            return True
        except Exception as e:
            print(f"[X] Error saving model: {e}")
            return False
                
    def _save_history(self, history):
        """
        Salva histórico de treino

        Ficheiro salvo:
        1. Histórico de treino (JSON)
        """
        try:
            os.makedirs('logs', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_path = os.path.join('logs', f'history_{timestamp}.json')
            
            with open(history_path, 'w') as f:
                json.dump(history, f)
            print(f"Training history saved to: {history_path}")
            
        except Exception as e:
            print(f"Error saving history: {str(e)}")
            # não levanta exceção aqui por não ser crítico

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RTTSSL Model Trainer')
    parser.add_argument('--mode', choices=['gesture', 'letter', 'word', 'face'],
                      help='Training mode (gesture, letter, word, face)')
    parser.add_argument('--type', choices=['realtime', 'dataset'],
                      help='Training type (realtime, dataset)')
    
    args = parser.parse_args()
    
    mode = args.mode
    training_type = args.type
    
    trainer = ModelTrainer(mode=mode, training_type=training_type)
    trainer.train()
