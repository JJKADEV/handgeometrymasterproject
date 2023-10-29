import sys
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from cryptography.fernet import Fernet
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QPushButton, QComboBox, QHBoxLayout

DATABASE_DIR = "hand_database"
KEY_PATH = "encryption_key.key"

# Générer une clé de chiffrement et la sauvegarder
def generate_encryption_key():
    key = Fernet.generate_key()
    with open(KEY_PATH, "wb") as key_file:
        key_file.write(key)

# Charger la clé de chiffrement
def load_encryption_key():
    return open(KEY_PATH, "rb").read()

# Chiffrer les données
def encrypt_data(data, key):
    f = Fernet(key)
    encrypted_data = f.encrypt(data.encode())
    return encrypted_data

# Déchiffrer les données
def decrypt_data(encrypted_data, key):
    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data).decode()
    return decrypted_data

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hand(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        landmarks_coords = None
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks_coords = landmarks.landmark
        return landmarks_coords, image

def register_hand(detector, username):
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)

    cap = cv2.VideoCapture(0)
    samples = []
    print("Maintenez votre main immobile pendant quelques secondes pour l'enregistrer....")

    while len(samples) < 100:  # Capture 100 samples
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, frame = detector.detect_hand(frame)
        cv2.namedWindow('Enregistrer la main', cv2.WINDOW_NORMAL)  # Créez une fenêtre redimensionnable
        cv2.moveWindow('Enregistrer la main', 100, 100)  # Déplacez la fenêtre à la position souhaitée
        cv2.resizeWindow('Enregistrer la main', 800, 600)  # Ajustez les dimensions selon vos préférences


        if landmarks:
            samples.append([(landmark.x, landmark.y) for landmark in landmarks])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate the average of the samples
    avg_landmarks = np.mean(samples, axis=0).tolist()

    # Chiffrer les données avant de les sauvegarder
    key = load_encryption_key()
    encrypted_data = encrypt_data(json.dumps(avg_landmarks), key)
    with open(os.path.join(DATABASE_DIR, f"{username}.json"), "wb") as file:
        file.write(encrypted_data)

    print(f"Main de {username} enregistrée avec succès!")

    cap.release()
    cv2.destroyAllWindows()

def authenticate(detector, username):
    filepath = os.path.join(DATABASE_DIR, f"{username}.json")
    if not os.path.exists(filepath):
        print(f"Pas de données trouvées pour {username}")
        return False

    # Déchiffrer les données pour l'authentification
    key = load_encryption_key()
    with open(filepath, "rb") as file:
        encrypted_data = file.read()
    registered_data = np.array(json.loads(decrypt_data(encrypted_data, key)))

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, frame = detector.detect_hand(frame)
        cv2.imshow('Authentification', frame)

        if landmarks:
            current_data = np.array([(landmark.x, landmark.y) for landmark in landmarks])
            diff = np.linalg.norm(registered_data - current_data, axis=1)
            if np.mean(diff) < 0.05:
                print("Authentification réussie!")
                return True
            else:
                print("Échec de l'authentification!")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False

class App(QWidget):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.init_ui()
        self.success_rates = []  # Liste pour stocker les taux de réussite
        self.execution_times = []  # Liste pour stocker les temps d'exécution

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)  # Espacement entre les éléments

        style = """
        QWidget {
            font: 14px;
        }
        QPushButton {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QLineEdit, QComboBox {
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        """
        self.setStyleSheet(style)

        user_layout = QHBoxLayout()
        user_layout.setSpacing(10)
        self.username_label = QLabel('Nom d\'utilisateur:')
        self.username_input = QLineEdit(self)
        user_layout.addWidget(self.username_label)
        user_layout.addWidget(self.username_input)

        action_layout = QHBoxLayout()
        action_layout.setSpacing(10)
        self.action_label = QLabel('Action:')
        self.action_combo = QComboBox(self)
        self.action_combo.addItems(['Enregistrement', 'Authentification'])
        action_layout.addWidget(self.action_label)
        action_layout.addWidget(self.action_combo)

        self.submit_button = QPushButton('Soumettre', self)
        self.submit_button.clicked.connect(self.on_submit)

        main_layout.addLayout(user_layout)
        main_layout.addLayout(action_layout)
        main_layout.addWidget(self.submit_button)

        self.success_plot = pg.PlotWidget(title="Taux de réussite d'authentification")
        self.success_plot.setLabel('left', 'Taux de réussite', units='%')  # Étiquette de l'axe Y
        self.success_plot.setLabel('bottom', 'Utilisateurs')  # Étiquette de l'axe X
        self.success_plot.showGrid(True, True)  # Grille sur le graphique
        main_layout.addWidget(self.success_plot)

        self.execution_time_plot = pg.PlotWidget(title="Temps d'exécution")
        self.execution_time_plot.setLabel('left', 'Temps (s)')  # Étiquette de l'axe Y
        self.execution_time_plot.setLabel('bottom', 'Utilisateurs')  # Étiquette de l'axe X
        self.execution_time_plot.showGrid(True, True)  # Grille sur le graphique
        main_layout.addWidget(self.execution_time_plot)

        self.setLayout(main_layout)
        self.setWindowTitle('Système d\'authentification biométrique')
        self.show()

    def on_submit(self):
        username = self.username_input.text()
        action = self.action_combo.currentText()
        if username:
            if action == 'Enregistrement':
                register_hand(self.detector, username)
            elif action == 'Authentification':
                start_time = time.time()
                result = authenticate(self.detector, username)
                end_time = time.time()
                self.execution_times.append(end_time - start_time)
                self.success_rates.append(100 if result else 0)  # Stockez le taux en pourcentage

        self.update_plots()

    def update_plots(self):
        # Mettez à jour le graphique du taux de réussite
        self.success_plot.clear()
        self.success_plot.plot(self.success_rates, pen='b', symbol='o', symbolSize=10)

        # Mettez à jour le graphique du temps d'exécution
        self.execution_time_plot.clear()
        self.execution_time_plot.plot(self.execution_times, pen='r', symbol='x', symbolSize=10)

if __name__ == '__main__':
    # Vérifier si la clé de chiffrement existe, sinon la générer
    if not os.path.exists(KEY_PATH):
        generate_encryption_key()

    detector = HandDetector()
    app = QApplication(sys.argv)
    ex = App(detector)
    sys.exit(app.exec_())
