import cv2
import mediapipe as mp
import math


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hand(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        digit_ratio = None

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, landmarks, self.mp_hands.HAND_CONNECTIONS)
                # Calculer le Digit Ratio (2D:4D)
                index_finger_length = math.dist([landmarks.landmark[5].x, landmarks.landmark[5].y], [landmarks.landmark[8].x, landmarks.landmark[8].y])
                ring_finger_length = math.dist([landmarks.landmark[13].x, landmarks.landmark[13].y], [landmarks.landmark[16].x, landmarks.landmark[16].y])
                digit_ratio = index_finger_length / ring_finger_length
                cv2.putText(image, f'Digit Ratio (2D:4D): {digit_ratio:.4f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        return image


def main(image_path):
    detector = HandDetector()
    image = cv2.imread(image_path)
    image = detector.detect_hand(image)

    # Redimensionner l'image pour l'affichage
    screen_res = 1280, 720
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)

    cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Detection', window_width, window_height)

    cv2.imshow('Hand Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = 'C:\contourmain\my_hand.jpeg'
    main(image_path)