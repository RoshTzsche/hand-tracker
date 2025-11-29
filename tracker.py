import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from actions import *

class HandEmojiSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Mapeo de gestos a Emojis
        self.emojis = {
            "THUMB_UP": "👍",
            "THUMB_DOWN": "👎",
            "OPEN_PALM": "🖐️",
            "FIST": "✊",
            "PEACE": "✌️",
            "POINT": "☝️",
            "UNKNOWN": ""
        }

    def _is_finger_up(self, lm_list, finger_tip_idx, finger_pip_idx):
        """
        Calcula si un dedo (no pulgar) está levantado basándose en la distancia a la muñeca (P0).
        Utiliza el teorema de pitágoras en 2D (plano imagen) para eficiencia.
        """
        # P0 es la muñeca
        wrist = lm_list[0]
        tip = lm_list[finger_tip_idx]
        pip = lm_list[finger_pip_idx]

        # Distancia euclidiana al cuadrado (evitamos raíz cuadrada para optimizar)
        dist_wrist_tip = (tip[1]-wrist[1])**2 + (tip[2]-wrist[2])**2
        dist_wrist_pip = (pip[1]-wrist[1])**2 + (pip[2]-wrist[2])**2

        return dist_wrist_tip > dist_wrist_pip

    def classify_gesture(self, lm_list):
        """
        Clasifica el gesto basado en la configuración geométrica de los dedos.
        lm_list: Lista de landmarks [id, x, y]
        """
        if not lm_list:
            return "UNKNOWN"

        fingers = []
        
        # --- Lógica del Pulgar ---
        # El pulgar es especial. Comprobamos su posición x relativa al nudillo para saber si está abierto horizontalmente
        # O su posición Y para pulgar arriba/abajo.
        
        # Para simplificar: Checamos si la punta del pulgar está a la derecha o izquierda del nudillo
        # Dependiendo de la mano (asumiremos mano derecha para la lógica básica, o lógica relativa)
        # Aquí usaremos una lógica de "Pulgar Arriba/Abajo" estricta basada en el eje Y
        
        thumb_tip = lm_list[4]
        thumb_ip = lm_list[3]
        thumb_mcp = lm_list[2]
        
        # Estado de los 4 dedos restantes (Índice, Medio, Anular, Meñique)
        # Tips: 8, 12, 16, 20 | PIPs: 6, 10, 14, 18
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        fingers_up = []
        for tip, pip in zip(finger_tips, finger_pips):
            fingers_up.append(self._is_finger_up(lm_list, tip, pip))
            
        total_fingers_up = fingers_up.count(True)

        # --- Detección de Gestos ---
        
        # 1. Puño (0 dedos levantados)
        if total_fingers_up == 0:
            # Diferenciar entre Puño, Pulgar Arriba y Pulgar Abajo
            
            # Pulgar Arriba: Punta del pulgar considerablemente ARRIBA del nudillo (menor valor Y)
            if thumb_tip[2] < thumb_mcp[2] - 20: # -20 es un umbral de píxeles
                return "THUMB_UP"
            
            # Pulgar Abajo: Punta del pulgar considerablemente ABAJO del nudillo
            elif thumb_tip[2] > thumb_mcp[2] + 20:
                return "THUMB_DOWN"
            
            else:
                return "FIST"

        # 2. Palma Abierta (4 o 5 dedos)
        if total_fingers_up >= 4:
            return "OPEN_PALM"

        # 3. Amor y Paz (Solo Índice y Medio)
        if fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
            return "PEACE"

        # 4. Señalar (Solo Índice)
        if fingers_up[0] and not fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
            return "POINT"

        return "UNKNOWN"

    def draw_emoji(self, img, gesture):
        if gesture == "UNKNOWN":
            return img
        
        emoji = self.emojis.get(gesture, "")
        
        # Convertir OpenCV (BGR) a PIL (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Cargar fuente (intenta usar una fuente del sistema que tenga emojis, como Noto Color Emoji en Linux)
        try:
            # En Fedora, Noto Color Emoji suele estar aquí. Si falla, usa default.
            font = ImageFont.truetype("default", 80)
        except IOError:
            font = ImageFont.load_default()

        # Dibujar emoji
        draw.text((50, 50), emoji, font=font, fill=(255, 255, 255))
        
        # Convertir de vuelta a OpenCV
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Instanciamos el controlador modificado (asegúrate de tener el nuevo actions.py)
        controller = ActionController()

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # MediaPipe necesita RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            # Por defecto, si no detecta nada, el gesto es UNKNOWN
            gesture = "UNKNOWN"
            lm_list = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar esqueleto de la mano
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extraer coordenadas en píxeles
                    h, w, c = image.shape
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([id, cx, cy])
                    
                    # Analizar gesto solo si tenemos landmarks
                    if lm_list:
                        gesture = self.classify_gesture(lm_list)

            # --- Lógica de Control (Hold to Trigger) ---
            # Obtenemos el progreso de carga (0.0 a 1.0) del controlador
            charge_progress = controller.process_gesture(gesture)

            # Dibujar el emoji del gesto detectado
            image = self.draw_emoji(image, gesture)
            
            # --- Visualización de la Barra de Carga ---
            # Solo dibujamos si hay progreso y no es un gesto desconocido
            if charge_progress > 0 and gesture != "UNKNOWN":
                # Coordenadas y dimensiones de la barra
                bar_x, bar_y = 50, 150
                bar_w, bar_h = 200, 20
                
                # 1. Fondo de la barra (gris oscuro)
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                
                # 2. Relleno dinámico (Interpolación de color: Verde -> Amarillo -> Rojo)
                fill_width = int(bar_w * charge_progress)
                
                # Calcular color (BGR): Verde al inicio, Rojo al final
                green = int(255 * (1 - charge_progress))
                blue = 0
                red = int(255 * charge_progress)
                color = (blue, green, red) 
                
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_h), color, -1)
                
                # 3. Texto de porcentaje
                cv2.putText(image, f"Hold: {int(charge_progress*100)}%", (bar_x, bar_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Mostrar Gesto en texto plano (debug)
            cv2.putText(image, f"Gesto: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Detector de Gestos Rosh', image)

            if cv2.waitKey(5) & 0xFF == 27: # Esc para salir
                break
        
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    app = HandEmojiSystem()
    app.run()