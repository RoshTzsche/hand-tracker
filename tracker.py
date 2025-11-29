import cv2
import mediapipe as mp
import numpy as np
from actions import ActionController
import os
import re 



class MultiModalSystem:
    def __init__(self):
        # 1. Configuración de MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
        
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 2. Controlador de Acciones
        self.controller = ActionController()

    def _is_finger_up(self, lm_list, finger_tip_idx, finger_pip_idx):
        """
        Lógica geométrica original: Compara distancias euclidianas al cuadrado
        desde la muñeca (P0) a la punta (Tip) vs al nudillo (PIP).
        """
        wrist = lm_list[0]
        tip = lm_list[finger_tip_idx]
        pip = lm_list[finger_pip_idx]

        # Distancia al cuadrado (d^2 = dx^2 + dy^2)
        dist_wrist_tip = (tip[1]-wrist[1])**2 + (tip[2]-wrist[2])**2
        dist_wrist_pip = (pip[1]-wrist[1])**2 + (pip[2]-wrist[2])**2

        return dist_wrist_tip > dist_wrist_pip

    def classify_hand(self, lm_list):
        """
        TU CLASIFICADOR ORIGINAL COMPLETO.
        Distingue con precisión entre Puño, Pulgar Arriba y Pulgar Abajo.
        """
        if not lm_list:
            return "UNKNOWN"

        # Coordenadas clave del Pulgar
        thumb_tip = lm_list[4]
        thumb_mcp = lm_list[2] # Nudillo base del pulgar
        
        # Índices de los otros 4 dedos
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        # Análisis de dedos levantados usando la función auxiliar
        fingers_up = []
        for tip, pip in zip(finger_tips, finger_pips):
            fingers_up.append(self._is_finger_up(lm_list, tip, pip))
            
        total_fingers_up = fingers_up.count(True)

        # --- LÓGICA DE DECISIÓN ---
        
        # 1. Puño o Variantes de Pulgar (0 dedos principales levantados)
        if total_fingers_up == 0:
            # Umbral de píxeles para considerar el pulgar arriba/abajo
            # Ajuste de sensibilidad (puedes cambiar 20 por otro valor)
            threshold = 20 
            
            # Pulgar Arriba: La punta está significativamente más ARRIBA (menor Y) que el nudillo
            if thumb_tip[2] < thumb_mcp[2] - threshold:
                return "THUMB_UP"
            
            # Pulgar Abajo: La punta está significativamente más ABAJO (mayor Y) que el nudillo
            elif thumb_tip[2] > thumb_mcp[2] + threshold:
                return "THUMB_DOWN"
            
            else:
                return "FIST"

        # 2. Palma Abierta (4 o 5 dedos)
        if total_fingers_up >= 4:
            return "OPEN_PALM"

        # 3. Amor y Paz (Índice y Medio)
        if fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
            return "PEACE"

        # 4. Señalar (Solo Índice)
        if fingers_up[0] and not fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
            return "POINT"

        return "UNKNOWN"

    def _get_euclidean_distance(self, p1, p2):
        """Calcula distancia euclidiana entre dos puntos (x, y)"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def _calculate_ear(self, landmarks, indices, w, h):
        """Calcula el Eye Aspect Ratio (Relación de Aspecto del Ojo)"""
        # Puntos convertidos a coordenadas de pixel
        coords = []
        for idx in indices:
            lm = landmarks[idx]
            coords.append((lm.x * w, lm.y * h))
        
        # Distancias verticales
        d_v1 = self._get_euclidean_distance(coords[1], coords[5])
        d_v2 = self._get_euclidean_distance(coords[2], coords[4])
        
        # Distancia horizontal
        d_h = self._get_euclidean_distance(coords[0], coords[3])
        
        if d_h == 0: return 0.0
        
        # Fórmula EAR
        return (d_v1 + d_v2) / (2.0 * d_h)

    def classify_face(self, landmarks, w, h):
        """
        Clasificador Facial Expandido: 
        Detecta: NEUTRAL, SURPRISED, SMILE, WINK_LEFT, WINK_RIGHT
        """
        # --- 1. Lógica de Sorpresa (MAR) ---
        # Labios: 13, 14, 61, 291
        top = (landmarks[13].x * w, landmarks[13].y * h)
        bot = (landmarks[14].x * w, landmarks[14].y * h)
        left = (landmarks[61].x * w, landmarks[61].y * h)
        right = (landmarks[291].x * w, landmarks[291].y * h)

        mouth_h = self._get_euclidean_distance(top, bot)
        mouth_w = self._get_euclidean_distance(left, right)
        
        if mouth_w == 0: return "NEUTRAL"
        mar = mouth_h / mouth_w

        if mar > 0.45:
            return "SURPRISED"

        # --- 2. Lógica de Guiños (EAR) ---
        # Índices MediaPipe para ojos (P1, P2, P3, P4, P5, P6)
        # Ojo Izquierdo (desde la perspectiva de la cámara en espejo es el DERECHO real del usuario)
        # Nota: Ajusta left/right según si usas espejo o no. Asumimos espejo.
        left_eye_indices = [362, 385, 387, 263, 373, 380] 
        right_eye_indices = [33, 160, 158, 133, 144, 153]

        ear_left = self._calculate_ear(landmarks, left_eye_indices, w, h)
        ear_right = self._calculate_ear(landmarks, right_eye_indices, w, h)
        
        blink_thresh = 0.2 # Umbral experimental

        # Lógica exclusiva: Para ser guiño, un ojo cerrado y el otro abierto
        if ear_left < blink_thresh and ear_right > blink_thresh:
            return "WINK_LEFT" # Ojo izquierdo en pantalla cerrado
        if ear_right < blink_thresh and ear_left > blink_thresh:
            return "WINK_RIGHT"

        # --- 3. Lógica de Sonrisa ---
        # Una sonrisa suele ensanchar la boca sin abrirla tanto verticalmente como la sorpresa
        # Usamos mouth_w relativo al ancho de la cara para normalizar
        # Puntos de la cara extremos: 234 (oreja derecha pantalla), 454 (oreja izq pantalla)
        face_left = (landmarks[234].x * w, landmarks[234].y * h)
        face_right = (landmarks[454].x * w, landmarks[454].y * h)
        face_width = self._get_euclidean_distance(face_left, face_right)

        if face_width > 0:
            smile_ratio = mouth_w / face_width
            # Si la boca ocupa más del 40% del ancho de la cara y no es sorpresa
            if smile_ratio > 0.42 and mar < 0.3: 
                return "SMILE"
        print(f"DEBUG: Ratio={smile_ratio:.2f} | MAR={mar:.2f}")
        return "NEUTRAL"
    def overlay_image(self, background, overlay, x, y):
        """Superpone imagen con transparencia de forma segura"""
        if overlay is None or background is None: return background
        
        h_ov, w_ov = overlay.shape[:2]
        h_bg, w_bg = background.shape[:2]
        
        # Clipping: Asegurar que la imagen no se salga del cuadro
        if x < 0: x = 0
        if y < 0: y = 0
        if x + w_ov > w_bg: w_ov = w_bg - x
        if y + h_ov > h_bg: h_ov = h_bg - y
        
        if w_ov <= 0 or h_ov <= 0: return background

        # Recorte de las regiones de interés
        overlay_crop = overlay[:h_ov, :w_ov]
        bg_slice = background[y:y+h_ov, x:x+w_ov]

        # Separación de canales segura
        # Como actions.py garantiza BGRA, esto ya no fallará
        b, g, r, a = cv2.split(overlay_crop)
        
        # Normalizar alpha a 0-1
        mask = a / 255.0
        mask_inv = 1.0 - mask
        
        # Composición Alpha Blending vectorial
        for c in range(3): # B, G, R del fondo
            bg_slice[:, :, c] = (mask * overlay_crop[:, :, c] + 
                               mask_inv * bg_slice[:, :, c])
            
        background[y:y+h_ov, x:x+w_ov] = bg_slice
        return background

    def run(self):
        # 1. RESOLUCIÓN DE HARDWARE (Mapeo de Symlink a Índice)
        # Tu ruta estable específica:
        stable_path = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd_USB_2.0_Camera_SN0001-video-index0"
        camera_index = 0 # Fallback por defecto

        if os.path.exists(stable_path):
            # Resolvemos a dónde apunta el enlace (ej: /dev/video2)
            real_path = os.path.realpath(stable_path)
            try:
                # Extraemos el número final de la cadena 'videoX'
                camera_index = int(re.findall(r'\d+', real_path)[-1])
                print(f"Cámara detectada en: {real_path} (Índice {camera_index})")
            except IndexError:
                print("No se pudo extraer el índice, usando 0 por defecto.")
        else:
            print(f"No se encontró la cámara Sonix en {stable_path}, usando índice 0.")

        # 2. INICIALIZACIÓN DEL TRANSDUCTOR
        # Usamos CAP_V4L2 explícitamente para mejor rendimiento en Fedora/Hyprland
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

        cap.set(cv2.CAP_PROP_FPS, 20)

        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
       
        # --- CONFIGURACIÓN DE UI ---
        window_name = 'Rosh Multimodal Tracker'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignorando frame vacío de la cámara.")
                continue
            
            # Espejo para que sea más natural
            image = cv2.flip(image, 1)
            h, w, _ = image.shape
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # --- PROCESAMIENTO ---
            # Bloqueamos escritura en image_rgb para mejorar rendimiento en memoria
            image_rgb.flags.writeable = False
            res_hands = self.hands.process(image_rgb)
            res_face = self.face_mesh.process(image_rgb)
            image_rgb.flags.writeable = True
            
            hand_gesture = "UNKNOWN"
            face_expression = "NEUTRAL"
            
            # 1. Analizar Mano
            if res_hands.multi_hand_landmarks:
                for hand_lms in res_hands.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(image, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    lm_list = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(hand_lms.landmark)]
                    hand_gesture = self.classify_hand(lm_list)

            # 2. Analizar Cara
            if res_face.multi_face_landmarks:
                for face_lms in res_face.multi_face_landmarks:
                    # Dibujar contornos (más ligero visualmente que la malla completa)
                    self.mp_draw.draw_landmarks(
                        image=image,
                        landmark_list=face_lms,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    face_expression = self.classify_face(face_lms.landmark, w, h)

            # --- LÓGICA DE ACCIÓN (OVERLAY) ---
            overlay_img = self.controller.get_overlay_image(hand_gesture, face_expression)
            
            if overlay_img is not None:
                image = self.overlay_image(image, overlay_img, w - 220, 20)
                cv2.putText(image, "COMBO!", (w - 200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Info Debug
            status = f"Gesto: {hand_gesture} | Cara: {face_expression}"
            cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, image)
            if cv2.waitKey(5) & 0xFF == 27: # ESC para salir
                break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MultiModalSystem()
    app.run()
