import cv2
import mediapipe as mp
import numpy as np
from actions import ActionController

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

    def classify_face(self, landmarks, w, h):
        """Detecta sorpresa usando MAR (Mouth Aspect Ratio)"""
        # Puntos del labio: 13 (Arriba), 14 (Abajo), 61 (Izq), 291 (Der)
        top = np.array([landmarks[13].x * w, landmarks[13].y * h])
        bot = np.array([landmarks[14].x * w, landmarks[14].y * h])
        left = np.array([landmarks[61].x * w, landmarks[61].y * h])
        right = np.array([landmarks[291].x * w, landmarks[291].y * h])

        height = np.linalg.norm(top - bot)
        width = np.linalg.norm(left - right)

        if width < 1: return "NEUTRAL" # Evitar división por cero

        mar = height / width
        
        # Si la boca es casi la mitad de alta que de ancha -> Sorpresa
        if mar > 0.45: 
            return "SURPRISED"
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
        cap = cv2.VideoCapture(0)

        # --- CONFIGURACIÓN DE UI ---
        # Definimos el nombre de la ventana primero
        window_name = 'Rosh Multimodal Tracker'
        
        # 1. WINDOW_NORMAL permite redimensionar la ventana (vital para Hyprland)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # 2. Establecemos un tamaño inicial grande (ej. 1280x720 o 1920x1080)
        cv2.resizeWindow(window_name, 1280, 720)
        
        # 3. (Opcional) Si la quieres en pantalla completa directamente:
        # cv2.setWindowProperty(window_name, cv2.WINDOW_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # ----------------------------------
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            
            # Espejo para que sea más natural
            image = cv2.flip(image, 1)
            h, w, _ = image.shape
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # --- PROCESAMIENTO ---
            res_hands = self.hands.process(image_rgb)
            res_face = self.face_mesh.process(image_rgb)
            
            hand_gesture = "UNKNOWN"
            face_expression = "NEUTRAL"
            
            # 1. Analizar Mano
            if res_hands.multi_hand_landmarks:
                for hand_lms in res_hands.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(image, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Convertir a pixeles
                    lm_list = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(hand_lms.landmark)]
                    
                    # Usar tu clasificador detallado
                    hand_gesture = self.classify_hand(lm_list)

            # 2. Analizar Cara
            if res_face.multi_face_landmarks:
                for face_lms in res_face.multi_face_landmarks:
                    
                    # --- BLOQUE DE DIBUJO FACIAL ---
                    
                    # A. Dibujar la Malla (Red tecnológica)
                    '''self.mp_draw.draw_landmarks(
                        image=image,
                        landmark_list=face_lms,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )'''

                    # B. Dibujar los Contornos (Ojos, cejas, labios más marcados)
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
                # Mostrar imagen en la esquina superior derecha
                image = self.overlay_image(image, overlay_img, w - 220, 20)
                
                cv2.putText(image, "COMBO!", (w - 200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Info Debug
            status = f"Gesto: {hand_gesture} | Cara: {face_expression}"
            cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, image)
            if cv2.waitKey(5) & 0xFF == 27: break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MultiModalSystem()
    app.run()