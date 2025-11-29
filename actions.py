import time
import os
import cv2

class ActionController:
    def __init__(self):
        # Mapeo de combinaciones: (Gesto Mano, Gesto Cara) -> Nombre de archivo
        self.combo_map = {
            # --- Grupo 1: Básicos (Rostro Neutral) ---
            ("THUMB_UP", "NEUTRAL"): "./images/like.png",
            ("THUMB_DOWN", "NEUTRAL"): "./images/dislike.png",
            ("FIST", "NEUTRAL"): "./images/rock.png",
            ("PEACE", "NEUTRAL"): "./images/peace.png",
            
            # --- Grupo 2: Emociones (Sorpresa) ---
            ("OPEN_PALM", "SURPRISED"): "./images/shocked.png",
            ("POINT", "SURPRISED"): "./images/look_there.png",
            ("PEACE", "SURPRISED"): "./images/party.png",
            
            # --- Grupo 3: Positividad (Sonrisa) ---
            ("THUMB_UP", "SMILE"): "./images/super_like.png",
            ("OPEN_PALM", "SMILE"): "./images/hello.png",
            ("PEACE", "SMILE"): "./images/happy_vibes.png",
            ("POINT", "SMILE"): "./images/idea.png",
            
            # --- Grupo 4: Guiños (Interacción coqueta/secreta) ---
            ("POINT", "WINK_LEFT"): "./images/secret.png",
            ("POINT", "WINK_RIGHT"): "./images/target_locked.png",
            ("FIST", "WINK_LEFT"): "./images/bro_fist.png",
            ("OPEN_PALM", "WINK_RIGHT"): "./images/high_five.png"
        }
        
        # Cache de imágenes cargadas para no leer disco en cada frame
        self.image_cache = {}
        self._load_images()

    def _load_images(self):
        """
        Carga imágenes y normaliza su formato a BGRA (4 canales) 
        para evitar errores de 'not enough values to unpack'.
        """
        for key, filename in self.combo_map.items():
            if os.path.exists(filename):
                # 1. Cargar la imagen sin modificar flags (para detectar alpha si existe)
                img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                
                if img is not None:
                    # 2. Verificar canales
                    # Si es RGB/BGR (3 canales), forzamos conversión a BGRA
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    
                    # 3. Redimensionar para consistencia (opcional, pero recomendado)
                    # Mantenemos un tamaño manejable (ej. 200px ancho)
                    target_w = 200
                    scale = target_w / img.shape[1]
                    target_h = int(img.shape[0] * scale)
                    img = cv2.resize(img, (target_w, target_h))
                    
                    self.image_cache[key] = img
                    print(f"[Sistema] Imagen cargada: {filename} {img.shape}")
                else:
                    print(f"[Error] No se pudo leer la imagen: {filename}")
            else:
                pass 
                # print(f"[Aviso] Falta imagen para combo: {key} -> {filename}")


    def get_overlay_image(self, hand_gesture, face_expression):
        """
        Devuelve la imagen (array numpy) correspondiente a la combinación,
        o None si no hay match.
        """
        key = (hand_gesture, face_expression)
        return self.image_cache.get(key, None)