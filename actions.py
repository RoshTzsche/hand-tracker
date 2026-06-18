################
## RoshTzsche ##
################
import os
import cv2

class ActionController:
    def __init__(self):
        # Dynamically resolve the absolute path to the images directory
        # so the script works regardless of where it is launched from
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.images_dir = os.path.join(base_dir, "images")

        # Combo map: (Hand Gesture, Face Expression) -> PNG file path
        # Add new entries here to extend the system with new combos
        self.combo_map = {
            # --- Group 1: Basics (Neutral Face) ---
            ("THUMB_UP",   "NEUTRAL"):   os.path.join(self.images_dir, "like.png"),
            ("THUMB_DOWN", "NEUTRAL"):   os.path.join(self.images_dir, "dislike.jpg"),
            ("FIST",       "NEUTRAL"):   os.path.join(self.images_dir, "bro_fist.png"),
            ("PEACE",      "NEUTRAL"):   os.path.join(self.images_dir, "peace.jpg"),
 ("FIST",   "SURPRISED"):   os.path.join(self.images_dir, "babyfist.png"),
            # --- Group 2: Surprise Combos ---
            ("OPEN_PALM",  "SURPRISED"): os.path.join(self.images_dir, "shocked.png"),
            ("POINT",      "SURPRISED"): os.path.join(self.images_dir, "look_there.png"),
            ("PEACE",      "SURPRISED"): os.path.join(self.images_dir, "party.png"),

            # --- Group 3: Positive / Smile Combos ---
            ("THUMB_UP",   "SMILE"):     os.path.join(self.images_dir, "like_mouth.jpg"),
            ("OPEN_PALM",  "SMILE"):     os.path.join(self.images_dir, "palm_smile.png"),
            ("POINT",      "SMILE"):     os.path.join(self.images_dir, "point_neutral.png"),
            ("PEACE",      "SMILE"):     os.path.join(self.images_dir, "peace_smile.png"),

            ("POINT",      "SMILE"):     os.path.join(self.images_dir, "point_neutral.png"),

            ("POINT",      "NEUTRAL"):     os.path.join(self.images_dir, "nerd.jpg"),
            # --- Group 4: Wink / Secret Combos ---
            ("POINT",      "WINK_LEFT"):  os.path.join(self.images_dir, "nerd.jpg"),
            ("POINT",      "WINK_RIGHT"): os.path.join(self.images_dir, "nerd.jpg"),
            ("FIST",       "WINK_LEFT"):  os.path.join(self.images_dir, "bro_fist.png"),
            ("OPEN_PALM",  "WINK_RIGHT"): os.path.join(self.images_dir, "high_five.png"),
        }

        # In-memory image cache — avoids reading from disk on every frame
        self.image_cache = {}
        self._load_images()

    def _load_images(self):
        """
        Pre-loads all combo images and normalizes them to BGRA (4 channels).
        This prevents 'not enough values to unpack' errors at runtime when
        OpenCV reads a 3-channel PNG without an alpha channel.
        """
        if not os.path.exists(self.images_dir):
            print(f"[Critical Error] Images folder not found at: {self.images_dir}")
            print(f"[Critical Error] Please create it and add the required PNG files.")
            return

        for key, filepath in self.combo_map.items():
            if not os.path.exists(filepath):
                # Skip silently — missing images are non-fatal; the combo simply won't fire
                continue

            # Load with UNCHANGED flag to preserve alpha channel if present
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"[Warning] Could not load image: {filepath}")
                continue

            # Normalize to 4-channel BGRA so overlay logic always receives the same shape
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            elif len(img.shape) == 2:
                # Grayscale image — convert to BGRA
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

            # Resize to a standard width of 200 px, preserving aspect ratio
            target_w = 200
            scale = target_w / img.shape[1]
            target_h = int(img.shape[0] * scale)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            self.image_cache[key] = img
            print(f"[System] Loaded: {os.path.basename(filepath)}  shape={img.shape}")

    def get_overlay_image(self, hand_gesture: str, face_expression: str):
        """
        Returns the numpy image array for the given (hand, face) combo,
        or None if the combination has no registered overlay.
        """
        return self.image_cache.get((hand_gesture, face_expression), None)
