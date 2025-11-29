import time
import subprocess
import sys

class ActionController:
    def __init__(self):
        # Estado actual
        self.current_gesture = "UNKNOWN"
        self.gesture_start_time = 0
        self.has_triggered = False  # Evita disparos múltiples mientras mantienes la mano

        # --- CONFIGURACIÓN MATEMÁTICA DE TIEMPOS ---
        
        # 1. HOLD TIME: Tiempo (segundos) que debes MANTENER el gesto para activarlo
        self.hold_map = {
            "THUMB_UP": 4.0,    # Tu petición: 4 segundos para animación
            "THUMB_DOWN": 1.0,  # 1 seg para volver al workspace anterior
            "POINT": 0.5,       # Rápido para cambiar workspace
            "OPEN_PALM": 2.0    # 2 seg para notificación
        }
        
        # 2. POST-COOLDOWN: Tiempo de "descanso" después de una ejecución exitosa
        self.post_cooldown = 2.0 
        self.last_execution_time = 0

    def process_gesture(self, gesture):
        """
        Procesa el gesto actual y gestiona la máquina de estados de tiempo.
        Retorna: Un valor float entre 0.0 y 1.0 indicando el progreso de 'carga'.
        """
        current_time = time.time()

        # A. RESET DE ESTADO
        # Si cambiamos de gesto o perdemos la mano, reseteamos el cronómetro
        if gesture != self.current_gesture:
            self.current_gesture = gesture
            self.gesture_start_time = current_time
            self.has_triggered = False
            return 0.0

        # B. VALIDACIÓN DE COOLDOWN POST-EJECUCIÓN
        # Si acabamos de disparar una acción, ignoramos todo hasta que pase el tiempo de enfriamiento
        if (current_time - self.last_execution_time) < self.post_cooldown:
            return 0.0

        # C. SI EL GESTO ES "UNKNOWN", NO HACEMOS NADA
        if gesture == "UNKNOWN":
            return 0.0

        # D. CÁLCULO DE TIEMPO ACUMULADO (INTEGRACIÓN)
        elapsed_time = current_time - self.gesture_start_time
        required_hold = self.hold_map.get(gesture, 3.0) # Default 3s si no está en lista

        # Calculamos progreso (0.0 a 1.0) para visualización
        progress = min(elapsed_time / required_hold, 1.0)

        # Imprimir barra de carga en terminal (Feedback visual estilo Linux)
        self._print_progress_bar(gesture, progress)

        # E. DISPARO DE ACCIÓN (TRIGGER)
        if elapsed_time >= required_hold and not self.has_triggered:
            self._execute_action(gesture)
            self.has_triggered = True # Bloqueamos para que no se repita en loop
            self.last_execution_time = current_time # Iniciamos cooldown post-acción
            print("\n") # Salto de línea limpio tras la barra de carga

        return progress

    def _print_progress_bar(self, gesture, progress):
        """Dibuja una barra de carga en la terminal"""
        bar_len = 30
        filled_len = int(bar_len * progress)
        bar = '█' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write(f'\r[{bar}] {int(progress * 100)}% : {gesture}   ')
        sys.stdout.flush()

    def _execute_action(self, gesture):
        print(f"\n>>> ¡ACCIÓN EJECUTADA!: {gesture}")
        
        if gesture == "THUMB_UP":
            self.run_animation_script()
            
        elif gesture == "THUMB_DOWN":
            subprocess.run("hyprctl dispatch workspace m-1", shell=True)
            
        elif gesture == "POINT":
            subprocess.run("hyprctl dispatch workspace m+1", shell=True)
            
        elif gesture == "OPEN_PALM":
            subprocess.Popen(['notify-send', 'System', 'Gesto Mantenido Completado'])

    def run_animation_script(self):
        try:
            #subprocess.Popen(['notify-send', '-u', 'critical', 'ANIMATION', 'Starting...'])
            # Aquí va tu script real
             subprocess.Popen(['python3', 'animacion.py'])
        except Exception as e:
            print(f"Error: {e}")

    def run_hyprland_command(self, command):
        """Envía comandos a hyprctl"""
        try:
            # shell=True permite argumentos complejos, pero cuidado con la inyección si usaras inputs externos
            subprocess.run(f"hyprctl {command}", shell=True)
        except Exception as e:
            print(f"Error comunicando con Hyprland: {e}")