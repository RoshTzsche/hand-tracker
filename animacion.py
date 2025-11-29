import matplotlib.pyplot as plt
import numpy as np
import time

def animacion_onda():
    # Configuración de la ventana
    x = np.linspace(0, 4*np.pi, 100)
    plt.ion() # Modo interactivo
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Animación Externa - Rosh System")
    
    line, = ax.plot(x, np.sin(x))
    ax.set_ylim(-1.5, 1.5)
    
    # Simulación de animación por 5 segundos (coincide con tu cooldown aprox)
    start_time = time.time()
    while time.time() - start_time < 5:
        phase = (time.time() - start_time) * 5
        line.set_ydata(np.sin(x + phase))
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01)
    
    plt.close()

if __name__ == "__main__":
    animacion_onda()