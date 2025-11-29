# Rosh Multimodal Tracker 🖐️🙂

Un sistema de interacción multimodal avanzado basado en visión por computadora que combina el rastreo de manos (Hand Tracking) y el reconocimiento de expresiones faciales (Face Mesh) para activar eventos y superposiciones visuales en tiempo real.

Este proyecto utiliza **MediaPipe** para la inferencia geométrica y **OpenCV** para el procesamiento de imágenes, diseñado para funcionar eficientemente en entornos Linux (específicamente optimizado para Fedora/Hyprland).

## 🚀 Características Principales

* **Detección Multimodal Simultánea:** Rastrea manos y rostro al mismo tiempo sin pérdida significativa de rendimiento.
* **Sistema de "Combos":** Una arquitectura lógica que mapea pares de `(Gesto Mano, Expresión Facial)` a acciones específicas.
    * *Ejemplo:* Un "Pulgar Arriba" con una "Sonrisa" genera un overlay diferente a un "Pulgar Arriba" con rostro "Neutral".
* **Feedback Visual en Tiempo Real:** Superposición de imágenes (overlays) con soporte de transparencia (Canal Alpha/BGRA).
* **Clasificación Geométrica Personalizada:** Algoritmos propios para determinar estados como "Sorpresa" o "Guiño" basados en distancias euclidianas y proporciones faciales.

## 🛠️ Requisitos del Sistema

* **Sistema Operativo:** Linux (Probado en Fedora 42 con Hyprland).
* **Python:** Versión 3.8 a 3.11.
    * *Nota importante:* El proyecto fue desarrollado y validado en **Python 3.11**. Versiones superiores (3.12+) presentan incompatibilidades con algunas dependencias (específicamente `mediapipe`/`distutils`) a fecha de Noviembre 2025.
* **Hardware:** Webcam funcional.

## 📦 Instalación

Sigue estos pasos para configurar el entorno desde cero:

### 1. Clonar el Repositorio
```bash
git clone [https://github.com/tu-usuario/hand-tracker.git](https://github.com/tu-usuario/hand-tracker.git)
cd hand-tracker
````

### 2\. Crear Entorno Virtual (Recomendado)

Para mantener las dependencias aisladas de tu sistema principal (Fedora):

```bash
python3 -m venv venv_gestos
source venv_gestos/bin/activate
```

### 3\. Instalar Dependencias

Instala las librerías necesarias ejecutando:

```bash
pip install opencv-python mediapipe numpy matplotlib
```

### 4\. ⚠️ Configuración de Recursos (CRÍTICO)

El sistema requiere una carpeta específica para los recursos gráficos que **no está incluida en el repositorio** por defecto. Debes crearla manualmente y añadir tus imágenes.

1.  Crea la carpeta `images` en la raíz del proyecto:

    ```bash
    mkdir images
    ```

2.  Añade archivos `.png` dentro de esa carpeta. Para que el sistema funcione, los nombres de archivo deben coincidir con los definidos a continuación (o puedes modificar las rutas en `actions.py`). Asegúrate de tener las siguientes imágenes:

      * **Básicos:** `like.png`, `dislike.png`, `rock.png`, `peace.png`
      * **Emociones:** `shocked.png`, `look_there.png`, `party.png`
      * **Positividad:** `super_like.png`, `hello.png`, `happy_vibes.png`, `idea.png`
      * **Guiños:** `secret.png`, `target_locked.png`, `bro_fist.png`, `high_five.png`

> **Nota:** El sistema normalizará automáticamente las imágenes a formato BGRA y las redimensionará, pero es recomendable usar imágenes PNG con fondo transparente para un mejor efecto visual.

## 📐 Fundamentos Técnicos (Desglose Matemático)

El núcleo de la clasificación no depende de redes neuronales de "caja negra" para la clasificación final, sino de **geometría analítica** aplicada sobre los *landmarks* extraídos por MediaPipe.

### 1\. Clasificación de Manos (Lógica Vectorial)

Para determinar si un dedo está levantado, no usamos aprendizaje profundo, sino la comparación de distancias euclidianas cuadráticas ($d^2$) para evitar el costo computacional de las raíces cuadradas en cada frame.

Sea $P_{wrist}$ la muñeca, $P_{tip}$ la punta del dedo y $P_{pip}$ la articulación intermedia:
$$d^2(P_{wrist}, P_{tip}) > d^2(P_{wrist}, P_{pip}) \implies \text{Dedo Levantado}$$

### 2\. Detección de Sorpresa (MAR - Mouth Aspect Ratio)

Para detectar una boca abierta (sorpresa), calculamos la relación de aspecto de la boca utilizando la distancia euclidiana:

$$MAR = \frac{||P_{top} - P_{bottom}||}{||P_{left} - P_{right}||}$$

Donde $|| \cdot ||$ es la norma euclidiana. Si $MAR > 0.45$, se clasifica como `SURPRISED`.

### 3\. Detección de Guiños (EAR - Eye Aspect Ratio)

Utilizamos la métrica estándar EAR para determinar la apertura del ojo. Se consideran 6 puntos de referencia por ojo ($p_1 \dots p_6$):

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

El sistema detecta un guiño intencional comparando los EAR de ambos ojos:

$$ \text{Si} (EAR_{left} < 0.2 \land EAR_{right} > 0.2) \implies \text{WINK\_LEFT}$$

## 🎮 Uso

Para iniciar el sistema principal de rastreo:

```bash
python tracker.py
```

### Controles

  * **ESC:** Cerrar la ventana y terminar el programa.

## ⚙️ Configuración Avanzada

### Selección de Cámara

El archivo `tracker.py` intenta localizar una cámara específica por su ID de hardware (`/dev/v4l/by-id/...`) para evitar problemas en sistemas con múltiples dispositivos de video en Linux.

Si tu cámara no es detectada, edita la línea en `tracker.py`:

```python
# Cambia esto por el índice de tu cámara (generalmente 0 o 1)
stable_path = "/ruta/a/tu/camara" 
# O fuerza el índice directamente en cv2.VideoCapture(0)
```

## 📂 Estructura del Proyecto

```text
hand-tracker/
├── actions.py       # Controlador de lógica de combos y carga de imágenes
├── tracker.py       # Punto de entrada principal (Loop de visión)
├── images/          # [TÚ DEBES CREAR ESTO] Carpeta de recursos PNG
├── .gitignore       # Configuración de exclusión de git
└── README.md        # Documentación
```

## 🤝 Contribución

Si deseas agregar nuevos combos, edita el diccionario `self.combo_map` en `actions.py` y añade la imagen correspondiente en la carpeta `images/`.

```python
# Ejemplo de nuevo combo
("FIST", "SMILE"): "./images/power_up.png",
```

-----

*Desarrollado por Rosh.*

