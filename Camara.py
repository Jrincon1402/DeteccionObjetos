import cv2
import numpy as np

def main():
    # Iniciar la captura de video desde la cámara (ajusta el índice según la cámara correcta)
    cap = cv2.VideoCapture(2)

    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    # Aumentar la resolución de la cámara al máximo soportado
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Ajustar a 1920 píxeles de ancho (Full HD)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Ajustar a 1080 píxeles de alto (Full HD)

    # Verificar si la resolución se configuró correctamente
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Resolución actual de la cámara: {int(actual_width)}x{int(actual_height)}")

    # Definir el rango de colores en HSV para detectar el color verde de la tarjeta
    lower_bound = np.array([38, 50, 50])   # Límite inferior en HSV para el verde de la tarjeta
    upper_bound = np.array([75, 255, 255]) # Límite superior en HSV para el verde de la tarjeta

    # Crear una ventana para el video que ocupe la pantalla completa pero sea una ventana redimensionada
    cv2.namedWindow("Seguimiento de tarjeta", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Seguimiento de tarjeta", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seguimiento de tarjeta", 1920, 1080)  # Ajustar resolución de la ventana

    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break

        # Convertir el frame de BGR a HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Crear una máscara para el rango de colores seleccionado (verde)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Procesar la máscara para reducir ruido (aplicar dilatación y erosión)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Encontrar contornos basados en la máscara
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Si se encontraron contornos, seguir el más grande (posible tarjeta)
        if len(contours) > 0:
            # Elegir el contorno más grande
            largest_contour = max(contours, key=cv2.contourArea)

            # Calcular el rectángulo que encierra el contorno
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Dibujar el rectángulo alrededor del objeto detectado
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar el frame original con el seguimiento de la tarjeta
        cv2.imshow("Seguimiento de tarjeta", frame)

        # Mostrar la máscara de color (solo para referencia)
        cv2.imshow("Máscara", mask)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
