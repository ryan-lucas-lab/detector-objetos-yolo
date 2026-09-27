import cv2
from ultralytics import YOLO

# Carrega o modelo YOLO já treinado
model = YOLO("yolo26n.pt")

# Abre a câmera
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    conectado, frame = video.read()

    if not conectado:
        print("Não foi possível conectar à câmera")
        break

    # Espelha a imagem
    frame = cv2.flip(frame, 1)

    # Faz a detecção
    deteccao = model(frame, verbose=False)[0]

    # Desenha as detecções no frame
    frame_detectado = deteccao.plot()

    # Mostra o resultado
    cv2.imshow("Deteccao", frame_detectado)

    # Aperte S para sair
    if cv2.waitKey(0) & 0xFF == ord("s"):
        break

video.release()
cv2.destroyAllWindows()