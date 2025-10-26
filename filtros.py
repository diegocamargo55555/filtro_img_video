import cv2
import numpy as np
from ultralytics import YOLO
import os
import pygame

try:
    pygame.mixer.init()
    caminho_musica = os.path.join('assets', 'musica.mp3')
    if os.path.exists(caminho_musica):
        pygame.mixer.music.load(caminho_musica)
        print(f"Música carregada: {caminho_musica}")
        music_loaded = True
    else:
        print(f"Arquivo de música não encontrado em: {caminho_musica}")
        music_loaded = False
except pygame.error as e:
    print(f"Erro ao inicializar o pygame.mixer ou carregar a música: {e}")
    music_loaded = False


model = YOLO('yolov8n.pt')
coco_names = model.names

def converter_para_cinza(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image

def converter_para_negativo(img):
    negative_img = cv2.bitwise_not(img)
    return negative_img

def aplicar_otsu(img):
    if len(img.shape) == 3:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img

    threshold_value, otsu_image = cv2.threshold(
        gray_image, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return otsu_image, threshold_value

def suavizar_pela_media(img, kernel_size):
    blurred_image = cv2.blur(img, (kernel_size, kernel_size))
    return blurred_image

def suavizar_pela_mediana(img, kernel_size):
    median_image = cv2.medianBlur(img, kernel_size)
    return median_image

def detectar_bordas_canny(img, t_lower=100, t_upper=200):
    if len(img.shape) == 3:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_edges = cv2.Canny(blurred_image, t_lower, t_upper)

    return canny_edges

def aplicar_erosao(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erode_image = cv2.erode(img, kernel, iterations=1)
    return erode_image

def aplicar_dilatacao(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilate_image = cv2.dilate(img, kernel, iterations=1)
    return dilate_image

def aplicar_abertura(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    open_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return open_image

def aplicar_fechamento(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    close_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return close_image

def detectar_com_yolo(img, classes_desejadas):
    results = model(img)
    dog_detected_in_frame = False 

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in classes_desejadas:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_name = coco_names[cls_id]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{cls_name} {conf:.2f}'
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if cls_id == 16: 
                    dog_detected_in_frame = True

    if music_loaded: 
        try:
            is_playing = pygame.mixer.music.get_busy()

            if dog_detected_in_frame and not is_playing:
                pygame.mixer.music.play(-1) 
                print("Cachorro detectado, tocando música...")
            elif not dog_detected_in_frame and is_playing:
                pygame.mixer.music.stop()
                print("Cachorro não detectado, parando música...")
        except pygame.error as e:
            print(f"Erro durante controle da música: {e}")

    return img

def detectar_pessoas(img):
    return detectar_com_yolo(img, classes_desejadas=[0]) 

def detectar_cachorros(img):
    return detectar_com_yolo(img, classes_desejadas=[16]) 

def gerar_histograma(img): 
    hist_height = 256 
    hist_width = 512 
    bin_width = int(round(hist_width / 256)) 

    if len(img.shape) == 3: 
        hist_image = np.zeros((hist_height * 2, hist_width, 3), dtype=np.uint8) 

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, color in enumerate(colors): 
            hist = cv2.calcHist([img], [i], None, [256], [0, 256]) 
            cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX) 

            for j in range(256): 
                cv2.line(hist_image, 
                        (bin_width * j, hist_height), 
                        (bin_width * j, hist_height - int(hist[j])), 
                        color, 
                        thickness=bin_width) 

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]) 
        cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX) 

        y_base_gray = hist_height * 2 

        for i in range(256): 
            cv2.line(hist_image, 
                    (bin_width * i, y_base_gray), 
                    (bin_width * i, y_base_gray - int(hist[i])), 
                    (255, 255, 255), 
                    thickness=bin_width) 

    else: 
        hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8) 

        hist = cv2.calcHist([img], [0], None, [256], [0, 256]) 
        cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX) 

        for i in range(256): 
            cv2.line(hist_image, 
                    (bin_width * i, hist_height), 
                    (bin_width * i, hist_height - int(hist[i])), 
                    (255, 255, 255), 
                    thickness=bin_width) 

    return hist_image 

def calcular_metricas_imagem(img): 
    height, width = img.shape[:2] 
    area = width * height 
    perimeter = 2 * (width + height) 
    diameter = np.sqrt(width**2 + height**2) 

    return area, perimeter, diameter 

def contagem_por_regiao(img): 
    if len(img.shape) == 3: 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    else: 
        img_gray = img 

    _, binary_image = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

    num_white_pixels = np.sum(binary_image == 255) 
    num_black_pixels = np.sum(binary_image == 0) 

    if num_white_pixels > num_black_pixels: 
        binary_image = cv2.bitwise_not(binary_image) 

    num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8) 

    num_objects = num_labels - 1 

    return num_objects 

