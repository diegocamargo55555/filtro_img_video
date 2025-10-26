import cv2
import numpy as np  
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
coco_names = model.names


person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
animal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

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
    # Roda a detecção do YOLO
    results = model(img)
    
    # Itera pelos resultados
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Pega o ID da classe
            cls_id = int(box.cls[0])
            
            # Verifica se a classe é uma das que queremos detectar
            if cls_id in classes_desejadas:
                # Pega as coordenadas da caixa
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Pega a confiança (score)
                conf = float(box.conf[0])
                # Pega o nome da classe
                cls_name = coco_names[cls_id]
                
                # Desenha o retângulo
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Escreve o texto (label)
                label = f'{cls_name} {conf:.2f}'
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    return img


# --- MODIFICAR A FUNÇÃO 'detectar_pessoas' ---
def detectar_pessoas(img):
    # Remove o código antigo do Haar Cascade
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    # pessoas = person_cascade.detectMultiScale(gray_image, 1.1, 4)
    # for (x, y, w, h) in pessoas:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # return img
    
    # Adiciona a nova lógica YOLO (classe 0 é 'person')
    return detectar_com_yolo(img, classes_desejadas=[0])


# --- ADICIONAR NOVA FUNÇÃO 'detectar_cachorros' ---
# (Coloque antes ou depois de 'gerar_histograma')
def detectar_cachorros(img):
    # Classe 16 é 'dog' no modelo COCO
    return detectar_com_yolo(img, classes_desejadas=[16])

    

def gerar_histograma(img):
    hist_height = 256
    hist_width = 512
    bin_width = int(round(hist_width / 256))
    
    if len(img.shape) == 3: 
        hist_image = np.zeros((hist_height * 2, hist_width, 3), dtype=np.uint8)
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # B, G, R
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
