import cv2
import numpy as np  # <-- ADICIONE ESTA IMPORTAÇÃO

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
        gray_image, 
        0, 
        255,  
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return otsu_image, threshold_value


def suavizar_pela_media(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred_image = cv2.blur(img, (kernel_size, kernel_size))
    
    return blurred_image


def suavizar_pela_mediana(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    median_image = cv2.medianBlur(img, kernel_size)
    
    return median_image


def detectar_bordas_canny(img, t_lower=100, t_upper=200):
    # Canny funciona melhor em escala de cinza
    if len(img.shape) == 3:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img
        
    # Aplica um leve blur para reduzir o ruído antes do Canny
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    canny_edges = cv2.Canny(blurred_image, t_lower, t_upper)
    
    return canny_edges

# --- NOVAS FUNÇÕES ADICIONADAS ---

def aplicar_erosao(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Cria o elemento estruturante (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Aplica a erosão
    erode_image = cv2.erode(img, kernel, iterations=1)
    return erode_image

def aplicar_dilatacao(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Cria o elemento estruturante (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Aplica a dilatação
    dilate_image = cv2.dilate(img, kernel, iterations=1)
    return dilate_image

def aplicar_abertura(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Cria o elemento estruturante (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Aplica a abertura (erosão seguida de dilatação)
    open_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return open_image

def aplicar_fechamento(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Cria o elemento estruturante (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Aplica o fechamento (dilatação seguida de erosão)
    close_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return close_image