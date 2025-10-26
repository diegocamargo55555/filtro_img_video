import cv2
import numpy as np


def converter_para_cinza(img):
    """
    Converte uma imagem colorida para escala de cinza.
    
    Args:
        img: Imagem em formato numpy array (BGR)
    
    Returns:
        Imagem em escala de cinza
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image


def converter_para_negativo(img):
    """
    Inverte os valores dos pixels da imagem (negativo).
    
    Args:
        img: Imagem em formato numpy array
    
    Returns:
        Imagem negativada
    """
    negative_img = cv2.bitwise_not(img)
    return negative_img


def aplicar_otsu(img):
    """
    Aplica limiarização automática pelo método de Otsu.
    
    Args:
        img: Imagem em formato numpy array (BGR ou escala de cinza)
    
    Returns:
        tuple: (imagem_binarizada, valor_do_limiar)
    """
    # Verifica se a imagem já está em escala de cinza
    if len(img.shape) == 3:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img
    
    # Aplicar limiarização de Otsu
    threshold_value, otsu_image = cv2.threshold(
        gray_image, 
        0,  # valor inicial (ignorado pelo Otsu)
        255,  # valor máximo
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return otsu_image, threshold_value


def suavizar_pela_media(img, kernel_size=5):
    """
    Aplica filtro de suavização pela média (blur).
    
    Args:
        img: Imagem em formato numpy array
        kernel_size: Tamanho do kernel (deve ser ímpar), padrão 5x5
    
    Returns:
        Imagem suavizada
    """
    # Garante que o kernel_size seja ímpar
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Aplica o filtro de média (blur)
    blurred_image = cv2.blur(img, (kernel_size, kernel_size))
    
    return blurred_image


def suavizar_pela_mediana(img, kernel_size=5):
    """
    Aplica filtro de suavização pela mediana.
    Este filtro é muito eficaz para remover ruído do tipo "sal e pimenta".
    
    Args:
        img: Imagem em formato numpy array
        kernel_size: Tamanho do kernel (deve ser ímpar), padrão 5
    
    Returns:
        Imagem suavizada
    """
    # Garante que o kernel_size seja ímpar
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Aplica o filtro de mediana
    median_image = cv2.medianBlur(img, kernel_size)
    
    return median_image