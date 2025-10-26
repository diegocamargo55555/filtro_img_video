import cv2


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