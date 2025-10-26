from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import base64
import filtros 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def ler_imagem(file):
    if 'image' not in request.files:
        return "Nenhuma imagem enviada", 400

    file = request.files['image']

    if file.filename == '':
        return "Nenhum arquivo selecionado", 400

    if file:
        image_np = np.frombuffer(file.read(), np.uint8)
        img_original = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if img_original is None:
            return "Erro ao decodificar a imagem", 400
        return img_original
    
def aplicar_filtro(file, filtro_func, kernel_size=None):
    img_original = ler_imagem(request.files) 
    
    if kernel_size is not None:
        img_processada = filtro_func(img_original, kernel_size)
    else:
        img_processada = filtro_func(img_original)
        
    _, buffer = cv2.imencode('.jpg', img_processada)
    final = base64.b64encode(buffer).decode('utf-8')
    cv2.imwrite('test_img.jpg', img_processada)

    return final
    
    
@app.route('/convert_to_grayscale', methods=['POST'])
def convert_to_grayscale():    
    return {'grayscale_image': aplicar_filtro(request.files, filtros.converter_para_cinza)}


@app.route('/convert_to_negative', methods=['POST'])
def convert_to_negative():
    return {'negative_img': aplicar_filtro(request.files, filtros.converter_para_negativo)}

@app.route('/convert_to_otsu', methods=['POST'])
def convert_to_otsu():
    img_original = ler_imagem(request.files)
    otsu_image, threshold_value = filtros.aplicar_otsu(img_original) #

    _, buffer = cv2.imencode('.jpg', otsu_image)
    otsu_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'otsu_image': otsu_image_base64
    }

@app.route('/suavizar_media', methods=['POST'])
def suavizar_media():
    kernel_size = int(request.form.get('kernel_size'))
    return {'blurred_image': aplicar_filtro(request.files, filtros.suavizar_pela_media, kernel_size)}


@app.route('/suavizar_mediana', methods=['POST'])
def suavizar_mediana():
    kernel_size = int(request.form.get('kernel_size'))
    return {'median_image': aplicar_filtro(request.files, filtros.suavizar_pela_mediana, kernel_size)}


@app.route('/detect_canny', methods=['POST'])
def detect_canny():
    return {'canny_image': aplicar_filtro(request.files, filtros.detectar_bordas_canny)}


@app.route('/gerar_histograma', methods=['POST'])
def gerar_histograma():
    return {'histogram_image': aplicar_filtro(request.files, filtros.gerar_histograma)}

@app.route('/morf_erode', methods=['POST'])
def morf_erode():
    kernel_size = int(request.form.get('kernel_size'))
    return {'erode_image': aplicar_filtro(request.files, filtros.aplicar_erosao, kernel_size)}


@app.route('/morf_dilate', methods=['POST'])
def morf_dilate():
    kernel_size = int(request.form.get('kernel_size'))
    return {'dilate_image': aplicar_filtro(request.files, filtros.aplicar_dilatacao, kernel_size)}


@app.route('/morf_open', methods=['POST'])
def morf_open():
    kernel_size = int(request.form.get('kernel_size'))
    return {'open_image': aplicar_filtro(request.files, filtros.aplicar_abertura, kernel_size)}


@app.route('/morf_close', methods=['POST'])
def morf_close():
    kernel_size = int(request.form.get('kernel_size'))
    return {'close_image': aplicar_filtro(request.files, filtros.aplicar_fechamento, kernel_size)}

@app.route('/detectar_pessoas', methods=['POST'])
def detectar_pessoas_route():
    return {'person_image': aplicar_filtro(request.files, filtros.detectar_pessoas)}


if __name__ == '__main__':
    app.run(debug=True)