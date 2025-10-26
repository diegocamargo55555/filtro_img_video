# app.py
from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import base64
from io import BytesIO
import filtros  # Importa o módulo com as funções de processamento

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/convert_to_grayscale', methods=['POST'])
def convert_to_grayscale():
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

        # Usa a função do módulo filtros
        gray_image = filtros.converter_para_cinza(img_original)

        _, buffer = cv2.imencode('.jpg', gray_image)
        gray_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {'grayscale_image': gray_image_base64}


@app.route('/convert_to_negative', methods=['POST'])
def convert_to_negative():
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

        # Usa a função do módulo filtros
        negative_img = filtros.converter_para_negativo(img_original)

        _, buffer = cv2.imencode('.jpg', negative_img)
        negative_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Opcional: salvar imagem processada
        # cv2.imwrite('img_aluno_negativa.jpg', negative_img)

        return {'negative_img': negative_img_base64}


@app.route('/convert_to_otsu', methods=['POST'])
def convert_to_otsu():
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

        # Usa a função do módulo filtros
        otsu_image, threshold_value = filtros.aplicar_otsu(img_original)

        _, buffer = cv2.imencode('.jpg', otsu_image)
        otsu_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print(f"Limiar calculado pelo método de Otsu: {threshold_value}")
        
        # Opcional: salvar imagem processada
        cv2.imwrite('img_aluno_otsu.jpg', otsu_image)

        return {
            'otsu_image': otsu_image_base64,
            'threshold_value': float(threshold_value)
        }


@app.route('/suavizar_media', methods=['POST'])
def suavizar_media():
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

        # Pega o tamanho do kernel dos parâmetros (padrão 5)
        kernel_size = int(request.form.get('kernel_size', 5))

        # Usa a função do módulo filtros
        blurred_image = filtros.suavizar_pela_media(img_original, kernel_size)

        _, buffer = cv2.imencode('.jpg', blurred_image)
        blurred_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {'blurred_image': blurred_image_base64}


@app.route('/suavizar_mediana', methods=['POST'])
def suavizar_mediana():
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

        # Pega o tamanho do kernel dos parâmetros (padrão 5)
        kernel_size = int(request.form.get('kernel_size', 5))

        # Usa a função do módulo filtros
        median_image = filtros.suavizar_pela_mediana(img_original, kernel_size)

        _, buffer = cv2.imencode('.jpg', median_image)
        median_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {'median_image': median_image_base64}


if __name__ == '__main__':
    app.run(debug=True)