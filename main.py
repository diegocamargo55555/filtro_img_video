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


@app.route('/detect_canny', methods=['POST'])
def detect_canny():
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
        canny_image = filtros.detectar_bordas_canny(img_original) 

        _, buffer = cv2.imencode('.jpg', canny_image)
        canny_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {'canny_image': canny_image_base64}


# --- NOVAS ROTAS ADICIONADAS ---

def process_morph_operation(operation_func):
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

        kernel_size = int(request.form.get('kernel_size', 5))
        
        # Chama a função de filtro específica (erosão, dilatação, etc.)
        processed_image = operation_func(img_original, kernel_size)

        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return processed_image_base64

@app.route('/morf_erode', methods=['POST'])
def morf_erode():
    image_base64 = process_morph_operation(filtros.aplicar_erosao)
    if isinstance(image_base64, tuple): # Em caso de erro
        return image_base64
    return {'erode_image': image_base64}

@app.route('/morf_dilate', methods=['POST'])
def morf_dilate():
    image_base64 = process_morph_operation(filtros.aplicar_dilatacao)
    if isinstance(image_base64, tuple):
        return image_base64
    return {'dilate_image': image_base64}

@app.route('/morf_open', methods=['POST'])
def morf_open():
    image_base64 = process_morph_operation(filtros.aplicar_abertura)
    if isinstance(image_base64, tuple):
        return image_base64
    return {'open_image': image_base64}

@app.route('/morf_close', methods=['POST'])
def morf_close():
    image_base64 = process_morph_operation(filtros.aplicar_fechamento)
    if isinstance(image_base64, tuple):
        return image_base64
    return {'close_image': image_base64}

# --- FIM DAS NOVAS ROTAS ---


if __name__ == '__main__':
    app.run(debug=True)