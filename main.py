# app.py
from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import base64 # Para codificar/decodificar imagens como string Base64
from io import BytesIO # Para trabalhar com bytes como se fossem um arquivo

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

        gray_image = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

        _, buffer = cv2.imencode('.jpg', gray_image)
        
        gray_image_base64 = base64.b64encode(buffer).decode('utf-8')
        print("bah")

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

        negative_img = cv2.bitwise_not(img_original)

        _, buffer = cv2.imencode('.jpg', negative_img)
        
        negative_img_base64 = base64.b64encode(buffer).decode('utf-8')
        print("oi")
        cv2.imwrite('img_aluno_negativa.jpg', negative_img)  

        return {'negative_img': negative_img_base64}


app.run(debug=True)