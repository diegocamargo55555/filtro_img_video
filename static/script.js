const btnCamera = document.getElementById('btnCamera');
const btnVideo = document.getElementById('btnVideo');
const btnImagem = document.getElementById('btnImagem');
const btnGrayscale = document.getElementById('btnGrayscale');
const btnNegative = document.getElementById('btnNegative');
const btnOtsu = document.getElementById('btnOtsu');
const btnMedia = document.getElementById('btnMedia');
const btnMediana = document.getElementById('btnMediana');
const kernelSize = document.getElementById('kernelSize');

const inputVideo = document.getElementById('inputVideo');
const inputImagem = document.getElementById('inputImagem');

const videoPlayer = document.getElementById('videoPlayer');
const videoCanvas = document.getElementById('videoCanvas');
const originalImage = document.getElementById('originalImage');
const originalVideo = document.getElementById('originalVideo');
const processedImage = document.getElementById('processedImage');
const processedCanvas = document.getElementById('processedCanvas');
const thresholdInfo = document.getElementById('thresholdInfo');

let currentImageFile = null;
let currentVideoStream = null;
let isProcessingVideo = false;
let currentFilter = null;
let animationFrameId = null;

function hideAllMedia() {
    originalImage.style.display = 'none';
    originalVideo.style.display = 'none';
    processedImage.style.display = 'none';
    processedCanvas.style.display = 'none';
    videoPlayer.style.display = 'none';
}

function enableFilterButtons() {
    btnGrayscale.disabled = false;
    btnNegative.disabled = false;
    btnOtsu.disabled = false;
    btnMedia.disabled = false;
    btnMediana.disabled = false;
}

function stopVideoProcessing() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    // NÃO definimos isProcessingVideo = false aqui.
    // Essa variável só deve mudar ao carregar uma IMAGEM.
    currentFilter = null;
}
// --- Acessar a Câmera ---
btnCamera.addEventListener('click', async () => {
    try {
        stopVideoProcessing();
        hideAllMedia();
        
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        currentVideoStream = stream;
        videoPlayer.srcObject = stream;
        originalVideo.srcObject = stream;
        
        originalVideo.style.display = 'block';
        videoPlayer.play();
        originalVideo.play();
        
        enableFilterButtons();
        isProcessingVideo = true;
        thresholdInfo.textContent = '';
    } catch (err) {
        console.error("Erro ao acessar a câmera: ", err);
        alert("Não foi possível acessar a câmera. Verifique as permissões do navegador.");
    }
});

// --- Carregar Vídeo ---
btnVideo.addEventListener('click', () => {
    inputVideo.click();
});

inputVideo.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        stopVideoProcessing();
        hideAllMedia();
        
        const fileURL = URL.createObjectURL(file);
        videoPlayer.src = fileURL;
        originalVideo.src = fileURL;
        
        originalVideo.style.display = 'block';
        videoPlayer.play();
        originalVideo.play();
        
        enableFilterButtons();
        isProcessingVideo = true;
        thresholdInfo.textContent = '';
    }
});

// --- Carregar Imagem ---
btnImagem.addEventListener('click', () => {
    inputImagem.click();
});

inputImagem.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        stopVideoProcessing();
        hideAllMedia();
        
        currentImageFile = file;
        const fileURL = URL.createObjectURL(file);
        originalImage.src = fileURL;
        originalImage.style.display = 'block';
        
        enableFilterButtons();
        isProcessingVideo = false;
        thresholdInfo.textContent = '';
    }
});

// Função para processar vídeo frame a frame
async function processVideoFrame(filterEndpoint, filterName) {
    if (!isProcessingVideo || currentFilter !== filterName) return;

    const canvas = videoCanvas;
    const ctx = canvas.getContext('2d');
    const video = videoPlayer;
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Processa com Canvas para filtros simples (mais rápido)
        if (filterName === 'grayscale') {
            applyGrayscaleCanvas(canvas);
            if (currentFilter === filterName) {
                animationFrameId = requestAnimationFrame(() => processVideoFrame(filterEndpoint, filterName));
            }
            return;
        } else if (filterName === 'negative') {
            applyNegativeCanvas(canvas);
            if (currentFilter === filterName) {
                animationFrameId = requestAnimationFrame(() => processVideoFrame(filterEndpoint, filterName));
            }
            return;
        }

        // Para filtros complexos (Otsu, Média, Mediana), envia ao servidor
        canvas.toBlob(async (blob) => {
            if (!blob || currentFilter !== filterName) return;

            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');
            if (filterName === 'media' || filterName === 'mediana') {
                formData.append('kernel_size', kernelSize.value);
            }

            try {
                const response = await fetch(filterEndpoint, {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok || currentFilter !== filterName) return;

                const data = await response.json();
                const imageKey = Object.keys(data).find(key => key.includes('image'));
                
                if (data[imageKey] && currentFilter === filterName) {
                    const processedCtx = processedCanvas.getContext('2d');
                    const img = new Image();
                    img.onload = () => {
                        if (currentFilter === filterName) {
                            processedCanvas.width = img.width;
                            processedCanvas.height = img.height;
                            processedCtx.drawImage(img, 0, 0);
                        }
                    };
                    img.src = 'data:image/jpeg;base64,' + data[imageKey];
                }
            } catch (error) {
                console.error('Erro ao processar frame:', error);
            }

            if (currentFilter === filterName) {
                // Reduz FPS para 10fps no processamento de servidor
                setTimeout(() => {
                    animationFrameId = requestAnimationFrame(() => processVideoFrame(filterEndpoint, filterName));
                }, 100);
            }
        }, 'image/jpeg', 0.6);
    } else {
        if (currentFilter === filterName) {
            animationFrameId = requestAnimationFrame(() => processVideoFrame(filterEndpoint, filterName));
        }
    }
}

// Filtro de escala de cinza usando Canvas (processamento local - rápido)
function applyGrayscaleCanvas(sourceCanvas) {
    const ctx = sourceCanvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
        const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
        data[i] = gray;
        data[i + 1] = gray;
        data[i + 2] = gray;
    }

    const processedCtx = processedCanvas.getContext('2d');
    processedCanvas.width = sourceCanvas.width;
    processedCanvas.height = sourceCanvas.height;
    processedCtx.putImageData(imageData, 0, 0);
}

// Filtro negativo usando Canvas (processamento local - rápido)
function applyNegativeCanvas(sourceCanvas) {
    const ctx = sourceCanvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
        data[i] = 255 - data[i];
        data[i + 1] = 255 - data[i + 1];
        data[i + 2] = 255 - data[i + 2];
    }

    const processedCtx = processedCanvas.getContext('2d');
    processedCanvas.width = sourceCanvas.width;
    processedCanvas.height = sourceCanvas.height;
    processedCtx.putImageData(imageData, 0, 0);
}

// --- Filtros ---
btnGrayscale.addEventListener('click', async () => {
    if (isProcessingVideo) {
        stopVideoProcessing();
        currentFilter = 'grayscale';
        processedCanvas.style.display = 'block';
        thresholdInfo.textContent = 'Processando vídeo (tempo real)...';
        processVideoFrame('/convert_to_grayscale', 'grayscale');
    } else if (currentImageFile) {
        const formData = new FormData();
        formData.append('image', currentImageFile);

        try {
            const response = await fetch('/convert_to_grayscale', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Erro do servidor: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.grayscale_image) {
                processedImage.src = 'data:image/jpeg;base64,' + data.grayscale_image;
                processedImage.style.display = 'block';
                thresholdInfo.textContent = '';
            }
        } catch (error) {
            console.error("Erro ao converter para cinza:", error);
            alert("Ocorreu um erro ao converter a imagem.");
        }
    }
});

btnNegative.addEventListener('click', async () => {
    if (isProcessingVideo) {
        stopVideoProcessing();
        currentFilter = 'negative';
        processedCanvas.style.display = 'block';
        thresholdInfo.textContent = 'Processando vídeo (tempo real)...';
        processVideoFrame('/convert_to_negative', 'negative');
    } else if (currentImageFile) {
        const formData = new FormData();
        formData.append('image', currentImageFile);

        try {
            const response = await fetch('/convert_to_negative', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Erro do servidor: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.negative_img) {
                processedImage.src = 'data:image/jpeg;base64,' + data.negative_img;
                processedImage.style.display = 'block';
                thresholdInfo.textContent = '';
            }
        } catch (error) {
            console.error("Erro ao inverter:", error);
            alert("Ocorreu um erro ao converter a imagem.");
        }
    }
});

btnOtsu.addEventListener('click', async () => {
    if (isProcessingVideo) {
        stopVideoProcessing();
        currentFilter = 'otsu';
        processedCanvas.style.display = 'block';
        thresholdInfo.textContent = 'Processando vídeo (pode estar lento)...';
        processVideoFrame('/convert_to_otsu', 'otsu');
    } else if (currentImageFile) {
        const formData = new FormData();
        formData.append('image', currentImageFile);

        try {
            const response = await fetch('/convert_to_otsu', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Erro do servidor: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.otsu_image) {
                processedImage.src = 'data:image/jpeg;base64,' + data.otsu_image;
                processedImage.style.display = 'block';
                
                if (data.threshold_value) {
                    thresholdInfo.textContent = `Limiar: ${data.threshold_value.toFixed(2)}`;
                }
            }
        } catch (error) {
            console.error("Erro ao aplicar Otsu:", error);
            alert("Ocorreu um erro ao processar a imagem.");
        }
    }
});

btnMedia.addEventListener('click', async () => {
    if (isProcessingVideo) {
        stopVideoProcessing();
        currentFilter = 'media';
        processedCanvas.style.display = 'block';
        thresholdInfo.textContent = `Processando vídeo (pode estar lento, Kernel: ${kernelSize.value}x${kernelSize.value})...`;
        processVideoFrame('/suavizar_media', 'media');
    } else if (currentImageFile) {
        const formData = new FormData();
        formData.append('image', currentImageFile);
        formData.append('kernel_size', kernelSize.value);

        try {
            const response = await fetch('/suavizar_media', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Erro do servidor: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.blurred_image) {
                processedImage.src = 'data:image/jpeg;base64,' + data.blurred_image;
                processedImage.style.display = 'block';
                thresholdInfo.textContent = `Média (Kernel: ${kernelSize.value}x${kernelSize.value})`;
            }
        } catch (error) {
            console.error("Erro ao suavizar:", error);
            alert("Ocorreu um erro ao processar a imagem.");
        }
    }
});

btnMediana.addEventListener('click', async () => {
    if (isProcessingVideo) {
        stopVideoProcessing();
        currentFilter = 'mediana';
        processedCanvas.style.display = 'block';
        thresholdInfo.textContent = `Processando vídeo (pode estar lento, Kernel: ${kernelSize.value}x${kernelSize.value})...`;
        processVideoFrame('/suavizar_mediana', 'mediana');
    } else if (currentImageFile) {
        const formData = new FormData();
        formData.append('image', currentImageFile);
        formData.append('kernel_size', kernelSize.value);

        try {
            const response = await fetch('/suavizar_mediana', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Erro do servidor: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.median_image) {
                processedImage.src = 'data:image/jpeg;base64,' + data.median_image;
                processedImage.style.display = 'block';
                thresholdInfo.textContent = `Mediana (Kernel: ${kernelSize.value}x${kernelSize.value})`;
            }
        } catch (error) {
            console.error("Erro ao suavizar:", error);
            alert("Ocorreu um erro ao processar a imagem.");
        }
    }
});