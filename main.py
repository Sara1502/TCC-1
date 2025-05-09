import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#import tensorflow_hub as hub
import numpy as np
import cv2
from matplotlib import pyplot as plt




# Substitua pelo caminho para a pasta que contém 'saved_model.pb' e 'variables'
caminho_para_o_modelo = "model"

# Carregar o modelo

model = tf.saved_model.load(caminho_para_o_modelo)
movenet = model.signatures['serving_default']

EDGES = {
    (0,1): 'm',
    (0,2): 'c',
    (1,3): 'm',
    (2,4): 'c',
    (0,5): 'm',
    (0,6): 'c',
    (5,7): 'm',
    (7,9): 'm',
    (6,8): 'c',
    (8,10): 'c',
    (5,6): 'y',
    (5,11): 'm',
    (6,12): 'c',
    (11,12): 'y',
    (11,13): 'm',
    (13,15): 'm',
    (12,14): 'c',
    (14,16):'c'
}

def loop_through_people(frame, keypoints_with_score, edges, confidence_threshold):
    for person in keypoints_with_score:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    input_height, input_width = 192, 192  # Ajuste para o tamanho da entrada do modelo
    shaped = np.squeeze(np.multiply(keypoints, [input_height, input_width, 1]))

    # Ajuste os keypoints para o tamanho real do frame
    shaped[:, 0] *= y / input_height  # Ajusta a altura dos keypoints
    shaped[:, 1] *= x / input_width   # Ajusta a largura dos keypoints


    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    input_height, input_width = 192, 192  # Ajuste para o tamanho da entrada do modelo
    shaped = np.squeeze(np.multiply(keypoints, [input_height, input_width, 1]))

    # Ajuste os keypoints para o tamanho real do frame
    shaped[:, 0] *= y / input_height  # Ajusta a altura dos keypoints
    shaped[:, 1] *= x / input_width   # Ajusta a largura dos keypoints
        
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


def calcular_angulos(pose):
        """Calcula ângulos articulares para uma pose"""
        angulos = {}
        
        # Cotovelos
        if pose[5][2] > 0.3 and pose[7][2] > 0.3 and pose[9][2] > 0.3:
            angulos['cotovelo_esq'] = calcular_angulo_3pontos(pose[5], pose[7], pose[9])
        if pose[6][2] > 0.3 and pose[8][2] > 0.3 and pose[10][2] > 0.3:
            angulos['cotovelo_dir'] = calcular_angulo_3pontos(pose[6], pose[8], pose[10])
        
        # Joelhos
        if pose[11][2] > 0.3 and pose[13][2] > 0.3 and pose[15][2] > 0.3:
            angulos['joelho_esq'] = calcular_angulo_3pontos(pose[11], pose[13], pose[15])
        if pose[12][2] > 0.3 and pose[14][2] > 0.3 and pose[16][2] > 0.3:
            angulos['joelho_dir'] = calcular_angulo_3pontos(pose[12], pose[14], pose[16])
        
        # Adicione outras articulações conforme necessário
        return angulos


def calcular_angulo_3pontos(a, b, c):
    """Calcula ângulo em graus entre os pontos a-b-c"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def calcular_sincronia(angulos_dançarinos):
    """Calcula quão sincronizados estão os dançarinos"""
    sincronias = []
    
    # Para cada articulação (ex: cotovelo esquerdo)
    articulacoes = ['cotovelo_esq', 'cotovelo_dir', 'joelho_esq', 'joelho_dir']  # Adicione todas que usar
    
    for artic in articulacoes:
        angulos = [d[artic] for d in angulos_dançarinos if artic in d]
        if len(angulos) >= 2:  # Pelo menos 2 dançarinos com a articulação detectada
            sincronias.append(np.std(angulos))  # Quanto menor, mais sincronizado
    
    return np.mean(sincronias) if sincronias else 0


frames_com_erro = 0
total_frames = 0

#LOOP PRICIPAL
cap = cv2.VideoCapture('Kpop-Dance-Practice\\4-pessoas\\Hip\\Hip.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    angulos_por_frame = []

    # No início do loop principal (antes de img = frame.copy())
    frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Reduz ruído
    kernel = np.ones((3, 3), dtype=np.uint8)
    frame = cv2.dilate(frame, kernel, iterations=1)  # Melhora contornos

    img = frame.copy()
    input_height, input_width = 192, 192  # Ajuste de acordo com a resolução esperada
    img_resized = cv2.resize(img, (input_width, input_height))
    img_input = tf.expand_dims(img_resized, axis=0)  # Adiciona a dimensão do batch (1, height, width, 3)
    img_input = tf.cast(img_input, dtype=tf.int32)


    # Detection
    result = movenet(img_input)
    keypoints_with_score = result['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))[:4]  

    if len(keypoints_with_score) == 4:
        dancer_positions = ['left', 'center_left', 'center_right', 'right']
        torso_centers = [np.mean([pose[5][1], pose[6][1]]) for pose in keypoints_with_score]
        keypoints_with_score = [keypoints_with_score[i] for i in np.argsort(torso_centers)]

    # Renderiza keypoints
    loop_through_people(frame, keypoints_with_score, EDGES, 0.3)

    angulos_dançarinos = [calcular_angulos(pose) for pose in keypoints_with_score]
    angulos_por_frame.append(angulos_dançarinos)

    sincronia_atual = calcular_sincronia(angulos_dançarinos)
    cv2.putText(frame, f"Sincronia: {sincronia_atual:.2f}°", 
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if sincronia_atual > 15.0: 
        frames_com_erro += 1
    total_frames += 1 
    
    cv2.imshow("Movenet Multipose", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

erros = (frames_com_erro * 100) / total_frames
confiabilidade = 100 - erros
print(f"Confiabilidade: {confiabilidade:.2f}%")

np.save('angulos.npy', angulos_por_frame)

cap.release()
cv2.destroyAllWindows()