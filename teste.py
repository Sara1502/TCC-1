import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_hub as hub
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


def calcular_coeficiente_angular(p1, p2):
    """Calcula o coeficiente angular entre dois keypoints"""
    if p1[2] < 0.3 or p2[2] < 0.3:  # Confiança mínima
        return None
    delta_y = p2[0] - p1[0]
    delta_x = p2[1] - p1[1]
    if delta_x == 0:
        return float('inf')
    return delta_y / delta_x

def avaliar_sincronia_grupo(keypoints_with_score):
    """Avalia sincronia considerando todas as articulações principais"""
    pares_juntas = [
        # Braços
        (5, 7),   # Ombro esquerdo -> Cotovelo esquerdo
        (7, 9),   # Cotovelo esquerdo -> Punho esquerdo
        (6, 8),   # Ombro direito -> Cotovelo direito
        (8, 10),  # Cotovelo direito -> Punho direito
        
        # Pernas
        (11, 13), # Quadril esquerdo -> Joelho esquerdo
        (13, 15), # Joelho esquerdo -> Tornozelo esquerdo
        (12, 14), # Quadril direito -> Joelho direito
        (14, 16)  # Joelho direito -> Tornozelo direito
    ]
    
    penalidade_total = 0
    num_comparacoes = 0
    
    for (junta_a, junta_b) in pares_juntas:
        coeficientes = []
        for pessoa in keypoints_with_score:
            # Verifica confiança dos keypoints
            if pessoa[junta_a][2] > 0.3 and pessoa[junta_b][2] > 0.3:
                coef = calcular_coeficiente_angular(pessoa[junta_a], pessoa[junta_b])
                if coef is not None:
                    coeficientes.append(coef)
        
        # Compara todos os pares de dançarinos
        for i in range(len(coeficientes)):
            for j in range(i+1, len(coeficientes)):
                # Diferença percentual normalizada
                avg = (abs(coeficientes[i]) + abs(coeficientes[j])) / 2
                diff_percent = abs(coeficientes[i] - coeficientes[j]) / (avg + 1e-6) * 100  # +1e-6 evita divisão por zero
                
                # Penalidade progressiva
                if diff_percent > 5:
                    penalidade = min(diff_percent - 5, 15)  # Limita penalidade máxima a 15% por comparação
                    penalidade_total += penalidade
                num_comparacoes += 1
    
    # Calcula nota final (0-100%)
    return max(0, 100 - (penalidade_total / num_comparacoes)) if num_comparacoes > 0 else 100


def calcular_coeficiente_angular(p1, p2):
    """Calcula o coeficiente angular entre dois keypoints"""
    if p1[2] < 0.3 or p2[2] < 0.3:  # Confiança mínima
        return None
    delta_y = p2[0] - p1[0]
    delta_x = p2[1] - p1[1]
    if delta_x == 0:
        return float('inf')
    return delta_y / delta_x

def avaliar_sincronia_grupo(keypoints_with_score):
    pares_juntas = [
        (5, 7), (6, 8),    # Braços
        (11, 13), (12, 14)  # Pernas
    ]
    
    penalidade_total = 0
    num_comparacoes = 0
    
    for (junta_a, junta_b) in pares_juntas:
        coeficientes = []
        for pessoa in keypoints_with_score:
            coef = calcular_coeficiente_angular(pessoa[junta_a], pessoa[junta_b])
            if coef is not None:
                coeficientes.append(coef)
        
        for i in range(len(coeficientes)):
            for j in range(i+1, len(coeficientes)):
                diff_percent = abs(coeficientes[i] - coeficientes[j]) / ((abs(coeficientes[i]) + abs(coeficientes[j]))/2) * 100
                if diff_percent > 5:
                    penalidade_total += min(10, diff_percent - 5)
                num_comparacoes += 1
    
    return max(0, 100 - (penalidade_total / num_comparacoes)) if num_comparacoes > 0 else 100



def calcular_confiabilidade_deteccao(keypoints_with_score):
    """Calcula a porcentagem de keypoints válidos detectados"""
    total_keypoints = 0
    keypoints_validos = 0
    
    for pessoa in keypoints_with_score:
        for kp in pessoa:
            total_keypoints += 1
            if kp[2] > 0.3:  # Confiança > 30%
                keypoints_validos += 1
                
    return (keypoints_validos / total_keypoints) * 100 if total_keypoints > 0 else 0

def verificar_erros_graves(keypoints_with_score):
    """Detecta erros grosseiros como fusão de pessoas"""
    if len(keypoints_with_score) == 0:
        return True
    
    # Verifica proporções corporais absurdas
    for pessoa in keypoints_with_score:
        ombro_esq = pessoa[5]
        ombro_dir = pessoa[6]
        quadril_esq = pessoa[11]
        
        if ombro_esq[2] > 0.3 and ombro_dir[2] > 0.3 and quadril_esq[2] > 0.3:
            largura_ombros = abs(ombro_esq[1] - ombro_dir[1])
            altura_tronco = abs(ombro_esq[0] - quadril_esq[0])
            
            # Razão inválida (indicando fusão de pessoas)
            if largura_ombros / (altura_tronco + 1e-6) > 2.5:
                return True
                
    return False

frames_com_erro = 0
total_frames = 0
angulos_por_frame = []  # Agora armazenará as notas de 0-100%



#LOOP PRICIPAL
cap = cv2.VideoCapture('Kpop-Dance-Practice\\7-pessoas\\O.O\\O.O.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


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

    
    nota_sincronia = avaliar_sincronia_grupo(keypoints_with_score)
    angulos_por_frame.append(nota_sincronia)  # Armazena a nota do frame atual


    if nota_sincronia < 70:
        frames_com_erro += 1
    cv2.putText(frame, "ERRO", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    if len(keypoints_with_score) == 4:
        dancer_positions = ['left', 'center_left', 'center_right', 'right']
        torso_centers = [np.mean([pose[5][1], pose[6][1]]) for pose in keypoints_with_score]
        keypoints_with_score = [keypoints_with_score[i] for i in np.argsort(torso_centers)]

    # Renderiza keypoints
    loop_through_people(frame, keypoints_with_score, EDGES, 0.3)

    
    status = "PERFEITA" if nota_sincronia >= 90 else "BOA" if nota_sincronia >= 70 else "FRACA"
    cv2.putText(frame, status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 
           (0, 0, 255) if status == "FRACA" else (0, 255, 0), 2)

    
    if nota_sincronia < 70: 
        frames_com_erro += 1
    total_frames += 1 
    
    
    # Dentro do loop, após a detecção:
    conf_deteccao = calcular_confiabilidade_deteccao(keypoints_with_score)
    erro_grave = verificar_erros_graves(keypoints_with_score)

    # Atualize a exibição
    cv2.putText(frame, f"Conf. Detecção: {conf_deteccao:.1f}%", (20, 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if erro_grave:
        cv2.putText(frame, "ERRO GRAVE", (20, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    
    cv2.imshow("Movenet Multipose", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


# Substitua o cálculo de confiabilidade por:
# Substitua o cálculo final por:
confiabilidade_modelo = np.mean([calcular_confiabilidade_deteccao(result['output_0'].numpy()[:,:,:51].reshape((6,17,3))[:4]) 
                               for _ in range(10)])  # Teste com 10 amostras


nota_final = np.mean(angulos_por_frame) if angulos_por_frame else 0


print("\n=== RELATÓRIO DE CONFIABILIDADE ===")
print("\nRELATÓRIO FINAL:")
print(f"Nota média de sincronia: {nota_final:.2f}%")
print(f"Frames analisados: {len(angulos_por_frame)}")
print(f"Precisão de Detecção: {confiabilidade_modelo:.2f}%")

cap.release()
cv2.destroyAllWindows()