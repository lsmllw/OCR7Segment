import cv2
import numpy as np
import json
import os

def nada(x): pass

CONFIG_FILE = "config_ocr_final.json"

def carregar_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f: return json.load(f)
    return [[0.5, 0.1], [0.8, 0.25], [0.8, 0.75], [0.5, 0.9], [0.2, 0.75], [0.2, 0.25], [0.5, 0.5]]

def detectar_completo():
    pos_segs = carregar_config()
    cap = cv2.VideoCapture(0)
    
    win_main = "Painel de Controle"
    cv2.namedWindow(win_main)

    #parametros
    cv2.createTrackbar("Limiar", win_main, 200, 255, nada)
    cv2.createTrackbar("Inclinacao", win_main, 50, 100, nada)
    cv2.createTrackbar("Sensib", win_main, 30, 100, nada)
    
    #kernel
    cv2.createTrackbar("Uniao_Vertical", win_main, 5, 20, nada)
    
    #ajuste
    cv2.createTrackbar("Eixo_C_X", win_main, 50, 100, nada)
    cv2.createTrackbar("Larg_Lados", win_main, 20, 100, nada)
    cv2.createTrackbar("Alt_Extremos", win_main, 25, 100, nada)
    
    #ajuste fino
    cv2.createTrackbar("EDIT_SEG", win_main, 0, 6, nada)
    cv2.createTrackbar("Fino_X", win_main, 50, 100, nada)
    cv2.createTrackbar("Fino_Y", win_main, 50, 100, nada)

    mapa = {(1,1,1,1,1,1,0):0, (0,1,1,0,0,0,0):1, (1,1,0,1,1,0,1):2, (1,1,1,1,0,0,1):3,
            (0,1,1,0,0,1,1):4, (1,0,1,1,0,1,1):5, (1,0,1,1,1,1,1):6, (1,1,1,0,0,0,0):7,
            (1,1,1,1,1,1,1):8, (1,1,1,1,0,1,1):9, (1,1,1,0,0,1,1):9}

    while True:
        ret, frame = cap.read()
        if not ret: break

        #parametros
        thr = cv2.getTrackbarPos("Limiar", win_main)
        tilt = (cv2.getTrackbarPos("Inclinacao", win_main) - 50) * 2
        sens = cv2.getTrackbarPos("Sensib", win_main) / 100.0
        v_join = cv2.getTrackbarPos("Uniao_Vertical", win_main)
        
        cX = cv2.getTrackbarPos("Eixo_C_X", win_main) / 100.0
        lados = cv2.getTrackbarPos("Larg_Lados", win_main) / 100.0
        alt = cv2.getTrackbarPos("Alt_Extremos", win_main) / 100.0

        #geometria
        temp_segs = [[cX, 0.10], [1-lados, alt], [1-lados, 1-alt], [cX, 0.90], [lados, 1-alt], [lados, alt], [cX, 0.50]]
        id_fino = cv2.getTrackbarPos("EDIT_SEG", win_main)
        temp_segs[id_fino][0] += (cv2.getTrackbarPos("Fino_X", win_main) - 50) / 100.0
        temp_segs[id_fino][1] += (cv2.getTrackbarPos("Fino_Y", win_main) - 50) / 100.0

        #processamento
        red = frame[:, :, 2]
        _, thresh = cv2.threshold(red, thr, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((v_join, v_join), np.uint8)
        thresh_unido = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh_unido, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digitos_encontrados = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            #filtro ponto decimal
            if 3 < h <= 25 and 2 < w <= 25:
                área = cv2.contourArea(cnt)
                if área > 5: #filtro poeira 
                    digitos_encontrados.append((x, "."))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                continue            

            if h > 20:
                #visualização
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                #ROI
                roi_limpa = thresh[y:y+h, x:x+w]
                
                altura_alvo = 150
                largura_alvo = 100
                
                escala = altura_alvo / float(h)
                nova_w = int(w * escala)
                
                if nova_w > largura_alvo:
                    roi_redimensionada = cv2.resize(roi_limpa, (largura_alvo, altura_alvo))
                    roi = roi_redimensionada
                else:
                    roi_redimensionada = cv2.resize(roi_limpa, (nova_w, altura_alvo))
                    
                    delta_w = largura_alvo - nova_w
                    left = delta_w // 2
                    right = delta_w - left
                    
                    left = max(0, left)
                    right = max(0, right)
                    
                    roi = cv2.copyMakeBorder(roi_redimensionada, 0, 0, left, right, cv2.BORDER_CONSTANT, value=0)
                
                #inclinação
                M = np.float32([[1, -tilt/100, 0], [0, 1, 0]])
                roi = cv2.warpAffine(roi, M, (100, 150))
                
                #filtro de aspecto para '1'
                if w/float(h) < 0.35:
                    digitos_encontrados.append((x, 1))
                    continue

                bits = []
                for i, (sx, sy) in enumerate(temp_segs):
                    px, py = int(sx*100), int(sy*150)
                    crop = roi[max(0,py-15):min(150,py+15), max(0,px-15):min(100,px+15)]
                    prench = cv2.countNonZero(crop)/float(crop.size) if crop.size > 0 else 0
                    bits.append(1 if prench > sens else 0)

                num = mapa.get(tuple(bits), "?")
                digitos_encontrados.append((x, num))

        #resultado
        digitos_encontrados.sort(key=lambda x: x[0])
        txt = "".join([str(d[1]) for d in digitos_encontrados])
        
        cv2.putText(frame, f"SISTEMA: {txt}", (20, 50), 1, 2, (0, 255, 255), 2)
        cv2.imshow(win_main, frame)
        cv2.imshow("Mascara de Uniao", thresh_unido)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord('s'): 
            with open(CONFIG_FILE, 'w') as f: json.dump(temp_segs, f)
            print("Salvo!")

    cap.release()
    cv2.destroyAllWindows()

detectar_completo()
