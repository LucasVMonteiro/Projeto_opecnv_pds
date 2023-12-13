import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import datetime

"""
    PRE-PROCESSAMENTOS:
    1-REDIMENCIONAMENTO OK
    2-BLUR OK
    3-GRAYSCALE OK
    
    PERFORMACE?
    threads
    
    FUNCOES
    em ordem de criacao
    
    1 - carregar_classificar
    2 - redimensionar
    3 - filtro_blur
    4 - carimbar_tempo

"""

def carimbar_tempo(img,fram_width):

    img = cv.rectangle(img,(0,0),(frame_width,20), (255, 255, 255), -1)


    font = cv.FONT_HERSHEY_PLAIN
    tempo_atual = datetime.datetime.now()
    tempo_decorrido = tempo_atual - tempo_inicial
    dt = str(tempo_decorrido)
    frame = cv.putText( img,
                        dt,
                        (0,20),
                        font,
                        1,
                        (0,0,0),
                        2,
                        cv.LINE_8)
    return frame


def filtro_blur(img):
    filtrado = cv.blur(img,(10,10))
    return filtrado


## FUNCAO QUE REDIMENCIONA O FRAME
def redimensionar(img):

    scale_down = 0.45

    scaled_f_down = cv.resize(img, None, fx= scale_down, fy= scale_down, interpolation= cv.INTER_LINEAR)

    return scaled_f_down

def carregar_classificar(img_gray,img_rgb_original,classificador):

    #retorna a imagem marcada

    found = classificador.detectMultiScale(img_gray, 
                                    minSize =(20, 20))

    amount_found = len(found)

    if amount_found != 0:
        
        for (x, y, width, height) in found:
            
 
            cv.rectangle(img_rgb_original, (x, y), 
                        (x + height, y + width), 
                        (0, 255, 0), 2)
            
    return img_rgb_original

args_count = len(sys.argv)
if args_count > 2:
    print(f"One argument expected, got {args_count - 1}")
    raise SystemExit(2)
elif args_count == 1:
    print("OK")

try:

    nome_video = sys.argv[1]
    cap = cv.VideoCapture(nome_video)
except:
    cap = cv.VideoCapture("amostra3.mp4")
    
classificador = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 

salva_video = False
if salva_video:
    result = cv.VideoWriter('video_processado.mp4',cv.VideoWriter_fourcc(*'MP4V'),30, size) 

tempo_inicial = datetime.datetime.now()

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    print(cap.isOpened())
    if not ret:
        print("Falha ao receber frame. Saindo ...")
        break
        
    frame_redimensionado = redimensionar(frame)
    frame_gray = cv.cvtColor(frame_redimensionado, cv.COLOR_RGB2GRAY)
    frame_gray = cv.convertScaleAbs(frame_gray,
                                 alpha=0.5,
                                 beta=0)

    frame_blur = filtro_blur(frame_gray)

    frame_entrada = frame_blur
    frame_redimensionado_saida = frame_redimensionado.copy()

    # identificacao utilizando um frame blur como entrada
    img_rgb0 = carregar_classificar(frame_entrada,
                                    frame_redimensionado,
                                    classificador)

    saidas_processadas = [img_rgb0, frame_redimensionado_saida]
    
    index = 0
    for frm in saidas_processadas:
        saidas_processadas[index] = carimbar_tempo(frm,frame_width)
        index += 1
    #Concatenando frames
    saidacon1 = cv.hconcat(saidas_processadas)
    # salvando video
    if salva_video:
        result.write(img_rgb0)
    # exibindo video
    
    #saida concatenada
    cv.imshow('saida',saidacon1)
    #saida separada

    #cv.imshow('Saida 1', img_rgb)
    #cv.imshow('Saida 2',frame_gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
