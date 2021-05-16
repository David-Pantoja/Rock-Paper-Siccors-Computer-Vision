import numpy as np
import cv2
import torch
import os
from model import Net

cap = cv2.VideoCapture(0)

cap.set(3, 700)
cap.set(4, 480)

modelo = torch.load("model_trained.pt")
modelo.eval()

signs = {'0': 'ROCK', '1': 'PAPER', '2': 'SCISSORS' }

while True:
    ret, frame = cap.read()

    # Lugar de la imagen donde se toma la muestra
    img = frame[20:250, 20:250]

    res = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_CUBIC)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    res1 = np.reshape(res, (1, 1, 28, 28)) / 255
    res1 = torch.from_numpy(res1)
    res1 = res1.type(torch.FloatTensor)

    out = modelo(res1)
    
    # Probabilidades
    probs, label = torch.topk(out, 2)
    probs = torch.nn.functional.softmax(probs, 1)*100   #multipied by 100 for nice %

    pred = out.max(1, keepdim=True)[1]  #multipied by 100 for nice %

    if float(probs[0,0]) < 25:            #this changes sensetivity for some reason?
        texto_mostrar = 'Sign not detected'
    else:
        texto_mostrar = signs[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0,0]))+ '%'

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, texto_mostrar, (60,285), font, 1, (255,0,0), 2, cv2.LINE_AA)

    frame = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)

    cv2.imshow('Cam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
