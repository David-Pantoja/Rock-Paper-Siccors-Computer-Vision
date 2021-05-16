import numpy as np
import cv2
import torch
#gets Net (Nueral Network) class from model.py
from model import Net       
#sets stream to webcam
cap = cv2.VideoCapture(0)
#sets resolution
cap.set(3, 700)
cap.set(4, 480)
#Creats Nueral Network object with trained .pt file
model = torch.load("model_trained.pt")
model.eval()
#Dictionary that gives information about output layer including how many, and whqat they correlate to
signs = {'0': 'ROCK', '1': 'PAPER', '2': 'SCISSORS' }
#Live feed of camera, will end when break conditon (press q is met)
while True:        
    #cap.read() gets the webcam output. frame is set to the webcam fram and ret returns whether the webcam worked or not
    ret, frame = cap.read()

    #Sets frame of where the Nueral Network looks
    img = frame[20:250, 20:250]      
    #Resizes image to fit desired pixelation
    res = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_CUBIC)  
    #Turns RGB color to Black and White      
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    #transforms black and white resized image into an array of grayscale values
    res1 = np.reshape(res, (1, 1, 28, 28)) / 255      
    #Makes array into a tensor 
    res1 = torch.from_numpy(res1)   
    #Adds class variable "type" of "torch.FloatTensor" 
    res1 = res1.type(torch.FloatTensor)       

    #Returns output layer
    out = model(res1)       
    
    #Sets probs (Probabilitys) and labels (In this case 0,1, and 2 (see signs Dictionary))
    probs, label = torch.topk(out, 2)       
    #"squishification" function on probabilitys  
    probs = torch.nn.functional.softmax(probs, 1)      

    #Choses most likely key from Dictionary (0,1,or2) and sets it as output prediction
    pred = out.max(1, keepdim=True)[1]
     #multipied to make output a percentage by adding a "%" substring later on
    probs *= 100   

    #Checks for minimum probability and then outputs desired text
    if float(probs[0,0]) < 25:       
        text_shown = 'Sign not detected'
    else:
        text_shown = signs[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0,0]))+ '%'

    #font
    font = cv2.FONT_HERSHEY_SIMPLEX
    #places text  
    frame = cv2.putText(frame, text_shown, (60,285), font, 1, (255,0,0), 2, cv2.LINE_AA)   
    #green rectangle
    frame = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)   

    #shows edited frame with labled output and probability to user
    cv2.imshow('Cam', frame)

     #press "q" to stop programm  
    if cv2.waitKey(1) & 0xFF == ord('q'):    
        break

#closes files
cap.release()   
#closes stream window   
cv2.destroyAllWindows()     
