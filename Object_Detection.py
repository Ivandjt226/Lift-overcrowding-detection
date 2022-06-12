import cv2
import numpy as np

weights = 'D:/DJSCE/6th Sem/IACV/Mini project/Video object detection/yolov4-tiny.weights'
cfg = 'D:/DJSCE/6th Sem/IACV/Mini project/Video object detection/yolov4-tiny.cfg'
net = cv2.dnn.readNet(weights,cfg)

classes = []
classes_file = 'D:/DJSCE/6th Sem/IACV/Mini project/Video object detection/classes.txt'
with open(classes_file, "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture('D:/DJSCE/6th Sem/IACV/Mini project/Video object detection/lift.mp4')
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    if(_):
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

        imS = cv2.resize(img, (960, 540))       
        cv2.imshow('Image', imS)
        key = cv2.waitKey(1)
        if key==27:
            break
        
        print(len(indexes))
        if(len(indexes) >= 8) :
            print("Lift capacity reached max limit !!!")

cap.release()
cv2.destroyAllWindows()