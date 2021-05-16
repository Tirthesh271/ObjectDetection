import cv2
import matplotlib.pyplot as plt

configFile = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
fModel = "frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(fModel, configFile)

classlabels = []
with open('names.txt','rt') as file:
    classlabels = file.read().rstrip('\n').split("\n")

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


'''
#Image object detection

img = cv2.imread('car.jpg')
#cv2.imshow('output',img)
#cv2.waitKey(0)

classInd , confidence, box = model.detect(img, confThreshold= 0.5)
print(classInd)

font = cv2.FONT_HERSHEY_PLAIN
for classI, conf, boxes in zip(classInd.flatten(), confidence.flatten(),box):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classlabels[classI -1],(boxes[0]+10,boxes[1]+40),font, fontScale=3, color=(0,255,0), thickness = 3)

plt.imshow(img)
plt.show()



'''

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("ERROR ")

while True:
    ret,frame = cap.read()

    try:classInd, confidence, box = model.detect(frame, confThreshold=0.6)
    except:pass

    if (len(classInd))!=0:
        for classI, conf, boxes in zip(classInd.flatten(), confidence.flatten(),box):
            if classI<=80 :
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classlabels[classI -1],(boxes[0]+10,boxes[1]+20), cv2.QT_FONT_NORMAL, fontScale=1, color=(0,255,128), thickness = 2)
                cv2.putText(frame, str((conf*100)%100)+"%", (boxes[0] + 10, boxes[1] + 50), cv2.QT_FONT_NORMAL,fontScale=0.5, color=(0, 255, 128),thickness=1)

                cv2.imshow('Object detection',frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break