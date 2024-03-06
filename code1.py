import os
import cv2

DAT_DIR = './data'
if not os.path.exists(DAT_DIR):
    os.makedirs(DAT_DIR)

number_of_classes = 5
dataset_size = 100

cap = cv2.VideoCapture(0) 
for j in range(number_of_classes):
    class_path = os.path.join(DAT_DIR, str(j))
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start', (100, 50), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
            file_path = os.path.join(DAT_DIR, str(j), '{}.jpg'.format(counter))  
            cv2.imwrite(file_path, frame)
            counter += 1

cap.release()
cv2.destroyAllWindows()
