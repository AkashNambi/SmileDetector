import cv2

#Face Classifier
face_detector = cv2.CascadeClassifier('harrcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('harrcascade_smile.xml')

#Grab webcam feed
webcam = cv2.VideoCapture(0)

while True:
    #Read the current frame from the user
    ret,frame = webcam.read()

    #If there's an error abort
    if not ret:
        break
    #Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)


    #Run Face detection for each of those faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,200,50), 4)

        #Get the sub-frame i.e)only the face (using N-dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]

        # Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        #Run smile detector on that little face
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)
        if(len(smiles)>0):
            cv2.putText(frame, 'smiling', (x,y+h+30), fontScale=2,fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0))

    #Show the current frame
    cv2.imshow('new one', frame)
    if cv2.waitKey(1) == ord('q'):
        break;



webcam.release()

cv2.destroyAllWindows()
