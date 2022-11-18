import cv2

#Load Web Camera
cap1 = cv2.VideoCapture(0) #load WebCamera
cap2 = cv2.VideoCapture(2) #load WebCamera
cap3 = cv2.VideoCapture(3) #load WebCamera
cap4 = cv2.VideoCapture(4) #load WebCamera

if not (cap1.isOpened()):
	print("File isn't opend!!")

#Set Video File Property
videoFileName_1 = 'output1.avi'
# videoFileName_2 = 'output2.avi'
# videoFileName_3 = 'output3.avi'
# videoFileName_4 = 'output4.avi'

w = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)) # width
h = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height
fps = cap1.get(cv2.CAP_PROP_FPS) #frame per second
fourcc = cv2.VideoWriter_fourcc(*'DIVX') #fourcc

#Save Video
out1 = cv2.VideoWriter(videoFileName_1, fourcc, fps, (w,h))
if not (out1.isOpened()):
	print("File isn't opend!!")
	cap1.release()


#Load frame and Save it
while(True): #Check Video is Available
	ret, frame1 = cap1.read() #read by frame (ret=TRUE/FALSE)s
	ret, frame2 = cap2.read() #read by frame (ret=TRUE/FALSE)s
	ret, frame3 = cap3.read() #read by frame (ret=TRUE/FALSE)s
	ret, frame4 = cap4.read() #read by frame (ret=TRUE/FALSE)s

	if ret:
		
		out1.write(frame1) #save video frame
		
		cv2.imshow('Original VIDEO', frame1)
		# cv2.imshow('Inversed VIDEO', inversed)

		if cv2.waitKey(1) == 27: #wait 10ms until user input 'esc'
			break
	else:
		print("ret is false")
		break

cap1.release() #release memory
out1.release() #release memory
cv2.destroyAllWindows() #destroy All Window