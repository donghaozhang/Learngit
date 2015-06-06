# The code for camera real-time
# Using the stable frame to classify the fur and background

# Sciki learn, opencv and opencv2 need to be installed first

import cv2
#import cv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

cap = cv2.VideoCapture(0)
width = cap.get(3)
height = cap.get(4)
print "width",width,"height",height

x_back = [460,560]
y_back = [27,127]
x_fur_long = [335,365]
y_fur_long = [175,215]
x_fur = [290,390]
y_fur = [290,420]
tree_num=1000
step = 10
area = np.power(step,2)
poly_times = 20


while(cap.isOpened()):
	ret,frame = cap.read()
	frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	frame_lab = cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
	cv2.namedWindow('frame_hsv')
	cv2.imshow('frame_hsv',frame_hsv)
	cv2.namedWindow('frame')
	cv2.imshow('frame',frame)
	cv2.namedWindow('frame_lab')
	cv2.imshow('frame_lab',frame_lab)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

Img = frame
Img_hsv = frame_hsv
Img_lab = frame_lab

# Background
Back = np.double(Img[y_back[0]:y_back[1],x_back[0]:x_back[1],:])
Back_hsv = np.double(Img_hsv[y_back[0]:y_back[1],x_back[0]:x_back[1],:])
Back_lab = np.double(Img_lab[y_back[0]:y_back[1],x_back[0]:x_back[1],:])
col,row,kernels = Back.shape
col_num_b = col/step
row_num_b = row/step

Feature_Back = np.zeros([col_num_b*row_num_b,9]);
for i in range(0,row_num_b):
	for j in range(0,col_num_b):
		Feature_Back[(i)*col_num_b+j][0] = np.sum(Back[(i)*step:((i+1)*step),(j)*step:((j+1)*step),0])/area;
		Feature_Back[(i)*col_num_b+j][1] = np.sum(Back[(i)*step:((i+1)*step),(j)*step:((j+1)*step),1])/area;
		Feature_Back[(i)*col_num_b+j][2] = np.sum(Back[(i)*step:((i+1)*step),(j)*step:((j+1)*step),2])/area;
		Feature_Back[(i)*col_num_b+j][3] = np.sum(Back_hsv[(i)*step:((i+1)*step),(j)*step:((j+1)*step),0])/area;
		Feature_Back[(i)*col_num_b+j][4] = np.sum(Back_hsv[(i)*step:((i+1)*step),(j)*step:((j+1)*step),1])/area;
		Feature_Back[(i)*col_num_b+j][5] = np.sum(Back_hsv[(i)*step:((i+1)*step),(j)*step:((j+1)*step),2])/area;
		Feature_Back[(i)*col_num_b+j][6] = np.sum(Back_lab[(i)*step:((i+1)*step),(j)*step:((j+1)*step),0])/area;
		Feature_Back[(i)*col_num_b+j][7] = np.sum(Back_lab[(i)*step:((i+1)*step),(j)*step:((j+1)*step),1])/area;
		Feature_Back[(i)*col_num_b+j][8] = np.sum(Back_lab[(i)*step:((i+1)*step),(j)*step:((j+1)*step),2])/area;
Feature_Back = np.float32(Feature_Back)
Back_label = np.float32(np.zeros([col_num_b*row_num_b,1]))

#Fur_Long
Fur_long = np.double(Img[y_fur_long[0]:y_fur_long[1],x_fur_long[0]:x_fur_long[1],:])
Fur_long_hsv = np.double(Img_hsv[y_fur_long[0]:y_fur_long[1],x_fur_long[0]:x_fur_long[1],:])
Fur_long_lab = np.double(Img_lab[y_fur_long[0]:y_fur_long[1],x_fur_long[0]:x_fur_long[1],:])
col,row,kernels = Fur_long.shape
col_num_fl = col/step
row_num_fl = row/step

Feature_Fur_long = np.zeros([row_num_fl*col_num_fl,9]);
for i in range(0,row_num_fl):
	for j in range(0,col_num_fl):
		Feature_Fur_long[(i)*col_num_fl+j][0] = np.sum(Fur_long[(i)*step:((i+1)*step),(j)*step:((j+1)*step),0])/area;
		Feature_Fur_long[(i)*col_num_fl+j][1] = np.sum(Fur_long[(i)*step:((i+1)*step),(j)*step:((j+1)*step),1])/area;
		Feature_Fur_long[(i)*col_num_fl+j][2] = np.sum(Fur_long[(i)*step:((i+1)*step),(j)*step:((j+1)*step),2])/area;
		Feature_Fur_long[(i)*col_num_fl+j][3] = np.sum(Fur_long_hsv[(i)*step:((i+1)*step),(j)*step:((j+1)*step),0])/area;
		Feature_Fur_long[(i)*col_num_fl+j][4] = np.sum(Fur_long_hsv[(i)*step:((i+1)*step),(j)*step:((j+1)*step),1])/area;
		Feature_Fur_long[(i)*col_num_fl+j][5] = np.sum(Fur_long_hsv[(i)*step:((i+1)*step),(j)*step:((j+1)*step),2])/area;
		Feature_Fur_long[(i)*col_num_fl+j][6] = np.sum(Fur_long_lab[(i)*step:((i+1)*step),(j)*step:((j+1)*step),0])/area;
		Feature_Fur_long[(i)*col_num_fl+j][7] = np.sum(Fur_long_lab[(i)*step:((i+1)*step),(j)*step:((j+1)*step),1])/area;
		Feature_Fur_long[(i)*col_num_fl+j][8] = np.sum(Fur_long_lab[(i)*step:((i+1)*step),(j)*step:((j+1)*step),2])/area;
Feature_Fur_long = np.float32(Feature_Fur_long)
Fur_long_label = np.float32(np.ones([row_num_fl*col_num_fl,1]))

#Fur
Fur = np.double(Img[y_fur[0]:y_fur[1],x_fur[0]:x_fur[1],:])
Fur_hsv = np.double(Img_hsv[y_fur[0]:y_fur[1],x_fur[0]:x_fur[1],:])
Fur_lab = np.double(Img_lab[y_fur[0]:y_fur[1],x_fur[0]:x_fur[1],:])
col,row,kernels = Fur.shape
col_num_f = col/step
row_num_f = row/step

Feature_Fur = np.zeros([row_num_f*col_num_f,9]);
for i in range(0,row_num_f):
	for j in range(0,col_num_f):
		Feature_Fur[(i)*col_num_f+j][0] = np.sum(Fur[(i)*step:((i+1)*step),(j)*step:((j+1)*step),0])/area;
		Feature_Fur[(i)*col_num_f+j][1] = np.sum(Fur[(i)*step:((i+1)*step),(j)*step:((j+1)*step),1])/area;
		Feature_Fur[(i)*col_num_f+j][2] = np.sum(Fur[(i)*step:((i+1)*step),(j)*step:((j+1)*step),2])/area;
		Feature_Fur[(i)*col_num_f+j][3] = np.sum(Fur_hsv[(i)*step:((i+1)*step),(j)*step:((j+1)*step),0])/area;
		Feature_Fur[(i)*col_num_f+j][4] = np.sum(Fur_hsv[(i)*step:((i+1)*step),(j)*step:((j+1)*step),1])/area;
		Feature_Fur[(i)*col_num_f+j][5] = np.sum(Fur_hsv[(i)*step:((i+1)*step),(j)*step:((j+1)*step),2])/area;
		Feature_Fur[(i)*col_num_f+j][6] = np.sum(Fur_lab[(i)*step:((i+1)*step),(j)*step:((j+1)*step),0])/area;
		Feature_Fur[(i)*col_num_f+j][7] = np.sum(Fur_lab[(i)*step:((i+1)*step),(j)*step:((j+1)*step),1])/area;
		Feature_Fur[(i)*col_num_f+j][8] = np.sum(Fur_lab[(i)*step:((i+1)*step),(j)*step:((j+1)*step),2])/area;
Feature_Fur = np.float32(Feature_Fur)
Fur_label = np.float32(np.ones([row_num_f*col_num_f,1])*2)

Feature_train = np.float32(np.zeros([row_num_b*col_num_b+row_num_fl*col_num_fl+row_num_f*col_num_f,9]))
Feature_train[0:row_num_b*col_num_b][:]= Feature_Back
Feature_train[row_num_b*col_num_b:(row_num_b*col_num_b+row_num_fl*col_num_fl)][:] = Feature_Fur_long;
Feature_train[(row_num_b*col_num_b+row_num_fl*col_num_fl):][:] = Feature_Fur;

Label_train = np.float32(np.zeros([row_num_b*col_num_b+row_num_fl*col_num_fl+row_num_f*col_num_f,1]))
Label_train[0:row_num_b*col_num_b] = Back_label;
Label_train[row_num_b*col_num_b:(row_num_b*col_num_b+row_num_fl*col_num_fl)] = Fur_long_label;
Label_train[(row_num_b*col_num_b+row_num_fl*col_num_fl):] = Fur_label;
print Label_train

#svm_params = dict(kernel_type = cv2.SVM_RBF,svm_type = cv2.SVM_C_SVC,C=7,gamma=0.7)

#svm = cv2.SVM();
#svm.train(Feature_train,Label_train,params = svm_params)
#svm.save('svm_data.dat')

rf = RandomForestClassifier(n_estimators=tree_num)
rf.fit(Feature_train,Label_train)

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
	ret,frame = cap.read()
	frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	frame_lab = cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
#	test_img = np.double(frame)
#	test_img_hsv = np.double(frame_hsv)
#	test_img_lab = np.double(frame_lab)

	#Entire Image
	col,row,kernel = frame.shape
	col_num = col/step
	row_num = row/step
	col_size = col_num*step
	row_size = row_num*step
	test_img = np.double(frame[0:col_size,0:row_size,:])
	test_img_hsv = np.double(frame_hsv[0:col_size,0:row_size,:])
	test_img_lab = np.double(frame_lab[0:col_size,0:row_size,:])

	Feature_test = np.zeros([col_num*row_num,9]);
	for i in range(0,col_num):
		for j in range(0,row_num):
			Feature_test[(i)*row_num+j][0] = np.sum(test_img[(i)*step:((i+1)*step),(j)*step:((j+1)*step),0])/area;
			Feature_test[(i)*row_num+j][1] = np.sum(test_img[(i)*step:((i+1)*step),(j)*step:((j+1)*step),1])/area;
			Feature_test[(i)*row_num+j][2] = np.sum(test_img[(i)*step:((i+1)*step),(j)*step:((j+1)*step),2])/area;
			Feature_test[(i)*row_num+j][3] = np.sum(test_img_hsv[(i)*step:((i+1)*step),(j)*step:((j+1)*step),0])/area;
			Feature_test[(i)*row_num+j][4] = np.sum(test_img_hsv[(i)*step:((i+1)*step),(j)*step:((j+1)*step),1])/area;
			Feature_test[(i)*row_num+j][5] = np.sum(test_img_hsv[(i)*step:((i+1)*step),(j)*step:((j+1)*step),2])/area;
			Feature_test[(i)*row_num+j][6] = np.sum(test_img_lab[(i)*step:((i+1)*step),(j)*step:((j+1)*step),0])/area;
			Feature_test[(i)*row_num+j][7] = np.sum(test_img_lab[(i)*step:((i+1)*step),(j)*step:((j+1)*step),1])/area;
			Feature_test[(i)*row_num+j][8] = np.sum(test_img_lab[(i)*step:((i+1)*step),(j)*step:((j+1)*step),2])/area;
	Feature_test = np.float32(Feature_test)

	#result = svm.predict_all(Feature_test)
	result = rf.predict(Feature_test)
	t = np.reshape(result,[col_num,row_num])
	col,row = t.shape
	
	view_img = np.uint8(np.zeros([col,row,3]))

	for i in range(0,col):
		for j in range(0,row):
			if t[i][j]==0:
				view_img[i][j][0] = 255
			else:
				if t[i][j]==1:
					view_img[i][j][1] = 255
				else:
					view_img[i][j][2] = 255

	cv2.imshow('classification_result',view_img)
	#np.save('classification_result_new',t)

	x_fur_long = step*np.array(range(row))
	y_fur_long = np.zeros(row)

	for i in range(row):
		for j in range(col):
			if t[j][i]==1:
				y_fur_long[i] = j*step
				break
	
	x_fur = step*np.array(range(row))
	y_fur = np.zeros(row)

	for i in range(row):
		for j in range(col):
			if (t[j][i]) ==2:
				y_fur[i] = j*step
				break
	
	p_1 = np.polyfit(x_fur_long,y_fur_long,poly_times)
	p_2 = np.polyfit(x_fur,y_fur,poly_times)

	y_fur_long_pred = np.float32(np.zeros(row*step))
	y_fur_pred= np.float32(np.zeros(row*step))

	func_1 = np.poly1d(p_1)
	x_full = np.array(range(row*step))
	y_fur_long_pred = func_1(x_full)

	func_2 = np.poly1d(p_2)
	y_fur_pred = func_2(x_full)

	point_draw_fur_long = np.zeros([row*step,2])
	point_draw_fur = np.zeros([row*step,2])
	for loop_1 in range(row*step):
		point_draw_fur_long[loop_1][0] = x_full[loop_1]
		point_draw_fur_long[loop_1][1] = y_fur_long_pred[loop_1]
		point_draw_fur[loop_1][0] = x_full[loop_1]
		point_draw_fur[loop_1][1] = y_fur_pred[loop_1]

	Img_show = frame[0:col_size,0:row_size,:]
	cv2.polylines(Img_show,np.int32([point_draw_fur_long]),0,(0,255,0),5)
	cv2.polylines(Img_show,np.int32([point_draw_fur]),0,(255,0,0),5)

	cv2.namedWindow('final_result',cv2.WINDOW_NORMAL)
	cv2.imshow('final_result',Img_show)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

