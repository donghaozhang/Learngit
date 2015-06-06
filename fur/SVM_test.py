import cv2
import cv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

Img = cv2.imread('IMG_0700.JPG')
Img_hsv = cv2.cvtColor(Img,cv2.COLOR_BGR2HSV);
Img_lab = cv2.cvtColor(Img,cv2.COLOR_BGR2LAB);

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img',Img)
cv2.namedWindow('img_hsv',cv2.WINDOW_NORMAL)
cv2.imshow('img_hsv',Img_hsv)
cv2.namedWindow('img_lab',cv2.WINDOW_NORMAL)
cv2.imshow('img_lab',Img_lab)

cv2.waitKey(0)
cv2.destroyAllWindows()

col,row,kernels = Img.shape
print "width",col,"height",row

# Background
Back = np.double(Img[114:914,1175:2175,:])
col,row,kernels = Back.shape
print col, row, kernels
cv2.imshow('Back',Back)
cv2.waitKey(0)
cv2.destroyAllWindows()
Back_hsv = np.double(Img_hsv[114:914,1175:2175,:])
Back_lab = np.double(Img_lab[114:914,1175:2175,:])

Feature_Back = np.zeros([100*80,9]);
for i in range(1,100):
	for j in range(1,80):
		Feature_Back[(i-1)*80+j-1][0] = np.sum(Back[(i-1)*10:(i*10),(j-1)*10:(j*10),0])/100;
		Feature_Back[(i-1)*80+j-1][1] = np.sum(Back[(i-1)*10:(i*10),(j-1)*10:(j*10),1])/100;
		Feature_Back[(i-1)*80+j-1][2] = np.sum(Back[(i-1)*10:(i*10),(j-1)*10:(j*10),2])/100;
		Feature_Back[(i-1)*80+j-1][3] = np.sum(Back_hsv[(i-1)*10:(i*10),(j-1)*10:(j*10),0])/100;
		Feature_Back[(i-1)*80+j-1][4] = np.sum(Back_hsv[(i-1)*10:(i*10),(j-1)*10:(j*10),1])/100;
		Feature_Back[(i-1)*80+j-1][5] = np.sum(Back_hsv[(i-1)*10:(i*10),(j-1)*10:(j*10),2])/100;
		Feature_Back[(i-1)*80+j-1][6] = np.sum(Back_lab[(i-1)*10:(i*10),(j-1)*10:(j*10),0])/100;
		Feature_Back[(i-1)*80+j-1][7] = np.sum(Back_lab[(i-1)*10:(i*10),(j-1)*10:(j*10),1])/100;
		Feature_Back[(i-1)*80+j-1][8] = np.sum(Back_lab[(i-1)*10:(i*10),(j-1)*10:(j*10),2])/100;
Feature_Back = np.float32(Feature_Back)
Back_label = np.float32(np.zeros([100*80,1]))

#Fur_Long
Fur_long = np.double(Img[1469:1769,1817:2417,:])
Fur_long_hsv = np.double(Img_hsv[1469:1769,1817:2417,:])
Fur_long_lab = np.double(Img_lab[1469:1769,1817:2417,:])

Feature_Fur_long = np.zeros([60*30,9]);
for i in range(1,60):
	for j in range(1,30):
		Feature_Fur_long[(i-1)*30+j-1][0] = np.sum(Fur_long[(i-1)*10:(i*10),(j-1)*10:(j*10),0])/100;
		Feature_Fur_long[(i-1)*30+j-1][1] = np.sum(Fur_long[(i-1)*10:(i*10),(j-1)*10:(j*10),1])/100;
		Feature_Fur_long[(i-1)*30+j-1][2] = np.sum(Fur_long[(i-1)*10:(i*10),(j-1)*10:(j*10),2])/100;
		Feature_Fur_long[(i-1)*30+j-1][3] = np.sum(Fur_long_hsv[(i-1)*10:(i*10),(j-1)*10:(j*10),0])/100;
		Feature_Fur_long[(i-1)*30+j-1][4] = np.sum(Fur_long_hsv[(i-1)*10:(i*10),(j-1)*10:(j*10),1])/100;
		Feature_Fur_long[(i-1)*30+j-1][5] = np.sum(Fur_long_hsv[(i-1)*10:(i*10),(j-1)*10:(j*10),2])/100;
		Feature_Fur_long[(i-1)*30+j-1][6] = np.sum(Fur_long_lab[(i-1)*10:(i*10),(j-1)*10:(j*10),0])/100;
		Feature_Fur_long[(i-1)*30+j-1][7] = np.sum(Fur_long_lab[(i-1)*10:(i*10),(j-1)*10:(j*10),1])/100;
		Feature_Fur_long[(i-1)*30+j-1][8] = np.sum(Fur_long_lab[(i-1)*10:(i*10),(j-1)*10:(j*10),2])/100;
Feature_Fur_long = np.float32(Feature_Fur_long)
Fur_long_label = np.float32(np.ones([60*30,1]))

#Fur
Fur = np.double(Img[2075:2375,1842:2442,:])
Fur_hsv = np.double(Img_hsv[2075:2375,1842:2442,:])
Fur_lab = np.double(Img_lab[2075:2375,1842:2442,:])

Feature_Fur = np.zeros([60*30,9]);
for i in range(1,60):
	for j in range(1,30):
		Feature_Fur[(i-1)*30+j-1][0] = np.sum(Fur[(i-1)*10:(i*10),(j-1)*10:(j*10),0])/100;
		Feature_Fur[(i-1)*30+j-1][1] = np.sum(Fur[(i-1)*10:(i*10),(j-1)*10:(j*10),1])/100;
		Feature_Fur[(i-1)*30+j-1][2] = np.sum(Fur[(i-1)*10:(i*10),(j-1)*10:(j*10),2])/100;
		Feature_Fur[(i-1)*30+j-1][3] = np.sum(Fur_hsv[(i-1)*10:(i*10),(j-1)*10:(j*10),0])/100;
		Feature_Fur[(i-1)*30+j-1][4] = np.sum(Fur_hsv[(i-1)*10:(i*10),(j-1)*10:(j*10),1])/100;
		Feature_Fur[(i-1)*30+j-1][5] = np.sum(Fur_hsv[(i-1)*10:(i*10),(j-1)*10:(j*10),2])/100;
		Feature_Fur[(i-1)*30+j-1][6] = np.sum(Fur_lab[(i-1)*10:(i*10),(j-1)*10:(j*10),0])/100;
		Feature_Fur[(i-1)*30+j-1][7] = np.sum(Fur_lab[(i-1)*10:(i*10),(j-1)*10:(j*10),1])/100;
		Feature_Fur[(i-1)*30+j-1][8] = np.sum(Fur_lab[(i-1)*10:(i*10),(j-1)*10:(j*10),2])/100;
Feature_Fur = np.float32(Feature_Fur)
Fur_label = np.float32(np.ones([60*30,1])*2)

Feature_train = np.float32(np.zeros([100*80+60*30*2,9]))
Feature_train[0:100*80][:]= Feature_Back
Feature_train[100*80:(100*80+60*30)][:] = Feature_Fur_long;
Feature_train[(100*80+60*30):][:] = Feature_Fur;

Label_train = np.float32(np.zeros([100*80+60*30*2,1]))
Label_train[0:100*80] = Back_label;
Label_train[100*80:(100*80+60*30)] = Fur_long_label;
Label_train[(100*80+60*30):] = Fur_label;
print Label_train

#svm_params = dict(kernel_type = cv2.SVM_RBF,svm_type = cv2.SVM_C_SVC,C=7,gamma=0.7)

#svm = cv2.SVM();
#svm.train(Feature_train,Label_train,params = svm_params)
#svm.save('svm_data.dat')

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(Feature_train,Label_train)

#Entire Image
test_img = np.double(Img[0:3450,0:4600,:])
test_img_hsv = np.double(Img_hsv[0:3450,0:4600,:])
test_img_lab = np.double(Img_lab[0:3450,0:4600,:])

Feature_test = np.zeros([345*460,9]);
for i in range(1,345):
	for j in range(1,460):
		Feature_test[(i-1)*460+j-1][0] = np.sum(test_img[(i-1)*10:(i*10),(j-1)*10:(j*10),0])/100;
		Feature_test[(i-1)*460+j-1][1] = np.sum(test_img[(i-1)*10:(i*10),(j-1)*10:(j*10),1])/100;
		Feature_test[(i-1)*460+j-1][2] = np.sum(test_img[(i-1)*10:(i*10),(j-1)*10:(j*10),2])/100;
		Feature_test[(i-1)*460+j-1][3] = np.sum(test_img_hsv[(i-1)*10:(i*10),(j-1)*10:(j*10),0])/100;
		Feature_test[(i-1)*460+j-1][4] = np.sum(test_img_hsv[(i-1)*10:(i*10),(j-1)*10:(j*10),1])/100;
		Feature_test[(i-1)*460+j-1][5] = np.sum(test_img_hsv[(i-1)*10:(i*10),(j-1)*10:(j*10),2])/100;
		Feature_test[(i-1)*460+j-1][6] = np.sum(test_img_lab[(i-1)*10:(i*10),(j-1)*10:(j*10),0])/100;
		Feature_test[(i-1)*460+j-1][7] = np.sum(test_img_lab[(i-1)*10:(i*10),(j-1)*10:(j*10),1])/100;
		Feature_test[(i-1)*460+j-1][8] = np.sum(test_img_lab[(i-1)*10:(i*10),(j-1)*10:(j*10),2])/100;
Feature_test = np.float32(Feature_test)

#result = svm.predict_all(Feature_test)
result = rf.predict(Feature_test)
t = np.reshape(result,[345,460])
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
cv2.waitKey(0)
cv2.destroyAllWindows()

x_fur_long = 10*np.array(range(row))
y_fur_long = np.zeros(row)

for i in range(row):
	for j in range(col):
		if t[j][i]==1:
			y_fur_long[i] = j*10
			break

x_fur = 10*np.array(range(row))
y_fur = np.zeros(row)

for i in range(row):
	for j in range(col):
		if t[j][i]==2:
			y_fur[i] = j*10
			break

p_1 = np.polyfit(x_fur_long,y_fur_long,20)
p_2 = np.polyfit(x_fur,y_fur,20)

y_fur_long_pred = np.float32(np.zeros(row*10))
y_fur_pred= np.float32(np.zeros(row*10))

func_1 = np.poly1d(p_1)
x_full = np.array(range(row*10))
y_fur_long_pred = func_1(x_full)

func_2 = np.poly1d(p_2)
y_fur_pred = func_2(x_full)

point_draw_fur_long = np.zeros([row*10-50,2])
point_draw_fur = np.zeros([row*10-50,2])
for loop_1 in range(row*10-50):
	point_draw_fur_long[loop_1][0] = x_full[loop_1]
	point_draw_fur_long[loop_1][1] = y_fur_long_pred[loop_1]
	point_draw_fur[loop_1][0] = x_full[loop_1]
	point_draw_fur[loop_1][1] = y_fur_pred[loop_1]

Img_show = Img[0:3450,0:4550,:]
cv2.polylines(Img_show,np.int32([point_draw_fur_long]),0,(0,255,0),5)
cv2.polylines(Img_show,np.int32([point_draw_fur]),0,(255,0,0),5)

cv2.namedWindow('final_result',cv2.WINDOW_NORMAL)
cv2.imshow('final_result',Img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
