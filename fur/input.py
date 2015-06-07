import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.simplefilter("ignore", np.RankWarning)
i = 1
x_mouse = []
y_mouse = []
# mouse callback function
def draw_circle(event,x,y,flags,param):
	global i, x_mouse, y_mouse
	if event == cv2.EVENT_LBUTTONDBLCLK:
		x_mouse.append(x)
		y_mouse.append(y)
		print i
		i = i + 1

# Create a black image, a window and bind the function to window
img = cv2.imread('IMG_0700.JPG',cv2.IMREAD_COLOR)
img = cv2.resize(img,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image',draw_circle)

while(i < 7):
	cv2.imshow('image',img)
	if cv2.waitKey(2) & 0xFF == 27:
		break
cv2.destroyAllWindows()

print x_mouse
x_back = [0, 0]; y_back = [0, 0]; x_fur_long = [0, 0]; y_fur_long = [0, 0]; x_fur = [0, 0]; y_fur = [0, 0];
x_back[0] =  x_mouse[0]; x_back[1] =  x_mouse[1];
y_back[0] =  y_mouse[0]; y_back[1] =  y_mouse[1];
x_fur[0] =  x_mouse[2]; x_fur[1] =  x_mouse[3];
y_fur[0] =  y_mouse[2]; y_fur[1] =  y_mouse[3];
x_fur_long[0] =  x_mouse[4]; x_fur_long[1] =  x_mouse[5];
y_fur_long[0] =  y_mouse[4]; y_fur_long[1] =  y_mouse[5];


tree_num=1000
step = 10
area = np.power(step,2)
poly_times = 20
print "test poly_time: ", poly_times

# Load an color image in grayscale
# img = cv2.imread('IMG_0700.JPG',cv2.IMREAD_COLOR)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)
Img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
Img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.namedWindow('frame_hsv', cv2.WINDOW_NORMAL)
cv2.imshow('frame_hsv', Img_hsv)
cv2.namedWindow('frame_lab', cv2.WINDOW_NORMAL)
cv2.imshow('frame_lab', Img_lab)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Background
Back = np.double(img[y_back[0] : y_back[1], x_back[0] : x_back[1], :])
Back_hsv = np.double(Img_hsv[y_back[0] : y_back[1], x_back[0] : x_back[1], :])
Back_lab = np.double(Img_lab[y_back[0] : y_back[1], x_back[0] : x_back[1], :])
col,row,kernels = Back.shape
col_num_b = col / step
row_num_b = row / step
print "col_num_b", col_num_b

Feature_Back = np.zeros([col_num_b * row_num_b, 9]);
for i in range(0, row_num_b):
	for j in range(0, col_num_b):
		Feature_Back[(i) * col_num_b + j][0] = np.sum(Back[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 0]) / area;
		Feature_Back[(i) * col_num_b + j][1] = np.sum(Back[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 1]) / area;
		Feature_Back[(i) * col_num_b + j][2] = np.sum(Back[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 2]) / area;
		Feature_Back[(i) * col_num_b + j][3] = np.sum(Back_hsv[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 0]) / area;
		Feature_Back[(i) * col_num_b + j][4] = np.sum(Back_hsv[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 1]) / area;
		Feature_Back[(i) * col_num_b + j][5] = np.sum(Back_hsv[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 2]) / area;
		Feature_Back[(i) * col_num_b + j][6] = np.sum(Back_lab[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 0]) / area;
		Feature_Back[(i) * col_num_b + j][7] = np.sum(Back_lab[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 1]) / area;
		Feature_Back[(i) * col_num_b + j][8] = np.sum(Back_lab[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 2]) / area;
Feature_Back = np.float32(Feature_Back)
Back_label = np.float32(np.zeros([col_num_b * row_num_b, 1]))

#Fur_Long
Fur_long = np.double(img[y_fur_long[0] : y_fur_long[1], x_fur_long[0] : x_fur_long[1], :])
Fur_long_hsv = np.double(Img_hsv[y_fur_long[0] : y_fur_long[1], x_fur_long[0] : x_fur_long[1], :])
Fur_long_lab = np.double(Img_lab[y_fur_long[0] : y_fur_long[1], x_fur_long[0] : x_fur_long[1], :])
col, row, kernels = Fur_long.shape
col_num_fl = col/step
row_num_fl = row/step

Feature_Fur_long = np.zeros([row_num_fl*col_num_fl,9]);
for i in range(0,row_num_fl):
	for j in range(0,col_num_fl):
		Feature_Fur_long[(i) * col_num_fl + j][0] = np.sum(Fur_long[(i) * step : ((i + 1) * step), (j) * step:((j + 1) * step), 0]) / area;
		Feature_Fur_long[(i) * col_num_fl + j][1] = np.sum(Fur_long[(i) * step : ((i + 1) * step), (j) * step:((j + 1) * step), 1]) / area;
		Feature_Fur_long[(i) * col_num_fl + j][2] = np.sum(Fur_long[(i) * step : ((i + 1) * step), (j) * step:((j + 1) * step), 2]) / area;
		Feature_Fur_long[(i) * col_num_fl + j][3] = np.sum(Fur_long_hsv[(i) * step : ((i + 1) * step), (j) * step:((j + 1) * step), 0]) / area;
		Feature_Fur_long[(i) * col_num_fl + j][4] = np.sum(Fur_long_hsv[(i) * step : ((i + 1) * step), (j) * step:((j + 1) * step), 1]) / area;
		Feature_Fur_long[(i) * col_num_fl + j][5] = np.sum(Fur_long_hsv[(i) * step : ((i + 1) * step), (j) * step:((j + 1) * step), 2]) / area;
		Feature_Fur_long[(i) * col_num_fl + j][6] = np.sum(Fur_long_lab[(i) * step : ((i + 1) * step), (j) * step:((j + 1) * step), 0]) / area;
		Feature_Fur_long[(i) * col_num_fl + j][7] = np.sum(Fur_long_lab[(i) * step : ((i + 1) * step), (j) * step:((j + 1) * step), 1]) / area;
		Feature_Fur_long[(i) * col_num_fl + j][8] = np.sum(Fur_long_lab[(i) * step : ((i + 1) * step), (j) * step:((j + 1) * step), 2]) / area;
Feature_Fur_long = np.float32(Feature_Fur_long)
Fur_long_label = np.float32(np.ones([row_num_fl * col_num_fl, 1]))



#Fur
Fur = np.double(img[y_fur[0] : y_fur[1], x_fur[0] : x_fur[1], :])
Fur_hsv = np.double(Img_hsv[y_fur[0] : y_fur[1], x_fur[0] : x_fur[1], :])
Fur_lab = np.double(Img_lab[y_fur[0] : y_fur[1], x_fur[0] : x_fur[1], :])
col,row,kernels = Fur.shape
col_num_f = col / step
row_num_f = row / step

Feature_Fur = np.zeros([row_num_f*col_num_f,9]);
for i in range(0,row_num_f):
	for j in range(0,col_num_f):
		Feature_Fur[(i) * col_num_f + j][0] = np.sum(Fur[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 0]) / area;
		Feature_Fur[(i) * col_num_f + j][1] = np.sum(Fur[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 1]) / area;
		Feature_Fur[(i) * col_num_f + j][2] = np.sum(Fur[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 2]) / area;
		Feature_Fur[(i) * col_num_f + j][3] = np.sum(Fur_hsv[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 0]) / area;
		Feature_Fur[(i) * col_num_f + j][4] = np.sum(Fur_hsv[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 1]) / area;
		Feature_Fur[(i) * col_num_f + j][5] = np.sum(Fur_hsv[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 2]) / area;
		Feature_Fur[(i) * col_num_f + j][6] = np.sum(Fur_lab[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 0]) / area;
		Feature_Fur[(i) * col_num_f + j][7] = np.sum(Fur_lab[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 1]) / area;
		Feature_Fur[(i) * col_num_f + j][8] = np.sum(Fur_lab[(i) * step : ((i + 1) * step), (j) * step : ((j + 1) * step), 2]) / area;
Feature_Fur = np.float32(Feature_Fur)
Fur_label = np.float32(np.ones([row_num_f*col_num_f,1])*2)

Feature_train = np.float32(np.zeros([row_num_b * col_num_b + row_num_fl * col_num_fl + row_num_f * col_num_f, 9]))
Feature_train[0 : row_num_b * col_num_b][:]= Feature_Back
Feature_train[row_num_b * col_num_b : (row_num_b * col_num_b + row_num_fl * col_num_fl)][:] = Feature_Fur_long;
Feature_train[(row_num_b * col_num_b + row_num_fl * col_num_fl) :][:] = Feature_Fur;



Label_train = np.float32(np.zeros([row_num_b * col_num_b + row_num_fl * col_num_fl + row_num_f * col_num_f,1]))
Label_train[0 : row_num_b * col_num_b] = Back_label;
Label_train[row_num_b * col_num_b : (row_num_b * col_num_b + row_num_fl * col_num_fl)] = Fur_long_label;
Label_train[(row_num_b * col_num_b + row_num_fl * col_num_fl) :] = Fur_label;
#print Label_train 

#cout time
e1 = cv2.getTickCount()
# your code execution
rf = RandomForestClassifier(n_estimators = tree_num)
rf.fit(Feature_train, Label_train)



col,row,kernel = img.shape
frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
frame_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
col_num = col / step
row_num = row / step
col_size = col_num * step
row_size = row_num * step
test_img = np.double(img[0 : col_size, 0 : row_size, :])
test_img_hsv = np.double(frame_hsv[0 : col_size, 0 : row_size, :])
test_img_lab = np.double(frame_lab[0 : col_size, 0 : row_size, :])
Feature_test = np.zeros([col_num * row_num, 9]);
for i in range(0, col_num):
		for j in range(0,row_num):
			Feature_test[(i) * row_num + j][0] = np.sum(test_img[(i) * step : ((i + 1) * step),(j)*step:((j+1)*step),0])/area;
			Feature_test[(i) * row_num + j][1] = np.sum(test_img[(i) * step : ((i + 1) * step),(j)*step:((j+1)*step),1])/area;
			Feature_test[(i) * row_num + j][2] = np.sum(test_img[(i) * step : ((i + 1) * step),(j)*step:((j+1)*step),2])/area;
			Feature_test[(i) * row_num + j][3] = np.sum(test_img_hsv[(i) * step : ((i + 1) * step),(j)*step:((j+1)*step),0])/area;
			Feature_test[(i) * row_num + j][4] = np.sum(test_img_hsv[(i) * step : ((i + 1) * step),(j)*step:((j+1)*step),1])/area;
			Feature_test[(i) * row_num + j][5] = np.sum(test_img_hsv[(i) * step : ((i + 1) * step),(j)*step:((j+1)*step),2])/area;
			Feature_test[(i) * row_num + j][6] = np.sum(test_img_lab[(i) * step : ((i + 1) * step),(j)*step:((j+1)*step),0])/area;
			Feature_test[(i) * row_num + j][7] = np.sum(test_img_lab[(i) * step : ((i + 1) * step),(j)*step:((j+1)*step),1])/area;
			Feature_test[(i) * row_num + j][8] = np.sum(test_img_lab[(i) * step : ((i + 1) * step),(j)*step:((j+1)*step),2])/area;
Feature_test = np.float32(Feature_test)

#result = svm.predict_all(Feature_test)
result = rf.predict(Feature_test)
t = np.reshape(result,[col_num, row_num])
col,row = t.shape
view_img = np.uint8(np.zeros([col, row, 3]))
for i in range(0,col):
	for j in range(0,row):
		if t[i][j] == 0:
			view_img[i][j][0] = 255
		elif t[i][j] == 1:
			view_img[i][j][1] = 255
		else:
			view_img[i][j][2] = 255

e2 = cv2.getTickCount()
time = (e2 - e1) / cv2.getTickFrequency()
print time

cv2.namedWindow('classfication_result', cv2.WINDOW_NORMAL);
cv2.imshow('classification_result',view_img);
cv2.waitKey(0)



# e1 = cv2.getTickCount()
x_fur_long = step*np.array(range(row))
y_fur_long = np.zeros(row)
for i in range(row):
	for j in range(col):
		if t[j][i]==1:
			y_fur_long[i] = j*step
			break
#print y_fur_long	
x_fur = step*np.array(range(row))
y_fur = np.zeros(row)

for i in range(row):
	for j in range(col):
		if (t[j][i]) ==2:
			y_fur[i] = j*step
			break
	
p_1 = np.polyfit(x_fur_long, y_fur_long, poly_times)
p_2 = np.polyfit(x_fur,y_fur, poly_times)

#print p_1

y_fur_long_pred = np.float32(np.zeros(row * step))
y_fur_pred = np.float32(np.zeros(row * step))

func_1 = np.poly1d(p_1)
x_full = np.array(range(row * step))
y_fur_long_pred = func_1(x_full)

func_2 = np.poly1d(p_2)
y_fur_pred = func_2(x_full)


cv2.namedWindow('final_result', cv2.WINDOW_NORMAL)
for loop_1 in range(row*step-2):
	img = cv2.line(img,(int(x_full[loop_1]),int(y_fur_long_pred[loop_1])),(int(x_full[loop_1+1]),int(y_fur_long_pred[loop_1+1])),(255,56,0),5)
for loop_1 in range(row*step-2):
	img = cv2.line(img,(int(x_full[loop_1]),int(y_fur_pred[loop_1])),(int(x_full[loop_1+1]),int(y_fur_pred[loop_1+1])),(25,56,90),5)	
cv2.imshow('final_result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()










#print y_fur_pred
# 	point_draw_fur_long = np.zeros([row * step,2])
# 	point_draw_fur = np.zeros([row * step,2])
# 	for loop_1 in range(row * step):
# 		point_draw_fur_long[loop_1][0] = x_full[loop_1]
# 		point_draw_fur_long[loop_1][1] = y_fur_long_pred[loop_1]
# 		point_draw_fur[loop_1][0] = x_full[loop_1]
# 		point_draw_fur[loop_1][1] = y_fur_pred[loop_1]

# 	Img_show = img[0 : col_size, 0 : row_size, :]
# 	cv2.polylines(Img_show,np.int32([point_draw_fur_long]), 0 , (0,255,0), 5)
# 	cv2.polylines(Img_show,np.int32([point_draw_fur]), 0 , (255,0,0), 5)

# cv2.namedWindow('final_result', cv2.WINDOW_NORMAL)
# cv2.imshow('final_result', Img_show)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# e2 = cv2.getTickCount()
# time = (e2 - e1) / cv2.getTickFrequency()
# print time
