# opencv 左上為0 ，往右
import cv2
import numpy as np 
from numba import jit

# 抓取圖片的點
def mouse_handler(event, x, y, flags, data): 
    if event == cv2.EVENT_LBUTTONDOWN:
        # 標記點位置
        cv2.circle(background_pic, (x,y), 3, (0,0,255), 5) 
        # 改變顯示 window 的內容
        cv2.imshow("image", background_pic)
        # 顯示 (x,y) 並儲存到 list中
        print("get points: (x, y) = ({}, {})".format(x, y))
        background_point.append([x,y])

### dff_warp
@jit(nopython = False)
def aff_wrap_pic(need_transpose, background, calculate_answer_inv): 
    need_transpose_x, need_transpose_y = need_transpose.shape[:2]
    for u in range(background.shape[1]):
        for v in range(background.shape[0]):
            # ans = calculate_answer_inv.dot([u,v,1])
            # np.dot 在 jit 不支援，用手刻
            # ans = np.zeros((3), dtype= np.float32)  --> jit 好像不支援
            ans_0 = calculate_answer_inv[0][0] * u + calculate_answer_inv[0][1] * v + calculate_answer_inv[0][2] * 1
            ans_1 = calculate_answer_inv[1][0] * u + calculate_answer_inv[1][1] * v + calculate_answer_inv[1][2] * 1
            ans_2 = calculate_answer_inv[2][0] * u + calculate_answer_inv[2][1] * v + calculate_answer_inv[2][2] * 1

            ans_0 = ans_0 / ans_2
            ans_1 = ans_1 / ans_2
            # ans = np.int16(ans)
            ans_0 = np.int16(ans_0)
            ans_1 = np.int16(ans_1)
            if (ans_0 > 0 and ans_1 > 0 and need_transpose_y > ans_0 and need_transpose_x > ans_1):  # ans[0] = 橫坐標 ans[1] = 縱座標
                background[v,u,:] = need_transpose[ans_1, ans_0, :]
    
    return background

people_path = './man.jpg'
background_path = './source picture.png'
people = cv2.imread(people_path)
background_pic = cv2.imread(background_path)
background_pic_1 = cv2.imread(background_path)
people_x, people_y = people.shape[:2]
#左上、左下、右下、右上 (Affine transform 只需三個點，but 老師要四個點來做) 
people_point = [[0,0], [0,people_x], [people_y,people_x]] 
background_point = []

#### 在背景上做的動作
Flag = 1
while Flag:
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_handler)
    cv2.imshow('image', background_pic)
    if cv2.waitKey(1) & len(background_point)>=3:
        Flag -= 1
cv2.destroyAllWindows()

#### 手刻矩陣  p' = Hp
# 目標點的矩陣
target = []
for index, num in enumerate(background_point):
    for i in range(len(background_point[index])):
        target.append(background_point[index][i])
target = np.array(target).reshape(-1,1)  # 6*1

# 原本點的矩陣
init = np.zeros((6,6))
x = 0
count = 0
for i in range(init.shape[0]):
    if (i%2) == 0:
        init[i][0:2] = people_point[x][0:2]
        init[i][2] = 1
        count += 1
    else:
        init[i][3:5] = people_point[x][0:2]
        init[i][5] = 1 
        count += 1  
    if count == 2:
        x += 1
        count = 0
# print(init)

### 求六個自由度的矩陣
# background_pic_1 = cv2.imread(background_path)
init_inv = np.linalg.inv(init)
calculate_answer = init_inv.dot(target)  # 6*1
calculate_answer = calculate_answer.reshape(2,3)
# print(calculate_answer)
calculate_answer = np.append(calculate_answer, [[0,0,1]], axis = 0)
calculate_answer_inv = np.linalg.inv(calculate_answer)
final = aff_wrap_pic(people, background_pic_1, calculate_answer_inv)

# cv2.imshow('man',people)
cv2.imshow('final', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
