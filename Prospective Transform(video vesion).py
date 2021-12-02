# opencv 左上為0 ，往右
import cv2
import numpy as np 
from numba import jit

# 貼照片到母照片上
@jit(nopython=True)
def warp_pic(need_transpose, background, calculate_answer_inv):
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
            # ans[0] = 橫坐標，ans[1] = 縱座標
            if (ans_0 > 0 and ans_1 > 0 and  need_transpose_y > ans_0 and  need_transpose_x > ans_1):
                background[v,u,:] = need_transpose[ans_1, ans_0, :]
    return background

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

cap = cv2.VideoCapture('Gura.mp4')
ret, frame = cap.read()  # frame 為 3_dim pic'
frame_x, frame_y = frame.shape[:2]
background_path = './source picture.png'
background_pic = cv2.imread(background_path)
background_pic_1 = cv2.imread(background_path)
#左上、左下、右下、右上 (Prospective transform 要四個點) 
people_point = [[0,0], [0,frame_x], [frame_y,frame_x], [frame_y,0]] 
background_point = []

#### 在背景上做的動作
Flag = 1
while Flag:
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_handler)
    cv2.imshow('image', background_pic)
    if cv2.waitKey(1) & len(background_point)>3:
        Flag -= 1
cv2.destroyAllWindows()
print(background_point)

# perspective transform (原本點的矩陣)
init = np.zeros((8,9))
x = 0
count = 0
for i in range(init.shape[0]):
    if (i%2) == 0:
        init[i][0:2] = people_point[x][0:2]
        init[i][2] = 1
        init[i][-3:-1] = people_point[x][0:2]
        init[i][-1] = 1 
        init[i][-3:] = init[i][-3:] * (background_point[x][count]) * -1
        count += 1
    else:
        init[i][3:5] = people_point[x][0:2]
        init[i][5] = 1 
        init[i][-3:-1] = people_point[x][0:2]
        init[i][-1] = 1 
        init[i][-3:] = init[i][-3:] * (background_point[x][count]) * -1
        count += 1  
    if count == 2:
        x += 1
        count = 0
# print(init)

#### SVD分解
svd_matrix = np.transpose(init).dot(init)
U, S, V = np.linalg.svd(svd_matrix)  # nx9 9x9 9x9
print(S) # eigenvector
min_eigenvalue = min(S)
new_S = np.zeros(9)
### S 的最小 eigenvalue 對應到 V 的 column 是我們的答案
new_S[8] = min_eigenvalue
calculate = (new_S).dot(V)
calculate = calculate.reshape(3,3)
calculate_answer = calculate / calculate[-1][-1]
print(calculate_answer)
calculate_answer_inv = np.linalg.inv(calculate_answer)

print('進入LOOP.....')
while(True):
    print('抓取照片')
    ret, frame = cap.read()
    final = warp_pic(frame, background_pic_1, calculate_answer_inv)       
    print('showing.....')
    cv2.imshow('final', final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


