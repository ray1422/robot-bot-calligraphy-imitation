
import math
from typing import List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt

def enhance(img: np.ndarray) -> np.ndarray:
    """
    params img: input image
    return: enhanced image in grayscale
    """
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img2 = cv2.resize(img2, (256, 256))
    ret, th2 = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY_INV)
    # erode and dilation
    kernel = np.ones((3, 3), np.uint8)
    th2 = cv2.erode(th2, kernel, iterations=2)   #需要隨著字的不同調整
    th2 = cv2.dilate(th2, kernel, iterations=2)  #需要隨著字的不同調整

    return th2


def sub_contour(img: np.ndarray) -> List[np.ndarray]:
    """
    param img: grayscale enhanced image
    """
    n, objs = cv2.connectedComponents(img)
    ret = []
    print(objs.shape)
    for i in range(n):
        mask = np.asarray(objs == i, dtype=np.uint8) * 255
        # remove background. 255 is threshold
        if np.sum(np.bitwise_and(mask, img)) < 255:
            continue
        x, y, w, h = cv2.boundingRect(mask)
        # should apply bounding check in the future.
        ret.append(mask[y-5:y+h+5, x-5:x+w+5])
    return ret


def edge_detection(img: np.ndarray) -> np.ndarray:
    """
    img: grayscale img of a part
    return: edges
    """
    edge = cv2.Canny(img, 100, 200)  #

    return edge


def dfs_outer_sort_points(map: List[List[str]], row: int, col: int, visited: List[List[bool]], outer_sorted_points: List[Tuple[int, int]], step: int):
    # 上、下、左、右、斜對角
    # 逆時針排序
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1),
                  (-1, -1), (1, -1), (1, 1), (-1, 1)]
    visited[row][col] = True
    outer_sorted_points.append((row, col))
    map[row][col] = str(step[0])  # 標記已經走過的點
    for direction in directions:
        new_row = row + direction[0]
        new_col = col + direction[1]
        # 檢查新的位置是否越界且未訪問過
        if 0 <= new_row < len(map) and 0 <= new_col < len(map[0]) and map[new_row][new_col] == '.' and not visited[new_row][new_col]:
            step[0] += 1
            if step[0] == 10:
                step[0] = 0  # 歸零(比較好看)
            if(map[row][col+1] == '@'):  # 確定正確方向後就改回來
                map[row][col+1] = '.'
            if(map[row-1][col+1] == '@'):
                map[row-1][col+1] = '.'
            if(map[row+1][col+1] == '@'):
                map[row+1][col+1] = '.'
            dfs_outer_sort_points(map, new_row, new_col,
                                  visited, outer_sorted_points, step)
        elif 0 <= new_row < len(map) and 0 <= new_col < len(map[0]) and new_row == row - 1 and new_col == col + 1:
            # 要擴大到第二層搜尋
            two_level_directions = [(-2, 0), (0, -2), (2, 0), (0, 2), (-2, -1), (-1, -2), (1, -2),
                                    (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -2), (2, -2), (2, 2), (-2, 2)]
            for two_level_direction in two_level_directions:
                new_row = row + two_level_direction[0]
                new_col = col + two_level_direction[1]
                # 檢查新的位置是否越界且未訪問過
                if 0 <= new_row < len(map) and 0 <= new_col < len(map[0]) and map[new_row][new_col] == '.' and not visited[new_row][new_col]:
                    step[0] += 1
                    if step[0] == 10:
                        step[0] = 0  # 歸零(比較好看)
                    if(map[row][col+1] == '@'):  # 確定正確方向後就改回來
                        map[row][col+1] = '.'
                    if(map[row-1][col+1] == '@'):
                        map[row-1][col+1] = '.'
                    if(map[row+1][col+1] == '@'):
                        map[row+1][col+1] = '.'
                    dfs_outer_sort_points(
                        map, new_row, new_col, visited, outer_sorted_points, step)


def dfs_outer_sort_map(map: List[List[str]]) -> List[Tuple[int, int]]:
    rows = len(map)
    cols = len(map[0])
    visited = [[False] * cols for _ in range(rows)]
    outer_sorted_points = []
    step = [0]
    jump_two_level = False  # 跳出兩層迴圈
    # 最外圈要逆時針排序
    for i in range(rows):
        for j in range(cols):
            if map[i][j] == '.' and not visited[i][j]:
                if(map[i][j+1] == '.'):
                    map[i][j+1] = '@'  # 先改成@不要讓他一開始往反方向跑
                if(map[i-1][j+1] == '.'):
                    map[i-1][j+1] = '@'
                if(map[i+1][j+1] == '.'):
                    map[i+1][j+1] = '@'
                dfs_outer_sort_points(map, i, j, visited,
                                      outer_sorted_points, step)
                jump_two_level = True
                break  # 跳出最外圈輪廓
        if jump_two_level:
            break

    return outer_sorted_points


def dfs_inter_sort_points(map: List[List[str]], row: int, col: int, visited: List[List[bool]], inter_sorted_points: List[Tuple[int, int]], inter_step: int):
    # 上、下、左、右、斜對角
    # 順時針排序
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1),
                  (-1, 1), (1, 1), (1, -1), (-1, -1)]
    visited[row][col] = True
    inter_sorted_points.append((row, col))
    map[row][col] = str(inter_step[0])  # 標記已經走過的點
    for direction in directions:
        new_row = row + direction[0]
        new_col = col + direction[1]
        # 檢查新的位置是否越界且未訪問過
        if 0 <= new_row < len(map) and 0 <= new_col < len(map[0]) and map[new_row][new_col] == '.' and not visited[new_row][new_col]:
            inter_step[0] += 1
            if inter_step[0] == 10:
                inter_step[0] = 0  # 歸零(比較好看)
            if(map[row][col-1] == '@'):  # 確定正確方向後就改回來
                map[row][col-1] = '.'
            if(map[row-1][col-1] == '@'):
                map[row-1][col-1] = '.'
            if(map[row+1][col-1] == '@'):
                map[row+1][col-1] = '.'

            dfs_inter_sort_points(map, new_row, new_col,
                                  visited, inter_sorted_points, inter_step)
        elif 0 <= new_row < len(map) and 0 <= new_col < len(map[0]) and new_row == row - 1 and new_col == col - 1:
            # 要擴大到第二層搜尋
            two_level_directions = [(-2, 0), (0, 2), (2, 0), (0, -2), (-2, 1), (-1, 2), (1, 2),
                                    (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 2), (2, 2), (2, -2), (-2, -2)]
            for two_level_direction in two_level_directions:
                new_row = row + two_level_direction[0]
                new_col = col + two_level_direction[1]
                # 檢查新的位置是否越界且未訪問過
                if 0 <= new_row < len(map) and 0 <= new_col < len(map[0]) and map[new_row][new_col] == '.' and not visited[new_row][new_col]:
                    inter_step[0] += 1
                    if inter_step[0] == 10:
                        inter_step[0] = 0  # 歸零(比較好看)
                    if(map[row][col-1] == '@'):  # 確定正確方向後就改回來
                        map[row][col-1] = '.'
                    if(map[row-1][col-1] == '@'):
                        map[row-1][col-1] = '.'
                    if(map[row+1][col-1] == '@'):
                        map[row+1][col-1] = '.'
                    dfs_inter_sort_points(
                        map, new_row, new_col, visited, inter_sorted_points, inter_step)


def dfs_inter_sort_map(map: List[List[str]]) -> List[Tuple[int, int]]:
    rows = len(map)
    cols = len(map[0])
    visited = [[False] * cols for _ in range(rows)]
    inter_sorted_points = []
    inter_step = [0]
    jump_two_level = False  # 跳出兩層迴圈
    # 內圈要順時針排序
    for i in range(rows):
        for j in range(cols):
            if map[i][j] == '.' and not visited[i][j]:
                if(map[i][j-1] == '.'):
                    map[i][j-1] = '@'  # 先改成@不要讓他一開始往反方向跑
                if(map[i-1][j-1] == '.'):
                    map[i-1][j-1] = '@'
                if(map[i+1][j-1] == '.'):
                    map[i+1][j-1] = '@'
                dfs_inter_sort_points(map, i, j, visited,
                                      inter_sorted_points, inter_step)
                jump_two_level = True
                break  # 跳出最外圈輪廓
        if jump_two_level:
            break

    return inter_sorted_points


def smooth_angles(angles):
    angles_origin = angles.copy()
    for i, _ in enumerate(angles_origin):
        # tuning the window size here
        # apply gaussian filter
        window_size = 3    # odd number
        kern = [math.exp(-0.5 * (x - 2)**2 / 1.5**2) for x in range(window_size)]
        kern = [x / sum(kern) for x in kern]
        # solve the boundary problem with circular list
        cand = None
        if i < window_size // 2:
            cand = angles_origin[i - window_size // 2:] + angles_origin[:i + window_size // 2 + 1]
        elif i >= len(angles_origin) - window_size // 2:
            cand = angles_origin[i - window_size // 2:] + angles_origin[:i + window_size // 2 + 1]
        else:
            cand = angles_origin[i - window_size // 2:i + window_size // 2 + 1]
        # apply gaussian filter
        
        angles[i] = sum([x * y for x, y in zip(kern, cand)])


def angle(map: List[List[str]], points_set):
    """
    my implementation
    """
    angles = []
    N = len(points_set)
    triangle = True
    inside = True
    # 先算各個點的內角角度
    for i in range(N):
        # 先判斷三點是否是垂直一條線(垂直無法算斜率判斷共線)
        if(points_set[(i + 3) % N][1] == points_set[(i - 3) % N][1] == points_set[i][1]):
            # 內角180度
            angles.append(180.0)
            triangle = False
        # 判斷三點是否共線(算斜率不可以有垂直部分)
        else:
            bc_x_diff = points_set[(i + 3) % N][1] - points_set[i][1]
            bc_y_diff = points_set[(i + 3) % N][0] - points_set[i][0]
            fc_x_diff = points_set[(i - 3) % N][1] - points_set[i][1]
            fc_y_diff = points_set[(i - 3) % N][0] - points_set[i][0]
            bf_x_diff = points_set[(i + 3) % N][1] - points_set[(i - 3) % N][1]
            bf_y_diff = points_set[(i + 3) % N][0] - points_set[(i - 3) % N][0]
            if bc_x_diff != 0 and fc_x_diff != 0 and bf_x_diff != 0:
                bc_m = bc_y_diff / bc_x_diff
                fc_m = fc_y_diff / fc_x_diff
                bf_m = bf_y_diff / bf_x_diff
                if(bc_m == fc_m == bf_m):  # 三點共線內角180度
                    angles.append(180.0)
                    triangle = False
                else:  # 三點沒共線可以形成三角形
                    triangle = True
            else:  # 如果有垂直線但三點也沒垂直共線也是三角形
                triangle = True

        # 判斷完是否共線後就可以用三角形來算角度
        # 利用三角形餘弦定理求內角度是否大於180
        if triangle == True:
            # 先判斷目前計算的三角形角度是書法字的內角還是外角
            # 先求前後兩點的中點
            midpoint_x = (points_set[(i + 3) % N][0] +
                          points_set[(i - 3) % N][0]) / 2
            midpoint_y = (points_set[(i + 3) % N][1] +
                          points_set[(i - 3) % N][1]) / 2
            # 如果中點不是整數就找附近的點判斷
            if isinstance(midpoint_x, int) != 0 and isinstance(midpoint_y, int):
                midpoint_x += 0.5
            elif isinstance(midpoint_x, int) and isinstance(midpoint_y, int) != 0:
                midpoint_y += 0.5
            elif isinstance(midpoint_x, int) != 0 and isinstance(midpoint_y, int) != 0:
                midpoint_x += 0.5
                midpoint_y += 0.5

            if map[int(midpoint_x)][int(midpoint_y)] == '#':
                inside = False
            else:
                inside = True

            # 算出來的角度是字的內角
            if inside == True:
                b = math.sqrt(fc_x_diff**2 + fc_y_diff**2)
                c = math.sqrt(bf_x_diff**2 + bf_y_diff**2)
                f = math.sqrt(bc_x_diff**2 + bc_y_diff**2)
                cosine_value = (f**2 + b**2 - c**2)/(2*f*b)
                angles.append((math.acos(cosine_value) * 180) / math.acos(-1))
            # 算出來的角度是字的外角所以要用360減一次
            else:
                b = math.sqrt(fc_x_diff**2 + fc_y_diff**2)
                c = math.sqrt(bf_x_diff**2 + bf_y_diff**2)
                f = math.sqrt(bc_x_diff**2 + bc_y_diff**2)
                cosine_value = (f**2 + b**2 - c**2)/(2*f*b)
                angles.append(
                    360 - ((math.acos(cosine_value) * 180) / math.acos(-1)))

            '''
            print("印出內角=============================")
            print(i, points_set[i][0], points_set[i][1])
            print(i, points_set[(i-3) % N][0], points_set[(i-3) % N][1])
            print(i, points_set[(i+3) % N][0], points_set[(i+3) % N][1])
            print(angles[i], inside)
            '''
    smooth_angles(angles)
    return angles


#從一區C_points中找出一點當C_points
def real_C_point(C_points):
    N = len(C_points)
    real_C = []
    seg_len = 0
    for i in range(N-1):
        seg_len += 1
        x_diff = C_points[i][0] - C_points[i+1][0]
        y_diff = C_points[i][1] - C_points[i+1][1]
        distance = math.sqrt(x_diff**2 + y_diff**2)
        if(distance > 5) or (i == N-2):
            seg_len += 1
            seg_len = seg_len // 2
            real_C.append(C_points[i - seg_len + 1])
            seg_len = 0

    return real_C


#判斷C-points的角度參數需隨著字調整還有C-points的準確率也要手動修正
def order_point_and_find_Cpoints(img: np.ndarray, con: np.ndarray) -> np.ndarray:
    """
    img: edges
    returns: sequence of points
    """
    all_points = []

    h0, w0 = np.shape(con)
    map = [[''] * w0 for _ in range(h0)]  # 將二值圖轉成map做dfs(視覺化)
    for i in range(h0):
        for j in range(w0):
            if con[i, j] > 128:
                map[i][j] = 'i'  # 輪廓內部設為i
            else:
                map[i][j] = '#'

    vs = []
    h, w = np.shape(img)
    # 將提取出的邊緣edge設為. 外部設為# 內部設為i
    for i in range(h):
        for j in range(w):
            if img[i, j] > 128:
                vs.append((i, j))
                map[i][j] = '.'
            else:
                if map[i][j] != 'i':
                    map[i][j] = '#'


    # 外輪廓逆時針排序後
    outer_sorted_points = dfs_outer_sort_map(map)
    for i in range(len(outer_sorted_points)):
        all_points.append((outer_sorted_points[i][0], outer_sorted_points[i][1]))
    # 排序後做貝賽爾曲線擬合(要評估看看要不要用)
    #bezier(outer_sorted_points)

    C_points = []
    outer_angle = angle(map, outer_sorted_points)
    for i in range(len(outer_angle)):
        if(outer_angle[i] > 220):     #220
            C_points.append((outer_sorted_points[i][0], outer_sorted_points[i][1]))
   
    # 動態生成多個陣列並以不同的變數名稱保存它們
    #inter_sorted_dict = {}
    #inter_count = 0  # 內圈數量 從0開始
    jump_three_level = False  # 跳出三層迴圈
    # 去判斷是否有內圈輪廓，因為內圈輪廓可能有多個所以用迴圈
    while True:
        for i in range(h):
            for j in range(w):
                if map[i][j] == '.':  # 尚有內圈輪廓
                    '''
                    inter_array_name = f"{inter_count}"
                    inter_sorted_dict[inter_array_name] = dfs_inter_sort_map(map)
                    for k in range(len(inter_sorted_dict[inter_array_name])):
                        all_points.append((inter_sorted_dict[k][0], inter_sorted_dict[k][1]))    
                    inter_angle = angle(map, inter_sorted_dict[inter_array_name])
                    for k in range(len(inter_angle)):
                        if(inter_angle[k] > 220):
                            #map[inter_sorted_dict[inter_array_name][i][0]][inter_sorted_dict[inter_array_name][i][1]] = 'C'
                            C_points.append((inter_sorted_dict[inter_array_name][k][0], inter_sorted_dict[inter_array_name][k][1]))
                    inter_count += 1
                    '''
                    inter_sorted_points = []
                    inter_sorted_points = dfs_inter_sort_map(map)
                    for k in range(len(inter_sorted_points)):
                        all_points.append((inter_sorted_points[k][0], inter_sorted_points[k][1]))
                    inter_angle = angle(map, inter_sorted_points)
                    for k in range(len(inter_angle)):
                        if(inter_angle[k] > 220):    #220
                            #map[inter_sorted_dict[inter_array_name][i][0]][inter_sorted_dict[inter_array_name][i][1]] = 'C'
                            C_points.append((inter_sorted_points[k][0], inter_sorted_points[k][1]))
                elif(i == h-1 and j == w-1):  # 如果找到最後一個還是沒有就代表沒有內圈輪廓了
                    jump_three_level = True
                    break
            if jump_three_level:
                break
        if jump_three_level:
            break

    #print(C_points)
    #將一區中多個C點取中間點一個當real C point
    real_C = real_C_point(C_points)
    #del real_C[:] 刪除所有real_C
    print(real_C)

    for i in range(len(real_C)):
        map[real_C[i][0]][real_C[i][1]] = 'C'
    
    print("印出排序且找出C點後的map")
    for i in range(h):
        print(i, end="")
        for j in range(w):
            if map[i][j]=='#' or map[i][j]=='i':
                print(" ",end=" ")
            else:
                print(map[i][j],end=" ")
        print()
    
    return map, real_C, all_points


#判斷水平跟垂直關係的距離也要判斷(需調整距離參數)還有判斷垂直跟水平的斜率差參數也需調整
def classify_C_point(map: List[List[str]], C_points):

    #將C_point定好名字 2a,2b,4c,4d,6e,6f...
    C_points_name = []
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','n','o','p','q','r','s','t','u','v','w','x','y','z']
    letters_count = 0
    number = 2
    for i in range(len(C_points)):
        if i%2 != 0:
            C_points_name.append(str(number)+ letters[letters_count])
            number += 2
            letters_count += 1
        else:
            C_points_name.append(str(number) + letters[letters_count])
            letters_count += 1
    #print(C_points_name)
    #將兩兩一對抓出來檢查相對關係
    
    total_pairs = []
    co_linear = []
    parallel = []
    same_sub_correspond = []
    co_linear_name = []
    parallel_name = []
    same_sub_correspond_name = []
    check_no_in_class = []
    #先全部初始為0
    for i in range(len(C_points)):
        check_no_in_class.append(0)

    for i in range(len(C_points_name)):
        for j in range(i+1, len(C_points_name)):
            pair = (C_points_name[i], C_points_name[j])
            total_pairs.append(pair)
            is_same_sub_correspond = True
            is_co_linear = True
            is_parallel = True

            #判斷是否co-linear 垂直 除了判斷垂直還要判斷距離是否離太遠
            if(abs(C_points[i][1] - C_points[j][1])) < 10 and math.sqrt((C_points[i][1] - C_points[j][1])**2 + (C_points[i][0] - C_points[j][0])**2) < 20:
                #判斷兩點連線中是否有切到字的內部
                center_x = (C_points[i][0] + C_points[j][0]) // 2
                center_y = (C_points[i][1] + C_points[j][1]) // 2
                if (map[center_x][center_y]=='i'):
                    #判斷中點往左右走是否會碰到邊界或到外面，如果有代表兩點可能在同一線上
                    for k in range(3):
                        if(map[center_x][center_y+k]!='i'):
                            is_co_linear = False
                    for k in range(3):
                        if(map[center_x][center_y-k]!='i'):
                            is_co_linear = False
                    if(is_co_linear == True):
                        co_linear_name.append(pair)
                        co_linear.append((i,j))
                        #改成有被分到類1
                        check_no_in_class[i] = 1
                        check_no_in_class[j] = 1
            #判斷是否parallel or same_sub_correspond 水平 除了判斷水平還要判斷距離是否離太遠
            elif(abs(C_points[i][0] - C_points[j][0])) < 10 and math.sqrt((C_points[i][1] - C_points[j][1])**2 + (C_points[i][0] - C_points[j][0])**2) < 20:
                #判斷是否為same_sub_correspond
                if C_points_name[i][0] == C_points_name[j][0]:
                    #判斷兩點連線中是否有切到字的內部
                    center_x = (C_points[i][0] + C_points[j][0]) // 2
                    center_y = (C_points[i][1] + C_points[j][1]) // 2
                    if (map[center_x][center_y]=='i'):
                        for k in range(3):
                            if(map[center_x+k][center_y]!='i'):
                                is_same_sub_correspond = False
                        for k in range(3):
                            if(map[center_x-k][center_y]!='i'):
                                is_same_sub_correspond = False
                        if(is_same_sub_correspond == True):
                            same_sub_correspond_name.append(pair)
                            same_sub_correspond.append((i,j))
                            #改成有被分到類1
                            check_no_in_class[i] = 1
                            check_no_in_class[j] = 1

                            
                else:
                    #判斷是否為parallel 
                    #判斷兩點連線中是否有切到字的內部
                    center_x = (C_points[i][0] + C_points[j][0]) // 2
                    center_y = (C_points[i][1] + C_points[j][1]) // 2
                    if (map[center_x][center_y]=='i'):
                        #判斷中點往左右走是否會碰到邊界或到外面，如果有代表兩點可能在同一線上
                        for k in range(3):
                            if(map[center_x+k][center_y]!='i'):
                                is_parallel = False
                        for k in range(3):
                            if(map[center_x-k][center_y]!='i'):
                                is_parallel = False
                        if(is_parallel == True):
                            parallel_name.append(pair)
                            parallel.append((i,j))
                            #改成有被分到類1
                            check_no_in_class[i] = 1
                            check_no_in_class[j] = 1

    #剩下未被分類的點就要沿著tangent line切
    no_in_class = []
    for i in range(len(check_no_in_class)):
        if check_no_in_class[i] == 0:
            no_in_class.append(i)
    
    print(total_pairs)
    print("印出Co_linear")
    #print(co_linear_name)
    print(co_linear)
    print("印出Parallel")
    #print(parallel_name)
    print(parallel)
    print("印出same_sub_correspond")
    #print(same_sub_correspond_name)
    print(same_sub_correspond)
    print("印出未分到類")
    print(no_in_class)
    
    return co_linear, parallel, same_sub_correspond, no_in_class


def sub_contour_cut_same(img: np.ndarray, order) -> List[np.ndarray]:
    """
    param img: grayscale enhanced image
    """
    n, objs = cv2.connectedComponents(img)
    ret = []
    for i in range(n):
        mask = np.asarray(objs == i, dtype=np.uint8) * 255
        # remove background. 255 is threshold
        if np.sum(np.bitwise_and(mask, img)) < 255:
            continue
        x, y, w, h = cv2.boundingRect(mask)
        # should apply bounding check in the future.
        ret.append(mask[y-5:y+h+5, x-5:x+w+5])
        for _, sub_contour in enumerate(ret):
            edge = edge_detection(sub_contour)
            _, real_C, _ = order_point_and_find_Cpoints(edge, sub_contour)
            if len(real_C) == 0:
                #cv2.imwrite(f"sub_contour_same{order}.png", sub_contour) 
                #cv2.imwrite(f"sub_contour0_same{order}.png", edge)  
                cv2.imwrite(f"output_{order}.png", sub_contour) 
                cv2.imwrite(f"output0_{order}.png", edge)
                inverted_img = ~sub_contour
                cv2.imwrite(f"output1_{order}.png",inverted_img)
                # Fill the corresponding region in the original image with black
                img[y:y+h, x:x+w] = 0
                order += 1
        ret.clear()
    return img, order


#判斷長方形的長寬差需調整參數
def sub_contour_cut_parallel(img: np.ndarray, co_linear, parallel, C_point, order) -> List[np.ndarray]:
    """
    param img: grayscale enhanced image
    """
    n, objs = cv2.connectedComponents(img)
    ret = []
    for i in range(n):
        mask = np.asarray(objs == i, dtype=np.uint8) * 255
        # remove background. 255 is threshold
        if np.sum(np.bitwise_and(mask, img)) < 255:
            continue
        x, y, w, h = cv2.boundingRect(mask)
        # should apply bounding check in the future.
        ret.append(mask[y-5:y+h+5, x-5:x+w+5])
        for _, sub_contour in enumerate(ret):
            edge = edge_detection(sub_contour)
            _, real_C, _ = order_point_and_find_Cpoints(edge, sub_contour)
            if len(real_C) == 0:    #因為C-points準確率的問題所以有些就先不考慮
                if w - h > 20: #判斷為長方形(橫線)
                    #cv2.imwrite(f"sub_contour_parallel{order}.png", sub_contour) 
                    #cv2.imwrite(f"sub_contour0_parallel{order}.png", edge)
                    cv2.imwrite(f"output_{order}.png", sub_contour) 
                    cv2.imwrite(f"output0_{order}.png", edge) 
                    inverted_img = ~sub_contour
                    cv2.imwrite(f"output1_{order}.png",inverted_img)
                    # Fill the corresponding region in the original image with black
                    img[y:y+h, x:x+w] = 0
                    order += 1
        ret.clear()
    cv2.imwrite("1.png",img)
    filled_img = img
    
    #要判斷是不是有切斷的地方
    #將切斷的部分連起來
    is_cut = False
    for i in range(len(co_linear)):
        for j in range(len(parallel)):
            if co_linear[i][0] == parallel[j][0] or co_linear[i][0] == parallel[j][1] or co_linear[i][1] == parallel[j][0] or co_linear[i][1] == parallel[j][1]:
                is_cut = True
                break
        if is_cut == True:
            # 两个點的坐標
            # 因為前面連線的寬度是2會有點連不到的問題，所以連長一點(+-10)
            if C_point[co_linear[i][0]][0] > C_point[co_linear[i][1]][0]:
                point1 = (C_point[co_linear[i][0]][1], C_point[co_linear[i][0]][0]+15)  #+
                point2 = (C_point[co_linear[i][1]][1], C_point[co_linear[i][1]][0]-15)  #-
            else:
                point1 = (C_point[co_linear[i][0]][1], C_point[co_linear[i][0]][0]-15)  #-
                point2 = (C_point[co_linear[i][1]][1], C_point[co_linear[i][1]][0]+15)  #+
            cv2.line(img, point1, point2, (255, 255, 255), 1)
            #連完線後改成-1 -1以免之後做切直線的時候重複連線
            co_linear = list(co_linear)  #將tuple轉乘list
            co_linear[i] = (-1, -1)  #再進行修改
            co_linear = tuple(co_linear)  #將list轉回tuple
    if is_cut == True:
        # 將輪廓內填滿白色
        # 尋找輪廓
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 創建一個與原始圖像大小一樣的空白圖像
        filled_img = np.zeros_like(img)
        # 填充輪廓内部為白色
        for contour in contours:
            cv2.fillConvexPoly(filled_img, contour, (255, 255, 255))

    return filled_img, co_linear, order

 
#判斷長方形的長寬差需調整參數
def sub_contour_cut_co_linear(img: np.ndarray, order) -> List[np.ndarray]:
    """
    param img: grayscale enhanced image
    """
    n, objs = cv2.connectedComponents(img)
    ret = []
    for i in range(n):
        mask = np.asarray(objs == i, dtype=np.uint8) * 255
        # remove background. 255 is threshold
        if np.sum(np.bitwise_and(mask, img)) < 255:
            continue
        x, y, w, h = cv2.boundingRect(mask)
        # should apply bounding check in the future.
        ret.append(mask[y-5:y+h+5, x-5:x+w+5])
        for _, sub_contour in enumerate(ret):
            edge = edge_detection(sub_contour)
            _, real_C, _ = order_point_and_find_Cpoints(edge, sub_contour)
            if len(real_C) == 0:    #因為C-points準確率的問題所以有些就先不考慮
                if (h - w) > 20:   #判斷為長方形(直線)

                    cv2.imwrite(f"output_{order}.png", sub_contour) 
                    cv2.imwrite(f"output0_{order}.png", edge)
                    inverted_img = ~sub_contour
                    cv2.imwrite(f"output1_{order}.png",inverted_img)
                    # Fill the corresponding region in the original image with black
                    img[y:y+h, x:x+w] = 0
                    order += 1
        ret.clear()
    return img, order


def sub_contour_cut_no_class(img: np.ndarray, order) -> List[np.ndarray]:
    """
    param img: grayscale enhanced image
    """
    n, objs = cv2.connectedComponents(img)
    ret = []
    print(objs.shape)
    for i in range(n):
        mask = np.asarray(objs == i, dtype=np.uint8) * 255
        # remove background. 255 is threshold
        if np.sum(np.bitwise_and(mask, img)) < 255:
            continue
        x, y, w, h = cv2.boundingRect(mask)
        # should apply bounding check in the future.
        ret.append(mask[y-5:y+h+5, x-5:x+w+5])
        for _, sub_contour in enumerate(ret):
            edge = edge_detection(sub_contour)
            cv2.imwrite(f"output_{order}.png", sub_contour) 
            cv2.imwrite(f"output0_{order}.png", edge) 
            inverted_img = ~sub_contour
            cv2.imwrite(f"output1_{order}.png",inverted_img)
            # Fill the corresponding region in the original image with black
            img[y:y+h, x:x+w] = 0
            order += 1
        ret.clear()

    return img, order


def quadratic_bezier(t, b_points):
    #計算二階貝賽爾曲線上的點座標(用來計算tangent line用)
    x = (1 - t) ** 2 * b_points[0, 0] + 2 * (1 - t) * t * b_points[1, 0] + t ** 2 * b_points[2, 0]
    y = (1 - t) ** 2 * b_points[0, 1] + 2 * (1 - t) * t * b_points[1, 1] + t ** 2 * b_points[2, 1]
    return x, y


def find_tangent_point(b_points, target_point):
    # 初始化参數值t和初始最小距離
    t = 0
    min_distance = float('inf')

    # 迭代逼近最接近目標點的参数值t
    for i in range(101):  # 使用101個等間距的t值進行迭代
        curr_t = i / 100
        x, y = quadratic_bezier(curr_t, b_points)
        distance = np.linalg.norm([x - target_point[0], y - target_point[1]])
        if distance < min_distance:
            min_distance = distance
            t = curr_t
    return t


def calculate_tangent_line(all_point, C_point_count, real_C):
    cut_x = 0
    cut_y = 0

    for i in range(len(all_point)):
        if (all_point[i][0] == real_C[C_point_count][0]) and (all_point[i][1] == real_C[C_point_count][1]):
            b_points = []
            #xy要顛倒
            b_points.append((all_point[(i-1) % len(all_point)][1] , all_point[(i-1) % len(all_point)][0]))
            b_points.append((all_point[i][1] , all_point[i][0]))
            #b_points.append((all_point[(i-2) % len(all_point)][1] , all_point[(i-2) % len(all_point)][0]))
            b_points.append((all_point[(i+1) % len(all_point)][1] , all_point[(i+1) % len(all_point)][0]))
            #b_points.append((all_point[(i+2) % len(all_point)][1] , all_point[(i+2) % len(all_point)][0]))
            b_points = np.array(b_points)
            #目標點
            target_point = b_points[1]
            #計算在目標點處的參數值t
            t = find_tangent_point(b_points, target_point)

            #在目標點處的切線向量
            x_derivative = 2 * (1 - t) * (b_points[1, 0] - b_points[0, 0]) + 2 * t * (b_points[2, 0] - b_points[1, 0])
            y_derivative = 2 * (1 - t) * (b_points[1, 1] - b_points[0, 1]) + 2 * t * (b_points[2, 1] - b_points[1, 1])
            tangent_vector = np.array([x_derivative, y_derivative])

            print(tangent_vector)
            # 切割距离（可根据需求调整）
            distance = 25
            # 计算切割点的坐标
            cut_x = int(target_point[0] + distance * tangent_vector[0])
            cut_y = int(target_point[1] + distance * tangent_vector[1])
            break
   
    return cut_x, cut_y


def cut_and_connect(img: np.ndarray, co_linear, parallel, same_sub_correspond, no_in_class, real_C, order, all_point):
    
    #先切出same_sub_correspond
    cut_img = img
    for i in range(len(same_sub_correspond)):
        # 两个點的坐標
        point1 = (real_C[same_sub_correspond[i][0]][1], real_C[same_sub_correspond[i][0]][0])
        point2 = (real_C[same_sub_correspond[i][1]][1], real_C[same_sub_correspond[i][1]][0])
        # 在圖像上繪製線段
        cv2.line(cut_img, point1, point2, (0, 0, 0), 2)
        # 畫完後去做切割
        img, order = sub_contour_cut_same(cut_img, order)

    cv2.imwrite('before_same_cut.png', img)

    #再切出橫的部分
    if len(parallel) != 0:
        # erode and dilation(再平滑一次，平滑後再去切割)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=2)
        img = cv2.dilate(img, kernel, iterations=2)
        cut_img = img
        #要切橫的部分要先把水平地方連起來
        for i in range(len(parallel)):
            # 两个點的坐標
            point1 = (real_C[parallel[i][0]][1], real_C[parallel[i][0]][0])
            point2 = (real_C[parallel[i][1]][1], real_C[parallel[i][1]][0])
            # 在圖像上繪製線段
            cv2.line(cut_img, point1, point2, (0, 0, 0), 2)
        #連完線後做切割
        img, co_linear, order = sub_contour_cut_parallel(cut_img, co_linear, parallel, real_C, order)
    cv2.imwrite('before_parallel_cut.png', img)

    
    #再切出直的部分
    if len(co_linear) != 0:
        # erode and dilation(再平滑一次，平滑後再去切割)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=2)
        img = cv2.dilate(img, kernel, iterations=2)
        cut_img = img
        #要切直的部分要先把垂直地方連起來
        for i in range(len(co_linear)):
            #沒有連過線
            if co_linear[i][0] != -1 and co_linear[i][1] != -1:
                # 两个點的坐標
                point1 = (real_C[co_linear[i][0]][1], real_C[co_linear[i][0]][0])
                point2 = (real_C[co_linear[i][1]][1], real_C[co_linear[i][1]][0])
                # 在圖像上繪製線段
                cv2.line(cut_img, point1, point2, (0, 0, 0), 2)
        img, order = sub_contour_cut_co_linear(img, order)
    cv2.imwrite('before_colinear_cut.png', img)
    

    #最後就是未分到類的沿著tangent line切
    #先算tangent line
    if len(no_in_class) != 0:
        # erode and dilation(再平滑一次，平滑後再去切割)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=2)
        img = cv2.dilate(img, kernel, iterations=2)
        cut_img = img
        for i in range(len(no_in_class)):
            cut_x, cut_y = calculate_tangent_line(all_point, no_in_class[i], real_C)
            # 两个點的坐標
            point1 = (real_C[no_in_class[i]][1], real_C[no_in_class[i]][0])
            point2 = (cut_x, cut_y)
            # 在圖像上繪製線段
            cv2.line(cut_img, point1, point2, (0, 0, 0), 2)
        cv2.imwrite('after_no_cut.png', img)

    #畫完要切的線後就做分割
    img, order = sub_contour_cut_no_class(img, order)
    
    '''切「山」時，直的地方要先切，所以橫的放在後面切
    cv2.imwrite('_cut.png', img)
    #再切出橫的部分
    if len(parallel) != 0:
        # erode and dilation(再平滑一次，平滑後再去切割)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=2)
        img = cv2.dilate(img, kernel, iterations=2)
        cut_img = img
        #要切橫的部分要先把水平地方連起來
        for i in range(len(parallel)):
            # 两个點的坐標
            point1 = (real_C[parallel[i][0]][1], real_C[parallel[i][0]][0])
            point2 = (real_C[parallel[i][1]][1], real_C[parallel[i][1]][0])
            # 在圖像上繪製線段
            cv2.line(cut_img, point1, point2, (0, 0, 0), 2)
        cv2.imwrite('after_parallel_cut.png', img)
        #連完線後做切割
        img, co_linear, order = sub_contour_cut_parallel(cut_img, co_linear, real_C, order)
    cv2.imwrite('before_parallel_cut.png', img)
    #畫完要切的線後就做分割
    img, order = sub_contour_cut_no_class(img, order)
    '''
        
    cv2.imwrite('before_no_cut.png', img)
    return order


def main():
    img = cv2.imread("sample_data/zuo.jpg")  # sample_data/syun.jpg
    img = enhance(img)
    #cv2.imwrite("./enhanced.png", img)  # outputs/enhanced.png
    cons = sub_contour(img)
    #輸出切割後的筆畫順序
    order = 0
    for i, con in enumerate(cons):
        
        edge = edge_detection(con)
        cv2.imwrite(f"./tmp0_{i}.png", con)  # outputs/tmp_{i}.png
        cv2.imwrite(f"./tmp_{i}.png", edge)  # outputs/tmp_{i}.png
        map, real_C, all_point = order_point_and_find_Cpoints(edge, con)
        co_linear, parallel, same_sub_correspond, no_in_class = classify_C_point(map, real_C)
        order = cut_and_connect(con, co_linear, parallel, same_sub_correspond, no_in_class, real_C, order, all_point)


if __name__ == "__main__":
    main()








#原論文方式(有問題)
def calculate_angle(points_set):

    angles = []
    N = len(points_set)
    for i in range(N - 1):
        x1, y1 = points_set[i]
        x2, y2 = points_set[i + 1]
        angle = math.atan2(y2 - y1, -(x2 - x1))
        angles.append(angle)

    x1, y1 = points_set[-1]
    x2, y2 = points_set[0]
    angle = math.atan2(y2 - y1, -(x2 - x1))
    angles.append(angle)
    # smooth angles here
    # smooth_angles(angles)
    return angles

def calculate_P(points_set, angle_set):
    P_values = []
    N = len(points_set)
    for i in range(N):
        pi_sum = 0.0
        for j in range(-1, 3):
            x_diff = points_set[(i + j) % N][0] - \
                points_set[(i + j - 1) % N][0]
            y_diff = points_set[(i + j) % N][1] - \
                points_set[(i + j - 1) % N][1]
            distance = math.sqrt(x_diff**2 + y_diff**2)
            pi_sum += distance
        pi_value = (angle_set[(i + 2) % N] -
                    angle_set[(i - 2) % N]) / pi_sum  # +2-2
        P_values.append(pi_value)
    return P_values

#smooth(有問題)
def Douglas_Peucker(points_set):

    #進行多邊形逼近
    epsilon = 0.1  # 控制逼近精度的參數
    approx = cv2.approxPolyDP(points_set, epsilon, closed=True)

    # 繪製逼近後的多邊形
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    approx[:, :, 1] = image.shape[0] - approx[:, :, 1]

    image = cv2.drawContours(image, [approx], contourIdx=0, color=(255, 255, 255), thickness=2)

    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    #cv2.imwrite("Douglas_Peucker.png", rotated_image)

    return rotated_image

#貝賽爾曲線(需評估是否需要放)
def decic_bezier(points, t):
    #計算十階貝賽爾曲線上的點座標
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = points
    x = (1 - t) ** 10 * p0[0] + 10 * (1 - t) ** 9 * t * p1[0] + 45 * (1 - t) ** 8 * t ** 2 * p2[0] + 120 * (1 - t) ** 7 * t ** 3 * p3[0] + 210 * (1 - t) ** 6 * t ** 4 * p4[0] + 252 * (1 - t) ** 5 * t ** 5 * p5[0] + 210 * (1 - t) ** 4 * t ** 6 * p6[0] + 120 * (1 - t) ** 3 * t ** 7 * p7[0] + 45 * (1 - t) ** 2 * t ** 8 * p8[0] + 10 * (1 - t) * t ** 9 * p9[0] + t ** 10 * p10[0]
    y = (1 - t) ** 10 * p0[1] + 10 * (1 - t) ** 9 * t * p1[1] + 45 * (1 - t) ** 8 * t ** 2 * p2[1] + 120 * (1 - t) ** 7 * t ** 3 * p3[1] + 210 * (1 - t) ** 6 * t ** 4 * p4[1] + 252 * (1 - t) ** 5 * t ** 5 * p5[1] + 210 * (1 - t) ** 4 * t ** 6 * p6[1] + 120 * (1 - t) ** 3 * t ** 7 * p7[1] + 45 * (1 - t) ** 2 * t ** 8 * p8[1] + 10 * (1 - t) * t ** 9 * p9[1] + t ** 10 * p10[1]
    return x, y

def bezier(points):
    #創建空白圖像
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    #原始輪廓點(一次抓三個出來做貝賽爾曲線擬合)
    for i in range(0, len(points), 11):
        b_points = []
        #xy要顛倒
        b_points.append((points[i][1] , points[i][0]))
        b_points.append((points[(i+1) % len(points)][1] , points[(i+1) % len(points)][0]))
        b_points.append((points[(i+2) % len(points)][1] , points[(i+2) % len(points)][0]))
        b_points.append((points[(i+3) % len(points)][1] , points[(i+3) % len(points)][0]))
        b_points.append((points[(i+4) % len(points)][1] , points[(i+4) % len(points)][0]))
        b_points.append((points[(i+5) % len(points)][1] , points[(i+5) % len(points)][0]))
        b_points.append((points[(i+6) % len(points)][1] , points[(i+6) % len(points)][0]))
        b_points.append((points[(i+7) % len(points)][1] , points[(i+7) % len(points)][0]))
        b_points.append((points[(i+8) % len(points)][1] , points[(i+8) % len(points)][0]))
        b_points.append((points[(i+9) % len(points)][1] , points[(i+9) % len(points)][0]))
        b_points.append((points[(i+10) % len(points)][1] , points[(i+10) % len(points)][0]))
        b_points = np.array(b_points)
        
        #生成更多點以繪製曲線
        t_values = np.linspace(0, 1, 100)
        curve_points = [decic_bezier(b_points, t) for t in t_values]

        #將輪廓點座標轉換為整數類型
        b_points = b_points.astype(int)
        curve_points = np.array(curve_points).astype(int)

        #繪製原始輪廓跟貝賽爾曲線
        #cv2.polylines(image, [b_points], isClosed=False, color=(0, 0, 255), thickness=1)
        cv2.polylines(image, [curve_points], isClosed=False, color=(255, 0, 0), thickness=1)
        

    #保存圖像
    cv2.imwrite('bezier_curve.png', image)

    return
