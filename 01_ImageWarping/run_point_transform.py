from math import log
import cv2
import numpy as np
import gradio as gr
from scipy.spatial.distance import cdist

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    source_pts: 2-dim variable, in 2nd dim: first is the width/x, second is the height/y
    注意图片的存储是高度在前，宽度在后
    Return
    ------
        A deformed image.
    """
    # warped_image = np.array(image)
    h, w = image.shape[:2]
    warped_image = np.zeros_like(image)

    ### FILL: 基于MLS or RBF 实现 image warping
    # 基于RBF(Radical Basis Function)

    # 建立目标点之间的距离矩阵，构建反向传播查找
    target_pts_dists = cdist(target_pts, target_pts)
    print(f"target points distances are \n{target_pts_dists}")

    target_pts_weight = rbf_kernel(target_pts_dists, eps, alpha)
    print(f"target points weight are \n{target_pts_weight}")

    """
    # 使用纯径向函数的方式
    warping_params_x = np.linalg.solve(target_pts_weight, source_pts[:, 0])
    warping_params_y = np.linalg.solve(target_pts_weight, source_pts[:, 1])
    print(f"shape of warping parameters is {np.shape(warping_params_x)}")
    # input(f"Warping Parameters is \n{warping_params_x}, {warping_params_y}")

    # 对warping后的图像每个像素进行反向遍历
    for w_index in range(w):
        for h_index in range(h):
            # 计算当前的距离和权重
            curr_pt_dist = cdist([[w_index, h_index]], target_pts)
            curr_pt_weight = rbf_kernel(curr_pt_dist, eps, alpha)

            origin_x = np.dot(curr_pt_weight, warping_params_x).astype(int)
            origin_y = np.dot(curr_pt_weight, warping_params_y).astype(int)
            if (origin_x in range(w)) and (origin_y in range(h)):
                print(f"Original coordinate for ({h_index, w_index})/{np.shape(image)[:2]} is ({origin_y, origin_x})")
                warped_image[h_index, w_index] = image[origin_y, origin_x]
            else:
                print(f"Original coordinate for ({h_index, w_index})/{np.shape(image)[:2]} is out of range")

    """
    # 添加一阶多项式项
    polyno = np.hstack(( target_pts, np.ones((target_pts.shape[0], 1)) ))   # 添加x, y, 1项
    bottom_matrix = np.hstack(( np.transpose(polyno), np.ones((3, 3)) ))
    target_pts_weight = np.hstack([target_pts_weight, polyno])
    target_pts_weight = np.vstack([target_pts_weight, bottom_matrix])

    goal_x = np.hstack([source_pts[:, 0], np.zeros(3)])
    goal_y = np.hstack([source_pts[:, 1], np.zeros(3)])
    warping_params_x = np.linalg.solve(target_pts_weight, goal_x)
    warping_params_y = np.linalg.solve(target_pts_weight, goal_y)
    # input(f"Warping Parameters is \n{warping_params_y}, {warping_params_x}")

    # 对warping后的图像每个像素进行反向遍历
    for w_index in range(w):
        for h_index in range(h):
            # 计算当前的距离和权重
            curr_pt_dist = cdist([[w_index, h_index]], target_pts)[0]
            curr_pt_weight = rbf_kernel(curr_pt_dist, eps, alpha)
            curr_pt_weight = np.hstack([curr_pt_weight, [w_index, h_index, 1]])

            origin_y = np.dot(curr_pt_weight, warping_params_y).astype(int)
            origin_x = np.dot(curr_pt_weight, warping_params_x).astype(int)
            if (origin_x in range(w)) and (origin_y in range(h)):
                print(f"Original coordinate for ({h_index, w_index})/{np.shape(image)[:2]} is ({origin_y, origin_x})")
                warped_image[h_index, w_index] = image[origin_y, origin_x]
            else:
                print(f"Original coordinate for ({h_index, w_index})/{np.shape(image)[:2]} is out of range")
 
    return warped_image


def rbf_kernel(distance, epsilon, alpha):
    """
    Radial Basis Function (RBF) kernel for calculating weights.
    """
    # return np.exp(-(distance ** 2) / (2 * (epsilon ** 2))) * alpha
    # return np.exp(-(distance ** 2)*1e-4) * alpha
    # return np.where(distance != 0, (distance**2) * log(distance+epsilon), 0)
    return (distance**2) * np.log(distance+epsilon)


def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=600, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=600, height=500)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
