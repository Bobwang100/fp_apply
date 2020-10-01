import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('carwheel.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

# template = cv2.imread('cir.jpg')
# w, h, _ = template.shape[::-1]
#
# res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where( res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img, pt, (pt[0] + 10*w, pt[1] + h), (0,0,255), 2)
def crop_img(img,img_name,ratio):     # ratio表示裁剪后左上角点的宽度坐标占宽度的比值
    H,W = img.shape
    for i in range(10):
        r = np.random.uniform(0,ratio)
        Len_l = int(np.around(W*r))           # 图片宽度上左边切除的尺寸，以此为最大基准,其他位置处剪裁小于此尺寸
        Len_rud = np.random.randint(0, Len_l)
        img_c = img[Len_rud:H-Len_rud,Len_l:W-Len_rud]
        cv2.imwrite('./' + img_name + 'crop' + str(i) + '.jpg', img_c)
def rotate_img(img, img_name, angle):
    H, W = img.shape
    for i in range(10):
        angle_ = np.random.randint(-angle,angle)
        cX, cY = W // 2, H // 2
        M = cv2.getRotationMatrix2D((cX, cY), -angle_, 1.0)
        rotate_img = cv2.warpAffine(img, M, (W, H))
        cv2.imwrite('./' + img_name + 'rotate' + str(i) + '.jpg', rotate_img)
def flip_img(img, img_name):
    Img_h = cv2.flip(img, 1)  # 水平镜像
    Img_v = cv2.flip(img, 0)  # 垂直镜像
    Img_d = cv2.flip(img, -1)  # 对角镜像
    cv2.imwrite('./' + img_name + 'fliph' + '.jpg', Img_h)
    cv2.imwrite('./' + img_name + 'flipv' + '.jpg', Img_v)
    cv2.imwrite('./' + img_name + 'flipd' + '.jpg', Img_d)
def contrast_bright_img(img, img_name, alpha, beta):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    for i in range(10):
        alpha_ = np.random.uniform(alpha, 1)
        beta_ = np.random.uniform(0,beta)
        img_cb = cv2.addWeighted(img, alpha_, blank, 1-alpha_, beta_)
        cv2.imwrite('./' + img_name + 'cb' + str(i) + '.jpg', img_cb)
def filter2d_img(img,img_name):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_f = cv2.filter2D(img, -1, kernel)
    cv2.imwrite('./' + img_name + 'filter'+ '.jpg', img_f)


def sp_noise_img(image,img_name, prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    cv2.imwrite('./' + img_name + 'spnoise' + '.jpg', output)


def gasuss_noise_img(image,img_name,mean=0, var=0.0001):
    '''
      添加高斯噪声
      mean : 均值
      var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    cv2.imwrite('./' + img_name + 'gasuss' + '.jpg',out)


# img_resize = cv2.resize(img, (W-30, H-30))
# cv2.imshow('img_crop',img_crop)
# cv2.imshow('img_resize',img_resize)
# print('shape:', img.shape, img_crop.shape, img_resize.shape)
# crop_img(img_gray,'compare', 0.2)
# rotate_img(img_gray,'coapare',50)
# flip_img(img_gray,'carwheel')
# contrast_bright_img(img_gray, 'carwheel',0.5,50)
# filter2d_img(img_gray, 'carwheel')
# sp_noise_img(img_gray,'haha',0.01)
# gasuss_noise_img(img_gray,'hehe')
# cv2.waitKey(0)
# cv2.destroyAllWindows()





