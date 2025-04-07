# mpl.rcParams['font.sans-serif'] = ['SimHei','SongTi', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
# mpl.rcParams['font.size'] = 12  # 字体大小
# mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import math
import os
import time

import cv2
import fitz  # 导入本模块需安装pymupdf库
import numpy as np
import xlrd
import xlutils.copy
import xlwt
from PIL import Image
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MultipleLocator

from xml_augmentation.read_xml import xml_parse

"计算检测结果中目标区域和相邻背景区域的相关数值. img = ./ResUnet/test_small_results/***.bmp"
def Img_out(img):
    im = cv2.imread(img)
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    switch_xml = {"cloud1": "XML/cloud1XML/", "cloud2": "XML/cloud2XML/", "cloud3": "XML/cloud3XML/", "cloud4": "XML/cloud4XML/",
                  "cloud5": "XML/cloud5XML/", "cloud6": "XML/cloud6XML/", "cloud7": "XML/cloud7XML/","cloud8": "XML/cloud8XML/"}  # XML目录
    img_xml = switch_xml[img.split("_")[1]] + img.split("/")[3].split(".")[0] + ".xml"
    status = xml_parse(img_xml)[0]  # GT,标准框 二维列表[[x,y,w,h]] 左上角坐标和宽高
    ob = im_gray[status[1]:status[1] + status[3], status[0]:status[0] + status[2]]  # 取图像框中的部分，先高后宽
    mean_t = np.mean(ob)  # 目标区域平均像素值

    "判断GT是否处在图像边缘"
    if status[1] - 20 < 0:
        ob_large = im_gray[0:status[1] + status[3] + 20, status[0] - 20: status[0] + status[2] + 20]  # 到顶
    else:
        ob_large = im_gray[status[1] - 20:status[1] + status[3] + 20,
                   status[0] - 20: status[0] + status[2] + 20]  # 框各边加20

    lar = np.array(ob_large).astype(np.int)

    "以目标区域GT为中间，向两边扩充20个像素，作为背景区域"
    if status[1] - 20 < 0:
        lar[status[1]:status[3], 20:20 + status[2]] = np.ones(
            lar[status[1]:status[3], 20:20 + status[2]].shape) * -1  # 将框内的像素值全部置-1
    else:
        lar[20:20 + status[3], 20:20 + status[2]] = np.ones(
            lar[20:20 + status[3], 20:20 + status[2]].shape) * -1  # 将框内的像素值全部置-1
    std = np.std(lar[lar > -1])  # 背景区域的标准差
    mean_b = np.mean(lar[lar > -1])  # 背景区域的平均像素值
    # print("\tmean_t = %f\tmean_b = %f" % (mean_t, mean_b))

    im_gray = im_gray.astype(int)
    im_gray[status[1]:status[1] + status[3], status[0]: status[0] + status[2]] = np.ones(
        im_gray[status[1]:status[1] + status[3], status[0]: status[0] + status[2]].shape) * -1  # 将框内的像素值全部置-1
    std_Img = np.std(im_gray[im_gray > -1])  # 整图,除目标框外的标准差

    # print("结果图的mean_t :",mean_t)
    # print("结果图的mean_b :", mean_b)
    CON_out = abs(mean_t - mean_b)

    return [CON_out, std, std_Img, status]


def Img_in(img, status):
    im = cv2.imread(img)
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    ob = im_gray[status[1]:status[1] + status[3], status[0]:status[0] + status[2]]  # 取图像框中的部分
    "计算原始图像中目标区域的平均像素值"
    mean_t = np.mean(ob)

    if status[1] - 20 < 0:
        ob_large = im_gray[0:status[1] + status[3] + 20, status[0] - 20: status[0] + status[2] + 20]  # 到顶
    else:
        ob_large = im_gray[status[1] - 20:status[1] + status[3] + 20,
                   status[0] - 20: status[0] + status[2] + 20]  # 框各边加20

    lar = np.array(ob_large).astype(np.int)
    lar[20:20 + status[3], 20: 20 + status[2]] = np.ones(
        lar[20:20 + status[3], 20: 20 + status[2]].shape) * -1  # 将框内的像素值全部置-1

    "背景区域的标准差和平均像素值"
    std = np.std(lar[lar > -1])
    mean_b = np.mean(lar[lar > -1])

    # im_gray = im_gray.astype(int)
    im_gray[status[1]:status[1] + status[3], status[0]: status[0] + status[2]] = np.ones(
        im_gray[status[1]:status[1] + status[3], status[0]: status[0] + status[2]].shape) * -1  # 将框内的像素值全部置-1

    std_Img = np.std(im_gray[im_gray > -1])
    # print("输入图的mean_t :", mean_t)
    # print("输入图的mean_b :", mean_b)
    return [abs(mean_t - mean_b), std, std_Img]


def cen(status):
    centroids_x = status[0] + status[2] / 2
    centroids_y = status[1] + status[3] / 2
    return np.array([centroids_x, centroids_y])


style = xlwt.XFStyle()  # 创建样式对象，初始化
al = xlwt.Alignment()
al.horz = 0x02  # 水平居中
al.vert = 0x01  # 垂直居中
style.alignment = al

def excel_SBC():
    print("=" * 100 + "\n\t\twriting excel...\n" + "=" * 100)
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet("sheet1")
    list1 = ["方法", "指标", "Bachang2", "Bachang5", "Bachang7", "Bachang9", "caochang1", "Gongkai1", "Goingkai2","Bachang3", "Bachang4"]
    for i in range(len(list1)):
        worksheet.write(0, i, list1[i], style)
    list2 = ["Ours", "PSRGAN", "IERN", "ChaSNet", "SPSR", "BSRN", "SRDenseNet", "MoCoPnet"]
    list3 = ["SCRG", "BSF", "CG"] * len(list2)
    "24根据方法个数定"
    for i in range(1, 32):
        if i in [1, 5, 9, 13, 17, 21, 25, 29]:
            worksheet.write_merge(i, i + 2, 0, 0, list2[0], style)
            list2 = list2[1:]
        if i in [4, 8, 12, 16, 20, 24, 28, 32]:
            continue
        worksheet.write(i, 1, list3[0])
        list3 = list3[1::]
    path = "./result/"
    if not os.path.exists(path):
        os.mkdir(path)
    workbook.save(path + "SBC.xls")  # 保存

"计算SCRG, BSF和CG. img_out = ./ResUnet/test_small_results/***.bmp"
def SBC(img_out, img_in):
    # Img_out针对的是测试结果图像
    # List_out中存着：CON_out目标区域的平均值-邻域背景区域的平均值, std背景区域的标准差, std_Img整图除目标框外的标准差, status目标的真实位置GT
    List_out = Img_out(img_out)

    "目标位置信息，左上角坐标和宽高"
    status = List_out[3]
    # Img_in针对的是原始测试图像
    # List_in中存着：CON_in目标区域的平均值-邻域背景区域的平均值, std背景区域的标准差, std_Img整图除目标框外的标准差
    List_in = Img_in(img_in, status)
    CON_out = List_out[0]  # 测试结果图的目标区域-背景区域平均值

    "SCR = CON_out / std"
    SCR_out = CON_out / List_out[1]  # 如果模型能力强，那么List_out[1]无限趋近于0，则SCR_out趋近于正无穷，即inf
    CON_in = List_in[0]  # 原始测试图像的目标区域-背景区域平均值
    SCR_in = CON_in / List_in[1]

    CG = CON_out / CON_in  # 如果模型能力强，那么背景区域就比较干净，使得CON_out就会变大，整体CG就会变大

    std_in = List_in[2]
    std_out = List_out[2]
    BSF = std_in / std_out  # 如果模型能力强，那么std_out就会无限趋近于0，最终使得BSF无限趋近于正无穷，即inf

    SCRG = SCR_out / SCR_in  # 如果模型能力强，那么SCRG无限趋近于正无穷，即inf

    # print("CON_out: " + str(CON_out))
    # print("CON_in: " + str(CON_in))
    # print("std_out: " + str(std_out))
    # print("std_in: " + str(std_in))
    # print("CG:",CG)
    if math.isnan(SCRG):
        print("-------------------")
        print("SCRG==nan")
        print("img_out: " + str(img_out))
        print("CON_out: " + str(CON_out))
        print("CON_in: " + str(CON_in))
        print("SCR_out: "+ str(SCR_out))

    if math.isnan(BSF):
        # print("")
        print("BSF==nan")
        print("std_out: " + str(std_out))
        print("std_in: " + str(std_in))
        print("BSF: " + str(BSF))

    return [SCRG, BSF, CG]

"计算SCRG, BSF, CG，调用SBC函数"
def SBC_mean(all_out, all_in):
    # 初始化表格
    excel_SBC()
    xls = xlrd.open_workbook("./result/SBC.xls")  # 读取excel文件
    excel = xlutils.copy.copy(xls)  # 将xlrd的对象转化为xlwt的对象
    sheet = excel.get_sheet(0)  # 获取第一个sheet
    # Row和Col是每一个内容咋表格中的行数和列数，行数和列数从0开始计数
    Row = {"MoCoPnet":29 , "SRDenseNet":25, "BSRN":21, "SPSR": 17, "ChaSNet": 13, "IERN": 9, "PSRGAN": 5, "Ours": 1}
    Col = {"Bachang2": 2, "Bachang5": 3, "Bachang7": 4, "Bachang9": 5, "caochang1": 6, "Gongkai1": 7, "Goingkai2": 8,"Bachang3": 9, "Bachang4":10}

    "获取检测结果和对应的原始图像"
    for i_Out, In in zip(all_out, all_in):
        start = time.time()
        for Out in i_Out:
            img_out = []
            img_in = []
            scrg = []
            bsf = []
            cg = []

            "Out = ./ResUnet/test_small_results/; img_out = ./ResUnet/test_small_results/***.bmp"
            for i in os.listdir(Out):
                img_out.append(Out + i)
            "In = ./testset/test_small/; img_in = ./testset/test_small/***.bmp"
            for j in os.listdir(In):
                img_in.append(In + j)
            img_out.sort()
            img_in.sort()

            "计算每一个检测结果的SCRG，BSF和CG"
            for i in range(len(img_in)):
                scrg.append(SBC(img_out[i], img_in[i])[0])
                bsf.append(SBC(img_out[i], img_in[i])[1])
                print("*************************************************")
                print("i", i)
                cg.append(SBC(img_out[i], img_in[i])[2])


            SCRG = np.mean(scrg)
            if SCRG == float('inf'):
                SCRG = 'inf'
            elif math.isnan(SCRG):
                SCRG = 'nan'
            else:
                SCRG = float("{:.2f}".format(SCRG))  # 保留两位小数
            BSF = np.mean(bsf)
            if BSF == float('inf'):
                BSF = 'inf'
            elif math.isnan(BSF):
                BSF = 'nan'
            else:
                BSF = float("{:.2f}".format(BSF))
            CG = float("{:.2f}".format(np.mean(cg)))
            write = [SCRG, BSF, CG]
            row = Row[Out.split("/")[1]]
            col = Col[Out.split("_")[1]]
            for i in range(3):
                sheet.write(row, col, write[i], style)
                row += 1
        print("\t%s Over\t\tcost %.3f seconds" % (i_Out[0].split("_")[1], time.time() - start))
    print("\twriting over!\n")
    excel.save("./result/SBC.xls")




dic = {"cloud1": "00561.bmp", "cloud2": "12477.bmp", "cloud3": "38869.bmp",
       "cloud4": "29054.bmp", "cloud5": "00146_cloud5.bmp", "cloud6": "00270_cloud6.bmp", "cloud7": "00102.bmp", "cloud8": "2098.bmp"  }  # 确定每一种数据集要画的图


"绘制带框的检测结果图，输入是all_in和all_out"


def pic2pdf_eps(img):
    portion = os.path.splitext(img)  # 分离文件名与后缀
    pdf = portion[0] + ".pdf"  # pdf
    doc = fitz.open()
    imgdoc = fitz.open(img)  # 打开图片
    pdfbytes = imgdoc.convertToPDF()  # 使用图片创建单页的 PDF
    imgpdf = fitz.open("pdf", pdfbytes)
    doc.insertPDF(imgpdf)  # 将当前页插入文档
    doc.save(pdf)
    doc.close()
    eps = portion[0] + ".eps"  # eps
    im = Image.open(img)
    im.save(eps)



def main():
    start_main = time.time()
    if not os.path.exists("result"):
        os.mkdir("result")

    "原始测试图像"
    img_in_cloud1 = "./testset/test_cloud1/"
    img_in_cloud2 = "./testset/test_cloud2/"
    img_in_cloud3 = "./testset/test_cloud3/"
    img_in_cloud4 = "./testset/test_cloud4/"
    img_in_cloud5 = "./testset/test_cloud5/"
    img_in_cloud6 = "./testset/test_cloud6/"
    img_in_cloud7 = "./testset/test_cloud7/"
    img_in_cloud8 = "./testset/test_cloud8/"

    img_in_Bachang2 = "/data/dl/SCINet-main/newdatasets/test_all/Bachang_Seq_2_test/LR/"


    "模型检测结果"
    path_cloud1 = [
                    "./UnetRDB/test_cloud1_results/"
                    # "./CDAE/test_cloud1_results/"
                   # "./DnCNN/test_cloud1_results/"
                   # "./PSTNN/test_cloud1_PSTNN_pat=slid=15_lambdaL=0.1/",
                   # "./RIPT/test_cloud1_ript_lambdaL=1h=1/",
                   # "./IPI/test_cloud1_IPI_lambdaL=5/"
                   ]


    path_Bachang2 = [
        "./Ours/Bachang_Seq_2"
    ]

    path_cloud2 = [
                    # "./UnetRDB/test_cloud2_results/",
                    # "./CDAE/test_cloud2_results/"
                    # "./DnCNN/test_cloud2_results/",
                    # "./PSTNN/test_cloud2_PSTNN_pat=slid=80_lambdaL=0.1/",
                    # "./RIPT/test_cloud2_ript_lambdaL=1h=1/",
                    "./IPI/test_cloud2_IPI_lambdaL=5/"
        ]

    path_cloud3 = [
                    # "./UnetRDB/test_cloud3_results/",
                    # "./CDAE/test_cloud3_results/"
                    # "./DnCNN/test_cloud3_results/"
                    # "./PSTNN/test_cloud3_PSTNN_pat=slid=20_lambdaL=0.1/",
                    # "./RIPT/test_cloud3_ript_lambdaL=0.4h=1/",
                    "./IPI/test_cloud3_IPI_lambdaL=2.5/"
                   ]

    path_cloud4 = [
                    # "./UnetRDB/test_cloud4_results/",
                    # "./CDAE/test_cloud4_results/",
                    # "./DnCNN/test_cloud4_results/",
                   # "./PSTNN/test_cloud4_PSTNN_pat=slid=20_lambdaL=0.1/",
                   # "./RIPT/test_cloud4_ript_lambdaL=0.3h=1/",
                   # "./IPI/test_cloud4_IPI_lambdaL=2/"
                   ]

    path_cloud5 = [
                    # "./UnetRDB/test_cloud5_results/"
                    # "./CDAE/test_cloud5_results/"
                   # "./DnCNN/test_cloud5_results/"
                   # "./PSTNN/test_cloud5_PSTNN_pat=slid=20_lambdaL=0.005/",
                   # "./RIPT/test_cloud5_ript_lambdaL=0.25h=1/",
                   # "./IPI/test_cloud5_IPI_lambdaL=4/"
                   ]

    path_cloud6 = [
                    # "./UnetRDB/test_cloud6_results/",
                    # "./CDAE/test_cloud6_results/"
                   # "./DnCNN/test_cloud6_results/"
                   # "./PSTNN/test_cloud6_PSTNN_pat=slid=20_lambdaL=1/",
                   # "./RIPT/test_cloud6_ript_lambdaL=1h=1/",
                   # "./IPI/test_cloud6_IPI_lambdaL=2.56/"
                   ]

    path_cloud7 = [
                   # "./UnetRDB/test_cloud7_results/",
                  # "./CDAE/test_cloud7_results/"
                  #  "./DnCNN/test_cloud7_results/",
                  #  "./PSTNN/test_cloud7_PSTNN_pat=slid=20_lambdaL=0.005/",
                  #  "./RIPT/test_cloud7_ript_lambdaL=0.25h=1/"
                  #  "./IPI/test_cloud7_IPI_lambdaL=4/"
                   ]

    path_cloud8 = [
                    "./UnetRDB/test_cloud8_results/"
                    # "./CDAE/test_cloud8_results/"
                    # "./DnCNN/test_cloud8_results/",
                    # "./PSTNN/test_cloud8_PSTNN_pat=slid=20_lambdaL=0.005/",
                    # "./RIPT/test_cloud8_ript_lambdaL=0.4h=1/",
                    # "./IPI/test_cloud8_IPI_lambdaL=1.625/"
    ]
    "writing excel...，all_out是原始图像，all_in是残差图像"
    # all_out = [path_cloud1, path_cloud2, path_cloud3,path_cloud4,path_cloud5,path_cloud6,path_cloud7,path_cloud8]  # all_out/all_in必须一一对应
    # all_in = [img_in_cloud1, img_in_cloud2, img_in_cloud3,img_in_cloud4,img_in_cloud5,img_in_cloud6,img_in_cloud7,img_in_cloud8]
    all_out = [path_Bachang2]# all_out/all_in必须一一对应
    all_in = [img_in_Bachang2]

    SBC_mean(all_out, all_in)

    # print("the program cost %f mins" % ((time.time() - start_main) / 60))


if __name__ == "__main__":
    main()