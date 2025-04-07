import os, shutil, random
import cv2
#
# input_imgHR_dir = '/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5/HR/'
#
# test_imgHR_dir  = '/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5_validtest/HR/'
#
#
# input_imgLR_dir = '/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5/LR/'
# test_imgLR_dir = '/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5_validtest/LR/'
# #
# input_imgBic_dir = '/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5/Bic/'
# test_imgBic_dir = '/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5_validtest/Bic/'
#
# os.mkdir('/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5_validtest/')
#
# if not os.path.exists(test_imgHR_dir):
#     os.mkdir(test_imgHR_dir)
#
# if not os.path.exists(test_imgLR_dir):
#     os.mkdir(test_imgLR_dir)
#
# if not os.path.exists(test_imgBic_dir):
#     os.mkdir(test_imgBic_dir)
#
# image_names = os.listdir(input_imgHR_dir)
# image_num = len(image_names)
# num_test = int(image_num * 0.2)
#
# test_names = random.sample(image_names, num_test)
# for name in test_names:
#
#     print(name)
#     print("----------------")
#     print(input_imgHR_dir  + name)
#     # 移动图片
#     shutil.move(input_imgHR_dir  + name,
#                 test_imgHR_dir  + name)
#
#     print("Image file moved.")
#
#     try:
#         shutil.move(input_imgLR_dir  + name,
#                     test_imgLR_dir  + name)
#     except:
#         print('No corresponding LR file.')
#
#     try:
#         shutil.move(input_imgBic_dir  + name,
#                     test_imgBic_dir  + name)
#     except:
#         print('No corresponding Bic file.')
#
#     print(" All file moved.")


input_imgHR_dir = '/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5_validtest/HR/'
test_imgHR_dir = '/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5_valid/HR/'

input_imgLR_dir = '/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5_validtest/LR/'
test_imgLR_dir = '/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5_valid/LR/'

input_imgBic_dir = '/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5_validtest/Bic/'
test_imgBic_dir = '/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5_valid/Bic/'

os.mkdir('/data/dl_2080/SPSR-master/newdataset/Gongkai_Seq_5_valid/')

if not os.path.exists(test_imgHR_dir):
    os.mkdir(test_imgHR_dir)

if not os.path.exists(test_imgLR_dir):
    os.mkdir(test_imgLR_dir)

if not os.path.exists(test_imgBic_dir):
    os.mkdir(test_imgBic_dir)

image_names = os.listdir(input_imgHR_dir)
image_num = len(image_names)

image_names = os.listdir(input_imgHR_dir)
image_num = len(image_names)
num_test = int(image_num * 0.5)

test_names = random.sample(image_names, num_test)
for name in test_names:

    print(name)
    print("----------------")
    print(input_imgHR_dir + name)
        # 移动图片
    shutil.move(input_imgHR_dir + name,
                test_imgHR_dir + name)

    print("Image file moved.")

    try:
        shutil.move(input_imgLR_dir + name,
                    test_imgLR_dir + name)
    except:
        print('No corresponding LR file.')

    try:
        shutil.move(input_imgBic_dir + name,
                    test_imgBic_dir + name)
    except:
        print('No corresponding Bic file.')

    print(" All file moved.")

    #
    # 移动XML
    # try:
    #     shutil.move(input_xml_dir + '\\' + name.replace('bmp', 'xml'),
    #                 output_xml_dir + '\\' + name.replace('bmp', 'xml'))
    # except:
    #     print('No corresponding xml file.')
    #
    # print('XML file moved.')
    # # input("PAUSE")
#
# train_txt_path = 'G:\\KKK\\640_512\\VOC2007\\ImageSets\\Main'
# valid_txt_path = 'G:\\KKK\\640_512\\VOC2007\\ImageSets\\Main'
#
# input_img_dir= 'G:\\KKK\\960_960random\\VOC2007\\JPEGImages\\'
# valid_img_dir = 'G:\\KKK\\960_960random\\VOC2007\\validImages\\'
# test_img_dir= 'G:\\KKK\\960_960random\\VOC2007\\testImages\\'

# 将测试集从所有数据集中移出去
# with open('G:\\KKK\\640_512\\VOC2007\\ImageSets\\Main\\test.txt','r',encoding='utf-8') as ft:
# with open('G:\\KKK\\960_960random\\VOC2007\\ImageSets\\Main\\test.txt', 'r', encoding='utf-8') as ft:
#     imageListtest = ft.readlines()
#     imageListtest = [line.strip("\n") for line in imageListtest]
#     print(len(imageListtest))
#     for i in imageListtest :
#         shutil.move(input_img_dir  + i + '.bmp',
#                         test_img_dir  + i + '.bmp')
#
#         print("Image file moved.")

# 从训练验证集中选取200张图片作为验证集，并且写入valid.txt中
# with open('G:\\KKK\\960_960random\\VOC2007\\ImageSets\\Main\\trainval.txt','r',encoding='utf-8') as f:
#     imageList = f.readlines()
#     imageList = [line.strip("\n") for line in imageList]
#     # print(imageList)
#     print(len(imageList))
#
#     #对于第二次划分，960×960的
#     valid_img_dir640 = 'G:\\KKK\\960_960\\VOC2007\\validImages'
#     list = os.listdir(valid_img_dir640)
#
#     for listn in list:
#         shutil.move(input_img_dir + listn,
#                     valid_img_dir + listn)
#
#         print("Image file moved.")
#
#         with open('G:\\KKK\\960_960random\\VOC2007\\ImageSets\\Main\\valid.txt', 'a', encoding='utf-8') as fw:
#             fw.write(listn + "\n")
#         fw.close()

    # 对于第一次划分，640×512的
    # rn =  random.sample(range(3297),200)
    # print(rn)
    # print(len(rn))
    #
    # for i in rn:
    #     lin = imageList[i]
    #     # 移动图片
    #     print(lin)
    #     shutil.move(input_img_dir  + lin + '.bmp',
    #                     valid_img_dir  + lin + '.bmp')
    #
    #     print("Image file moved.")
    #
    #     with open('G:\\KKK\\640_512\\VOC2007\\ImageSets\\Main\\valid.txt', 'a', encoding='utf-8') as fw:
    #             fw.write(lin + "\n")
    #     fw.close()

# 根据训练集图片创建训练集的txt文件
# train_img_dir= 'G:\\KKK\\960_960random\\VOC2007\\trainImages'
# list = os.listdir(train_img_dir)
# print(list)
# print(len(list))
# with open('G:\\KKK\\960_960random\\VOC2007\\ImageSets\\Main\\train.txt', 'a', encoding='utf-8') as fr:
#     for i in list:
#         fr.write(i + "\n")
# fr.close()


# 将训练集的xml标记信息划分出来，存入trainLabelsXML里
# train_img_dir= 'G:\\KKK\\960_960random\\VOC2007\\trainImages\\'
# xml_dir= 'G:\\KKK\\960_960random\\VOC2007\\Annotations\\'
# train_xml_dir= 'G:\\KKK\\960_960random\\VOC2007\\trainLabelsXML\\'
# list = os.listdir(train_img_dir)
# print(list)
# print(len(list))
# for j in list:
#     try:
#         shutil.move(xml_dir +  '\\' + j.replace('bmp', 'xml'),
#                 train_xml_dir  +  '\\' + j.replace('bmp', 'xml'))
#     except:
#         print('No corresponding xml file.')

# 将验证集的xml标记信息划分出来，存入validLabelsXML里
# valid_img_dir= 'G:\\KKK\\960_960random\\VOC2007\\validImages\\'
# xml_dir= 'G:\\KKK\\960_960random\\VOC2007\\Annotations\\'
# valid_xml_dir= 'G:\\KKK\\960_960random\\VOC2007\\validLabelsXML\\'
# list = os.listdir(valid_img_dir)
# print(list)
# print(len(list))
# for j in list:
#     try:
#         shutil.move(xml_dir +  '\\' + j.replace('bmp', 'xml'),
#                 valid_xml_dir  +  '\\' + j.replace('bmp', 'xml'))
#     except:
#         print('No corresponding xml file.')


# 将测试集的xml标记信息划分出来，存入testLabelsXML里
# test_img_dir = 'G:/KKK/640_512/VOC2007/testImages/'
# xml_dir= 'G:/KKK/640_512/VOC2007/Annotations/'
# test_xml_dir = 'G:/KKK/640_512/VOC2007/testLabelsXML/'
# list = os.listdir(test_img_dir)
# print(list)
# print(len(list))
# for j in list:
#     print(xml_dir + j.replace('bmp', 'xml'))
#     try:
#         shutil.move(xml_dir + j.replace('bmp', 'xml'),
#                 test_xml_dir + j.replace('bmp', 'xml'))
#     except:
#         print('No corresponding xml file.')

# with open('G:\\KKK\\640_512\\VOC2007\\ImageSets\\Main\\valid.txt', 'r', encoding='utf-8') as fr:
#     imageList2 = fr.readlines()
#     imageList2 = [line.strip("\n") for line in imageList]
#
#     print(len(rn))




# with open('F:/学习资源/论文代码/SSD测试/val.txt','r',encoding='utf-8') as f:
#     imageList=f.readlines()
#     imageList = [line.strip("\n") for line in imageList]
#     print(imageList)
# f.close()
#
# for name in imageList:
#     print(name)
#     try:
#         shutil.move(input_img_label + name + '.txt',
#                 valid_img_label  + name+ '.txt')
#     except:
#         print('No corresponding xml file.')



# count=2
# for i in range(1, count):
#     print(i)
#     print("sa")
# print("t")
#
# mdef={'mask':'6,7,8','anchors':'22.0,21.8, 24.8,27.1,  31.9,26.3,  27.2,34.0,  33.8,35.1,  28.5,49.8,  33.3,43.9,  53.1,28.9,  35.6,55.9'}
# mask = [int(x) for x in mdef['mask'].split(',')]
# a = [float(x) for x in mdef['anchors'].split(',')]
# a1 = [(a[i], a[i + 1]) for i in range(0, len(a), 2)]
# print(mask)
# print(a)
# print(a1)
# print("--------------------------")
#
# anchorq=[]
# for i in mask:
#     print(i)
#     print(a1[i])
#     anchorq.append(a1[i])
# print(anchorq)
#
# anchors=[a1[i] for i in mask]
# print(anchors)


#
# img_path = 'C:/Users/Administrator/Desktop/10994.bmp'
# resize_img_path='C:/Users/Administrator/Desktop/'
# # img = os.listdir(img_path)
# # for file in img:
# #     imga = cv2.imread(os.path.join(img_path,file), cv2.IMREAD_GRAYSCALE)
# imga = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# imgb = cv2.resize(imga,(320,256))
# cv2.imwrite('C:/Users/Administrator/Desktop/10994_new.bmp',imgb)
# print("success")

# from PIL import Image
# def letterbox_image(image, size):
#     # 对图片进行resize，使图片不失真。在空缺的地方进行padding
#     iw, ih = image.size
#     w, h = size
#     scale = min(w/iw, h/ih)
#     nw = int(iw*scale)
#     nh = int(ih*scale)
#
#     image = image.resize((nw,nh), Image.BICUBIC)
#     new_image = Image.new('RGB', size, (128,128,128))
#     new_image.paste(image, ((w-nw)//2, (h-nh)//2))
#     return new_image
#
#
# img = Image.open("2007_000175.jpg")
# imagea = img.resize((416,416), Image.BICUBIC)
# imagea.show()
# new_image = letterbox_image(img,[416,416])
# new_image.show()

