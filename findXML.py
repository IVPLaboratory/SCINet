import os, shutil
#
#
# ori_xml_path = '/data/dl/yolov5-master-yolov5m-640/paper_data/XMLAll/'
# xml_path = '/data/dl_2080/SPSR-master/newdataset/公开数据集1/'
# find_xml_path = '/data/dl_2080/SPSR-master/newdataset/公开数据集1_XML/'
# xmllist = os.listdir(xml_path)
# nolabellist = []
# for xml in xmllist:
#     # name = xml.split('.')[0]
#     ori_name = ori_xml_path + xml.replace('bmp','xml')
#     try:
#         shutil.copy(ori_name, find_xml_path)
#     except:
#         nolabellist.append(xml)
#         print('No corresponding label file.')
# print(nolabellist)
# print(len(nolabellist))

# no_label =['0231.bmp', '1239.bmp', '1487.bmp', '2197.bmp', '0323.bmp', '0023.bmp', '0235.bmp', '1495.bmp', '2213.bmp', '0413.bmp', '1710.bmp', '0233.bmp', '0411.bmp', '2217.bmp', '0631.bmp', '1503.bmp', '1512.bmp', '1527.bmp', '1612.bmp', '2215.bmp', '1243.bmp', '1497.bmp', '2216.bmp', '1411.bmp', '1528.bmp', '1413.bmp', '1500.bmp', '1525.bmp', '1810.bmp', '1501.bmp', '0373.bmp', '2211.bmp', '1444.bmp', '1524.bmp', '1521.bmp', '1523.bmp', '2195.bmp', '2193.bmp', '1498.bmp', '1526.bmp', '0634.bmp', '1493.bmp', '1692.bmp', '1491.bmp', '1533.bmp', '1520.bmp', '1518.bmp', '1201.bmp', '1499.bmp', '1517.bmp', '2212.bmp', '1530.bmp', '0636.bmp', '1203.bmp', '1522.bmp', '1516.bmp', '1514.bmp', '1513.bmp', '1529.bmp', '1698.bmp', '1532.bmp', '1515.bmp', '1241.bmp', '1531.bmp', '1519.bmp', '1489.bmp', '2214.bmp', '1511.bmp', '0229.bmp', '2207.bmp']
#
# input_img_dir = '/data/dl_2080/SPSR-master/newdataset/靶场无人机序列5/'
# test_img_dir  = '/data/dl_2080/SPSR-master/newdataset/bachang5_nolabel_img/'
#
# for name in no_label:
#
#     print(name)
#     print(input_img_dir  + name)
#     # 移动图片
#     shutil.copy(input_img_dir  + name,
#                 test_img_dir)
#
#     print("Image file moved.")
#

xml_dir = '/data/dl_2080/SPSR-master/newdataset/公开数据集5_XML/'
xmllist = os.listdir(xml_dir)
for xml in xmllist:
    name = xml.split('.')[0]
    name_new = name + '_Gongkai_Seq5'
    os.rename(xml_dir + xml, xml_dir + name_new + '.xml')

