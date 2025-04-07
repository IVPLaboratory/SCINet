b = a["gt_rect"]
print(b)

strex = '00105'
q = int(strex)
print(q)
import argparse
import glob
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse
import xml.dom.minidom
import shutil
from PIL import Image
import random
from PIL import Image , ImageEnhance

def txtLabel_to_xmlLabel(source_txt_path,source_img_path,save_xml_path):
    if not os.path.exists(save_xml_path):
        os.makedirs(save_xml_path)
    classes = ['uav']
    print(classes)
    for file in os.listdir(source_txt_path):
        img_path = os.path.join(source_img_path,file.replace('.txt','.bmp'))
        img_file = Image.open(img_path)
        txt_file = open(os.path.join(source_txt_path,file)).read().splitlines()
        print(txt_file)
        xml_file = open(os.path.join(save_xml_path,file.replace('.txt','.xml')), 'w')
        width, height = img_file.size
        # width, height = 640, 512
        xml_file.write('<annotation>\n')
        xml_file.write('\t<folder>simple</folder>\n')
        xml_file.write('\t<filename>' + str(file) + '</filename>\n')
        xml_file.write('\t<size>\n')
        xml_file.write('\t\t<width>' + str(width) + ' </width>\n')
        xml_file.write('\t\t<height>' + str(height) + '</height>\n')
        xml_file.write('\t\t<depth>' + str(3) + '</depth>\n')
        xml_file.write('\t</size>\n')

        for line in txt_file:
            print(line)
            line_split = line.split(' ')
            name = int(line_split[0])
            x_center = float(line_split[1])
            y_center = float(line_split[2])
            w = float(line_split[3])
            h = float(line_split[4])
            xmax = int((2*x_center*width + w*width)/2)
            xmin = int((2*x_center*width - w*width)/2)
            ymax = int((2*y_center*height + h*height)/2)
            ymin = int((2*y_center*height - h*height)/2)

            xml_file.write('\t<object>\n')
            xml_file.write('\t\t<name>'+ classes[name] +'</name>\n')
            xml_file.write('\t\t<pose>Unspecified</pose>\n')
            xml_file.write('\t\t<truncated>0</truncated>\n')
            xml_file.write('\t\t<difficult>0</difficult>\n')
            xml_file.write('\t\t<bndbox>\n')
            xml_file.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
            xml_file.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
            xml_file.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
            xml_file.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
            xml_file.write('\t\t</bndbox>\n')
            xml_file.write('\t</object>\n')
        xml_file.write('</annotation>')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_txt_path', type=str, default="F:/学习资源/论文代码/拉下来的数据/testLabels")
    parser.add_argument('--source_img_path', type=str, default="F:/学习资源/论文代码/拉下来的数据/x4_test")
    parser.add_argument('--save_xml_path', type=str, default="F:/学习资源/论文代码/拉下来的数据/testXML")
    opt = parser.parse_args()

    txtLabel_to_xmlLabel(opt.source_txt_path,opt.source_img_path,opt.save_xml_path)