import os
import json
import shutil
import os.path as opath

from pathlib import Path
from lxml import etree


def write_xml(json_path: str,
              xml_output_dir: str,
              ids_output_dir: str):
    assert opath.exists(json_path) and opath.isfile(json_path), 'Not a valid json file!'
    typ = 'trainval' if 'trainval' in Path(json_path).stem else 'test'

    with open(json_path, 'r') as file:
        contents = json.load(file)
    
    images = {x['id']: dict(filename=x['file_name'],
                            width=x['width'],
                            height=x['height']) for x in contents['images']}
    classes = {x['id']: x['name'] for x in contents['categories']}
    annotations = {x['image_id']: list() for x in contents['annotations']}
    for anno in contents['annotations']:
        annotations[anno['image_id']].append(dict(bbox=anno['bbox'], category_id=anno['category_id']))

    ids = []
    
    for id, info in images.items():
        filename = info['filename']
        width, height = info['width'], info['height']
        annos = annotations[id]

        ids.append(Path(filename).stem + '\n')

        _root = etree.Element('annotation')
        _folder = etree.SubElement(_root, 'folder')
        _folder.text = ''
        _filename = etree.SubElement(_root, 'filename')
        _filename.text = filename
        _path = etree.SubElement(_root, 'path')
        _path.text = ''
        _size = etree.SubElement(_root, 'size')
        _width = etree.SubElement(_size, 'width')
        _width.text = str(width)
        _height = etree.SubElement(_size, 'height')
        _height.text = str(height)
        _depth = etree.SubElement(_size, 'depth')
        _depth.text = str(1)
        _segmented = etree.SubElement(_root, 'segmented')
        _segmented.text = str(0)
        
        for anno in annos:
            _object = etree.SubElement(_root, 'object')

            bbox, clsname = anno['bbox'], classes[anno['category_id']]
            bbox[2:] = [bbox[0] + bbox[2], bbox[1] + bbox[3]]

            _name = etree.SubElement(_object, 'name')
            _name.text = str(clsname)
            _pose = etree.SubElement(_object, 'pose')
            _pose.text = 'Unspecified'
            _truncated = etree.SubElement(_object, 'truncated')
            _truncated.text = str(0)
            _difficult = etree.SubElement(_object, 'difficult')
            _difficult.text = str(0)
            _bndbox = etree.SubElement(_object, 'bndbox')
            _xmin = etree.SubElement(_bndbox, 'xmin')
            _xmin.text = str(bbox[0])
            _ymin = etree.SubElement(_bndbox, 'ymin')
            _ymin.text = str(bbox[1])
            _xmax = etree.SubElement(_bndbox, 'xmax')
            _xmax.text = str(bbox[2])
            _ymax = etree.SubElement(_bndbox, 'ymax')
            _ymax.text = str(bbox[3])
        
        tree = etree.ElementTree(_root)
        tree.write(opath.join(xml_output_dir, filename.replace('bmp', 'xml')),
                   pretty_print=True, xml_declaration=False, encoding='utf-8')
    
    with open(opath.join(ids_output_dir, f'{typ}.txt'), 'w+') as file:
        file.writelines(sorted(ids))



def convert(coco_dir: str,
            voc_dir: str):
    assert opath.exists(coco_dir) and opath.isdir(coco_dir), 'Not a valid COCO directory!'
    assert opath.exists(opath.join(coco_dir, 'annotations/')), 'No valid COCO annotation directory!'
    img_dirs_bool = opath.exists(opath.join(coco_dir, 'trainval2017/')) and opath.exists(opath.join(coco_dir, 'test2017/'))
    assert img_dirs_bool, 'No valid COCO image directories!'

    coco_trainval_dir = opath.join(coco_dir, 'trainval2017/')
    coco_test_dir = opath.join(coco_dir, 'test2017/')
    coco_anno_dir = opath.join(coco_dir, 'annotations/')

    voc_anno_dir = opath.join(voc_dir, 'Annotations/')
    voc_txt_dir = opath.join(voc_dir, 'ImageSets/Main/')
    voc_img_dir = opath.join(voc_dir, 'JPEGImages/')
    os.makedirs(voc_anno_dir, exist_ok=True)
    os.makedirs(voc_txt_dir, exist_ok=True)
    os.makedirs(voc_img_dir, exist_ok=True)

    write_xml(opath.join(coco_anno_dir, 'instances_trainval2017.json'), voc_anno_dir, voc_txt_dir)
    write_xml(opath.join(coco_anno_dir, 'instances_test2017.json'), voc_anno_dir, voc_txt_dir)

    shutil.copytree(coco_trainval_dir, voc_img_dir, dirs_exist_ok=True)
    shutil.copytree(coco_test_dir, voc_img_dir, dirs_exist_ok=True)


if __name__ == '__main__':
    convert('操场无人机序列1/COCO/', '操场无人机序列1/VOC2007/')
    