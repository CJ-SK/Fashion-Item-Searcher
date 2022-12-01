import cv2 as cv
import json
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.data.datasets import register_coco_instances
import torch
import numpy as np
from PIL import Image
import requests
import webbrowser
import re
import bs4
from bs4 import BeautifulSoup


class Detector:

    def __init__(self):

        # set model and test set
        self.model = 'mask_rcnn_R_101_FPN_3x.yaml'
        
        register_coco_instances("deepfashion_validation", {}, './deepfashion2_validation.json', "./static/")

        # obtain detectron2's default config
        self.cfg = get_cfg()

        # load values from a file
        self.cfg.merge_from_file("./config.yml")

        #self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+self.model))
        self.cfg.DATASETS.TRAIN = ("deepfashion_validation",)
        self.cfg.DATASETS.TEST = ("deepfashion_validation",)
        self.cfg.SOLVER.IMS_PER_BATCH = 4
        self.cfg.SOLVER.BASE_LR = 0.001
        self.cfg.SOLVER.WARMUP_ITERS = 1000
        self.cfg.SOLVER.MAX_ITER = 1500
        self.cfg.SOLVER.STEPS = (1000, )
        self.cfg.SOLVER.GAMMA = 0.05
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
        self.cfg.TEST.EVAL_PERIOD = 500
        self.cfg.INPUT.MAX_SIZE_TEST = 2048
        # set device to cpu
        self.cfg.MODEL.DEVICE = "cpu"

        # get weights
        #self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+self.model)
        self.cfg.MODEL.WEIGHTS = "./model_final.pth"

        # set the testing threshold for this model
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        # build model from weights
        # self.cfg.MODEL.WEIGHTS = self.convert_model_for_inference()

    # build model and convert for inference
    def convert_model_for_inference(self):

        # build model
        model = build_model(self.cfg)

        # save as checkpoint
        torch.save(model.state_dict(), 'checkpoint.pth')

        # return path to inference model
        return 'checkpoint.pth'

    def resize_input(self, filepath):
        # height, width, c = image.shape
        # resized = image.resize((int(width / 2), int(height / 2)))
        basewidth = 300
        img = Image.open(filepath)
        # wpercent = (basewidth/float(img.size[0]))
        # hsize = int((float(img.size[1])*float(wpercent)))
        # resized = img.resize((basewidth,hsize), Image.ANTIALIAS)
        resized = img.resize((int(img.width / 2), int(img.height / 2)), Image.LANCZOS)
        # img.save('sompic.jpg') 
        # res_0 = width * height
        # res_1 = 25000

        # # You need a scale factor to resize the image to res_1
        # scale_factor = (res_1/res_0)**0.5
        # resized = image.resize((int(width * scale_factor), int(height * scale_factor)))
        return resized

    # detectron model
    # adapted from detectron2 colab notebook: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
    def inference(self, file):

        predictor = DefaultPredictor(self.cfg)
        print('reading img..')
        #im = Image.open(file)
        #print('read successfully', im.size)

        file_path = './static/file.jpg'

        # img = Image.open(file_path)
        # resized_img = self.resize_input(file_path)
        # resized_img.save(file_path)
        im = cv.imread(file_path)
        print('resized and read successfully', im.shape)

        outputs = predictor(im)
        print('predicted')
        # with open(self.curr_dir+'/data.txt', 'w') as fp:
        # 	json.dump(outputs['instances'], fp)
        # 	# json.dump(cfg.dump(), fp)

        # get metadata
        #metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        metadata = MetadataCatalog.get("deepfashion_validation")
        #dataset_dict = DatasetCatalog.get("deepfashion_validation")
        # visualise
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2, instance_mode=ColorMode.SEGMENTATION)
        instances = outputs["instances"].to('cpu')
        #mask_instances = outputs["instances"].get("pred_masks").to('cpu')
        #v.draw_instance_predictions(instances)
        #for m in mask_instances:
        v = v.draw_instance_predictions(instances)
        #v = v.draw_soft_mask(instances)
        #v = v.get_output()

        # get image
        img = Image.fromarray(np.uint8(v.get_image()[:, :, ::-1]))
        print('writing img..')
        # write to jpg
        
        # file.save(file_path, 'JPEG')
        result_img_path = './static/result_img.jpg'
        cv.imwrite(result_img_path, v.get_image())

        ##### crop ####

        # 인식된 객체 카테고리명 추출, 바운딩박스 좌표대로 이미지 크롭
        pred_classes = instances.pred_classes
        boxes = instances.pred_boxes
        print('boxesss', boxes)
        if isinstance(boxes, detectron2.structures.boxes.Boxes):
            boxes = boxes.tensor.numpy()
        else:
            boxes = np.asarray(boxes)

        img_original = Image.open(file)
        #metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        #metadata = MetadataCatalog.get(img_original)
        
        #class_catalog = metadata.things_classes

        initial_html_path = "./templates/initial_html.html"
        # load inital HTML file
        with open(initial_html_path) as inf:
            file1 = inf.read()
            soup = bs4.BeautifulSoup(file1, 'lxml')

        # create new box
        original_img = cv.imread(file_path, cv.IMREAD_COLOR)
        # image = soup.new_tag("img", src=file_path, height=h/2,
        #                     width=w/2, )
        result_img = cv.imread(result_img_path, cv.IMREAD_COLOR)

        h, w, c = original_img.shape
        # h2, w2, c2 = result_img.shape

        for idx, coordinates in enumerate(boxes):
            class_index = pred_classes[idx]
            #class_name = class_catalog[class_index] # 인식된 객체 카테고리명 ex.'vest_dress'
            #print('classsss', class_name)
            box = boxes[idx]  # pred_boxes
            x_top_left = box[0]
            y_top_left = box[1]
            x_bottom_right = box[2]
            y_bottom_right = box[3]

            crop_img = img_original.crop((int(x_top_left), int(
                y_top_left), int(x_bottom_right), int(y_bottom_right)))
            print('saving cropped img..')
            crop_img.save("./static/crop_result_img_"+str(idx)+".jpg", "JPEG")

            ####### search #######

            #filePath = '/content/drive/MyDrive/Colab Notebooks/AI-ICT Project/LV3.jpeg'
            filePath = "./static/crop_result_img.jpg"
            # searchUrl = 'http://www.google.hr/searchbyimage/upload'
            # multipart = {'encoded_image': (filePath, open(
            #     filePath, 'rb')), 'image_content': ''}

            # response = requests.post(searchUrl, files=multipart, allow_redirects=False)
            # # fetchUrl = response.headers['Location']
            # # webbrowser.open(fetchUrl)

            # y = response.content
            # z = str(y)
            # print("response!!!", z)

            # regularex = regularex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|(([^\s()<>]+|(([^\s()<>]+)))))+(?:(([^\s()<>]+|(([^\s()<>]+))))|[^\s`!()[]{};:'\".,<>?«»“”‘’]))"
            # urlsrc = re.findall(regularex, z)
            # print(urlsrc[0][0])

            # generated_link = urlsrc[0][0][:-1]
            # print('generated link', generated_link)

            # ##### filter URL #####

            # url = generated_link

            # headers = {
            #     'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            #     'accept-language': 'en-KR',
            #     'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
            # }

            all_links = ['https://www.lotteon.com/p/product/LE1211849808?sitmNo=LE1211849808_1267224482&ch_no=100999&ch_dtl_no=1025549&entryPoint=pcs&dp_infw_cd=CHT&gclid=Cj0KCQiAm5ycBhCXARIsAPldzoWgxu0jw_JcXPJyxnN5-XtlVKGykGOGao5q0EPzxlLfVNI3vWzv5R8aArz_EALw_wcB','https://www.highsnobiety.com/en-us/shop/product/lemaire-italian-woven-denim-sailor-pants-saltpeter/']
            # soup = BeautifulSoup(requests.get(url, headers=headers).text, 'lxml')
            # # print(soup)
            # for href in soup.find('div', {'id': 'search'}).find_all('a'):
            #     print('href', href)
            #     # only take amazon shopping website
            #     if href.get('href').startswith('https://'):
            #         all_links.append(href.get('href'))

            search_api_URL_list = list(set(all_links))
            print('amazon',search_api_URL_list)

            try:
                x_top_left = box[0] + 300
                y_top_left = box[1] + 130
                x_bottom_right = box[2] + 300
                y_bottom_right = box[3] + 130
                coords = str(box[0]+300) + "," + str(box[1]+130) + "," + str(box[2]+300) + "," + str(box[3]+130)
                if idx == 0:
                    image2 = soup.new_tag("img", src=result_img_path, height=h, width=w, usemap="#imagemap", id="detected")
                    soup.div.append(image2)
                
                print('detected boxes', boxes)
                new_box = soup.new_tag(
                    "area", shape="rect",coords=box, href=search_api_URL_list[1], target="blank")

                # insert it into the document
                # soup.h1.append(image)
                soup.map.append(new_box)

            # failure
            except:
                print("error")


        ##### modify HTML #####

        # path for HTML beatiful soup
        # In PC, need original image (ori_desktop_img_path) and output image (desktop_result_img_path)  from detectron model
        desktop_result_img_path = 'result_img.jpg'
        # ori_desktop_img_path = "G:/My Drive/Colab Notebooks/AI-ICT Project/LV3.JPEG"

        # In google drive, need initial HTML file (initial_html_path)

        # FInal html (modified_html_path) will be generated in google drive
        modified_html_path = "./templates/modified_html.html"


        # save the file again
        with open(modified_html_path, "w") as outf:
            outf.write(str(soup))

        print("Done!")

        return img
