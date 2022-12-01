from FashionDetection import Detector
import io
from flask import Flask, render_template, redirect, url_for, request, send_from_directory, send_file
from PIL import Image
import os
import img_transforms
import requests
import webbrowser
import re
import bs4

app = Flask(__name__)
detector = Detector()

RENDER_FACTOR = 5
file_path = './static/file.jpg'
# endpoint = 'http://13.57.247.28:5000/'
# function to load img from url

def load_image_url(url):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    return img

# run inference using image transform to reduce memory
def run_inference_transform(img_path='./static/file.jpg', transformed_path='./static/file_transformed.jpg'):

    # get height, width of image
    original_img = Image.open(img_path)

    # transform to square, using render factor
    # transformed_img = img_transforms._scale_to_square(
    #    original_img, targ=RENDER_FACTOR*16)
    # transformed_img.save(transformed_path)

    # run inference using detectron2
    result_img = detector.inference(img_path)

    # unsquare
    # result_img = img_transforms._unsquare(untransformed_result, original_img)

    # clean up
    #try:

    #os.remove(img_path)
    #    os.remove(transformed_path)
    #except:
    #    pass
    return result_img


@app.route("/")
def index():
    print('index')
    return render_template('index.html')


@app.route("/detect")
def result():
    # run inference
    print('infer')
    result_img = run_inference_transform()
    # result_img = run_inference(file_path)

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    result_img.save(file_object, 'PNG')
    print('wow')
    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)
    return render_template('modified_html.html')
    # return send_file(file_object, mimetype='image/jpeg')

def inferencing():
    return render_template("inferencing_html.html")

@app.route("/inference", methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':

        try:

            # open image
            file = Image.open(request.files['file'].stream)

            # remove alpha channel
            rgb_im = file.convert('RGB')
            rgb_im.save(file_path)
            print('post file saved in path')
            # return send_file(file_object, mimetype='image/jpeg')

        # failure
        except:

            return render_template("failure.html")

    elif request.method == 'GET':

        # get url
        url = request.args.get("url")

        # save
        try:
            # save image as jpg
            # urllib.request.urlretrieve(url, 'file.jpg')
            rgb_im = load_image_url(url)
            rgb_im = rgb_im.convert('RGB')
            rgb_im.save(file_path)
            print('file saved')
            render_template("inferencing_html.html")
            return redirect(url_for('result'))

        # failure
        except:
            return render_template("failure.html")

    # run inference
    print('inferencing')
    result_img = run_inference_transform()
    # result_img = run_inference(file_path)

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    result_img.save(file_object, 'PNG')
    print('writing result img')
    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)
    return render_template('modified_html.html')



if __name__ == "__main__":

    # get port. Default to 8080
    # port = int(os.environ.get('PORT', 8080))i

    # run app
    app.debug = True
    app.run(host='0.0.0.0')
