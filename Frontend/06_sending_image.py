from flask import Flask, request, render_template
import cv2 as cv
import numpy as np
import datetime

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("01.html")

@app.route('/sending_image', methods=['POST'])
def sending_image():
    file = request.files['file']
    print("Posted file - SENDING: {}".format(file))
    img = cv.imdecode(np.frombuffer(file.read(), np.uint8), cv.IMREAD_UNCHANGED)
    width = 500
    scale_percent = img.shape[1] / width
    height = int(img.shape[0] / scale_percent)
    dim = (width, height)
    img = cv.resize(img, dim, interpolation=cv.INTER_CUBIC)
    extension = file.filename.split(".")[-1]
    filename = f'static/images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f.") + extension}'
    cv.imwrite(filename, img)
    return render_template("01_out.html", img_file=filename)

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
