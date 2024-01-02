from flask import Flask, request, render_template, jsonify
import cv2 as cv
import numpy as np
import datetime

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("03.html")

@app.route('/demo_json', methods=['POST'])
def demo_json():
    file = request.files['file']
    img = cv.imdecode(np.frombuffer(file.read(), np.uint8), cv.IMREAD_UNCHANGED)

    width = 500
    scale_percent = img.shape[1] / width
    height = int(img.shape[0] / scale_percent)
    dim = (width, height)
    img = cv.resize(img, dim, interpolation=cv.INTER_CUBIC)

    pixels = np.float32(img.reshape(-1, 3))
    n_colors = 15
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    average = img.mean(axis=0).mean(axis=0)

    json_structure = {
        "originalImage": {
            "width": img.shape[1],
            "height": img.shape[0]
        },
        "average_color": np.array2string(average, precision=2, separator=',', suppress_small=True),
        "dominant_color": np.array2string(dominant, precision=2, separator=',', suppress_small=True)
    }

    avg_patch = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(average)
    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices] / float(counts.sum())]))
    rows = np.int_(img.shape[0] * freqs)
    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    extension = file.filename.split(".")[-1]
    f_img = f'static/images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f.") + extension}'
    cv.imwrite(f_img, img)
    f_avg = f'static/images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f_avg.") + extension}'
    cv.imwrite(f_avg, avg_patch)
    f_dom = f'static/images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f_dom.") + extension}'
    cv.imwrite(f_dom, dom_patch)
    return render_template("03_out.html", img_file=f_img, avg_file=f_avg, dom_file=f_dom, json_structure=json_structure)


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
