from flask import Flask

app = Flask(__name__)

@app.route('/')
def no_color():
    return "You didn't pick a color!"

@app.route('/red')
def red_color():
    return "RED"

@app.route('/green')
def green_color():
    return "GREEN"

@app.route('/blue')
def blue_color():
    return "BLUE"

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)