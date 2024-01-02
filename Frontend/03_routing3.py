from flask import Flask

app = Flask(__name__)

@app.route('/')
def no_color():
    return "You didn't pick a color!"

def red_color():
    return "RED"

def green_color():
    return "GREEN"

def blue_color():
    return "BLUE"

app.add_url_rule('/red',view_func=red_color)
app.add_url_rule('/green', view_func=green_color)
app.add_url_rule('/blue', view_func=blue_color)

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
