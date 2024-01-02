from flask import Flask

app = Flask(__name__)

@app.route('/')
def no_name():
    return "I don't know your name!"

@app.route('/<name>')
def hello_name(name):
    return f"Hello {name}!"

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
