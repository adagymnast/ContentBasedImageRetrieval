from flask import Flask

app = Flask(__name__)


@app.route('/')
def do_not_hello():
    return "Can't you say hello?"


@app.route('/hello')
def do_hello():
    return 'Hello to you, too!'

@app.route('/cvl')
def cvl():
    return 'Hello CVL!'

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
