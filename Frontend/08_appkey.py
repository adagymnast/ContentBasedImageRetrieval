from flask import Flask, request, abort
from functools import wraps

API_KEY = "TVDkJWEQAyGfWIDdW3EmknifBt1JiJ2Q"

app = Flask(__name__)

def require_appkey(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        if request.args.get('key') and request.args.get('key') == API_KEY:
            return view_function(*args, **kwargs)
        else:
            abort(401)
    return decorated_function

@app.route('/')
@require_appkey
def hello_world():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
