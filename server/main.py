import subprocess
from flask import Flask, request
from flask_cors import CORS
import hashlib
from werkzeug.utils import secure_filename
import os
import random
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from tfsocket import TFSockets

app: Flask = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/'
app.config['ALLOWED_EXTENSIONS'] = {'py'}
CORS(app, resources=r'/*')
sockets = TFSockets(app)
token2data: dict[str, any] = {}


def allowed_file(filename: str):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET'])
def hello_world() -> None:
    return 'hello world'


@app.route('/upload', methods=['POST'])
def upload() -> None:
    upload_file = request.files['script']
    if upload_file and allowed_file(upload_file.filename):
        token = hashlib.md5(secure_filename(upload_file.filename).encode(
            'utf-8')+random.randbytes(random.randint(0, 10))).hexdigest()
        filename = token + ".py"
        upload_file.save(os.path.join(
            app.root_path, app.config['UPLOAD_FOLDER'], filename))
        ret = subprocess.run(
            "start powershell.exe cmd /k '{cmd}'".format(cmd='python -u ./static/{file}'.format(file=filename)), shell=True)
        print(ret)
        return {"data": token, "msg": "success", "code": 200}
    else:
        return 'failed'


@app.route('/local', methods=['POST'])
def local() -> None:
    local_file = request.files['file']
    return local_file.stream.read()


@sockets.route('/simulate/get', websocket=True)
def stimulate_get(ws) -> None:
    while not ws.closed:
        msg: str = ws.receive()
        if '[INIT]' in msg:
            token = msg.replace('[INIT]', '')
            token2data[token] = {"kline": "", "profits": "", "cash": ""}
            ws.send('[ACK]')
        elif '[KLINE]' in msg:
            msg = msg.replace('[KLINE]', '')
            print(msg[0:32])
            ws.send('[KLINE]')
        elif '[CASH]' in msg:
            ws.send('[CASH]')
        elif '[PROFITS]' in msg:
            ws.send('[PROFITS]')
        print(msg)


@sockets.route('/simulate/update', websocket=True)
def stimulate_update(ws) -> None:
    while not ws.closed:
        msg = ws.receive()
        if msg == '[CLOSE]':
            ws.close()
            break
        elif '[KLINE]' in msg:
            pass
        elif '[PROFIT]' in msg:
            pass
        elif '[CASH]' in msg:
            pass
        else:
            ws.close()
            break


if __name__ == '__main__':
    server = pywsgi.WSGIServer(
        ('0.0.0.0', 5000), application=app, handler_class=WebSocketHandler)
    server.serve_forever()
