import eventlet
from livevideo import app

import socketio
from waitress import serve
import socket
 
hostname=socket.gethostname()   
IPAddr=socket.gethostbyname(hostname)  
sio = socketio.Server()
appServer = socketio.WSGIApp(sio, app)

# @sio.event
# def connect(sid, environ):
#     print('connect ', sid)

# @sio.event
# def my_message(sid, data):
#     print('message ', data)

# @sio.event
# def disconnect(sid):
#     print('disconnect ', sid)

if __name__ == '__main__':
    # eventlet.wsgi.server(eventlet.listen(('localhost', 5000)), appServer)
    print("Server started connect with http://127.0.0.1:8080 or http://"+IPAddr+":8080 - 6 clients max")
    serve(appServer, host='0.0.0.0', port=8080, url_scheme='http', threads=6)
    # threads=6,expose_tracebacks=True,log_untrusted_proxy_headers=True

