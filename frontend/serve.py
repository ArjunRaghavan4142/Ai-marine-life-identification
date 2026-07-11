import http.server
import socketserver
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(("0.0.0.0", 3000), handler)
print("Frontend serving on http://localhost:3000", flush=True)
httpd.serve_forever()
