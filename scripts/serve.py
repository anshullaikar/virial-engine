"""Static dev server on http://127.0.0.1:5173 that tells the browser not to
cache, so a rebuilt .wasm is always picked up on reload."""
import http.server

class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

http.server.ThreadingHTTPServer(("127.0.0.1", 5173), NoCache).serve_forever()
