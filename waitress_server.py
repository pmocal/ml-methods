from waitress import serve
import app
serve(app.app, listen='*:8080', trusted_proxy='*', clear_untrusted_proxy_headers=False, threads=1)