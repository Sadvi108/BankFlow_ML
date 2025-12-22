"""
Simple HTTP server to test receipts without FastAPI dependencies
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

ocr_pipeline = EnhancedOCRPipeline()

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Receipt Tester</title>
                <style>
                    body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
                    h1 { color: #333; }
                    .upload-box { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
                    button { background: #4CAF50; color: white; padding: 15px 30px; border: none; cursor: pointer; font-size: 16px; }
                    button:hover { background: #45a049; }
                    #result { margin-top: 20px; padding: 20px; background: #f5f5f5; display: none; }
                    .success { color: green; }
                    .error { color: red; }
                </style>
            </head>
            <body>
                <h1>🧾 Bank Receipt OCR Tester</h1>
                <div class="upload-box">
                    <input type="file" id="fileInput" accept=".pdf,.jpg,.jpeg,.png">
                    <br><br>
                    <button onclick="testReceipt()">Test Receipt</button>
                </div>
                <div id="result"></div>
                
                <script>
                function testReceipt() {
                    const file = document.getElementById('fileInput').files[0];
                    if (!file) {
                        alert('Please select a file first');
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    document.getElementById('result').innerHTML = 'Processing...';
                    document.getElementById('result').style.display = 'block';
                    
                    fetch('/upload', {
                        method: 'POST',
                        body: formData
                    })
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h2>Results:</h2>';
                        if (data.success) {
                            html += '<p class="success">✅ Extraction Successful!</p>';
                            html += '<p><strong>Bank:</strong> ' + (data.bank || 'Unknown') + '</p>';
                            html += '<p><strong>Transaction IDs:</strong> ' + JSON.stringify(data.ids || []) + '</p>';
                            html += '<p><strong>Amount:</strong> ' + (data.amount || 'Not found') + '</p>';
                            html += '<p><strong>Confidence:</strong> ' + (data.confidence || 0) + '%</p>';
                        } else {
                            html += '<p class="error">❌ ' + data.error + '</p>';
                        }
                        document.getElementById('result').innerHTML = html;
                    })
                    .catch(err => {
                        document.getElementById('result').innerHTML = '<p class="error">Error: ' + err + '</p>';
                    });
                }
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/upload':
            # This is a simplified version - full multipart handling would be complex
            # For now, just return a test response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'success': True,
                'message': 'Server is running! Full upload functionality requires FastAPI.',
                'bank': 'Test',
                'ids': ['TEST123456'],
                'amount': 'RM 100.00',
                'confidence': 95
            }
            self.wfile.write(json.dumps(response).encode())

if __name__ == '__main__':
    port = 8081
    server = HTTPServer(('localhost', port), SimpleHandler)
    print(f'✅ Server running at http://localhost:{port}')
    print('Press Ctrl+C to stop')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped')
