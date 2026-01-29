"""
Robust BankFlow Server - Fresh Restructure
Uses standard libraries and advanced error handling for reliable extraction.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys
import os
from pathlib import Path
import re
import datetime
import traceback
import logging

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3
from app.history_manager import history_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BankFlowServer")

# Internal state
ocr_pipeline = EnhancedOCRPipeline()
debug_log_path = Path("debug_server.log")
SESSION_ID = os.urandom(4).hex()
SERVER_START_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_debug(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(debug_log_path, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

# Clear log on start
debug_log_path.write_text(f"--- SERVER STARTED AT {datetime.datetime.now()} ---\n", encoding="utf-8")

class RobustHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/':
            try:
                with open('templates/simple_upload.html', 'r', encoding='utf-8') as f:
                    html = f.read()
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            except Exception as e:
                self.send_error(500, f"Error loading template: {e}")
        
        elif self.path == '/debug_logs':
            try:
                content = debug_log_path.read_text(encoding='utf-8')
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            except:
                self.send_error(404)

        elif self.path == '/history':
            try:
                history = history_manager.get_all()
                self._send_json({"success": True, "history": history})
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, 500)

        elif self.path == '/export':
            try:
                import csv
                import io
                history = history_manager.get_all()
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=["timestamp", "filename", "bank_name", "reference_id", "amount", "date", "status"])
                writer.writeheader()
                for entry in history:
                    writer.writerow({k: entry.get(k, "") for k in writer.fieldnames})
                
                self.send_response(200)
                self.send_header('Content-type', 'text/csv')
                self.send_header('Content-Disposition', 'attachment; filename=extraction_history.csv')
                self.end_headers()
                self.wfile.write(output.getvalue().encode('utf-8'))
            except Exception as e:
                self.send_error(500, f"Export failed: {e}")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/extract' or self.path == '/upload':
            try:
                log_debug(f"Handling POST request to {self.path}")
                
                from email.parser import BytesParser
                from email.policy import default

                # Extract boundary
                content_type = self.headers.get('Content-Type', '')
                if 'multipart/form-data' not in content_type:
                    return self._send_json({"success": False, "error": "Content-Type must be multipart/form-data"}, 400)

                # Get the full body
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)

                # Parse as email message to handle multipart
                msg_bytes = b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
                msg = BytesParser(policy=default).parsebytes(msg_bytes)

                file_data = None
                original_filename = "unknown_receipt"

                for part in msg.iter_parts():
                    if part.get_content_disposition() == 'form-data' and part.get_param('name', header='content-disposition') == 'file':
                        file_data = part.get_payload(decode=True)
                        original_filename = part.get_filename() or "uploaded_receipt"
                        break

                if file_data is None:
                    log_debug("FAILED: No 'file' field found in multipart payload")
                    return self._send_json({"success": False, "error": "No file uploaded"}, 400)
                
                log_debug(f"FILE RECEIVED: {original_filename} ({len(file_data)} bytes)")
                
                # Save to disk for OCR processing
                temp_dir = Path("data/uploads")
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                # Clean filename
                clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', original_filename)
                temp_path = temp_dir / f"web_{os.urandom(4).hex()}_{clean_name}"
                
                with open(temp_path, 'wb') as f:
                    f.write(file_data)
                
                log_debug(f"SAVED: {temp_path.name}. Starting AI Analysis...")
                
                # Run OCR Pipeline
                ocr_result = ocr_pipeline.process_file(str(temp_path))
                text = ocr_result.get('text', '')
                conf = ocr_result.get('confidence', 0)
                method = ocr_result.get('method', 'unknown')
                
                log_debug(f"OCR FINISHED: {len(text)} characters found with {conf:.1f}% confidence. Method: {method}")
                
                # Extract Fields
                log_debug("RUNNING PATTERN MATCHER...")
                extraction = extract_all_fields_v3(text) if text else {}
                
                bank = extraction.get('bank_name', 'Unknown')
                ref_id = extraction.get('transaction_id')
                all_ids = extraction.get('all_ids', [])
                amount = extraction.get('amount')
                date = extraction.get('date')
                
                log_debug(f"RESULTS: Bank={bank}, Primary ID={ref_id}, IDs Found={len(all_ids)}, Amount={amount}, Date={date}")
                
                # Save to History
                try:
                    history_entry = {
                        "filename": original_filename,
                        "bank_name": bank,
                        "reference_id": ref_id or (all_ids[0] if all_ids else None),
                        "amount": amount,
                        "date": date,
                        "confidence": conf
                    }
                    entry_id = history_manager.add_entry(history_entry)
                    log_debug(f"HISTORY UPDATED: Entry ID {entry_id}")
                except Exception as he:
                    log_debug(f"HISTORY ERROR: {he}")
                    entry_id = "error_" + os.urandom(4).hex()

                # Clean up extracted artifacts
                try:
                    os.remove(temp_path)
                except:
                    pass

                # Final response construction
                response_data = {
                    'success': True,
                    'data': {
                        'entry_id': entry_id,
                        'bank_name': bank,
                        'transaction_id': ref_id or (all_ids[0] if all_ids else None),
                        'all_ids': all_ids,
                        'date': date,
                        'amount': amount,
                        'ocr_confidence': conf / 100.0, # UI expects 0.0 - 1.0
                        'method': method,
                        'text_snippet': text[:200].replace('\n', ' ') if text else "",
                        'session_id': SESSION_ID,
                        'server_start': SERVER_START_TIME
                    }
                }
                
                log_debug("RESPONSE SENT SUCCESSFULLY")
                return self._send_json(response_data)

            except Exception as e:
                err_trace = traceback.format_exc()
                log_debug(f"CRITICAL POST ERROR: {e}\n{err_trace}")
                return self._send_json({"success": False, "error": str(e), "trace": err_trace}, 500)

        elif self.path.startswith('/history/update/'):
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                updates = json.loads(body)
                entry_id = self.path.split('/')[-1]
                
                success = history_manager.update_entry(entry_id, updates)
                return self._send_json({"success": success})
            except Exception as e:
                return self._send_json({"success": False, "error": str(e)}, 500)
        else:
            self.send_error(404)

if __name__ == '__main__':
    PORT = 8081
    try:
        server = HTTPServer(('0.0.0.0', PORT), RobustHandler)
        log_debug(f"READY: Server running at http://localhost:{PORT}")
        server.serve_forever()
    except KeyboardInterrupt:
        log_debug("SHUTDOWN: Server stopped by user")
    except Exception as e:
        log_debug(f"SHUTDOWN: Fatal error: {e}")
