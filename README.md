# Bank Receipt OCR and Structured Extraction

A specialized Optical Character Recognition (OCR) and information-extraction
system for Malaysian bank receipts. It combines bounded OCR with label-aware
extraction, so account numbers, bank references, clearing references, and
payer-supplied references remain separate.

## 🚀 Features

*   **Measured Regression Coverage**: The labeled IBG corpus currently passes 168/168 scalar field checks and captures 72/72 expected references. These are regression results, not a claim of universal real-world accuracy.
*   **Multi-Bank Support**: Supports Maybank, CIMB, Public Bank, RHB, Hong Leong, AmBank, HSBC, UOB, Standard Chartered, DuitNow, and more.
*   **Robust Extraction**:
    *   **Ultimate Pattern Matcher V3**: Advanced regex engine with flexible spacing, OCR error repair, and noise filtering.
    *   **Layout-Awareness**: Intelligent fallback to layout analysis when patterns are ambiguous.
    *   **OCR Repair**: Automatically fixes common OCR glitches (e.g., `H5BC` -> `HSBC`, `l` -> `1`).
*   **Adaptive OCR**: One fast pass for clean documents, then at most one orientation check and one photo-optimized retry for weak reads.
*   **Complete Reference Output**: Returns every labeled reference with its role while preserving the legacy scalar ID.
*   **Simple UI**: Web interface for easy testing and upload.

## 📂 Project Structure

```
├── app/                    # Core application logic
│   ├── ultimate_patterns_v3.py  # The extraction brain (100% accuracy engine)
│   ├── enhanced_ocr_pipeline.py # OCR processing
│   └── ...
├── tests/                  # Test suites
│   ├── test_100_percent_accuracy.py # Main validation script
│   └── ...
├── scripts/                # Utility scripts (debug, training, verification)
├── logs/                   # Logs and test outputs
├── docs/                   # Documentation
├── static/                 # Static assets
├── templates/              # HTML templates
├── simple_server.py        # Lightweight FastAPI server
├── run_tests.py            # Unified test runner
└── requirements.txt        # Dependencies
```

## 🛠️ Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python simple_server.py
```
Open **[http://localhost:8081](http://localhost:8081)** in your browser.

### 3. Run Validation Tests

To verify the system's accuracy:

```bash
python run_tests.py
```

## 🧠 Key Components

### Ultimate Pattern Matcher V3 (`app/ultimate_patterns_v3.py`)
The core engine that handles:
*   **Bank Detection**: Identifies the bank from keywords and logos.
*   **ID Extraction**: Extracts Transaction IDs, Reference Numbers, and DuitNow IDs.
*   **Normalization**: Cleans noise, fixes spacing (`Ref :` -> `Ref:`), and standardizes separators.
*   **OCR Repair**: intelligently repairs digit/letter confusion based on context.

### Simple Server (`simple_server.py`)
A FastAPI backend that serves the UI and processes uploads using the V3 engine.

### Portal response contract

`POST /extract` keeps the existing `transaction_id` and `all_ids` fields and
also returns unambiguous reference collections:

* `primary_reference_id`: the bank's primary transaction reference, or `null`.
* `reference_ids`: every extracted reference, deduplicated.
* `bank_reference_ids`: primary plus bank/clearing-system references.
* `payer_reference_ids`: customer, invoice, remittance, and other payer-entered references.
* `references`: objects containing `value`, `label`, `role`, `confidence`, and `source`.
* `processing_time_ms`, `timings`, and `ocr_details`: production latency diagnostics.

### OCR deployment controls

The defaults in `.env.example` and `render.yaml` cap rendered images at 2800
pixels and use two OCR passes at most. `OCR_ENABLE_HEAVY_PASS=1` enables a third
CPU-heavy pass and should only be used after staging benchmarks show a benefit.

## 🧪 Testing

The system is validated against a comprehensive suite of edge cases in `tests/test_100_percent_accuracy.py`, covering:
*   Standard formats (e.g., `Ref: 123456`)
*   OCR errors (e.g., `H5BC...`)
*   Weird spacing (e.g., `Ref : 123`)
*   Multiline IDs
*   Noise and clutter

## 📜 License
Internal use only.
