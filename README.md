# Bank Receipt OCR - High Accuracy Extraction System

A specialized Optical Character Recognition (OCR) and Information Extraction system for Malaysian Bank Receipts.
Designed to achieve **100% accuracy** on reference ID extraction using a hybrid approach of enhanced pattern matching (V3) and layout-aware extraction.

## 🚀 Features

*   **High Accuracy**: Achieves 100% accuracy on the validation dataset (35/35 challenging cases).
*   **Multi-Bank Support**: Supports Maybank, CIMB, Public Bank, RHB, Hong Leong, AmBank, HSBC, UOB, Standard Chartered, DuitNow, and more.
*   **Robust Extraction**:
    *   **Ultimate Pattern Matcher V3**: Advanced regex engine with flexible spacing, OCR error repair, and noise filtering.
    *   **Layout-Awareness**: Intelligent fallback to layout analysis when patterns are ambiguous.
    *   **OCR Repair**: Automatically fixes common OCR glitches (e.g., `H5BC` -> `HSBC`, `l` -> `1`).
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

## 🧪 Testing

The system is validated against a comprehensive suite of edge cases in `tests/test_100_percent_accuracy.py`, covering:
*   Standard formats (e.g., `Ref: 123456`)
*   OCR errors (e.g., `H5BC...`)
*   Weird spacing (e.g., `Ref : 123`)
*   Multiline IDs
*   Noise and clutter

## 📜 License
Internal use only.
