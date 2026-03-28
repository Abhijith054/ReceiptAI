"""
Demo script: runs extraction on 3 sample CORD receipts and prints results.
Shows the full pipeline without needing to start the API server.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


SAMPLE_RECEIPTS = [
    {
        "name": "cafe_receipt.txt",
        "text": """
SUNSHINE CAFE
123 Main Street, Downtown
Tel: 555-0123

Date: 2024-01-15
Receipt No: RC-20240115-001

Cappuccino x2          $8.00
Blueberry Muffin       $3.50
Avocado Toast          $12.00

Subtotal:             $23.50
Tax (8.5%):            $2.00
TOTAL:                $25.50

Thank you for visiting!
""",
    },
    {
        "name": "grocery_receipt.txt",
        "text": """
FRESH MART GROCERY
Invoice #: INV-2024-8842
Date: 15/03/2024

Organic Milk 2L        4.99
Whole Wheat Bread      3.29
Free Range Eggs 12ct   5.49
Cheddar Cheese 500g    7.99
Apples 1kg             2.99

Grand Total:          24.75
Cash Paid:            30.00
Change:                5.25
""",
    },
    {
        "name": "restaurant_receipt.txt",
        "text": """
THE GOLDEN DRAGON RESTAURANT
Fine Asian Cuisine

Server: Table 7
Date: 2024/02/20   Time: 19:45
Order #: 2024-0220-T7

Spring Rolls (2)       $9.00
Kung Pao Chicken      $16.50
Steamed Rice           $3.00
Jasmine Tea            $4.50

Subtotal:             $33.00
Service Charge (10%):  $3.30
Total Amount:         $36.30
""",
    },
]


def run_demo():
    print("=" * 60)
    print("  ReceiptAI — Information Extraction Demo")
    print("=" * 60)

    from src.extractor import ReceiptExtractor
    from src.storage import RecordStorage

    extractor = ReceiptExtractor()
    storage = RecordStorage("data/demo_records.jsonl")

    print(f"\nExtraction method: {'🤖 Fine-tuned NER Model' if extractor._model_loaded else '📐 Regex Fallback'}")
    print()

    for i, sample in enumerate(SAMPLE_RECEIPTS, 1):
        print(f"{'─' * 60}")
        print(f"  Receipt {i}: {sample['name']}")
        print(f"{'─' * 60}")

        # Extract
        result = extractor.extract(sample["text"])

        # Store
        record = storage.save_record(result, filename=sample["name"])

        # Display
        print(f"  Doc ID : {record['doc_id']}")
        print(f"  Method : {result.get('method', '?')}")
        print()
        print("  Extracted Fields:")
        ex = result
        print(f"    Total Amount : {ex.get('total_amount') or '—'}")
        print(f"    Date         : {ex.get('date') or '—'}")
        print(f"    Vendor       : {ex.get('vendor_name') or '—'}")
        print(f"    Receipt ID   : {ex.get('receipt_id') or '—'}")
        print()
        print("  Full JSON output:")
        print("  " + json.dumps({
            k: v for k, v in ex.items()
            if k in ("total_amount", "date", "vendor_name", "receipt_id", "method")
        }, indent=4).replace("\n", "\n  "))
        print()

    print("=" * 60)
    print(f"  Results saved to data/demo_records.jsonl")
    print("=" * 60)

    # QA demo
    print("\n  Question Answering Demo")
    print("─" * 60)

    from src.qa_engine import QAEngine
    qa = QAEngine(storage)

    questions = [
        "What is the total amount?",
        "What is the date of receipt?",
        "Who is the vendor?",
        "Show me all receipts",
    ]

    all_records = storage.list_all()
    if all_records:
        first_doc = all_records[0]["doc_id"]
        for q in questions:
            print(f"\n  Q: {q}")
            result = qa.answer(q, doc_id=first_doc)
            print(f"  A: {result['answer']}")

    print("\n" + "=" * 60)
    print("  Demo complete! Run the API with:")
    print("    uvicorn app.main:app --reload --port 8000")
    print("  Then open: http://localhost:8000")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
