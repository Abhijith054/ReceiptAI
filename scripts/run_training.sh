#!/usr/bin/env bash
# scripts/run_training.sh
# One-command script to process data and run fine-tuning.

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "  ReceiptAI — Training Pipeline"
echo "========================================"

# Check for virtual environment
if [ -d "venv" ]; then
  source venv/bin/activate
  echo "→ Activated venv"
fi

# Default params (override with env vars)
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_TRAIN=${MAX_TRAIN:-600}
MAX_VAL=${MAX_VAL:-100}
MAX_TEST=${MAX_TEST:-100}

echo "→ Configuration:"
echo "   Epochs:    $EPOCHS"
echo "   Batch:     $BATCH_SIZE"
echo "   Train set: $MAX_TRAIN samples"
echo ""

echo "→ Starting training..."
python -m src.train \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-train "$MAX_TRAIN" \
  --max-val "$MAX_VAL" \
  --max-test "$MAX_TEST"

echo ""
echo "========================================"
echo "  Training complete!"
echo "  Model saved to: models/receipt_ner/"
echo ""
echo "  Start the API server:"
echo "    uvicorn app.main:app --reload --port 8000"
echo ""
echo "  Or run the demo:"
echo "    python scripts/demo_extract.py"
echo "========================================"
