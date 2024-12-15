# ML-project

## 1. To start training the model
```bash
export PYTHONPATH=.
./scripts/train.sh
```

## 2. To create report with the model predictions
```bash
python scripts/create_report.py
```

Evaluate the model on the train data
```bash
python scripts/pred.py --model-path checkpoints/catboost.cbm --input-csv data/train.csv --output-csv outputs/train-pred.csv
```

Create report with the model predictions
```bash
python scripts/create_report.py --pred-path outputs/train-pred.csv
```

## Data Information

refer to [DATA_DESCRIPTION.md](DATA_DESCRIPTION.md)