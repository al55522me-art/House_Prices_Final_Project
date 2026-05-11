# House Prices Final Project

Учебный ML-проект на данных Kaggle House Prices - Advanced Regression Techniques.
Проект содержит EDA, classic ML regression и простую DL-модель на PyTorch.

## Структура

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   └── EDA.ipynb
├── outputs/
│   ├── metrics/
│   ├── models/
│   └── submissions/
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── training.py
│   └── dl/
│       ├── model.py
│       └── training.py
├── main.py
└── requirements.txt
```

## Быстрый запуск

```bash
pip install -r requirements.txt
python main.py --mode classic
```

DL-часть:

```bash
python main.py --mode dl --epochs 250
```

Все пайплайны:

```bash
python main.py --mode all
```

## Метрика

Для Kaggle House Prices основная метрика - RMSLE. Поэтому модели обучаются на
`log1p(SalePrice)`, а submission возвращает цены в исходной шкале.

## Classic ML модели

- Ridge
- Lasso
- ElasticNet
- SVR
- Random Forest
- Extra Trees
- Gradient Boosting
- LightGBM

## Артефакты

После запуска создаются:

- `outputs/metrics/classic_ml_metrics.csv`
- `outputs/metrics/classic_ml_metrics.json`
- `outputs/models/best_classic_model.joblib`
- `outputs/submissions/classic_ml_submission.csv`
- `outputs/metrics/dl_metrics.csv`
- `outputs/metrics/dl_metrics.json`
- `outputs/models/house_price_mlp.pt`
- `outputs/models/dl_preprocessor.joblib`
- `outputs/submissions/dl_submission.csv`
