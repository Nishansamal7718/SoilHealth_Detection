from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
FEATURES = ['pH', 'N', 'P', 'K', 'OC', 'moisture']
MODEL_PATH = ROOT / 'model.joblib'
METRICS_PATH = ROOT / 'metrics.json'

def demo_dataset(n=1200, seed=42):
    """Synthetic, rule-labelled examples ONLY; not laboratory observations."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'pH': rng.uniform(3.5, 9.5, n), 'N': rng.uniform(0, 200, n),
                       'P': rng.uniform(0, 120, n), 'K': rng.uniform(0, 250, n),
                       'OC': rng.uniform(0.1, 3.5, n), 'moisture': rng.uniform(0, 100, n)})
    df['health_class'] = [demo_label(row) for _, row in df.iterrows()]
    return df

def demo_label(row):
    # Illustrative classroom thresholds; NOT agronomic recommendations.
    scores = [5.8 <= row['pH'] <= 7.5, 40 <= row['N'] <= 160,
              20 <= row['P'] <= 90, 40 <= row['K'] <= 200,
              row['OC'] >= 1.0, 25 <= row['moisture'] <= 75]
    count = sum(scores)
    return 'Healthy' if count >= 5 else ('Moderate' if count >= 3 else 'Poor')

def train(data_path=None):
    if data_path:
        df = pd.read_csv(data_path)
        missing = set(FEATURES + ['health_class']) - set(df.columns)
        if missing:
            raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
        source = 'uploaded CSV'
    else:
        df = demo_dataset()
        source = 'synthetic demonstration data (rule-labelled)'
        (ROOT / 'data' / 'demo_soil_data.csv').parent.mkdir(exist_ok=True)
        df.to_csv(ROOT / 'data' / 'demo_soil_data.csv', index=False)
    X = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    y = df['health_class'].astype(str)
    if y.nunique() < 2:
        raise ValueError('At least two health classes are needed for training.')
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model = Pipeline([('imputer', SimpleImputer(strategy='median')),
                      ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))])
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {'accuracy': round(float(accuracy_score(yte, pred)), 4), 'source': source,
               'rows': int(len(df)), 'labels': sorted(y.unique().tolist()),
               'report': classification_report(yte, pred, output_dict=True, zero_division=0),
               'confusion_matrix': confusion_matrix(yte, pred, labels=sorted(y.unique())).tolist()}
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='CSV containing pH,N,P,K,OC,moisture,health_class')
    args = parser.parse_args()
    print(json.dumps(train(args.csv), indent=2))
