import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if not file.filename.endswith('.csv'):
         return jsonify({"error": "Invalid file format, must be CSV"}), 400
         
    try:
        df = pd.read_csv(file)
        
        # Auto-detect target column
        targets = ["churn", "churned", "target", "label"]
        target_col = next((c for c in df.columns if c.lower().strip() in targets), df.columns[-1])
        
        # Exclude IDs
        ignore = ["id", "customerid", "customer_id", "userid"]
        features = [c for c in df.columns if c != target_col and c.lower().strip() not in ignore]
        
        df = df.dropna(subset=[target_col])
        
        # Map target to 0/1
        target_map = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
        
        # Safely convert target to strings/lowercase if applicable, then map
        if df[target_col].dtype == object or df[target_col].dtype == str:
            df[target_col] = df[target_col].astype(str).str.lower().str.strip().map(target_map).fillna(0).astype(int)
            
        # Encode categorical variables robustly
        df_feats = df[features].copy()
        for col in df_feats.columns:
            if df_feats[col].dtype == object or df_feats[col].dtype == str:
                df_feats[col] = df_feats[col].astype('category').cat.codes
            else:
                df_feats[col] = pd.to_numeric(df_feats[col], errors='coerce').fillna(0)
                
        X = df_feats.values
        y = df[target_col].values
        
        if len(X) < 10:
            return jsonify({"error": "Dataset too small (needs at least 10 rows)"}), 400
            
        # Split and train
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(X)), test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else np.zeros(len(y_test))
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        roc_data = [{"x": float(f), "y": float(t)} for f, t in zip(fpr, tpr)]
        
        # Feature importances
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:10]
        feat_imp = [{"name": features[i], "value": float(importances[i])} for i in sorted_idx]
        
        # Dist
        total_pred_pos = int(np.sum(y_pred == 1))
        total_pred_neg = int(len(y_pred) - total_pred_pos)
        
        # High Risk
        test_df = df.iloc[idx_test].copy()
        test_df['Prob'] = y_proba
        high_risk = test_df.sort_values('Prob', ascending=False).head(10)
        
        hr_data = []
        for _, row in high_risk.iterrows():
            id_col = next((c for c in df.columns if "id" in c.lower()), None)
            hr_data.append({
                "id": str(row[id_col]) if id_col else "Unknown",
                "F1": str(row[features[sorted_idx[0]]] if len(sorted_idx) > 0 else ""),
                "F2": str(row[features[sorted_idx[1]]] if len(sorted_idx) > 1 else ""),
                "F3": str(row[features[sorted_idx[2]]] if len(sorted_idx) > 2 else ""),
                "F4": str(row[features[sorted_idx[3]]] if len(sorted_idx) > 3 else ""),
                "prob": float(row['Prob'])
            })
            
        # Prob Distribution
        bins = np.histogram(y_proba, bins=5, range=(0, 1))[0].tolist()

        return jsonify({
            "metrics": {
                "acc": acc, "prec": prec, "rec": rec, "f1": f1
            },
            "cm": {
                "tn": tn, "fp": fp, "fn": fn, "tp": tp
            },
            "roc": {
                "data": roc_data, "auc": roc_auc
            },
            "features": feat_imp,
            "dist": {
                "not_churn": total_pred_neg, "churn": total_pred_pos, "total": len(y_test)
            },
            "risk_table": hr_data,
            "prob_bins": bins
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)