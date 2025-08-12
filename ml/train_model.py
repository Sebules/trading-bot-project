import os
import json
import pickle
import pandas as pd
import glob
from datetime import datetime

from nltk import accuracy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def load_report(report_dir, filename):
    path = os.path.join(report_dir, filename)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df.dropna(inplace=True)
    df["Return"] = df["Close"].pct_change().shift(-1)
    df["Target"] = (df["Return"] > 0).astype(int)
    return df

def get_candidate_features(df):
    return [col for col in df.columns if df[col].dtype != 'O' and col not in ['Return', 'Target']]

def train_model(df, selected_features, model_type="RandomForestClassifier", n_estimators=100, random_state=42):
    X = df[selected_features]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    if model_type == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model_params = {"n_estimators": n_estimators, "random_state": random_state}
    elif model_type == "SVC":
        model = SVC()
        model_params = {"model": "SVC (par défaut)"}
    else:
        raise ValueError("Modèle non supporté")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, model_params, accuracy, report

def save_model(model, model_params, report, selected_features, rapport_filename,
               model_type, model_dir, accuracy, strategy_params={}):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = rapport_filename.replace(".csv", "")
    model_name = f"{model_type}_{base_name}_{timestamp}"

    # Save model
    pickle_path = os.path.join(model_dir, model_name + ".pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)

    # Save metadata
    meta_path = os.path.join(model_dir, model_name + ".json")
    metadata = {
        "model_name": model_name,
        "created_at": timestamp,
        "strategies_used": list(set([col.split("_")[0] for col in selected_features])),
        "features_used": selected_features,
        "model_params": model_params,
        "strategy_params": strategy_params,
        "metrics": report,
        "accuracy": accuracy
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model_name, accuracy


def list_models(model_dir):
    """
    Retourne la liste des modèles disponibles (nom de base, sans extension).
    """
    # On cherche tous les .pkl, on enlève le .pkl pour ne garder que le nom
    paths = glob.glob(os.path.join(model_dir, "*.pkl"))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]

def load_model(model_dir, model_name):
    """
    Charge un modèle entraîné et ses métadonnées.

    Args:
      model_dir (str): dossier contenant model_name.pkl et model_name.json
      model_name (str): nom de base du modèle (sans .pkl ni .json)

    Returns:
      model: l’objet sklearn désérialisé
      metadata: dict de métadonnées (le contenu du JSON)
    """
    pkl_path  = os.path.join(model_dir, model_name + ".pkl")
    json_path = os.path.join(model_dir, model_name + ".json")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Le fichier pickle n'existe pas : {pkl_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Le fichier metadata n'existe pas : {json_path}")

    # Charge le modèle
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    # Charge les méta-infos
    with open(json_path, "r") as f:
        metadata = json.load(f)

    return model, metadata