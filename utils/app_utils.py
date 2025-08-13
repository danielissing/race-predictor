import os, json

def load_saved_app_creds( app_credits_path: str):
    try:
        with open(app_credits_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_app_creds(client_id: str, client_secret: str, data_dir: str, app_credits_path: str):
    os.makedirs(data_dir, exist_ok=True)
    with open(app_credits_path, "w") as f:
        json.dump({"client_id": str(client_id), "client_secret": str(client_secret)}, f)

def forget_app_creds(app_credits_path: str):
    try:
        os.remove(app_credits_path)
    except FileNotFoundError:
        pass

def fmt(sec):
    sec = int(sec);
    h = sec // 3600;
    m = (sec % 3600) // 60;
    return f"{h:d}:{m:02d}"