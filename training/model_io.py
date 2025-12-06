import os
import lightgbm as lgb

def save_model(model, model_path):
    model.save_model(model_path)
    print(f"[+] Model saved to: {model_path}")

def load_existing_model(model_path):
    if os.path.exists(model_path):
        print(f"[*] Loading existing model: {model_path}")
        try:
            model = lgb.Booster(model_file=model_path)
            print("[+] Existing model loaded successfully")
            return model
        except Exception as e:
            print(f"[!] Model loading failed: {e}")
            return None
    else:
        print(f"[-] Existing model not found: {model_path}")
        return None