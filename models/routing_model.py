import os
import torch
import lightgbm as lgb
import numpy as np
from models.gating import create_gating_model
from config.config import (
    GATING_MODE, GATING_INPUT_DIM, GATING_HIDDEN_DIM, GATING_OUTPUT_DIM, 
    GATING_THRESHOLD, EXPERT_NORMAL_MODEL_PATH, EXPERT_PACKED_MODEL_PATH,
    GATING_MODEL_PATH
)

class RoutingModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.gating_model = None
        self.expert_normal = None
        self.expert_packed = None
        self.load_models()

    def load_models(self):
        # Load Gating Model
        if os.path.exists(GATING_MODEL_PATH):
            print(f"[*] Loading Gating Model from {GATING_MODEL_PATH}")
            self.gating_model = create_gating_model(GATING_MODE, GATING_INPUT_DIM, GATING_HIDDEN_DIM, GATING_OUTPUT_DIM)
            self.gating_model.load_state_dict(torch.load(GATING_MODEL_PATH, map_location=self.device))
            self.gating_model.to(self.device)
            self.gating_model.eval()
        else:
            print(f"[!] Gating model not found at {GATING_MODEL_PATH}")

        # Load Expert Models
        if os.path.exists(EXPERT_NORMAL_MODEL_PATH):
            print(f"[*] Loading Normal Expert from {EXPERT_NORMAL_MODEL_PATH}")
            self.expert_normal = lgb.Booster(model_file=EXPERT_NORMAL_MODEL_PATH)
        else:
            print(f"[!] Normal expert model not found at {EXPERT_NORMAL_MODEL_PATH}")

        if os.path.exists(EXPERT_PACKED_MODEL_PATH):
            print(f"[*] Loading Packed Expert from {EXPERT_PACKED_MODEL_PATH}")
            self.expert_packed = lgb.Booster(model_file=EXPERT_PACKED_MODEL_PATH)
        else:
            print(f"[!] Packed expert model not found at {EXPERT_PACKED_MODEL_PATH}")

    def predict(self, features):
        """
        Predicts using the routing mechanism.
        features: numpy array of shape (n_samples, n_features)
        Returns: predictions (n_samples,), routing_decisions (n_samples,)
        """
        if self.gating_model is None:
            raise RuntimeError("Gating model is not loaded.")
        
        # Prepare features for Gating Model
        x_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            logits = self.gating_model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            # Probability of being 'Packed' (class 1)
            packed_probs = probs[:, 1].cpu().numpy()
        
        # Routing Decision
        # 1 means Packed (send to Expert Packed), 0 means Normal (send to Expert Normal)
        routing_decisions = (packed_probs > GATING_THRESHOLD).astype(int)
        
        predictions = np.zeros(len(features))
        
        # Indices for each expert
        normal_indices = np.where(routing_decisions == 0)[0]
        packed_indices = np.where(routing_decisions == 1)[0]
        
        # Inference with Expert Normal
        if len(normal_indices) > 0:
            if self.expert_normal:
                X_normal = features[normal_indices]
                pred_normal = self.expert_normal.predict(X_normal)
                predictions[normal_indices] = pred_normal
            else:
                print("[!] Expert Normal not loaded, skipping predictions for normal samples.")
        
        # Inference with Expert Packed
        if len(packed_indices) > 0:
            if self.expert_packed:
                X_packed = features[packed_indices]
                pred_packed = self.expert_packed.predict(X_packed)
                predictions[packed_indices] = pred_packed
            else:
                print("[!] Expert Packed not loaded, skipping predictions for packed samples.")
                
        return predictions, routing_decisions

    def get_routing_stats(self, routing_decisions):
        total = len(routing_decisions)
        packed_count = np.sum(routing_decisions)
        normal_count = total - packed_count
        return {
            'total': total,
            'normal': normal_count,
            'packed': packed_count,
            'packed_ratio': packed_count / total if total > 0 else 0
        }
