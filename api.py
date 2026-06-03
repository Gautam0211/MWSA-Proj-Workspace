from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import xgboost as xgb
import numpy as np
import pickle
import os

app = FastAPI(
    title="Institutional Credit Risk & Default Prediction API",
    description="Production endpoint featuring inline Z-score normalization matrices and automated K-Means clustering.",
    version="1.4.0"
)

# 1. Verify model registry paths
XGB_PATH = "final_detective_model.json"
KMEANS_PATH = "final_kmeans_model.pkl"

if not os.path.exists(XGB_PATH) or not os.path.exists(KMEANS_PATH):
    raise FileNotFoundError("🚨 Registry error: One or more model artifacts are missing from the root directory.")

# 2. Load native trained booster and clustering engine into memory
detective_model = xgb.Booster()
detective_model.load_model(XGB_PATH)

with open(KMEANS_PATH, "rb") as f:
    kmeans_model = pickle.load(f)

SUB_GRADE_ORDER = [
    'A1','A2','A3','A4','A5','B1','B2','B3','B4','B5',
    'C1','C2','C3','C4','C5','D1','D2','D3','D4','D5',
    'E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5'
]

SubGradeOptions = Literal[
    'A1','A2','A3','A4','A5','B1','B2','B3','B4','B5',
    'C1','C2','C3','C4','C5','D1','D2','D3','D4','D5',
    'E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5'
]

SCALER_METRICS = {
    'int_rate':   {'mean': 13.26, 'scale': 4.76},
    'loan_amnt':  {'mean': 15046.0, 'scale': 9190.0},
    'annual_inc': {'mean': 78000.0, 'scale': 49000.0},
    'bc_util':    {'mean': 57.85, 'scale': 28.35}
}

class ProductionApplicantData(BaseModel):
    sub_grade: SubGradeOptions  
    int_rate: float = Field(..., ge=4.0, le=40.0, description="True Interest Rate percentage (4% to 40%)")
    loan_amnt: float = Field(..., ge=500.0, le=40000.0, description="True Requested Loan Amount in dollars ($500 to $40,000)")
    annual_inc: float = Field(..., ge=1000.0, le=1000000.0, description="True Annual Income in dollars ($1,000 to $1,000,000)")
    bc_util: float = Field(..., ge=0.0, le=150.0, description="True Bankcard Utilization percentage (0% to 150%)")

@app.get("/")
def home():
    return {"status": "ONLINE", "framework": "FastAPI", "engine": "Credit Risk Scoring"}

@app.post("/evaluate_loan")
def evaluate_loan(applicant: ProductionApplicantData):
    try:
        # A. Scale features using historical benchmarks
        sub_grade_rank = SUB_GRADE_ORDER.index(applicant.sub_grade) + 1
        z_sub_grade = (sub_grade_rank - 13.2) / 6.8  
        
        baseline_model_prob = 0.12 if sub_grade_rank <= 10 else (0.22 if sub_grade_rank <= 20 else 0.45)
        
        z_int_rate   = (applicant.int_rate - SCALER_METRICS['int_rate']['mean']) / SCALER_METRICS['int_rate']['scale']
        z_loan_amnt  = (applicant.loan_amnt - SCALER_METRICS['loan_amnt']['mean']) / SCALER_METRICS['loan_amnt']['scale']
        z_annual_inc = (applicant.annual_inc - SCALER_METRICS['annual_inc']['mean']) / SCALER_METRICS['annual_inc']['scale']
        z_bc_util    = (applicant.bc_util - SCALER_METRICS['bc_util']['mean']) / SCALER_METRICS['bc_util']['scale']
        
        feature_mapping = {
            "sub_grade": z_sub_grade,
            "int_rate": z_int_rate,
            "loan_amnt": z_loan_amnt,
            "annual_inc": z_annual_inc,
            "bc_util": z_bc_util
        }
        
        expected_names = detective_model.feature_names
        padded_vector = [feature_mapping.get(name, 0.0) for name in expected_names]
        matrix_array = np.array([padded_vector], dtype=np.float32)
        dmatrix_payload = xgb.DMatrix(matrix_array, feature_names=expected_names)
        
        # B. Generate tree prediction and isolate numerical scalar value
        raw_predictions = detective_model.predict(dmatrix_payload)
        xgboost_correction_score = float(raw_predictions.item())
        
        # C. Synthesize continuous risk metrics
        final_calibrated_risk = max(0.0, min(1.0, baseline_model_prob + xgboost_correction_score))
        credit_decision = "REJECT - HIGH DEFAULT RISK" if final_calibrated_risk >= 0.214 else "APPROVE - SAFE CREDIT PROFILE"
        
        # D. Automated K-Means segment assignment via the true model coordinates
        predicted_cluster_id = int(kmeans_model.predict(matrix_array)[0])
        
        cluster_names = {
            0: "Cluster 0: Over-Penalized Low-Ticket Rejection Trap",
            1: "Cluster 1: Stable Mainstream Prime Portfolio",
            2: "Cluster 2: Toxic Large-Ticket Default Leak"
        }
        assigned_cluster = cluster_names.get(predicted_cluster_id, "Unknown Segment Profile")
            
        return {
            "status": "SUCCESS",
            "selected_tier": applicant.sub_grade,
            "components_breakdown": {
                "stage_1_baseline_probability": round(baseline_model_prob, 4),
                "stage_2_xgboost_correction_score": round(xgboost_correction_score, 6),
                "final_calculated_combined_probability": round(final_calibrated_risk, 4)
            },
            "mrmg_automated_cluster_id": predicted_cluster_id,
            "mrmg_assigned_segment": assigned_cluster,
            "final_credit_decision": credit_decision
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Processing Exception: {str(e)}")
