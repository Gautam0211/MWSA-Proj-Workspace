import gradio as gr
import xgboost as xgb
import numpy as np
import pickle
import shap
import os

# =====================================================================
# 1. LOAD RUNTIME MODEL ENGINES & ARTIFACTS
# =====================================================================
XGB_PATH = "final_detective_model.json"
KMEANS_PATH = "final_kmeans_model.pkl"
LOGISTIC_PATH = "final_logistic_model.pkl"

if not os.path.exists(XGB_PATH) or not os.path.exists(KMEANS_PATH) or not os.path.exists(LOGISTIC_PATH):
    raise FileNotFoundError(
        "🚨 Registry error: One or more model artifacts (XGBoost, K-Means, or Logistic) are missing from the root directory."
    )

# Load native trained booster artifact
detective_model = xgb.Booster()
detective_model.load_model(XGB_PATH)

# Load the clustering engine
with open(KMEANS_PATH, "rb") as f:
    kmeans_model = pickle.load(f)

# Load the baseline Logistic Regression scorecard model
with open(LOGISTIC_PATH, "rb") as f:
    logistic_model = pickle.load(f)

# Initialize the tree explainer dynamically directly on the loaded booster artifact
explainer = shap.TreeExplainer(detective_model)

# =====================================================================
# 2. CONFIGURATION & BENCHMARK CONSTANTS
# =====================================================================
SUB_GRADE_ORDER = [
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

# =====================================================================
# 3. CORE RISK SCORING & PIPELINE LOGIC
# =====================================================================
def predict_credit_risk(sub_grade, int_rate, loan_amnt, annual_inc, bc_util):
    try:
        # Strict input boundary constraints matching your Pydantic boundaries
        if annual_inc < 1000.0 or annual_inc > 1000000.0:
            return "<div style='padding:15px; background-color:#fff0f0; border-left:5px solid red; color:red;'><b>Validation Error:</b> Annual Income must be between $1,000 and $1,000,000.</div>"
        if int_rate < 4.0 or int_rate > 40.0:
            return "<div style='padding:15px; background-color:#fff0f0; border-left:5px solid red; color:red;'><b>Validation Error:</b> Interest Rate must be between 4% and 40%.</div>"
        if loan_amnt < 500 or loan_amnt > 40000:
            return "<div style='padding:15px; background-color:#fff0f0; border-left:5px solid red; color:red;'><b>Validation Error:</b> Loan Amount must be between $500 and $40,000.</div>"
        if bc_util < 0.0 or bc_util > 150.0:
            return "<div style='padding:15px; background-color:#fff0f0; border-left:5px solid red; color:red;'><b>Validation Error:</b> Bankcard Utilization must be between 0% and 150%.</div>"

        # A. Run inline Z-score normalization mapping
        sub_grade_rank = SUB_GRADE_ORDER.index(sub_grade) + 1
        z_sub_grade = (sub_grade_rank - 11.45) / 6.52  
        
        z_int_rate   = (int_rate - SCALER_METRICS['int_rate']['mean']) / SCALER_METRICS['int_rate']['scale']
        z_loan_amnt  = (loan_amnt - SCALER_METRICS['loan_amnt']['mean']) / SCALER_METRICS['loan_amnt']['scale']
        z_annual_inc = (annual_inc - SCALER_METRICS['annual_inc']['mean']) / SCALER_METRICS['annual_inc']['scale']
        z_bc_util    = (bc_util - SCALER_METRICS['bc_util']['mean']) / SCALER_METRICS['bc_util']['scale']
        
        # B. Align features precisely to XGBoost & Logistic expected mapping
        feature_mapping = {"sub_grade": z_sub_grade, "int_rate": z_int_rate, "loan_amnt": z_loan_amnt, "annual_inc": z_annual_inc, "bc_util": z_bc_util}
        expected_names = detective_model.feature_names
        padded_vector = [feature_mapping.get(name, 0.0) for name in expected_names]
        matrix_array = np.array([padded_vector], dtype=np.float32)
        
        # C. FIX: Predict baseline probability using your ACTUAL Logistic Regression model file
        # Using predict_proba to extract the true probability of class 1 (default risk)
        raw_logistic_probs = logistic_model.predict_proba(matrix_array)
        baseline_model_prob = float(raw_logistic_probs[0][1])
        
        # D. Predict XGBoost Risk Score Correction
        dmatrix_payload = xgb.DMatrix(matrix_array, feature_names=expected_names)
        xgboost_correction_score = float(detective_model.predict(dmatrix_payload).item())
        
        final_probability = max(0.0, min(1.0, baseline_model_prob + xgboost_correction_score))
        
        # Dynamically assign style coordinates based on threshold breach
        if final_probability >= 0.214:
            credit_decision = "❌ REJECT - HIGH DEFAULT RISK"
            card_bg = "#fdf2f2"       
            card_border = "#ff4b4b"   
            card_text = "#9b1c1c"     
            highlight_color = "#ff4b4b"
        else:
            credit_decision = "✅ APPROVE - SAFE CREDIT PROFILE"
            card_bg = "#e6f9ed"       
            card_border = "#00cc66"   
            card_text = "#008040"     
            highlight_color = "#00cc66"
        
        # E. Dynamic on-the-fly SHAP values extraction for segment profiling
        raw_shap_matrix = explainer.shap_values(matrix_array)
        if hasattr(raw_shap_matrix, "values"):
            shap_vector = raw_shap_matrix.values
        else:
            shap_vector = np.array(raw_shap_matrix)
            
        cluster_predictions = kmeans_model.predict(shap_vector)
        predicted_cluster_id = int(cluster_predictions.item())
        
        cluster_names = {
            0: "Cluster 0: Over-Penalized Low-Ticket Rejection Trap",
            1: "Cluster 1: Stable Mainstream Prime Portfolio",
            2: "Cluster 2: Toxic Large-Ticket Default Leak"
        }
        assigned_cluster = cluster_names.get(predicted_cluster_id, "Unknown Segment Profile")
        
        # F. Render the detailed breakdown presentation layout
        results_html = f"""
        <div style='padding: 20px; border-radius: 8px; background-color: {card_bg}; border-left: 5px solid {card_border}; font-family: sans-serif;'>
            <h2 style='margin-top:0; color: {card_text};'>{credit_decision}</h2>
            <hr style='border:0; border-top:1px solid #eee; margin:15px 0;'>
            
            <h3 style='margin: 0 0 10px 0; color:#444;'>Components Breakdown</h3>
            <p style='margin: 4px 0;'><b>Stage 1 Baseline Probability:</b> {round(baseline_model_prob, 4)}</p>
            <p style='margin: 4px 0;'><b>Stage 2 XGBoost Correction Score:</b> {round(xgboost_correction_score, 6)}</p>
            <p style='margin: 4px 0; font-size: 1.05em; color: {highlight_color};'>
                <b>Final Calculated Combined Probability:</b> {round(final_probability, 4)} ({round(final_probability * 100, 2)}%)
            </p>
            
            <hr style='border:0; border-top:1px solid #eee; margin:15px 0;'>
            <p style='margin: 0; color:#333;'><b>Assigned Risk Segment:</b> {assigned_cluster} (ID: {predicted_cluster_id})</p>
        </div>
        """
        return results_html

    except Exception as e:
        return f"<div style='padding:15px; background-color:#fff0f0; border-left:5px solid red; color:red;'><b>API Processing Exception:</b> {str(e)}</div>"

# =====================================================================
# 5. INTERACTIVE FRONTEND UI LAYOUT BUILDER
# =====================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏦 Institutional Credit Risk & Default Prediction System")
    gr.Markdown("Input an applicant's financial attributes below to evaluate live loan risk scoring and automated portfolio segmentation profiling.")
    
    with gr.Row():
        with gr.Column():
            sub_grade = gr.Dropdown(choices=SUB_GRADE_ORDER, value="C2", label="Applicant Sub-Grade Tier")
            int_rate = gr.Slider(minimum=4.0, maximum=40.0, value=14.5, label="Interest Rate (%)")
            loan_amnt = gr.Slider(minimum=500, maximum=40000, value=12000, step=500, label="Requested Loan Amount ($)")
            annual_inc = gr.Number(value=75000, label="Annual Income ($)")
            bc_util = gr.Slider(minimum=0.0, maximum=150.0, value=58.2, label="Bankcard Utilization Rate (%)")
            submit_btn = gr.Button("Analyze Credit Application", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Real-Time Production Risk Output")
            output_display = gr.HTML(value="<p style='color:#777;'>Awaiting applicant input values...</p>")
            
    submit_btn.click(
        fn=predict_credit_risk, 
        inputs=[sub_grade, int_rate, loan_amnt, annual_inc, bc_util], 
        outputs=output_display
    )

if __name__ == "__main__":
    demo.launch()
