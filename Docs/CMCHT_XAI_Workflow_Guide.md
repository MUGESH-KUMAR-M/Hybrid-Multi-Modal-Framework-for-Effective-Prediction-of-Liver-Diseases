# 🔄 CMCHT-XAI: Full Project Workflow & Execution Pipeline

This study material breaks down the exact **step-by-step workflow** of how the CMCHT-XAI project was built and executed. If an examiner asks, "Walk me through how your system processes a patient from raw data to the final diagnosis," this is the exact flow you will describe.

---

## Phase 1: Data Preprocessing Pipeline
**File:** `src/data/preprocessing.py` and `src/data/dataset.py`

Before the AI can learn, the raw data must be cleaned, transformed, and split.

1. **Loading Raw Data:** The system loads the 3 raw datasets from the `data/raw/` directory:
   - **NAFLD (Mendeley):** 87 patients (Images + 11 Tabular features).
   - **ILPD (UCI):** 583 patients (Tabular only).
   - **Cirrhosis (Kaggle):** 418 patients (Tabular only).
2. **Handling Missing Values:** If a patient is missing a blood test value (e.g., in ILPD), the pipeline fills it using **Median Imputation** (replacing the missing value with the middle value of all other patients).
3. **Handling Skewed Data:** Some blood markers (like Bilirubin) are exponentially distributed. The pipeline applies a **Log Transform** (`log1p`) to normalize them.
4. **Data Splitting (Crucial Step):** The data is split into **Train (70%)**, **Validation (15%)**, and **Test (15%)**.
   - *Important:* This is done at the **Patient-Level**. If you split by image slice, the model might see slice 1 of Patient A in training, and slice 2 of Patient A in testing. It would "memorize" the patient's anatomy instead of learning the disease. Patient-level splitting prevents this data leakage.
5. **Handling Class Imbalance:** The dataset is heavily imbalanced (e.g., very few 'Normal' patients). The pipeline applies **SMOTE (Synthetic Minority Over-sampling Technique)** *only to the training fold* to generate synthetic examples of the minority class, ensuring the model doesn't just blindly guess the majority class.
6. **Output:** The cleaned data is saved to `data/processed/` as CSV files and paired `.npy` image arrays.

---

## Phase 2: Architecture Initialization
**Files:** Everything inside `src/models/`

Once the data is ready, the system builds the neural network in memory.

1. **Imaging Encoder:** Initializes a Swin-Tiny Transformer combined with a ResNet-style CNN stem. It runs a "dummy forward pass" to figure out the exact tensor sizes dynamically (preventing shape mismatch bugs).
2. **Tabular Encoder:** Initializes the FT-Transformer. It creates a dedicated weight matrix (`nn.Parameter`) for every single tabular feature to map them into the embedding space, then passes them through a Multi-Head Self-Attention block.
3. **CSG-Fusion Layer:** Initializes the cross-attention blocks that will eventually merge the image and tabular vectors.
4. **CUSP-Cascade Heads:** Initializes three separate prediction modules: 
   - Detection (Binary classification)
   - Staging (Multi-class classification)
   - Severity (Continuous regression)

---

## Phase 3: Model Training (The Learning Phase)
**File:** `src/train.py` and `src/training/confidence_gated_training.py`

This is where the model looks at the training data and adjusts its weights to minimize errors.

1. **The Forward Pass:**
   - A batch of 16 patients (images + tabular data) enters the encoders.
   - The encoders produce separate image and tabular embeddings.
   - The **CSG-Fusion** module applies a slight mathematical perturbation to the tabular data (the "counterfactual probe") and calculates epistemic uncertainty. Based on these two things, it weights the features and fuses them into one single vector.
2. **The CUSP-Cascade Prediction:**
   - The fused vector goes into the **Detection Head**. It outputs a prediction (Sick or Healthy?) and an uncertainty score.
   - The fused vector + the Detection Prediction + the Detection Uncertainty all flow into the **Staging Head**. It outputs the disease stage.
   - Everything flows into the **Severity Head** to output a continuous score.
3. **Confidence-Gated Cascade Training (CGCT):** 
   - Early in training (Epoch 1-10), the Detection Head is guessing blindly. If we pass its garbage guesses into the Staging head, the Staging head breaks.
   - CGCT dynamically intercepts the flow. It looks at the model's confidence. If the confidence is low, it injects the **Real Ground Truth** detection label into the Staging head instead (Teacher Forcing). As epochs increase, it fades this out.
4. **Loss Calculation & Backpropagation:** 
   - The system calculates the error using BCE (for Detection), CrossEntropy (for Staging), and Huber Loss (for Severity). 
   - The AdamW optimizer updates the network's weights to reduce this error.
5. **Validation & Checkpointing:** At the end of every epoch, it checks the Validation Set. If the loss is the lowest it has ever been, it saves the weights to `checkpoints/cmcht_xai_best.pth`.

---

## Phase 4: Evaluation and Ablation (Proving it Works)
**File:** `src/evaluate.py`

After training 50 epochs, the script evaluates the model on the 15% **Test Set** (patients it has never seen before) to prove the architecture works. It runs an **Ablation Study**.

1. **Step 1:** It turns off your contributions (loads a basic flat model with normal cross-attention) and tests it. (Baseline: Staging F1 = 0.070).
2. **Step 2:** It tests with just CSG-Fusion turned on.
3. **Step 3:** It tests with just CUSP-Cascade turned on.
4. **Step 4:** It tests the Full System using the `cmcht_xai_best.pth` checkpoint.
5. **CGCT Isolation:** It runs a completely separate 50-epoch training session *without* CGCT to prove that CGCT was responsible for the massive jump in staging accuracy (+0.240 F1 score).
6. **Output:** It generates `results/ablation_table.md` showing exactly how much each contribution improved the detection accuracy, staging F1, and ECE (calibration error).

---

## Phase 5: Explainability (XAI)
**File:** `src/explainability/run_explain.py`

A doctor cannot trust a black box. The final step is running the trained model through the Explainability Layer to generate visual and text proofs for its decisions.

1. **SHAP (Tabular Importance):** The script runs a `GradientExplainer` over the FT-Transformer. It calculates which blood tests (e.g., ALT, BMI) caused the model to make its decision, outputting a ranking chart (`shap_summary.png`).
2. **Grad-CAM (Image Attention):** It hooks into the final convolutional layer of the Swin-CNN encoder. It looks at the gradients flowing backward to see which pixels "lit up" the most when predicting fatty liver, saving heatmaps (`gradcam_sample_X.png`).
3. **DiCE (Counterfactuals):** Using gradient descent, it asks the model: "What is the smallest change I can make to this patient's blood work to make you predict they are Healthy instead of Sick?" It saves this "What-If" scenario to `counterfactuals.txt`.
4. **MC-Dropout (Uncertainty Flagging):** During inference (testing), it runs the same patient through the network 10 times, randomly turning off (dropping out) different neurons each time. If the 10 answers vary wildly, the model is uncertain. It flags these highly uncertain patients so a human doctor can manually review them.

---
*End of Workflow. By understanding this pipeline step-by-step (Data $\rightarrow$ Initialization $\rightarrow$ Training $\rightarrow$ Evaluation $\rightarrow$ XAI), you can confidently answer any question about how your system was engineered.*
