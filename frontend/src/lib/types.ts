export interface PatientMeta {
  id: number;
  stage: string;
  detection: number;
}

export interface PatientsResponse {
  count: number;
  feature_names: string[];
  patients: PatientMeta[];
}

export interface ShapItem {
  feature: string;
  value: number;
}

export interface AnalyzeResponse {
  patient_id: number;
  feature_names: string[];
  features: Record<string, number>;
  predictions: {
    detection_probability: number;
    detection_label: string;
    staging: string;
    staging_index: number;
    severity_score: number;
    severity_name: string;
  };
  uncertainty: {
    detection_variance: number;
    threshold: number;
    needs_review: boolean;
    confidence: string;
  };
  gradcam_b64: string | null;
  shap: ShapItem[];
  counterfactuals: string[];
}
