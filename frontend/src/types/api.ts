export type Severity = "Green" | "Yellow" | "Red";

export interface FeatureTestResult {
  feature_name: string;
  test_name: string; // "ks" | "psi" | "chi_squared"
  statistic: number;
  p_value: number | null;
  severity: Severity;
}

export interface PredictionDriftResult {
  statistic: number;
  raw_distance: number;
  severity: Severity;
}

export interface FeatureScores {
  features: Record<string, Record<string, FeatureTestResult>>;
  prediction_drift: PredictionDriftResult | null;
  overall_severity: Severity;
}

export interface DriftReport {
  id: number;
  model_version_id: number | null;
  window_start: string;
  window_end: string;
  feature_scores: FeatureScores;
  overall_severity: Severity;
  html_report_path: string | null;
  created_at: string;
}

export interface DriftSummaryFeatureCard {
  feature_name: string;
  severity: Severity;
  ks_p_value: number;
  psi_score: number;
}

export interface DriftSummary {
  report_id: number;
  overall_severity: Severity;
  evaluated_at: string;
  feature_cards: DriftSummaryFeatureCard[];
  prediction_drift: PredictionDriftResult | null; // shape unconfirmed since it's currently always null on your data
}

export interface AlertEvent {
  id: number;
  drift_report_id: number | null;
  feature_name: string;
  severity: Severity;
  channel: "slack" | "email";
  notified_at: string;
}

export interface ModelVersion {
  id: number;
  name: string;
  version: string;
  stage: string;
  training_metrics: Record<string, number>;
  mlflow_run_id: string | null;
  created_at: string;
}