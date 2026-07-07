"use client";

import { useCallback, useEffect, useState } from "react";
import { CircleNotch } from "@phosphor-icons/react";
import { analyzePatient, getPatients } from "@/lib/api";
import type { AnalyzeResponse, PatientMeta } from "@/lib/types";
import { CounterfactualList } from "@/components/counterfactual-list";
import { DashboardHeader } from "@/components/dashboard-header";
import { GradCamPanel } from "@/components/gradcam-panel";
import { MetricCard } from "@/components/metric-card";
import { PatientSidebar } from "@/components/patient-sidebar";
import { ShapChart } from "@/components/shap-chart";
import { UncertaintyBanner } from "@/components/uncertainty-banner";

export default function DashboardPage() {
  const [patients, setPatients] = useState<PatientMeta[]>([]);
  const [selectedId, setSelectedId] = useState(0);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAnalysis = useCallback(async (id: number) => {
    setLoading(true);
    setError(null);
    try {
      const data = await analyzePatient(id);
      setAnalysis(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to connect to the CMCHT-XAI API.",
      );
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    getPatients()
      .then((res) => {
        setPatients(res.patients);
        return loadAnalysis(0);
      })
      .catch(() => {
        setError(
          "Cannot reach the backend. Start it with: uvicorn api.server:app --reload --port 8000",
        );
        setLoading(false);
      });
  }, [loadAnalysis]);

  const handleSelect = (id: number) => {
    setSelectedId(id);
    loadAnalysis(id);
  };

  const detProb = analysis?.predictions.detection_probability ?? 0;
  const detTone =
    detProb >= 0.5 ? ("danger" as const) : ("success" as const);

  return (
    <div className="flex min-h-[100dvh] flex-col">
      <DashboardHeader />

      <div className="mx-auto grid w-full max-w-[1600px] flex-1 grid-cols-1 gap-6 px-4 py-6 md:px-6 lg:grid-cols-[280px_1fr] lg:px-8 lg:py-8">
        <PatientSidebar
          patients={patients}
          selectedId={selectedId}
          features={analysis?.features ?? {}}
          onSelect={handleSelect}
          loading={loading}
        />

        <main className="min-w-0 space-y-6">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">
              Multi-task clinical analysis
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
              Cross-modal fusion of liver imaging and tabular labs with
              explainable AI outputs for detection, staging, and severity.
            </p>
          </div>

          {error && (
            <div className="panel border-danger/30 bg-danger/5 p-5 text-sm text-danger">
              {error}
            </div>
          )}

          {loading && !analysis && !error && (
            <div className="panel flex items-center justify-center gap-3 p-12 text-muted">
              <CircleNotch size={22} className="animate-spin text-accent" />
              Running inference and explainability pipeline...
            </div>
          )}

          {analysis && (
            <>
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                <MetricCard
                  label="Detection probability"
                  value={`${(detProb * 100).toFixed(1)}%`}
                  sublabel={analysis.predictions.detection_label}
                  tone={detTone}
                  icon="chart"
                  delay={0}
                />
                <MetricCard
                  label="Disease staging"
                  value={analysis.predictions.staging}
                  sublabel={`Class ${analysis.predictions.staging_index}`}
                  icon="brain"
                  delay={0.06}
                />
                <MetricCard
                  label={`Severity (${analysis.predictions.severity_name})`}
                  value={analysis.predictions.severity_score.toFixed(1)}
                  sublabel="Model regression output"
                  icon="pulse"
                  delay={0.12}
                />
              </div>

              <UncertaintyBanner
                variance={analysis.uncertainty.detection_variance}
                threshold={analysis.uncertainty.threshold}
                needsReview={analysis.uncertainty.needs_review}
              />

              <div className="grid gap-6 xl:grid-cols-2">
                <GradCamPanel imageB64={analysis.gradcam_b64} />
                <div>
                  <div className="mb-4">
                    <h2 className="text-lg font-medium">SHAP feature impact</h2>
                    <p className="mt-1 text-sm text-muted">
                      Clinical variables influencing the detection decision
                    </p>
                  </div>
                  <ShapChart data={analysis.shap} />
                </div>
              </div>

              <section>
                <div className="mb-4">
                  <h2 className="text-lg font-medium">
                    Counterfactual explanations
                  </h2>
                  <p className="mt-1 text-sm text-muted">
                    Actionable changes that could shift the model toward a
                    healthier prediction
                  </p>
                </div>
                <CounterfactualList items={analysis.counterfactuals} />
              </section>
            </>
          )}

          <footer className="border-t border-border pt-6 text-xs text-muted">
            CMCHT-XAI is for research use only. Not intended for clinical
            decision-making without validation on real patient data.
          </footer>
        </main>
      </div>
    </div>
  );
}
