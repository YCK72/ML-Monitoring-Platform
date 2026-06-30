import { useMemo, useState } from "react";
import { useDriftReports, useDriftSummary } from "@/api/hooks";
import { Badge } from "@/components/ui/badge";
import type { Severity } from "@/types/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

function severityBadgeVariant(severity: Severity): "default" | "secondary" | "destructive" {
  if (severity === "Red") return "destructive";
  if (severity === "Yellow") return "secondary";
  return "default";
}

export default function FeatureDetail() {
  const { data: summary } = useDriftSummary();
  const { data: reports, isLoading, isError, error } = useDriftReports();

  const features = summary?.feature_cards?.map((feature) => feature.feature_name) ?? [];
  const [selectedFeature, setSelectedFeature] = useState<string>("");

  const activeFeature = selectedFeature || features[0];

  const chartData = useMemo(() => {
    if (!reports || !activeFeature) return [];

    return reports
      .slice()
      .reverse()
      .map((report) => {
        const featureScores = report.feature_scores?.features?.[activeFeature];
        const ks = featureScores?.ks;
        const psi = featureScores?.psi;

        return {
          reportId: report.id,
          time: new Date(report.created_at).toLocaleTimeString(),
          ksPValue: ks?.p_value ?? null,
          psiScore: psi?.statistic ?? null,
          severity: ks?.severity ?? psi?.severity ?? report.overall_severity,
        };
      })
      .filter((row) => row.ksPValue !== null || row.psiScore !== null);
  }, [reports, activeFeature]);

  if (isLoading) {
    return <div className="p-6 text-muted-foreground">Loading feature details...</div>;
  }

  if (isError) {
    return (
      <div className="p-6 text-red-600">
        Failed to load feature details: {(error as Error).message}
      </div>
    );
  }

  if (!activeFeature) {
    return <div className="p-6 text-muted-foreground">No feature drift data available yet.</div>;
  }

  const latest = summary?.feature_cards?.find(
    (feature) => feature.feature_name === activeFeature
  );

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Feature Detail</h1>
        <p className="text-sm text-muted-foreground">
          Inspect per-feature drift scores across evaluation windows.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <select
          className="rounded-md border bg-background px-3 py-2 text-sm"
          value={activeFeature}
          onChange={(event) => setSelectedFeature(event.target.value)}
        >
          {features.map((feature) => (
            <option key={feature} value={feature}>
              {feature}
            </option>
          ))}
        </select>

        {latest && (
          <Badge variant={severityBadgeVariant(latest.severity)}>
            {latest.severity}
          </Badge>
        )}
      </div>

      {latest && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="rounded-xl border p-4">
            <p className="text-sm text-muted-foreground">Latest KS p-value</p>
            <p className="text-2xl font-bold">{latest.ks_p_value.toFixed(4)}</p>
          </div>

          <div className="rounded-xl border p-4">
            <p className="text-sm text-muted-foreground">Latest PSI score</p>
            <p className="text-2xl font-bold">{latest.psi_score.toFixed(4)}</p>
          </div>
        </div>
      )}

      {chartData.length === 0 ? (
        <div className="rounded-xl border p-6 text-sm text-muted-foreground">
          No historical chart data available for {activeFeature}.
        </div>
      ) : (
        <div className="h-[380px] rounded-xl border p-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="psiScore" strokeWidth={2} dot />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}