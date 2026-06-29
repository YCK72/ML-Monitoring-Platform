import { useDriftSummary } from "@/api/hooks";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Severity } from "@/types/api";

function severityBadgeVariant(severity: Severity): "default" | "secondary" | "destructive" {
  if (severity === "Red") return "destructive";
  if (severity === "Yellow") return "secondary";
  return "default";
}

export default function Overview() {
  const { data, isLoading, isError, error } = useDriftSummary();

  if (isLoading) {
    return <div className="p-6 text-muted-foreground">Loading drift summary...</div>;
  }

  if (isError) {
    return (
      <div className="p-6 text-red-600">
        Failed to load drift summary: {(error as Error).message}
      </div>
    );
  }

  if (!data) {
    return <div className="p-6 text-muted-foreground">No drift summary available yet.</div>;
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Overview</h1>
        <p className="text-sm text-muted-foreground">
          Drift report #{data.report_id} — evaluated {new Date(data.evaluated_at).toLocaleString()}
        </p>
        <div className="mt-2">
          <span className="font-medium">Overall severity: </span>
          <Badge variant={severityBadgeVariant(data.overall_severity)}>
            {data.overall_severity}
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {data.feature_cards.map((feature) => (
          <Card key={feature.feature_name}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">{feature.feature_name}</CardTitle>
              <Badge variant={severityBadgeVariant(feature.severity)}>
                {feature.severity}
              </Badge>
            </CardHeader>
            <CardContent className="text-xs text-muted-foreground space-y-1">
              <div>KS p-value: {feature.ks_p_value.toFixed(4)}</div>
              <div>PSI score: {feature.psi_score.toFixed(4)}</div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}