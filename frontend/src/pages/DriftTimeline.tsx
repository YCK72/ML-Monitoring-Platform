import { useDriftReports } from "@/api/hooks";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

function severityToScore(severity: string) {
  if (severity === "Red") return 3;
  if (severity === "Yellow") return 2;
  return 1;
}

export default function DriftTimeline() {
  const { data, isLoading, isError, error } = useDriftReports();

  if (isLoading) {
    return <div className="p-6 text-muted-foreground">Loading drift timeline...</div>;
  }

  if (isError) {
    return (
      <div className="p-6 text-red-600">
        Failed to load reports: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return <div className="p-6 text-muted-foreground">No drift reports available yet.</div>;
  }

  const chartData = data
    .slice()
    .reverse()
    .map((report) => ({
      id: report.id,
      time: new Date(report.created_at).toLocaleTimeString(),
      severityScore: severityToScore(report.overall_severity),
      severity: report.overall_severity,
    }));

  return (
    <div className="p-6 space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Drift Timeline</h1>
        <p className="text-sm text-muted-foreground">
          Overall drift severity over recent evaluation windows.
        </p>
      </div>

      <div className="h-[420px] rounded-xl border p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis
              domain={[1, 3]}
              ticks={[1, 2, 3]}
              tickFormatter={(value) =>
                value === 1 ? "Green" : value === 2 ? "Yellow" : "Red"
              }
            />
            <Tooltip
              formatter={(_value, _name, props) => [
                props.payload.severity,
                "Severity",
              ]}
              labelFormatter={(label) => `Time: ${label}`}
            />
            <Line
              type="monotone"
              dataKey="severityScore"
              strokeWidth={2}
              dot
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}