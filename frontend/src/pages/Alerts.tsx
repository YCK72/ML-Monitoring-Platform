import { useState } from "react";
import { useAlerts } from "@/api/hooks";
import { Badge } from "@/components/ui/badge";
import type { Severity } from "@/types/api";

function severityBadgeVariant(severity: Severity): "default" | "secondary" | "destructive" {
  if (severity === "Red") return "destructive";
  if (severity === "Yellow") return "secondary";
  return "default";
}

export default function Alerts() {
  const [page, setPage] = useState(1);
  const pageSize = 20;

  const { data, isLoading, isError, error } = useAlerts(page, pageSize);

  if (isLoading) {
    return <div className="p-6 text-muted-foreground">Loading alerts...</div>;
  }

  if (isError) {
    return (
      <div className="p-6 text-red-600">
        Failed to load alerts: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.items.length === 0) {
    return <div className="p-6 text-muted-foreground">No alerts fired yet.</div>;
  }

  const totalPages = Math.ceil(data.total / data.page_size);

  return (
    <div className="p-6 space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Alert History</h1>
        <p className="text-sm text-muted-foreground">
          Showing {data.items.length} of {data.total} alerts
        </p>
      </div>

      <div className="overflow-x-auto rounded-md border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50 text-left">
            <tr>
              <th className="p-3 font-medium">Feature</th>
              <th className="p-3 font-medium">Severity</th>
              <th className="p-3 font-medium">Channel</th>
              <th className="p-3 font-medium">Drift Report</th>
              <th className="p-3 font-medium">Notified At</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((alert) => (
              <tr key={alert.id} className="border-t">
                <td className="p-3">{alert.feature_name}</td>
                <td className="p-3">
                  <Badge variant={severityBadgeVariant(alert.severity)}>
                    {alert.severity}
                  </Badge>
                </td>
                <td className="p-3 capitalize">{alert.channel}</td>
                <td className="p-3 text-muted-foreground">
                  {alert.drift_report_id ?? "—"}
                </td>
                <td className="p-3 text-muted-foreground">
                  {new Date(alert.notified_at).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between">
        <button
          className="rounded-md border px-3 py-2 text-sm disabled:opacity-50"
          disabled={page <= 1}
          onClick={() => setPage((current) => current - 1)}
        >
          Previous
        </button>

        <span className="text-sm text-muted-foreground">
          Page {page} of {totalPages}
        </span>

        <button
          className="rounded-md border px-3 py-2 text-sm disabled:opacity-50"
          disabled={page >= totalPages}
          onClick={() => setPage((current) => current + 1)}
        >
          Next
        </button>
      </div>
    </div>
  );
}