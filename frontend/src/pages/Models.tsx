import { useModels } from "@/api/hooks";
import { Badge } from "@/components/ui/badge";

export default function Models() {
  const { data, isLoading, isError, error } = useModels();

  if (isLoading) return <div className="p-6 text-muted-foreground">Loading models...</div>;

  if (isError) {
    return <div className="p-6 text-red-600">Failed to load models: {(error as Error).message}</div>;
  }

  if (!data || data.length === 0) {
    return <div className="p-6 text-muted-foreground">No model versions available yet.</div>;
  }

  return (
    <div className="p-6 space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Model Performance</h1>
        <p className="text-sm text-muted-foreground">
          Registered model versions and training metrics from MLflow.
        </p>
      </div>

      <div className="overflow-x-auto rounded-md border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50 text-left">
            <tr>
              <th className="p-3 font-medium">Name</th>
              <th className="p-3 font-medium">Version</th>
              <th className="p-3 font-medium">Stage</th>
              <th className="p-3 font-medium">Metrics</th>
              <th className="p-3 font-medium">Created At</th>
            </tr>
          </thead>
          <tbody>
            {data.map((model) => (
              <tr key={model.id} className="border-t">
                <td className="p-3 font-medium">{model.name}</td>
                <td className="p-3">{model.version}</td>
                <td className="p-3">
                  <Badge variant="secondary">{model.stage}</Badge>
                </td>
                <td className="p-3 text-muted-foreground">
                  {Object.entries(model.training_metrics ?? {}).map(([key, value]) => (
                    <div key={key}>
                      {key}: {Number(value).toFixed(4)}
                    </div>
                  ))}
                </td>
                <td className="p-3 text-muted-foreground">
                  {new Date(model.created_at).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}