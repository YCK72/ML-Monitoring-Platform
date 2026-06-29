import { useQuery } from "@tanstack/react-query";
import apiClient from "./client";
import type { DriftSummary, AlertEvent, DriftReport, ModelVersion } from "@/types/api";

export function useDriftSummary() {
  return useQuery({
    queryKey: ["drift-summary"],
    queryFn: async (): Promise<DriftSummary> => {
      const { data } = await apiClient.get("/drift/summary");
      return data;
    },
  });
}

export function useAlerts() {
  return useQuery({
    queryKey: ["alerts"],
    queryFn: async (): Promise<AlertEvent[]> => {
      const { data } = await apiClient.get("/alerts");
      return data;
    },
  });
}

export function useDriftReports() {
  return useQuery({
    queryKey: ["drift-reports"],
    queryFn: async (): Promise<DriftReport[]> => {
      const { data } = await apiClient.get("/drift/reports");
      return data;
    },
  });
}

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: async (): Promise<ModelVersion[]> => {
      const { data } = await apiClient.get("/models");
      return data;
    },
  });
}
