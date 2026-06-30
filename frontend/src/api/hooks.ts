import { useQuery } from "@tanstack/react-query";
import apiClient from "./client";
import type {
  DriftSummary,
  PaginatedAlerts,
  DriftReport,
  ModelVersion,
} from "@/types/api";

const POLL_INTERVAL_MS = 30_000;

export function useDriftSummary() {
  return useQuery({
    queryKey: ["drift-summary"],
    queryFn: async (): Promise<DriftSummary> => {
      const { data } = await apiClient.get("/drift/summary");
      return data;
    },
    refetchInterval: POLL_INTERVAL_MS,
  });
}

export function useAlerts(page = 1, pageSize = 20) {
  return useQuery({
    queryKey: ["alerts", page, pageSize],
    queryFn: async (): Promise<PaginatedAlerts> => {
      const { data } = await apiClient.get("/alerts", {
        params: { page, page_size: pageSize },
      });
      return data;
    },
    refetchInterval: POLL_INTERVAL_MS,
  });
}

export function useDriftReports() {
  return useQuery({
    queryKey: ["drift-reports"],
    queryFn: async (): Promise<DriftReport[]> => {
      const { data } = await apiClient.get("/drift/reports");

      if (Array.isArray(data)) {
        return data;
      }

      if (Array.isArray(data.items)) {
        return data.items;
      }

      if (Array.isArray(data.reports)) {
        return data.reports;
      }

      return [];
    },
    refetchInterval: POLL_INTERVAL_MS,
  });
}

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: async (): Promise<ModelVersion[]> => {
      const { data } = await apiClient.get("/models");

      if (Array.isArray(data)) {
        return data;
      }

      if (Array.isArray(data.items)) {
        return data.items;
      }

      if (Array.isArray(data.models)) {
        return data.models;
      }

      return [];
    },
    refetchInterval: POLL_INTERVAL_MS,
  });
}