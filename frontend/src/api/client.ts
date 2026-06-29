import axios from "axios";

const apiClient = axios.create({
  baseURL: "/api",
  headers: {
    "X-API-Key": import.meta.env.VITE_API_KEY,
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error("API request failed:", error?.response?.data ?? error.message);
    return Promise.reject(error);
  }
);

export default apiClient;