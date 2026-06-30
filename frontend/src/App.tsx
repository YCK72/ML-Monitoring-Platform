import { Routes, Route } from "react-router-dom";
import Layout from "@/components/layout/Layout";
import Overview from "@/pages/Overview";
import Alerts from "@/pages/Alerts";
import DriftTimeline from "@/pages/DriftTimeline";
import FeatureDetail from "@/pages/FeatureDetail";
import Models from "@/pages/Models";

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Overview />} />
        <Route path="/drift" element={<DriftTimeline />} />
        <Route path="/features" element={<FeatureDetail />} />
        <Route path="/models" element={<Models />} />
        <Route path="/alerts" element={<Alerts />} />
      </Route>
    </Routes>
  );
}

export default App;