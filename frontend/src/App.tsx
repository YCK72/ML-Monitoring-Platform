import { Routes, Route } from "react-router-dom";
import Overview from "@/pages/Overview";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Overview />} />
    </Routes>
  );
}

export default App;