import { useState } from "react";
import { Routes, Route, Link } from "react-router-dom";
import { predictImage } from "./api";
import UploadCard from "./components/UploadCard";
import ResultCard from "./components/ResultCard";
import HistoryPage from "./components/HistoryPage";
import Loader from "./components/Loader";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const handleAnalyze = async () => {
    setLoading(true);
    const data = await predictImage(file);
    setResult(data);
    setHistory(prev => [
      { ...data, date: new Date().toLocaleString() },
      ...prev
    ]);
    setLoading(false);
  };

  return (
    <div className="app-container">
      <nav>
        <Link to="/">Home</Link>
        <Link to="/history">History</Link>
      </nav>

      <Routes>
        <Route path="/" element={
          <>
            <UploadCard file={file} setFile={setFile} onAnalyze={handleAnalyze} />
            {loading && <Loader />}
            {result && <ResultCard result={result} file={file} />}
          </>
        } />

        <Route path="/history" element={<HistoryPage history={history} />} />
      </Routes>
    </div>
  );
}