export default function HistoryPage({ history }) {
  return (
    <div className="card">
      <h3>Prediction History</h3>

      {history.length === 0 && <p>No predictions yet.</p>}

      {history.map((item, idx) => (
        <div key={idx} className="history-item">
          <strong>{item.prediction}</strong> – {(item.confidence * 100).toFixed(1)}%
          <br />
          <small>{item.date}</small>
        </div>
      ))}
    </div>
  );
}