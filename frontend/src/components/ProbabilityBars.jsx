export default function ProbabilityBars({ probabilities }) {
  return (
    <div className="prob-bars">
      {Object.entries(probabilities).map(([label, value]) => (
        <div key={label} className="bar-row">
          <span>{label}</span>
          <div className="bar-bg">
            <div
              className="bar-fill"
              style={{ width: `${value * 100}%` }}
            ></div>
          </div>
          <span>{(value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}