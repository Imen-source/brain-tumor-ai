import ProbabilityBars from "./ProbabilityBars";

export default function ResultCard({ result, file }) {
  return (
    <div className="card">
      <h3>Prediction Result</h3>

      <div className="image-row">
        <div>
          <p>Original</p>
          <img src={URL.createObjectURL(file)} className="img-box" />
        </div>

        <div>
          <p>Grad‑CAM</p>
          <img
            src={result.gradcam_image_base64}
            className="img-box"
          />
        </div>
      </div>

      <h4>{result.prediction}</h4>
      <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>

      <ProbabilityBars probabilities={result.probabilities} />
    </div>
  );
}