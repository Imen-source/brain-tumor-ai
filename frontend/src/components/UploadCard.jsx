
export default function UploadCard({ file, setFile, onAnalyze }) {
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  return (
    <div className="card">
      <h3>Upload MRI Image</h3>

      <input type="file" accept="image/*" onChange={handleFileChange} />

      {file && (
        <img
          src={URL.createObjectURL(file)}
          alt="preview"
          className="preview"
        />
      )}

      <button disabled={!file} onClick={onAnalyze}>
        Analyze
      </button>
    </div>
  );
}
