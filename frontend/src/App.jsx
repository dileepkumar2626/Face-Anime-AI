import { useState } from "react";
import "./index.css";

const API_URL = "http://127.0.0.1:8000/predict";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResultUrl(null);
    setError(null);
  };

  const handleGenerate = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResultUrl(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Server error (${response.status}): ${detail}`);
      }

      const blob = await response.blob();
      setResultUrl(URL.createObjectURL(blob));
    } catch (err) {
      setError(err.message || "Something went wrong while generating the image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>Face → Anime AI</h1>

      <div className="upload-section">
        <label className="file-button">
          Choose Image
          <input type="file" accept="image/*" onChange={handleFileChange} hidden />
        </label>

        {previewUrl && (
          <div className="image-block">
            <h3>Your selected image</h3>
            <img src={previewUrl} alt="Selected face" />
          </div>
        )}

        <button
          className="generate-button"
          onClick={handleGenerate}
          disabled={!selectedFile || loading}
        >
          {loading ? "Generating..." : "Generate Anime"}
        </button>

        {error && <p className="error">{error}</p>}

        {resultUrl && (
          <div className="image-block">
            <h3>Generated result</h3>
            <img src={resultUrl} alt="Generated anime" />
            <a href={resultUrl} download="anime_result.jpg" className="download-button">
              Download
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;