import { useState, useRef } from "react";
import "./index.css";

const API_URL = "http://127.0.0.1:8000/predict";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

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
      setError(
        err.message || "Transformation failed. Check that the server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="masthead">
        <div className="masthead-mark">
          FACE<span className="arrow">→</span>ANIME
        </div>
        <div className="masthead-tag">image-to-image · cyclegan</div>
      </header>

      <main className="spread">
        {/* Panel 01 — input */}
        <section className="panel panel--input">
          <div className="panel-label">
            <span className="panel-number">01</span>
            <span className="panel-name">original</span>
          </div>

          <label
            className={`dropzone ${previewUrl ? "dropzone--filled" : ""}`}
            onClick={() => !previewUrl && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              hidden
            />
            {previewUrl ? (
              <img src={previewUrl} alt="Selected face" className="panel-image" />
            ) : (
              <div className="dropzone-empty">
                <div className="dropzone-icon">＋</div>
                <p>Choose a face photo</p>
                <span className="dropzone-hint">JPG, PNG, or WEBP</span>
              </div>
            )}
          </label>

          {previewUrl && (
            <button
              className="ghost-button"
              onClick={() => fileInputRef.current?.click()}
            >
              swap image
            </button>
          )}
        </section>

        {/* Center action */}
        <div className="gutter">
          <button
            className={`burst-button ${loading ? "burst-button--loading" : ""}`}
            onClick={handleGenerate}
            disabled={!selectedFile || loading}
          >
            <span className="burst-shape" aria-hidden="true" />
            <span className="burst-label">
              {loading ? "working" : "transform"}
            </span>
          </button>
        </div>

        {/* Panel 02 — output */}
        <section className="panel panel--output">
          <div className="panel-label">
            <span className="panel-number">02</span>
            <span className="panel-name">transformed</span>
          </div>

          <div className={`resultzone ${loading ? "resultzone--loading" : ""}`}>
            {loading && (
              <div className="speed-lines" aria-hidden="true">
                <span />
                <span />
                <span />
                <span />
                <span />
              </div>
            )}

            {!loading && resultUrl && (
              <img src={resultUrl} alt="Generated anime" className="panel-image" />
            )}

            {!loading && !resultUrl && (
              <div className="resultzone-empty">
                <div className="dropzone-icon">〜</div>
                <p>
                  {selectedFile
                    ? "Ready — hit transform"
                    : "Your anime version appears here"}
                </p>
              </div>
            )}
          </div>

          {resultUrl && (
            <a href={resultUrl} download="anime_result.jpg" className="ghost-button">
              download result
            </a>
          )}
        </section>
      </main>

      {error && (
        <div className="error-strip" role="alert">
          <span className="error-mark">!</span>
          {error}
        </div>
      )}

      <footer className="footer">
        running locally · <span className="footer-mono">127.0.0.1:8000</span>
      </footer>
    </div>
  );
}

export default App;