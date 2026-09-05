import { useEffect, useRef, useState } from "react";
import "./index.css";

const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

const MODELS_URL = `${API_BASE_URL}/models`;
const PREDICT_URL = `${API_BASE_URL}/predict`;

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] =
    useState("anime2_v2");

  const [loadingModels, setLoadingModels] = useState(true);
  const [loading, setLoading] = useState(false);

  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);
  useEffect(() => {
    const loadModels = async () => {
      try {
        setLoadingModels(true);
        setError(null);

        const response = await fetch(MODELS_URL);

        if (!response.ok) {
          throw new Error(
            `Could not load models (${response.status})`
          );
        }

        const data = await response.json();

        const availableModels = data.models || [];

        setModels(availableModels);

        // Prefer Anime2 V2 if available.
        const anime2V2 = availableModels.find(
          (model) => model.id === "anime2_v2"
        );

        if (anime2V2) {
          setSelectedModel("anime2_v2");
        } else if (availableModels.length > 0) {
          setSelectedModel(
            availableModels[0].id
          );
        }
      } catch (err) {
        console.error(
          "Failed to load models:",
          err
        );

        setError(
          "Could not connect to the Face-Anime API. " +
            "Make sure the backend is running."
        );
      } finally {
        setLoadingModels(false);
      }
    };

    loadModels();
  }, []);
  const handleFileChange = (e) => {
    const file = e.target.files[0];

    if (!file) {
      return;
    }

    setSelectedFile(file);

    const newPreviewUrl =
      URL.createObjectURL(file);

    setPreviewUrl(newPreviewUrl);

    setResultUrl(null);
    setError(null);
  };
  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);

    // Clear previous output because it was generated
    // using another model.
    setResultUrl(null);
    setError(null);
  };
  const handleGenerate = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    if (!selectedModel) {
      setError("Please select a model.");
      return;
    }

    setLoading(true);
    setError(null);
    setResultUrl(null);

    try {
      const formData = new FormData();

      formData.append(
        "file",
        selectedFile
      );

      formData.append(
        "model",
        selectedModel
      );

      const response = await fetch(
        PREDICT_URL,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const detail =
          await response.text();

        throw new Error(
          `Server error (${response.status}): ${detail}`
        );
      }

      const blob =
        await response.blob();

      const generatedUrl =
        URL.createObjectURL(blob);

      setResultUrl(generatedUrl);

    } catch (err) {
      console.error(
        "Transformation failed:",
        err
      );

      setError(
        err.message ||
          "Transformation failed. " +
            "Check that the server is running."
      );

    } finally {
      setLoading(false);
    }
  };

  // ==========================================================
  // Current model information
  // ==========================================================

  const currentModel =
    models.find(
      (model) =>
        model.id === selectedModel
    );

  // ==========================================================
  // UI
  // ==========================================================

  return (
    <div className="page">

      <header className="masthead">
        <div className="masthead-mark">
          FACE
          <span className="arrow">
            →
          </span>
          ANIME
        </div>

        <div className="masthead-tag">
          image-to-image · cyclegan
        </div>
      </header>


      <main className="spread">

        {/* ==================================================
            PANEL 01 — INPUT
        ================================================== */}

        <section className="panel panel--input">

          <div className="panel-label">
            <span className="panel-number">
              01
            </span>

            <span className="panel-name">
              original
            </span>
          </div>


          <label
            className={`dropzone ${
              previewUrl
                ? "dropzone--filled"
                : ""
            }`}
            onClick={() =>
              !previewUrl &&
              fileInputRef.current?.click()
            }
          >

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={
                handleFileChange
              }
              hidden
            />


            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Selected face"
                className="panel-image"
              />
            ) : (
              <div className="dropzone-empty">

                <div className="dropzone-icon">
                  ＋
                </div>

                <p>
                  Choose a face photo
                </p>

                <span className="dropzone-hint">
                  JPG, PNG, or WEBP
                </span>

              </div>
            )}

          </label>


          {previewUrl && (
            <button
              className="ghost-button"
              onClick={() =>
                fileInputRef.current?.click()
              }
            >
              swap image
            </button>
          )}

        </section>


        {/* ==================================================
            CENTER CONTROL
        ================================================== */}

        <div className="gutter">

          <div className="model-selector">

            <label htmlFor="model-select">
              model
            </label>

            <select
              id="model-select"
              value={selectedModel}
              onChange={
                handleModelChange
              }
              disabled={
                loadingModels || loading
              }
            >

              {loadingModels ? (
                <option>
                  loading models...
                </option>
              ) : (
                models.map((model) => (
                  <option
                    key={model.id}
                    value={model.id}
                  >
                    {model.name}
                  </option>
                ))
              )}

            </select>


            {currentModel && (
              <span className="model-description">
                {currentModel.description}
              </span>
            )}

          </div>


          <button
            className={`burst-button ${
              loading
                ? "burst-button--loading"
                : ""
            }`}
            onClick={
              handleGenerate
            }
            disabled={
              !selectedFile ||
              !selectedModel ||
              loading ||
              loadingModels
            }
          >

            <span
              className="burst-shape"
              aria-hidden="true"
            />

            <span className="burst-label">
              {loading
                ? "working"
                : "transform"}
            </span>

          </button>

        </div>


        {/* ==================================================
            PANEL 02 — OUTPUT
        ================================================== */}

        <section className="panel panel--output">

          <div className="panel-label">

            <span className="panel-number">
              02
            </span>

            <span className="panel-name">
              transformed
            </span>

          </div>


          <div
            className={`resultzone ${
              loading
                ? "resultzone--loading"
                : ""
            }`}
          >

            {loading && (
              <div
                className="speed-lines"
                aria-hidden="true"
              >
                <span />
                <span />
                <span />
                <span />
                <span />
              </div>
            )}


            {!loading &&
              resultUrl && (
                <img
                  src={resultUrl}
                  alt="Generated anime"
                  className="panel-image"
                />
              )}


            {!loading &&
              !resultUrl && (
                <div className="resultzone-empty">

                  <div className="dropzone-icon">
                    〜
                  </div>

                  <p>
                    {selectedFile
                      ? "Ready — hit transform"
                      : "Your anime version appears here"}
                  </p>

                </div>
              )}

          </div>


          {resultUrl && (
            <a
              href={resultUrl}
              download={`${selectedModel}_result.jpg`}
              className="ghost-button"
            >
              download result
            </a>
          )}

        </section>

      </main>


      {/* ====================================================
          ERROR
      ==================================================== */}

      {error && (
        <div
          className="error-strip"
          role="alert"
        >

          <span className="error-mark">
            !
          </span>

          {error}

        </div>
      )}


      {/* ====================================================
          FOOTER
      ==================================================== */}

      <footer className="footer">

        Face → Anime AI ·{" "}

        <span className="footer-mono">
          {currentModel
            ? currentModel.name
            : "CycleGAN"}
        </span>

      </footer>

    </div>
  );
}

export default App;