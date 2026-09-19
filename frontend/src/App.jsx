import {
  useEffect,
  useRef,
  useState
} from "react";

import {
  Brain,
  CircleAlert,
  CircleCheck,
  LoaderCircle,
  RefreshCw,
  ShieldAlert
} from "lucide-react";

import ImageUpload from "./components/ImageUpload";
import PredictionResult from "./components/PredictionResult";

import {
  checkHealth,
  predictMRI
} from "./services/api";

import "./App.css";


const ALLOWED_IMAGE_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp"
];

const MAX_FILE_SIZE =
  10 * 1024 * 1024;

const CONFIDENCE_THRESHOLD = 0.70;


function App() {
  const [
    selectedFile,
    setSelectedFile
  ] = useState(null);

  const [
    previewUrl,
    setPreviewUrl
  ] = useState("");

  const [
    prediction,
    setPrediction
  ] = useState(null);

  const [
    loading,
    setLoading
  ] = useState(false);

  const [
    error,
    setError
  ] = useState("");

  const [
    dragActive,
    setDragActive
  ] = useState(false);

  const [
    apiStatus,
    setApiStatus
  ] = useState("checking");

  const [
    healthInformation,
    setHealthInformation
  ] = useState(null);

  const previewUrlRef = useRef("");


  useEffect(() => {
    handleHealthCheck();
  }, []);


  useEffect(() => {
    previewUrlRef.current =
      previewUrl;
  }, [previewUrl]);


  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(
          previewUrlRef.current
        );
      }
    };
  }, []);


  async function handleHealthCheck() {
    setApiStatus("checking");

    try {
      const health =
        await checkHealth();

      setHealthInformation(
        health
      );

      if (health.model_loaded) {
        setApiStatus("connected");
      } else {
        setApiStatus("unavailable");
      }
    } catch {
      setHealthInformation(null);
      setApiStatus("disconnected");
    }
  }


  function clearCurrentPreview() {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(
        previewUrlRef.current
      );

      previewUrlRef.current = "";
    }
  }


  function clearFileInput() {
    const fileInput =
      document.getElementById(
        "mri-file"
      );

    if (fileInput) {
      fileInput.value = "";
    }
  }


  function validateAndSelectFile(
    file
  ) {
    setError("");
    setPrediction(null);

    if (!file) {
      return;
    }

    if (
      !ALLOWED_IMAGE_TYPES.includes(
        file.type
      )
    ) {
      clearCurrentPreview();
      clearFileInput();

      setSelectedFile(null);
      setPreviewUrl("");

      setError(
        "Upload a valid JPG, PNG, or WEBP image."
      );

      return;
    }

    if (
      file.size > MAX_FILE_SIZE
    ) {
      clearCurrentPreview();
      clearFileInput();

      setSelectedFile(null);
      setPreviewUrl("");

      setError(
        "The image must be 10 MB or smaller."
      );

      return;
    }

    clearCurrentPreview();

    const newPreviewUrl =
      URL.createObjectURL(file);

    previewUrlRef.current =
      newPreviewUrl;

    setSelectedFile(file);

    setPreviewUrl(
      newPreviewUrl
    );
  }


  function handleFileChange(
    event
  ) {
    const file =
      event.target.files?.[0];

    validateAndSelectFile(file);
  }


  function handleDragEnter(
    event
  ) {
    event.preventDefault();
    event.stopPropagation();

    setDragActive(true);
  }


  function handleDragOver(
    event
  ) {
    event.preventDefault();
    event.stopPropagation();
  }


  function handleDragLeave(
    event
  ) {
    event.preventDefault();
    event.stopPropagation();

    setDragActive(false);
  }


  function handleDrop(
    event
  ) {
    event.preventDefault();
    event.stopPropagation();

    setDragActive(false);

    const file =
      event.dataTransfer
        .files?.[0];

    validateAndSelectFile(file);
  }


  async function handlePrediction() {
    if (!selectedFile) {
      setError(
        "Select an MRI image first."
      );

      return;
    }

    if (
      apiStatus !== "connected"
    ) {
      setError(
        "The prediction API is not connected."
      );

      return;
    }

    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const result =
        await predictMRI(
          selectedFile,
          CONFIDENCE_THRESHOLD
        );

      setPrediction(result);
    } catch (requestError) {
      setError(
        requestError.message ||
        "The prediction request failed."
      );
    } finally {
      setLoading(false);
    }
  }


  function handleReset() {
    clearCurrentPreview();
    clearFileInput();

    setSelectedFile(null);
    setPreviewUrl("");
    setPrediction(null);
    setError("");
    setDragActive(false);
  }


  function renderApiStatusIcon() {
    if (apiStatus === "checking") {
      return (
        <LoaderCircle
          size={18}
          className="spinner"
        />
      );
    }

    if (apiStatus === "connected") {
      return (
        <CircleCheck size={18} />
      );
    }

    return (
      <CircleAlert size={18} />
    );
  }


  function getApiStatusText() {
    if (apiStatus === "checking") {
      return "Checking API";
    }

    if (apiStatus === "connected") {
      return "API connected";
    }

    if (apiStatus === "unavailable") {
      return "Model unavailable";
    }

    return "API disconnected";
  }


  return (
    <main className="application">

      
      <div className="page-container">
        <header className="page-header">
          <div className="brand-row">
            <div>
              <h1>
                Brain Tumor MRI Classification
              </h1>
            </div>
          </div>

          

          <p className="page-description">
            Upload a brain MRI image to classify
            it as glioma, meningioma, no tumor,
            or pituitary tumor using a custom
            PyTorch convolutional neural network.
          </p>

          <div
            className={
              `api-panel ${apiStatus}`
            }
          >
            

            {apiStatus ===
              "disconnected" && (
              <button
                type="button"
                className="retry-button"
                onClick={
                  handleHealthCheck
                }
              >
                <RefreshCw size={16} />
                Retry
              </button>
            )}
          </div>
        </header>

        <div className="main-grid">
          <ImageUpload
            file={selectedFile}
            previewUrl={previewUrl}
            error={error}
            loading={loading}
            dragActive={dragActive}
            onFileChange={
              handleFileChange
            }
            onDrop={handleDrop}
            onDragEnter={
              handleDragEnter
            }
            onDragLeave={
              handleDragLeave
            }
            onDragOver={
              handleDragOver
            }
            onPredict={
              handlePrediction
            }
            onReset={handleReset}
          />

          <PredictionResult
            prediction={prediction}
          />
        </div>

            <footer className="page-footer">
          

          <p>
            <ShieldAlert size={12} />
              Educational demonstration only.
            This model is not clinically
            validated and must not be used for
            diagnosis, treatment, or medical
            decision-making.
          </p>
          </footer>
        
      </div>
    </main>
  );
}


export default App;