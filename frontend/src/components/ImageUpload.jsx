import React from "react";

import {
  AlertTriangle,
  FileImage,
  ImageUp,
  LoaderCircle,
  RefreshCw
} from "lucide-react";


function ImageUpload({
  file,
  previewUrl,
  error,
  loading,
  dragActive,
  onFileChange,
  onDrop,
  onDragEnter,
  onDragLeave,
  onDragOver,
  onPredict,
  onReset
}) {
  const imagePreview = previewUrl
    ? React.createElement(
        "img",
        {
          src: previewUrl,
          alt: "Selected brain MRI preview",
          className: "preview-image"
        }
      )
    : null;


  return (
    <section className="card">
      <div className="card-heading">
        <div>
          <h2>Upload MRI Image</h2>
        </div>
      </div>

      <label
        className={
          dragActive
            ? "upload-area drag-active"
            : "upload-area"
        }
        htmlFor="mri-file"
        onDragEnter={onDragEnter}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        {previewUrl ? (
          imagePreview
        ) : (
          <div className="upload-placeholder">
            <div className="upload-icon">
              <ImageUp size={42} />
            </div>

            <p>Choose an MRI image</p>

            <span>
              Click or drag and drop
            </span>

            <small>
              JPG, PNG or WEBP, maximum 10 MB
            </small>
          </div>
        )}
      </label>

      <input
        id="mri-file"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={onFileChange}
        hidden
      />

      {file && (
        <div className="file-information">
          <div>
            <FileImage size={18} />

            <span className="file-name">
              {file.name}
            </span>
          </div>

          <span className="file-size">
            {(file.size / 1024).toFixed(1)} KB
          </span>
        </div>
      )}

      {error && (
        <div className="error-message">
          <AlertTriangle size={18} />

          <span>{error}</span>
        </div>
      )}

      <div className="action-row">
        <button
          type="button"
          className="predict-button"
          onClick={onPredict}
          disabled={!file || loading}
        >
          {loading ? (
            <>
              <LoaderCircle
                size={20}
                className="spinner"
              />

              Processing MRI
            </>
          ) : (
            <>
              <ImageUp size={20} />

              Run Prediction
            </>
          )}
        </button>

        <button
          type="button"
          className="reset-button"
          onClick={onReset}
          disabled={loading}
        >
          <RefreshCw size={19} />

          Reset
        </button>
      </div>
    </section>
  );
}


export default ImageUpload;