import {
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2
} from "lucide-react";


function formatClassName(className) {
  if (!className) {
    return "Unknown";
  }

  if (className.toLowerCase() === "notumor") {
    return "No Tumor";
  }

  return className
    .replaceAll("_", " ")
    .replace(
      /\b\w/g,
      (character) =>
        character.toUpperCase()
    );
}


function PredictionResult({
  prediction
}) {
  if (!prediction) {
    return (
      <section className="card">
        <div className="card-heading">
          <div>
            <h2>Prediction Result</h2>
          </div>
        </div>

        <div className="empty-result">
          <div className="empty-result-icon">
          </div>

          <h3>No prediction yet</h3>

          <p>
            Select a brain MRI image and run the
            model to view the predicted class,
            confidence, and class probabilities.
          </p>
        </div>
      </section>
    );
  }


  const probabilityEntries =
    Object.entries(
      prediction.class_probabilities ||
      {}
    ).sort(
      (firstEntry, secondEntry) =>
        secondEntry[1] - firstEntry[1]
    );

  const confidencePercentage =
    Number(prediction.confidence) * 100;

  const thresholdPercentage =
    Number(
      prediction.confidence_threshold
    ) * 100;


  return (
    <section className="card">
      <div className="card-heading">
        <div>
          <p className="section-label">
            Output
          </p>

          <h2>Prediction Result</h2>
        </div>
      </div>

      <div className="primary-result">
        <div>
          <span>Predicted category</span>

          <h3>
            {formatClassName(
              prediction.predicted_class
            )}
          </h3>
        </div>

        <CheckCircle2 size={38} />
      </div>

      <div className="confidence-section">
        <div className="metric-row">
          <span>Model confidence</span>

          <strong>
            {confidencePercentage.toFixed(2)}%
          </strong>
        </div>

        <div className="confidence-track">
          <div
            className="confidence-value"
            style={{
              width: `${Math.min(
                confidencePercentage,
                100
              )}%`
            }}
          />
        </div>
      </div>

      <div className="probability-section">
        <h4>Class Probabilities</h4>

        {probabilityEntries.map(
          ([className, probability]) => {
            const percentage =
              Number(probability) * 100;

            return (
              <div
                className="probability-item"
                key={className}
              >
                <div className="metric-row">
                  <span>
                    {formatClassName(
                      className
                    )}
                  </span>

                  <span>
                    {percentage.toFixed(2)}%
                  </span>
                </div>

                <div className="probability-track">
                  <div
                    className="probability-value"
                    style={{
                      width: `${Math.min(
                        percentage,
                        100
                      )}%`
                    }}
                  />
                </div>
              </div>
            );
          }
        )}
      </div>

      <div
        className={
          prediction.requires_review
            ? (
                "review-message "
                + "review-warning"
              )
            : (
                "review-message "
                + "review-accepted"
              )
        }
      >
        {prediction.requires_review ? (
          <AlertTriangle size={19} />
        ) : (
          <CheckCircle2 size={19} />
        )}

        <div>
          <strong>
            {prediction.requires_review
              ? "Manual review recommended"
              : "Confidence threshold passed"}
          </strong>

          <p>
            {prediction.requires_review
              ? (
                  "The prediction confidence "
                  + "is below the configured "
                  + "review threshold."
                )
              : (
                  "The prediction confidence "
                  + "is above the configured "
                  + "review threshold."
                )}
          </p>
        </div>
      </div>

      <div className="prediction-metadata">
        <span>
          File:{" "}
          {prediction.filename ||
            "Uploaded image"}
        </span>

        <span>
          Threshold:{" "}
          {thresholdPercentage.toFixed(0)}%
        </span>
      </div>
    </section>
  );
}


export default PredictionResult;