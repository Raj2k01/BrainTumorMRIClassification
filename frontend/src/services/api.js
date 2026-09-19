const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

async function parseResponse(response) {
  let responseData;

  try {
    responseData = await response.json();
  } catch {
    responseData = null;
  }

  if (!response.ok) {
    const errorMessage =
      responseData?.detail ||
      responseData?.message ||
      `Request failed with status ${response.status}.`;

    throw new Error(errorMessage);
  }

  return responseData;
}

export async function checkHealth() {
  const response = await fetch(
    `${API_URL}/health`,
    {
      method: "GET",
      headers: {
        Accept: "application/json"
      }
    }
  );

  return parseResponse(response);
}

export async function getModelInfo() {
  const response = await fetch(
    `${API_URL}/model-info`,
    {
      method: "GET",
      headers: {
        Accept: "application/json"
      }
    }
  );

  return parseResponse(response);
}

export async function predictMRI(
  file,
  confidenceThreshold = 0.7
) {
  const formData = new FormData();

  formData.append("file", file);

  formData.append(
    "confidence_threshold",
    String(confidenceThreshold)
  );

  const response = await fetch(
    `${API_URL}/predict`,
    {
      method: "POST",
      body: formData
    }
  );

  return parseResponse(response);
}

export { API_URL };