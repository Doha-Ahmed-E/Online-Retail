// API Base URL
const API_BASE_URL = "http://127.0.0.1:3000/api";

// Tab Navigation
function showTab(tabName) {
  // Hide all tabs
  const tabs = document.querySelectorAll(".tab-content");
  tabs.forEach((tab) => tab.classList.remove("active"));

  // Remove active class from all buttons
  const buttons = document.querySelectorAll(".tab-button");
  buttons.forEach((btn) => btn.classList.remove("active"));

  // Show selected tab
  document.getElementById(`${tabName}-tab`).classList.add("active");

  // Add active class to clicked button
  event.target.classList.add("active");

  // Load segments if segments tab is clicked
  if (tabName === "segments") {
    const segmentsList = document.getElementById("segments-list");
    if (segmentsList.innerHTML === "") {
      loadSegments();
    }
  }
}

// Single Prediction Form Handler
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("predict-form");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    // Get form values
    const recency = parseFloat(document.getElementById("recency").value);
    const frequency = parseFloat(document.getElementById("frequency").value);
    const monetary = parseFloat(document.getElementById("monetary").value);

    // Hide previous results and errors
    document.getElementById("result").classList.add("hidden");
    document.getElementById("error").classList.add("hidden");

    // Disable submit button
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = "Predicting...";
    submitBtn.disabled = true;

    try {
      // Make API call
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          recency: recency,
          frequency: frequency,
          monetary: monetary,
        }),
      });

      const data = await response.json();

      if (data.success) {
        // Display result
        displayPrediction(data.prediction);
      } else {
        // Display error
        showError(data.error || "Prediction failed");
      }
    } catch (error) {
      showError("Network error: " + error.message);
    } finally {
      // Re-enable submit button
      submitBtn.textContent = originalText;
      submitBtn.disabled = false;
    }
  });
});

// Display Prediction Result
function displayPrediction(prediction) {
  // Show result section
  const resultDiv = document.getElementById("result");
  resultDiv.classList.remove("hidden");

  // Set segment name
  document.getElementById("segment-name").textContent = prediction.segment_name;

  // Set confidence
  document.getElementById("confidence").textContent =
    prediction.confidence.toFixed(2);

  // Set cluster characteristics
  document.getElementById("avg-recency").textContent =
    prediction.cluster_characteristics.avg_recency;
  document.getElementById("avg-frequency").textContent =
    prediction.cluster_characteristics.avg_frequency;
  document.getElementById("avg-monetary").textContent =
    prediction.cluster_characteristics.avg_monetary.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });

  // Set recommendations
  const recommendationsList = document.getElementById("recommendations-list");
  recommendationsList.innerHTML = "";
  prediction.recommendations.forEach((rec) => {
    const li = document.createElement("li");
    li.textContent = rec;
    recommendationsList.appendChild(li);
  });

  // Scroll to result
  resultDiv.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Show Error Message
function showError(message) {
  const errorDiv = document.getElementById("error");
  errorDiv.textContent = message;
  errorDiv.classList.remove("hidden");
  errorDiv.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Batch Prediction
async function predictBatch() {
  const batchInput = document.getElementById("batch-input").value;
  const batchResults = document.getElementById("batch-results");
  const batchError = document.getElementById("batch-error");

  // Hide previous results
  batchResults.classList.add("hidden");
  batchError.classList.add("hidden");

  try {
    // Parse JSON input
    const data = JSON.parse(batchInput);

    // Make API call
    const response = await fetch(`${API_BASE_URL}/batch-predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    const result = await response.json();

    if (result.success) {
      displayBatchResults(result.predictions);
    } else {
      showBatchError(result.error || "Batch prediction failed");
    }
  } catch (error) {
    if (error instanceof SyntaxError) {
      showBatchError("Invalid JSON format. Please check your input.");
    } else {
      showBatchError("Error: " + error.message);
    }
  }
}

// Display Batch Results
function displayBatchResults(predictions) {
  const resultsDiv = document.getElementById("batch-results");
  const resultsContent = document.getElementById("batch-results-content");

  resultsContent.innerHTML = "";

  predictions.forEach((pred, index) => {
    const card = document.createElement("div");
    card.className = "batch-result-item";

    if (pred.error) {
      card.innerHTML = `
                <h4>Customer ${pred.customer_index + 1} ❌</h4>
                <p style="color: var(--danger-color);">Error: ${pred.error}</p>
            `;
    } else {
      card.innerHTML = `
                <h4>Customer ${pred.customer_index + 1}</h4>
                <p><strong>Segment:</strong> ${pred.segment_name}</p>
                <p><strong>Cluster ID:</strong> ${pred.cluster_id}</p>
                <p><strong>Recency:</strong> ${
                  pred.input_values.recency
                } days</p>
                <p><strong>Frequency:</strong> ${
                  pred.input_values.frequency
                } purchases</p>
                <p><strong>Monetary:</strong> $${pred.input_values.monetary.toLocaleString(
                  "en-US",
                  { minimumFractionDigits: 2 }
                )}</p>
            `;
    }

    resultsContent.appendChild(card);
  });

  resultsDiv.classList.remove("hidden");
  resultsDiv.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Show Batch Error
function showBatchError(message) {
  const errorDiv = document.getElementById("batch-error");
  errorDiv.textContent = message;
  errorDiv.classList.remove("hidden");
  errorDiv.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Load Segments Information
async function loadSegments() {
  const segmentsList = document.getElementById("segments-list");
  segmentsList.innerHTML =
    '<p style="text-align: center; padding: 20px;">Loading segments...</p>';

  try {
    const response = await fetch(`${API_BASE_URL}/segments`);
    const data = await response.json();

    if (data.success) {
      displaySegments(data.segments);
    } else {
      segmentsList.innerHTML = `<p style="color: var(--danger-color); text-align: center;">Error loading segments: ${data.error}</p>`;
    }
  } catch (error) {
    segmentsList.innerHTML = `<p style="color: var(--danger-color); text-align: center;">Network error: ${error.message}</p>`;
  }
}

// Display Segments
function displaySegments(segments) {
  const segmentsList = document.getElementById("segments-list");
  segmentsList.innerHTML = "";

  segments.forEach((segment) => {
    const card = document.createElement("div");
    card.className = "segment-card";

    card.innerHTML = `
            <h3>${getSegmentEmoji(segment.segment_name)} ${
      segment.segment_name
    }</h3>
            <p><strong>Cluster ID:</strong> ${segment.cluster_id}</p>
            <div class="segment-stats">
                <p><strong>Average Recency:</strong> ${
                  segment.avg_recency
                } days</p>
                <p><strong>Average Frequency:</strong> ${
                  segment.avg_frequency
                } purchases</p>
                <p><strong>Average Monetary:</strong> $${segment.avg_monetary.toLocaleString(
                  "en-US",
                  { minimumFractionDigits: 2 }
                )}</p>
            </div>
        `;

    segmentsList.appendChild(card);
  });
}

// Get Emoji for Segment
function getSegmentEmoji(segmentName) {
  const emojiMap = {
    Champions: "🏆",
    "Loyal Customers": "💎",
    "Big Spenders": "💰",
    "At Risk": "⚠️",
    "New Customers": "🌱",
    "Regular Customers": "👤",
  };

  return emojiMap[segmentName] || "👥";
}

// Health Check on Load
async function checkAPIHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();

    if (data.status === "healthy") {
      console.log(" API is healthy");
      console.log(" Model loaded:", data.model_loaded);
      console.log(" Scaler loaded:", data.scaler_loaded);
    } else {
      console.warn(" API health check failed");
    }
  } catch (error) {
    console.error(" API is not accessible:", error.message);
  }
}

// Run health check on page load
document.addEventListener("DOMContentLoaded", function () {
  checkAPIHealth();
});
