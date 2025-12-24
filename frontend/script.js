const API = "http://127.0.0.1:8000";

let csvData = [];
let currentRow = 0;

/* Load Models When Page Loads */
window.onload = async () => {
    checkHealth();

    const res = await fetch(`${API}/models`);
    const data = await res.json();

    let select = document.getElementById("model-select");
    data.available_models.forEach(model => {
        let option = document.createElement("option");
        option.value = model;
        option.innerText = model;
        select.appendChild(option);
    });
};

/* HEALTH CHECK */
async function checkHealth() {
    const box = document.getElementById("health-indicator");

    try {
        const res = await fetch(`${API}/health`);
        if (res.ok) {
            box.innerText = "✔ API Online";
            box.classList.add("health-online");
            box.classList.remove("health-offline");
        } else {
            box.innerText = "❌ API Offline";
            box.classList.add("health-offline");
            box.classList.remove("health-online");
        }
    } catch (error) {
        box.innerText = "❌ API Offline";
        box.classList.add("health-offline");
        box.classList.remove("health-online");
    }
}


/* LOAD CSV */
async function loadCSV() {
    const res = await fetch(`${API}/dataset/preview`);
    csvData = await res.json();
    currentRow = 0;
    displayRow();
}

function displayRow() {
    document.getElementById("csv-row").textContent =
        JSON.stringify(csvData[currentRow], null, 2);
}

function nextRow() {
    if (currentRow < csvData.length - 1) {
        currentRow++;
        displayRow();
    }
}

function prevRow() {
    if (currentRow > 0) {
        currentRow--;
        displayRow();
    }
}

/* Autofill Prediction Inputs */
function useRow() {
    const row = csvData[currentRow];
    document.getElementById("distance").value = row["Distance"];
    document.getElementById("pressure").value = row["Pressure"];
    document.getElementById("hrv").value = row["HRV"];
    document.getElementById("sugar").value = row["Sugar level"];
    document.getElementById("spo2").value = row["SpO2"];
    document.getElementById("accel").value = row["Accelerometer"];
}

/* PREDICT */
async function predict() {
    const payload = {
        model_name: document.getElementById("model-select").value,
        distance: +document.getElementById("distance").value,
        pressure: +document.getElementById("pressure").value,
        hrv: +document.getElementById("hrv").value,
        sugar_level: +document.getElementById("sugar").value,
        spo2: +document.getElementById("spo2").value,
        accelerometer: +document.getElementById("accel").value
    };

    const res = await fetch(`${API}/predict/model`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });

    const data = await res.json();
    const box = document.getElementById("result");

    const colors = {
        "No Fall": "#C9E7C3",
        "Fall Predicted": "#F4E3A0",
        "Fall Detected": "#E5A1A1"
    };

    box.style.background = colors[data.prediction_label];
    box.innerHTML = `
        <strong>Model:</strong> ${data.model_used}<br>
        <strong>Prediction:</strong> ${data.prediction_label}<br><br>
        No Fall: ${(data.probabilities["No Fall"]*100).toFixed(2)}%<br>
        Fall Predicted: ${(data.probabilities["Fall Predicted"]*100).toFixed(2)}%<br>
        Fall Detected: ${(data.probabilities["Fall Detected"]*100).toFixed(2)}%
    `;
    
    box.classList.remove("hidden");
}

/* BATCH PREDICTION */
async function predictBatch() {
    const txt = document.getElementById("batchInput").value;
    const res = await fetch(`${API}/predict/batch`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: txt
    });

    const data = await res.json();
    let box = document.getElementById("batch-result");
    box.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    box.classList.remove("hidden");
}

/* MODEL ACCURACY */
async function getAccuracy() {
    const res = await fetch(`${API}/model/accuracy`);
    const data = await res.json();
    let box = document.getElementById("metrics-result");
    box.innerHTML = `<strong>Accuracy:</strong> ${data.accuracy}`;
    box.classList.remove("hidden");
}

/* FULL METRICS */

function renderTable(data) {
    let html = "<table>";

    data.forEach((row, index) => {
        html += "<tr>";
        row.forEach(cell => {
            if (index === 0) {
                html += `<th>${cell}</th>`;
            } else {
                html += `<td>${cell}</td>`;
            }
        });
        html += "</tr>";
    });

    html += "</table>";
    return html;
}

function renderClassReport(report) {
    let html = `
    <table>
        <tr>
            <th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th>
        </tr>
    `;

    report.forEach(row => {
        html += `
        <tr>
            <td>${row.class}</td>
            <td>${row.precision}</td>
            <td>${row.recall}</td>
            <td>${row.f1_score}</td>
            <td>${row.support}</td>
        </tr>
        `;
    });

    html += "</table>";
    return html;
}


async function getMetrics() {
    const res = await fetch(`${API}/model/metrics`);
    const data = await res.json();

    let output = `
        <h3>Accuracy: ${(data.accuracy * 100).toFixed(2)}%</h3>
        <h3>Confusion Matrix</h3>
        ${renderTable(data.confusion_matrix_table)}

        <h3>Classification Report</h3>
        ${renderClassReport(data.classification_report_table)}
    `;

    document.getElementById("metrics-output").innerHTML = output;
}



/* RETRAIN MODEL */
async function retrainModel() {
    const res = await fetch(`${API}/model/retrain`, { method: "POST" });
    const data = await res.json();
    let box = document.getElementById("metrics-result");
    box.innerHTML = `<strong>Retrained!</strong><br>New Accuracy: ${data.new_test_accuracy}`;
    box.classList.remove("hidden");
}

/* Load a JSON template for batch prediction */
function loadBatchTemplate() {
    const template = {
        "inputs": [
            {
                "model_name": "logreg",
                "distance": 10.0,
                "pressure": 1,
                "hrv": 90.0,
                "sugar_level": 50.0,
                "spo2": 95.0,
                "accelerometer": 1
            },
            {
                "model_name": "svm",
                "distance": 25.0,
                "pressure": 2,
                "hrv": 110.0,
                "sugar_level": 20.0,
                "spo2": 67.0,
                "accelerometer": 1
            }
        ]
    };

    document.getElementById("batchInput").value =
        JSON.stringify(template, null, 4);
}

/* Load batch data from first 5 rows of CSV */
async function loadBatchFromCSV() {
    const res = await fetch(`${API}/dataset/preview`);
    const rows = await res.json();

    const batch = {
        "inputs": rows.map(r => ({
            model_name: document.getElementById("model-select").value,
            distance: r["Distance"],
            pressure: r["Pressure"],
            hrv: r["HRV"],
            sugar_level: r["Sugar level"],
            spo2: r["SpO2"],
            accelerometer: r["Accelerometer"]
        }))
    };

    document.getElementById("batchInput").value =
        JSON.stringify(batch, null, 4);
}

/* Clear the batch input */
function clearBatch() {
    document.getElementById("batchInput").value = "";
}

async function compareModels() {
    const res = await fetch(`${API}/model/compare`);
    const data = await res.json();

    let html = "<h3>Model Comparison</h3><table><tr><th>Model</th><th>Accuracy</th></tr>";

    data.comparison.forEach(row => {
        html += `<tr><td>${row.model}</td><td>${(row.accuracy * 100).toFixed(2)}%</td></tr>`;
    });

    html += "</table>";

    document.getElementById("metrics-output").innerHTML = html;
}


function renderTable(data) {
    let html = "<table>";
    data.forEach((row, i) => {
        html += "<tr>";
        row.forEach(cell => {
            html += i === 0 ? `<th>${cell}</th>` : `<td>${cell}</td>`;
        });
        html += "</tr>";
    });
    html += "</table>";
    return html;
}
async function getAccuracy() {
    const res = await fetch(`${API}/model/accuracy`);
    const data = await res.json();

    document.getElementById("accuracy-output").innerHTML =
        `<strong>Accuracy:</strong> ${(data.accuracy * 100).toFixed(2)}%`;
}
async function getConfusion() {
    const res = await fetch(`${API}/model/metrics`);
    const data = await res.json();

    document.getElementById("confusion-output").innerHTML =
        renderTable(data.confusion_matrix_table);
}
async function getClassReport() {
    const res = await fetch(`${API}/model/metrics`);
    const data = await res.json();

    let rows = [["Class", "Precision", "Recall", "F1", "Support"]];

    data.classification_report_table.forEach(r => {
        rows.push([r.class, r.precision, r.recall, r.f1_score, r.support]);
    });

    document.getElementById("classreport-output").innerHTML =
        renderTable(rows);
}
async function compareModels() {
    const res = await fetch(`${API}/model/compare`);
    const data = await res.json();

    let rows = [["Model", "Accuracy"]];

    data.comparison.forEach(m => {
        rows.push([m.model, (m.accuracy * 100).toFixed(2) + "%"]);
    });

    document.getElementById("compare-output").innerHTML =
        renderTable(rows);
}

async function getROC() {
    let model = document.getElementById("model-select").value;

    const res = await fetch(`${API}/model/roc/${model}`);
    const data = await res.json();

    document.getElementById("roc-output").innerHTML =
        `<img src="data:image/png;base64,${data.roc_curve}" style="width:100%;">`;
}



async function getPR() {
    let model = document.getElementById("model-select").value;

    const res = await fetch(`${API}/model/pr/${model}`);
    const data = await res.json();

    document.getElementById("pr-output").innerHTML =
        `<img src="data:image/png;base64,${data.pr_curve}" style="width:100%;">`;
}

