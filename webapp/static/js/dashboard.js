// ----------------------------------------------------
// CROWDSHIELD AI - CORE DASHBOARD LOGIC
// ----------------------------------------------------

// Global State
let currentViewMode = 'heatmap';
let currentCount = 0;
let lastRiskState = 'safe'; // Track changes for logging
let alertTimerStart = null;  // For calculating alert delay duration
let isAlertLogged = false;

const thresholds = {
    maxPeople: 9,
    sensitivity: 5.0,
    alertDelay: 5
};

// DOM Elements cache
const elLiveCountBadge = document.getElementById('live-count-badge');
const elConfidenceBadge = document.getElementById('live-confidence-badge');
const elStatusPill = document.getElementById('status-pill');
const elStatusText = document.getElementById('status-text');
const elRiskLevelDisplay = document.getElementById('risk-level-display');
const elPeopleNow = document.getElementById('stat-people-now');
const elLimit = document.getElementById('stat-limit');
const elProgressBar = document.getElementById('capacity-progress-bar');
const elCapacityText = document.getElementById('capacity-text');
const elLogList = document.getElementById('log-list');

// Sliders and Value spans
const sliderMaxPeople = document.getElementById('slider-max-people');
const valMaxPeople = document.getElementById('val-max-people');
const sliderSensitivity = document.getElementById('slider-sensitivity');
const valSensitivity = document.getElementById('val-sensitivity');
const sliderAlertDelay = document.getElementById('slider-alert-delay');
const valAlertDelay = document.getElementById('val-alert-delay');

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
    // 1. Initial State Sync
    updateCrowdState(currentCount, thresholds.maxPeople);
    
    // 2. Load Mock Logs
    addLogEntry("System initialized successfully. Monitoring active.", "safe", "18:00:00");
    addLogEntry("Warning: Limit adjusted to 9 people.", "caution", "18:01:25");
    addLogEntry("Self-check completed. All systems functional.", "safe", "18:02:10");
    
    // 3. Setup Threshold controls listeners
    setupSliders();
    
    // 4. Start Demo loop
    startDemoSimulation();
});

// ----------------------------------------------------
// VIEW MODE TOGGLES
// ----------------------------------------------------
function setViewMode(mode) {
    currentViewMode = mode;
    console.log(`View mode changed to: ${mode}`);
    
    // Update button styling
    const buttons = ['btn-mode-raw', 'btn-mode-heatmap', 'btn-mode-grid'];
    buttons.forEach(id => {
        const btn = document.getElementById(id);
        if (id.includes(mode)) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // Later: Add logic to update the web socket / canvas video stream query parameters
}

// ----------------------------------------------------
// CORE RISK CALCULATION AND UPDATE
// ----------------------------------------------------
function updateCrowdState(count, limit) {
    currentCount = count;
    
    // Update live metrics display
    elPeopleNow.textContent = count;
    elLimit.textContent = limit;
    elLiveCountBadge.textContent = count;
    
    // Calculate percentage capacity
    const percentage = Math.round((count / limit) * 100);
    elProgressBar.style.width = `${Math.min(percentage, 100)}%`;
    elCapacityText.textContent = `${count} of ${limit} people · ${percentage}% full`;
    
    // Determine risk state
    let riskState = 'safe'; // Under 70%
    if (percentage >= 70 && percentage <= 90) {
        riskState = 'caution'; // 70-90%
    } else if (percentage > 90) {
        riskState = 'danger'; // Above 90%
    }
    
    // Alert Delay Evaluation logic
    let activeState = riskState;
    if (riskState === 'danger') {
        if (!alertTimerStart) {
            alertTimerStart = Date.now();
        }
        const elapsedSeconds = Math.floor((Date.now() - alertTimerStart) / 1000);
        if (elapsedSeconds < thresholds.alertDelay) {
            // Keep status as Caution while the delay timer counts down
            activeState = 'caution';
        } else if (!isAlertLogged) {
            // Trigger log entry when alert is officially breached after delay
            addLogEntry(`CRITICAL: Capacity exceeded (${count}/${limit}) for over ${thresholds.alertDelay}s!`, "danger");
            isAlertLogged = true;
        }
    } else {
        // Reset delay timer
        alertTimerStart = null;
        isAlertLogged = false;
    }
    
    // Apply visual styling based on risk state
    applyRiskStyling(activeState, count, limit, percentage);
    
    // Log state transitions
    if (activeState !== lastRiskState) {
        if (activeState === 'caution' && lastRiskState === 'safe') {
            addLogEntry("Warning: Crowd size approaching capacity limit.", "caution");
        } else if (activeState === 'safe' && lastRiskState !== 'safe') {
            addLogEntry("Restored: Crowd size has returned to safe levels.", "safe");
        }
        lastRiskState = activeState;
    }
}

// Helper to update colors/badges across elements
function applyRiskStyling(state, count, limit, percentage) {
    // 1. Reset Classes
    elStatusPill.className = 'status-pill';
    elRiskLevelDisplay.className = 'risk-value';
    elProgressBar.className = 'progress-bar-fill';
    
    // 2. Add specific colors
    if (state === 'safe') {
        elStatusPill.classList.add('safe');
        elStatusText.textContent = "Crowd levels normal";
        
        elRiskLevelDisplay.classList.add('text-safe');
        elRiskLevelDisplay.textContent = "Safe";
        
        elProgressBar.classList.add('fill-safe');
    } else if (state === 'caution') {
        elStatusPill.classList.add('caution');
        elStatusText.textContent = "Warning — crowd approaching limit";
        
        elRiskLevelDisplay.classList.add('text-caution');
        elRiskLevelDisplay.textContent = "Caution";
        
        elProgressBar.classList.add('fill-caution');
    } else if (state === 'danger') {
        elStatusPill.classList.add('danger');
        elStatusText.textContent = "Critical — immediate action needed";
        
        elRiskLevelDisplay.classList.add('text-danger');
        elRiskLevelDisplay.textContent = "Danger";
        
        elProgressBar.classList.add('fill-danger');
    }
}

// ----------------------------------------------------
// INCIDENT LOG SYSTEM
// ----------------------------------------------------
function addLogEntry(message, severity, timestamp = null) {
    const timeStr = timestamp || new Date().toLocaleTimeString();
    
    const entryDiv = document.createElement('div');
    entryDiv.className = 'log-entry';
    
    // Map severity to log dot class
    let dotClass = 'dot-safe';
    if (severity === 'caution') dotClass = 'dot-caution';
    if (severity === 'danger') dotClass = 'dot-danger';
    
    entryDiv.innerHTML = `
        <span class="log-dot ${dotClass}"></span>
        <div class="log-content">
            <span class="log-message">${message}</span>
            <span class="log-time">${timeStr}</span>
        </div>
    `;
    
    // Prepend to top of log list
    elLogList.insertBefore(entryDiv, elLogList.firstChild);
}

// ----------------------------------------------------
// THRESHOLDS CONTROL SLIDERS
// ----------------------------------------------------
function setupSliders() {
    // Max people slider
    sliderMaxPeople.addEventListener('input', (e) => {
        thresholds.maxPeople = parseInt(e.target.value);
        valMaxPeople.textContent = thresholds.maxPeople;
        updateCrowdState(currentCount, thresholds.maxPeople);
    });

    // Sensitivity slider
    sliderSensitivity.addEventListener('input', (e) => {
        thresholds.sensitivity = parseFloat(e.target.value);
        valSensitivity.textContent = thresholds.sensitivity.toFixed(1);
    });

    // Alert delay slider
    sliderAlertDelay.addEventListener('input', (e) => {
        thresholds.alertDelay = parseInt(e.target.value);
        valAlertDelay.textContent = `${thresholds.alertDelay}s`;
        updateCrowdState(currentCount, thresholds.maxPeople);
    });
}

// ----------------------------------------------------
// DEMO SIMULATION LOOP (DEMO ONLY)
// ----------------------------------------------------
let demoInterval = null;

function startDemoSimulation() {
    // Start count at 4 people
    currentCount = 4;
    
    // Periodically fluctuate counts simulating a random walk
    demoInterval = setInterval(() => {
        // Random change of -2, -1, 0, +1, or +2
        const delta = Math.floor(Math.random() * 5) - 2;
        let newCount = currentCount + delta;
        
        // Boundaries: 0 to 18 people
        newCount = Math.max(0, Math.min(newCount, 18));
        
        // Simulate slight confidence drift
        const confidence = Math.floor(Math.random() * 6) + 90; // 90% - 95%
        elConfidenceBadge.textContent = `${confidence}%`;
        
        updateCrowdState(newCount, thresholds.maxPeople);
    }, 3000); // Trigger walk every 3 seconds
}

// NOTE FOR PRODUCTION:
// To wire this up to real detector output, comment out the `startDemoSimulation()` call in `DOMContentLoaded`
// and call `updateCrowdState(realCount, thresholds.maxPeople)` from your WebSocket / SSE event handler.
