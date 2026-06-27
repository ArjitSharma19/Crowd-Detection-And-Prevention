// ----------------------------------------------------
// CROWDSHIELD AI - CORE DASHBOARD LOGIC (PROD CONNECTED)
// ----------------------------------------------------

// Global State
let currentViewMode = 'raw';
let currentCount = 0;

const thresholds = {
    maxPeople: 25,
    cautionAt: 70,
    alertDelay: 20,
    confidence: 0.25,
    resolution: 960,
    modelType: 'general'
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
const sliderCautionAt = document.getElementById('slider-caution-at');
const valCautionAt = document.getElementById('val-caution-at');
const sliderAlertDelay = document.getElementById('slider-alert-delay');
const valAlertDelay = document.getElementById('val-alert-delay');

// Technical Sliders
const sliderConfidence = document.getElementById('slider-confidence');
const valConfidence = document.getElementById('val-confidence');
const sliderResolution = document.getElementById('slider-resolution');
const valResolution = document.getElementById('val-resolution');
const selectModelType = document.getElementById('select-model-type');

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
    // 1. Fetch initial configuration from backend and sync UI
    syncConfigFromBackend().then(() => {
        // 2. Setup Threshold control listeners
        setupSliders();
        
        // 3. Start real metrics polling loop
        startMetricsPolling();
        
        // 4. Set initial view mode
        setViewMode(currentViewMode);
    });
});

// ----------------------------------------------------
// SYNC CONFIGURATION WITH BACKEND
// ----------------------------------------------------
async function syncConfigFromBackend() {
    try {
        const response = await fetch('/api/config');
        if (!response.ok) throw new Error('Could not fetch initial config');
        const config = await response.json();
        
        // Sync local thresholds state
        thresholds.maxPeople = config.max_capacity;
        thresholds.cautionAt = config.caution_at || 70;
        thresholds.alertDelay = Math.round(config.trigger_delay);
        thresholds.confidence = config.confidence_threshold;
        thresholds.resolution = config.imgsz;
        thresholds.modelType = config.model_type;
        
        // Sync UI Sliders & Labels
        if (sliderMaxPeople) {
            sliderMaxPeople.value = thresholds.maxPeople;
            valMaxPeople.textContent = thresholds.maxPeople;
        }
        if (sliderCautionAt) {
            sliderCautionAt.value = thresholds.cautionAt;
            valCautionAt.textContent = `${thresholds.cautionAt}%`;
        }
        if (sliderAlertDelay) {
            sliderAlertDelay.value = thresholds.alertDelay;
            valAlertDelay.textContent = `${thresholds.alertDelay}s`;
        }
        if (sliderConfidence) {
            sliderConfidence.value = thresholds.confidence;
            valConfidence.textContent = thresholds.confidence.toFixed(2);
        }
        if (sliderResolution) {
            sliderResolution.value = thresholds.resolution;
            valResolution.textContent = `${thresholds.resolution}px`;
        }
        if (selectModelType) {
            selectModelType.value = thresholds.modelType;
        }
        
        console.log('Successfully synced initial config from backend:', config);
    } catch (error) {
        console.error('Error syncing config from backend:', error);
    }
}

async function sendConfigUpdate() {
    const payload = {
        max_capacity: thresholds.maxPeople,
        caution_at: parseInt(thresholds.cautionAt),
        trigger_delay: parseFloat(thresholds.alertDelay),
        confidence_threshold: thresholds.confidence,
        imgsz: thresholds.resolution,
        model_type: thresholds.modelType
    };
    
    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error('Failed to update config');
        const data = await response.json();
        console.log('Backend config updated successfully:', data.config);
    } catch (err) {
        console.error('Error updating backend config:', err);
    }
}

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
        if (btn) {
            if (id.includes(mode)) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        }
    });
    
    // Update the video feed element's source dynamically
    const elCameraFeed = document.getElementById('camera-feed-img');
    if (elCameraFeed) {
        elCameraFeed.src = `/video_feed?mode=${mode}`;
    }
}

// Helper to calculate risk tier consistently
function getRiskTier(count, limit, backendStatus) {
    if (backendStatus === 'CRITICAL') {
        return 'danger';
    }
    if (backendStatus === 'WARNING') {
        return 'caution';
    }
    
    const percentage = (count / limit) * 100;
    if (percentage >= 100) {
        return 'danger';
    } else if (percentage >= thresholds.cautionAt) {
        return 'caution';
    }
    return 'safe';
}

function updateCrowdState(count, limit, backendStatus, backendMessage) {
    currentCount = count;
    const roundedCount = Math.round(count);
    
    // Update live metrics display
    elPeopleNow.textContent = roundedCount;
    elLimit.textContent = limit;
    elLiveCountBadge.textContent = roundedCount;
    
    // Calculate percentage capacity
    const percentage = Math.round((count / limit) * 100);
    elProgressBar.style.width = `${Math.min(percentage, 100)}%`;
    elCapacityText.textContent = `${roundedCount} of ${limit} people · ${percentage}% full`;
    
    // Map backend status to risk state
    const riskState = getRiskTier(count, limit, backendStatus);
    
    // Determine message
    let message = backendMessage;
    if (riskState === 'danger' && (!message || message.includes("normal") || message.includes("Warning") || message.includes("safe"))) {
        message = `CRITICAL: Capacity exceeded (${roundedCount}/${limit})`;
    } else if (riskState === 'caution' && (!message || message.includes("normal") || message.includes("safe"))) {
        message = `Warning — capacity at ${percentage}%`;
    } else if (!message) {
        message = "Crowd levels within safe parameters.";
    }
    
    // Apply styling and message
    applyRiskStyling(riskState, message);
}

// Helper to update colors/badges across elements
function applyRiskStyling(state, message) {
    // 1. Reset Classes
    elStatusPill.className = 'status-pill';
    elRiskLevelDisplay.className = 'risk-value';
    elProgressBar.className = 'progress-bar-fill';
    
    // 2. Add specific colors & update text
    if (state === 'safe') {
        elStatusPill.classList.add('safe');
        elStatusText.textContent = message || "Crowd levels normal";
        
        elRiskLevelDisplay.className = 'risk-value text-safe';
        elRiskLevelDisplay.textContent = "Safe";
        
        elProgressBar.classList.add('fill-safe');
    } else if (state === 'caution') {
        elStatusPill.classList.add('caution');
        elStatusText.textContent = message || "Warning — crowd approaching limit";
        
        elRiskLevelDisplay.className = 'risk-value text-caution';
        elRiskLevelDisplay.textContent = "Caution";
        
        elProgressBar.classList.add('fill-caution');
    } else if (state === 'danger') {
        elStatusPill.classList.add('danger');
        elStatusText.textContent = message || "Critical — capacity exceeded";
        
        elRiskLevelDisplay.className = 'risk-value text-danger';
        elRiskLevelDisplay.textContent = "Danger";
        
        elProgressBar.classList.add('fill-danger');
    }
}

// ----------------------------------------------------
// METRICS POLLING AND INCIDENT LOG UPDATER
// ----------------------------------------------------
let metricsInterval = null;

function startMetricsPolling() {
    // Poll every 500ms for responsiveness
    metricsInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/metrics');
            if (!response.ok) throw new Error('Failed to fetch metrics');
            const data = await response.json();
            
            // Update UI count states using actual detector readings and capacity limits from backend
            updateCrowdState(data.current_count, data.max_capacity || thresholds.maxPeople, data.status, data.status_message);
            
            // Set the real confidence readout from the active model
            elConfidenceBadge.textContent = data.avg_confidence || 'N/A';
            
            // Update the Active Model indicator badge
            const elLiveModelBadge = document.getElementById('live-model-badge');
            const elModelDot = document.getElementById('model-dot');
            if (elLiveModelBadge) {
                elLiveModelBadge.textContent = data.model_used;
                if (elModelDot) {
                    if (data.model_used === 'YOLO') {
                        elLiveModelBadge.style.color = '#38bdf8'; // sky blue
                        elModelDot.style.backgroundColor = '#38bdf8';
                    } else {
                        elLiveModelBadge.style.color = '#a855f7'; // purple
                        elModelDot.style.backgroundColor = '#a855f7';
                    }
                }
            }
            
            // Sync incident logs list from backend manager history
            updateLogsList(data.alert_history, data.caution_at || thresholds.cautionAt);
            
        } catch (error) {
            console.error('Error in metrics polling:', error);
        }
    }, 500);
}

function updateLogsList(history, cautionAt) {
    if (!history) return;
    
    // Clear list
    elLogList.innerHTML = '';
    
    if (history.length === 0) {
        elLogList.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 32px 0; border: none; width: 100%; gap: 12px;">
                <div style="width: 48px; height: 48px; border-radius: 50%; background-color: rgba(16, 185, 129, 0.1); border: 1px dashed var(--color-safe); display: flex; align-items: center; justify-content: center; color: var(--color-safe);">
                    <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                </div>
                <span style="font-size: 14px; font-weight: 500; color: var(--text-primary);">No incidents recorded</span>
                <span class="label-muted" style="text-align: center; margin-bottom: 0;">Alerts will appear here once capacity crosses ${cautionAt || 70}%</span>
            </div>
        `;
        return;
    }
    
    history.forEach(item => {
        const entryDiv = document.createElement('div');
        entryDiv.className = 'log-entry';
        
        let dotClass = 'dot-safe';
        if (item.status === 'WARNING') dotClass = 'dot-caution';
        if (item.status === 'CRITICAL') dotClass = 'dot-danger';
        
        // Extract time part from "YYYY-MM-DD HH:MM:SS"
        const timePart = item.timestamp.split(' ')[1] || item.timestamp;
        
        entryDiv.innerHTML = `
            <span class="log-dot ${dotClass}"></span>
            <div class="log-content">
                <span class="log-message">${item.message}</span>
                <span class="log-time">${timePart}</span>
            </div>
        `;
        elLogList.appendChild(entryDiv);
    });
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
    sliderMaxPeople.addEventListener('change', sendConfigUpdate);

    // Caution at slider
    if (sliderCautionAt) {
        sliderCautionAt.addEventListener('input', (e) => {
            thresholds.cautionAt = parseInt(e.target.value);
            valCautionAt.textContent = `${thresholds.cautionAt}%`;
        });
        sliderCautionAt.addEventListener('change', sendConfigUpdate);
    }

    // Alert delay slider
    sliderAlertDelay.addEventListener('input', (e) => {
        thresholds.alertDelay = parseInt(e.target.value);
        valAlertDelay.textContent = `${thresholds.alertDelay}s`;
    });
    sliderAlertDelay.addEventListener('change', sendConfigUpdate);

    // Confidence slider
    sliderConfidence.addEventListener('input', (e) => {
        thresholds.confidence = parseFloat(e.target.value);
        valConfidence.textContent = thresholds.confidence.toFixed(2);
    });
    sliderConfidence.addEventListener('change', sendConfigUpdate);

    // Resolution slider
    sliderResolution.addEventListener('input', (e) => {
        thresholds.resolution = parseInt(e.target.value);
        valResolution.textContent = `${thresholds.resolution}px`;
    });
    sliderResolution.addEventListener('change', sendConfigUpdate);

    // Model type selector
    if (selectModelType) {
        selectModelType.addEventListener('change', (e) => {
            thresholds.modelType = e.target.value;
            sendConfigUpdate();
        });
    }
}
