/**
 * FL_ImagePicker.js
 * Interactive modal for selecting images from a batch
 */

import { app } from "../../../../../scripts/app.js";
import { api } from "../../../../../scripts/api.js";

// Track active modal to prevent duplicates
let activeModal = null;
let countdownInterval = null;
let previewModal = null;
let currentPreviewIndex = -1;
let currentSessionId = null;
let currentBatchSize = 0;

// Listen for the show selector event from the backend
api.addEventListener("fl_image_picker_show", (event) => {
    const { session_id, images, batch_size, timeout_seconds } = event.detail;
    console.log(`[FL_ImagePicker] Received ${batch_size} images for selection (timeout: ${timeout_seconds}s)`);
    showImagePickerModal(session_id, images, batch_size, timeout_seconds || 300);
});

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function showImagePickerModal(sessionId, images, batchSize, timeoutSeconds) {
    // Close any existing modal
    if (activeModal) {
        activeModal.remove();
        activeModal = null;
    }
    if (countdownInterval) {
        clearInterval(countdownInterval);
        countdownInterval = null;
    }

    // Store session info for preview functionality
    currentSessionId = sessionId;
    currentBatchSize = batchSize;

    // Track selected indices
    const selectedIndices = new Set();
    let remainingSeconds = timeoutSeconds;

    // Create modal overlay
    const overlay = document.createElement('div');
    overlay.className = 'fl-image-picker-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        backdrop-filter: blur(4px);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.2s ease-out;
    `;

    // Create modal container - takes up most of the screen for better image viewing
    const container = document.createElement('div');
    container.className = 'fl-image-picker-container';
    container.style.cssText = `
        background: linear-gradient(145deg, #2d2d2d, #252525);
        border-radius: 12px;
        border: 1px solid #3a3a3a;
        width: 95%;
        height: 95%;
        max-width: 1900px;
        display: flex;
        flex-direction: column;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8);
        animation: slideIn 0.3s ease-out;
        overflow: hidden;
    `;

    // Create header
    const header = document.createElement('div');
    header.style.cssText = `
        padding: 20px 24px;
        border-bottom: 1px solid #404040;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0) 100%);
        border-radius: 12px 12px 0 0;
    `;

    const titleContainer = document.createElement('div');
    titleContainer.innerHTML = `
        <h2 style="margin: 0; color: #fff; font-size: 20px; font-weight: 600;">
            Select Images to Keep
        </h2>
        <p style="margin: 4px 0 0 0; color: #888; font-size: 13px;">
            Click images to select/deselect. Selected images will be passed through.
        </p>
    `;

    // Header right side - stats and timer
    const headerRight = document.createElement('div');
    headerRight.style.cssText = `
        display: flex;
        align-items: center;
        gap: 16px;
    `;

    const statsDisplay = document.createElement('div');
    statsDisplay.id = 'fl-picker-stats';
    statsDisplay.style.cssText = `
        color: #4ECDC4;
        font-size: 16px;
        font-weight: 600;
        padding: 8px 16px;
        background: rgba(78, 205, 196, 0.1);
        border-radius: 6px;
        border: 1px solid rgba(78, 205, 196, 0.3);
    `;

    // Timer display
    const timerDisplay = document.createElement('div');
    timerDisplay.id = 'fl-picker-timer';
    timerDisplay.style.cssText = `
        display: flex;
        align-items: center;
        gap: 8px;
        color: #fff;
        font-size: 14px;
        font-weight: 500;
        padding: 8px 16px;
        background: rgba(255, 193, 7, 0.1);
        border-radius: 6px;
        border: 1px solid rgba(255, 193, 7, 0.3);
    `;
    timerDisplay.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FFC107" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
        <span id="timer-text">${formatTime(remainingSeconds)}</span>
    `;

    function updateStats() {
        statsDisplay.textContent = `${selectedIndices.size} of ${batchSize} selected`;
    }

    function updateTimer() {
        const timerText = timerDisplay.querySelector('#timer-text');
        if (timerText) {
            timerText.textContent = formatTime(remainingSeconds);
        }

        // Change color when low on time
        if (remainingSeconds <= 30) {
            timerDisplay.style.background = 'rgba(217, 83, 79, 0.2)';
            timerDisplay.style.borderColor = 'rgba(217, 83, 79, 0.5)';
            timerDisplay.style.color = '#d9534f';
            const svg = timerDisplay.querySelector('svg');
            if (svg) svg.style.stroke = '#d9534f';
        } else if (remainingSeconds <= 60) {
            timerDisplay.style.background = 'rgba(255, 193, 7, 0.2)';
            timerDisplay.style.borderColor = 'rgba(255, 193, 7, 0.5)';
            timerDisplay.style.color = '#FFC107';
            const svg = timerDisplay.querySelector('svg');
            if (svg) svg.style.stroke = '#FFC107';
        }
    }

    updateStats();
    updateTimer();

    headerRight.appendChild(statsDisplay);
    headerRight.appendChild(timerDisplay);

    header.appendChild(titleContainer);
    header.appendChild(headerRight);

    // Start countdown timer
    countdownInterval = setInterval(async () => {
        remainingSeconds--;
        updateTimer();

        if (remainingSeconds <= 0) {
            clearInterval(countdownInterval);
            countdownInterval = null;
            // Timeout - cancel the job instead of sending images through
            await sendSelection(sessionId, [], true);
            closeModal();
        }
    }, 1000);

    // Create image grid container - uses fixed minimum width to prevent shrinking
    const gridContainer = document.createElement('div');
    gridContainer.className = 'fl-image-grid';
    gridContainer.style.cssText = `
        flex: 1;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 20px;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
        gap: 20px;
        align-content: start;
    `;

    // Create image cards
    images.forEach((imgData, index) => {
        const card = document.createElement('div');
        card.className = 'fl-image-card';
        card.dataset.index = index;
        card.style.cssText = `
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 3px solid transparent;
            background: #1a1a1a;
            height: 280px;
        `;

        // Image element - use object-fit: contain to show full image without cropping
        const img = document.createElement('img');
        img.src = imgData.data;
        img.style.cssText = `
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
            transition: opacity 0.2s ease;
            background: #0a0a0a;
        `;

        // Index badge
        const badge = document.createElement('div');
        badge.style.cssText = `
            position: absolute;
            top: 8px;
            left: 8px;
            background: rgba(0, 0, 0, 0.7);
            color: #fff;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        `;
        badge.textContent = `#${index + 1}`;

        // Dimensions badge
        const dimBadge = document.createElement('div');
        dimBadge.style.cssText = `
            position: absolute;
            bottom: 8px;
            right: 8px;
            background: rgba(0, 0, 0, 0.7);
            color: #888;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 10px;
        `;
        dimBadge.textContent = `${imgData.width}x${imgData.height}`;

        // Selection checkmark overlay
        const checkOverlay = document.createElement('div');
        checkOverlay.className = 'check-overlay';
        checkOverlay.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(78, 205, 196, 0.3);
            display: none;
            align-items: center;
            justify-content: center;
            pointer-events: none;
        `;
        checkOverlay.innerHTML = `
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#4ECDC4" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
        `;

        // Preview button (magnifying glass) - appears on hover
        const previewBtn = document.createElement('button');
        previewBtn.className = 'preview-btn';
        previewBtn.title = 'View full resolution';
        previewBtn.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            width: 36px;
            height: 36px;
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.2);
            cursor: pointer;
            display: none;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
            z-index: 10;
        `;
        previewBtn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                <line x1="11" y1="8" x2="11" y2="14"></line>
                <line x1="8" y1="11" x2="14" y2="11"></line>
            </svg>
        `;
        previewBtn.onmouseenter = () => {
            previewBtn.style.background = 'rgba(78, 205, 196, 0.8)';
            previewBtn.style.borderColor = '#4ECDC4';
        };
        previewBtn.onmouseleave = () => {
            previewBtn.style.background = 'rgba(0, 0, 0, 0.7)';
            previewBtn.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        };
        previewBtn.onclick = (e) => {
            e.stopPropagation(); // Don't trigger card selection
            showFullResPreview(index, imgData);
        };

        card.appendChild(img);
        card.appendChild(badge);
        card.appendChild(dimBadge);
        card.appendChild(checkOverlay);
        card.appendChild(previewBtn);

        // Click handler
        card.onclick = () => {
            if (selectedIndices.has(index)) {
                selectedIndices.delete(index);
                card.style.border = '3px solid transparent';
                checkOverlay.style.display = 'none';
            } else {
                selectedIndices.add(index);
                card.style.border = '3px solid #4ECDC4';
                checkOverlay.style.display = 'flex';
            }
            updateStats();
        };

        // Hover effect
        card.onmouseenter = () => {
            if (!selectedIndices.has(index)) {
                card.style.border = '3px solid rgba(78, 205, 196, 0.4)';
            }
            card.style.transform = 'scale(1.02)';
            previewBtn.style.display = 'flex';
        };
        card.onmouseleave = () => {
            if (!selectedIndices.has(index)) {
                card.style.border = '3px solid transparent';
            }
            card.style.transform = 'scale(1)';
            previewBtn.style.display = 'none';
        };

        gridContainer.appendChild(card);
    });

    // Create footer
    const footer = document.createElement('div');
    footer.style.cssText = `
        padding: 16px 24px;
        border-top: 1px solid #404040;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
    `;

    // Quick action buttons
    const quickActions = document.createElement('div');
    quickActions.style.cssText = `
        display: flex;
        gap: 10px;
    `;

    const selectAllBtn = createButton('Select All', '#555', () => {
        images.forEach((_, idx) => {
            selectedIndices.add(idx);
            const card = gridContainer.querySelector(`[data-index="${idx}"]`);
            if (card) {
                card.style.border = '3px solid #4ECDC4';
                card.querySelector('.check-overlay').style.display = 'flex';
            }
        });
        updateStats();
    });

    const selectNoneBtn = createButton('Select None', '#555', () => {
        selectedIndices.clear();
        gridContainer.querySelectorAll('.fl-image-card').forEach(card => {
            card.style.border = '3px solid transparent';
            card.querySelector('.check-overlay').style.display = 'none';
        });
        updateStats();
    });

    const invertBtn = createButton('Invert', '#555', () => {
        images.forEach((_, idx) => {
            const card = gridContainer.querySelector(`[data-index="${idx}"]`);
            if (selectedIndices.has(idx)) {
                selectedIndices.delete(idx);
                card.style.border = '3px solid transparent';
                card.querySelector('.check-overlay').style.display = 'none';
            } else {
                selectedIndices.add(idx);
                card.style.border = '3px solid #4ECDC4';
                card.querySelector('.check-overlay').style.display = 'flex';
            }
        });
        updateStats();
    });

    quickActions.appendChild(selectAllBtn);
    quickActions.appendChild(selectNoneBtn);
    quickActions.appendChild(invertBtn);

    // Main action buttons
    const mainActions = document.createElement('div');
    mainActions.style.cssText = `
        display: flex;
        gap: 12px;
    `;

    const cancelBtn = createButton('Cancel', '#d9534f', async () => {
        await sendSelection(sessionId, [], true);
        closeModal();
    });

    const confirmBtn = createButton('Continue with Selected', '#4ECDC4', async () => {
        const selection = Array.from(selectedIndices).sort((a, b) => a - b);
        await sendSelection(sessionId, selection, false);
        closeModal();
    }, true);

    mainActions.appendChild(cancelBtn);
    mainActions.appendChild(confirmBtn);

    footer.appendChild(quickActions);
    footer.appendChild(mainActions);

    // Assemble modal
    container.appendChild(header);
    container.appendChild(gridContainer);
    container.appendChild(footer);
    overlay.appendChild(container);

    // Add CSS animations
    addStyles();

    // Add to DOM
    document.body.appendChild(overlay);
    activeModal = overlay;

    // Keyboard shortcuts (only when preview modal is NOT open)
    const keyHandler = async (e) => {
        // Don't handle if preview modal is open - let the preview handler deal with it
        if (previewModal) return;

        if (e.key === 'Escape') {
            e.preventDefault();
            await sendSelection(sessionId, [], true);
            closeModal();
        } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            const selection = Array.from(selectedIndices).sort((a, b) => a - b);
            await sendSelection(sessionId, selection, false);
            closeModal();
        }
    };
    document.addEventListener('keydown', keyHandler);

    function closeModal() {
        // Clear countdown timer
        if (countdownInterval) {
            clearInterval(countdownInterval);
            countdownInterval = null;
        }

        document.removeEventListener('keydown', keyHandler);
        if (overlay.parentNode) {
            overlay.style.animation = 'fadeIn 0.15s ease-in reverse';
            container.style.animation = 'slideIn 0.15s ease-in reverse';
            setTimeout(() => {
                if (overlay.parentNode) {
                    overlay.remove();
                }
                activeModal = null;
            }, 150);
        }
    }
}

function createButton(text, color, onClick, isPrimary = false) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.style.cssText = `
        padding: ${isPrimary ? '10px 24px' : '8px 16px'};
        background: ${isPrimary ? color : 'transparent'};
        border: 2px solid ${color};
        border-radius: 6px;
        color: ${isPrimary ? '#fff' : color};
        cursor: pointer;
        font-size: ${isPrimary ? '14px' : '13px'};
        font-weight: ${isPrimary ? '600' : '500'};
        transition: all 0.2s ease;
    `;
    btn.onmouseenter = () => {
        btn.style.background = color;
        btn.style.color = '#fff';
        btn.style.transform = 'scale(1.02)';
    };
    btn.onmouseleave = () => {
        btn.style.background = isPrimary ? color : 'transparent';
        btn.style.color = isPrimary ? '#fff' : color;
        btn.style.transform = 'scale(1)';
    };
    btn.onclick = onClick;
    return btn;
}

async function sendSelection(sessionId, selection, cancelled) {
    console.log(`[FL_ImagePicker] Sending selection: session=${sessionId}, selection=${JSON.stringify(selection)}, cancelled=${cancelled}`);
    try {
        const response = await fetch('/fl_image_picker/select', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId,
                selection: selection,
                cancelled: cancelled
            })
        });
        const result = await response.json();
        console.log('[FL_ImagePicker] Selection sent successfully:', result);
        return result;
    } catch (error) {
        console.error('[FL_ImagePicker] Error sending selection:', error);
        throw error;
    }
}

// Full-resolution preview modal
async function showFullResPreview(index, imgData) {
    currentPreviewIndex = index;

    // Close existing preview if any
    if (previewModal) {
        previewModal.remove();
        previewModal = null;
    }

    // Create preview overlay
    const overlay = document.createElement('div');
    overlay.className = 'fl-preview-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.95);
        z-index: 10001;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.2s ease-out;
    `;

    // Create container
    const container = document.createElement('div');
    container.style.cssText = `
        position: relative;
        max-width: 95%;
        max-height: 95%;
        display: flex;
        flex-direction: column;
        align-items: center;
    `;

    // Header with image info and close button
    const header = document.createElement('div');
    header.style.cssText = `
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        padding: 12px 16px;
        background: rgba(0, 0, 0, 0.6);
        border-radius: 8px 8px 0 0;
        margin-bottom: -1px;
    `;

    const infoText = document.createElement('div');
    infoText.id = 'preview-info';
    infoText.style.cssText = `
        color: #fff;
        font-size: 14px;
        font-weight: 500;
    `;
    infoText.textContent = `Image #${index + 1} of ${currentBatchSize} - ${imgData.width}x${imgData.height}`;

    const closeBtn = document.createElement('button');
    closeBtn.style.cssText = `
        background: transparent;
        border: none;
        cursor: pointer;
        padding: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    closeBtn.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
    `;
    closeBtn.onclick = closePreview;

    header.appendChild(infoText);
    header.appendChild(closeBtn);

    // Image container with loading state
    const imgContainer = document.createElement('div');
    imgContainer.style.cssText = `
        position: relative;
        background: #0a0a0a;
        border-radius: 0 0 8px 8px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 300px;
        min-height: 300px;
    `;

    // Loading spinner
    const spinner = document.createElement('div');
    spinner.id = 'preview-spinner';
    spinner.style.cssText = `
        position: absolute;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 12px;
        color: #888;
    `;
    spinner.innerHTML = `
        <div style="
            width: 40px;
            height: 40px;
            border: 3px solid #333;
            border-top-color: #4ECDC4;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        "></div>
        <span>Loading full resolution...</span>
    `;

    // The actual image
    const img = document.createElement('img');
    img.id = 'preview-image';
    img.style.cssText = `
        max-width: 90vw;
        max-height: 85vh;
        object-fit: contain;
        opacity: 0;
        transition: opacity 0.3s ease;
    `;

    imgContainer.appendChild(spinner);
    imgContainer.appendChild(img);

    // Navigation arrows
    const leftArrow = document.createElement('button');
    leftArrow.className = 'preview-nav-btn';
    leftArrow.style.cssText = `
        position: absolute;
        left: 20px;
        top: 50%;
        transform: translateY(-50%);
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: rgba(0, 0, 0, 0.6);
        border: 2px solid rgba(255, 255, 255, 0.3);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
        z-index: 10;
    `;
    leftArrow.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="15 18 9 12 15 6"></polyline>
        </svg>
    `;
    leftArrow.onmouseenter = () => {
        leftArrow.style.background = 'rgba(78, 205, 196, 0.6)';
        leftArrow.style.borderColor = '#4ECDC4';
    };
    leftArrow.onmouseleave = () => {
        leftArrow.style.background = 'rgba(0, 0, 0, 0.6)';
        leftArrow.style.borderColor = 'rgba(255, 255, 255, 0.3)';
    };
    leftArrow.onclick = () => navigatePreview(-1);

    const rightArrow = document.createElement('button');
    rightArrow.className = 'preview-nav-btn';
    rightArrow.style.cssText = `
        position: absolute;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: rgba(0, 0, 0, 0.6);
        border: 2px solid rgba(255, 255, 255, 0.3);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
        z-index: 10;
    `;
    rightArrow.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="9 18 15 12 9 6"></polyline>
        </svg>
    `;
    rightArrow.onmouseenter = () => {
        rightArrow.style.background = 'rgba(78, 205, 196, 0.6)';
        rightArrow.style.borderColor = '#4ECDC4';
    };
    rightArrow.onmouseleave = () => {
        rightArrow.style.background = 'rgba(0, 0, 0, 0.6)';
        rightArrow.style.borderColor = 'rgba(255, 255, 255, 0.3)';
    };
    rightArrow.onclick = () => navigatePreview(1);

    // Help text
    const helpText = document.createElement('div');
    helpText.style.cssText = `
        position: absolute;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        color: #666;
        font-size: 12px;
        background: rgba(0, 0, 0, 0.6);
        padding: 8px 16px;
        border-radius: 4px;
    `;
    helpText.textContent = 'Use arrow keys to navigate, Escape to close';

    container.appendChild(header);
    container.appendChild(imgContainer);
    overlay.appendChild(container);
    overlay.appendChild(leftArrow);
    overlay.appendChild(rightArrow);
    overlay.appendChild(helpText);

    // Click outside to close
    overlay.onclick = (e) => {
        if (e.target === overlay) {
            closePreview();
        }
    };

    document.body.appendChild(overlay);
    previewModal = overlay;

    // Load full-res image
    await loadFullResImage(index);
}

async function loadFullResImage(index) {
    const spinner = document.getElementById('preview-spinner');
    const img = document.getElementById('preview-image');
    const infoText = document.getElementById('preview-info');

    if (!spinner || !img) {
        console.error('[FL_ImagePicker] Preview elements not found');
        return;
    }

    if (!currentSessionId) {
        console.error('[FL_ImagePicker] No active session for preview');
        spinner.innerHTML = `<span style="color: #d9534f;">No active session</span>`;
        return;
    }

    // Show loading state
    spinner.style.display = 'flex';
    spinner.innerHTML = `
        <div style="
            width: 40px;
            height: 40px;
            border: 3px solid #333;
            border-top-color: #4ECDC4;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        "></div>
        <span>Loading full resolution...</span>
    `;
    img.style.opacity = '0';

    console.log(`[FL_ImagePicker] Loading full-res image: session=${currentSessionId}, index=${index}`);

    try {
        const response = await fetch(`/fl_image_picker/full_image/${currentSessionId}/${index}`);
        const data = await response.json();

        console.log(`[FL_ImagePicker] Full-res response:`, data.status, data.status === 'ok' ? `${data.width}x${data.height}` : data.message);

        if (data.status === 'ok') {
            img.src = data.data;
            img.onload = () => {
                spinner.style.display = 'none';
                img.style.opacity = '1';
            };
            img.onerror = () => {
                console.error('[FL_ImagePicker] Failed to load image data');
                spinner.innerHTML = `<span style="color: #d9534f;">Failed to decode image</span>`;
                spinner.style.display = 'flex';
            };
            if (infoText) {
                infoText.textContent = `Image #${index + 1} of ${currentBatchSize} - ${data.width}x${data.height}`;
            }
            currentPreviewIndex = index;
        } else {
            console.error('[FL_ImagePicker] Server error:', data.message);
            spinner.innerHTML = `<span style="color: #d9534f;">Error: ${data.message || 'Unknown error'}</span>`;
        }
    } catch (error) {
        console.error('[FL_ImagePicker] Error loading full-res image:', error);
        spinner.innerHTML = `<span style="color: #d9534f;">Network error loading image</span>`;
    }
}

function navigatePreview(direction) {
    let newIndex = currentPreviewIndex + direction;

    // Wrap around
    if (newIndex < 0) {
        newIndex = currentBatchSize - 1;
    } else if (newIndex >= currentBatchSize) {
        newIndex = 0;
    }

    loadFullResImage(newIndex);
}

function closePreview() {
    if (previewModal) {
        previewModal.style.animation = 'fadeIn 0.15s ease-in reverse';
        setTimeout(() => {
            if (previewModal && previewModal.parentNode) {
                previewModal.remove();
            }
            previewModal = null;
            currentPreviewIndex = -1;
        }, 150);
    }
}

// Keyboard handler for preview navigation
document.addEventListener('keydown', (e) => {
    if (!previewModal) return;

    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        navigatePreview(-1);
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        navigatePreview(1);
    } else if (e.key === 'Escape') {
        e.preventDefault();
        closePreview();
    }
});

function addStyles() {
    if (document.getElementById('fl-image-picker-styles')) return;

    const style = document.createElement('style');
    style.id = 'fl-image-picker-styles';
    style.textContent = `
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: scale(0.95) translateY(-20px);
            }
            to {
                opacity: 1;
                transform: scale(1) translateY(0);
            }
        }
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        .fl-image-grid {
            scrollbar-width: thin;
            scrollbar-color: #4ECDC4 #1a1a1a;
        }
        .fl-image-grid::-webkit-scrollbar {
            width: 12px;
        }
        .fl-image-grid::-webkit-scrollbar-track {
            background: #1a1a1a;
            border-radius: 6px;
        }
        .fl-image-grid::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #4ECDC4 0%, #45B7AA 100%);
            border-radius: 6px;
            border: 2px solid #1a1a1a;
        }
        .fl-image-grid::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #5FE0D3 0%, #4ECDC4 100%);
        }
        .fl-image-card:hover {
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }
        .fl-image-card:hover .preview-btn {
            display: flex !important;
        }
    `;
    document.head.appendChild(style);
}

// Register extension (for node-specific enhancements if needed)
app.registerExtension({
    name: "FillNodes.ImagePicker",
    async nodeCreated(node) {
        if (node.comfyClass === "FL_ImagePicker") {
            // Add any node-specific UI enhancements here
            // For example, we could add a status indicator widget
        }
    }
});
