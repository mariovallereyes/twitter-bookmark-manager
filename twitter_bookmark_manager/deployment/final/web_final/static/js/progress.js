/**
 * progress.js - Handles tracking and displaying progress for vector rebuilding
 * 
 * This module provides functions to:
 * 1. Track ongoing vector rebuild operations
 * 2. Display progress updates and status messages
 * 3. Handle errors gracefully with retry options
 * 4. Allow resuming interrupted operations
 */

// Configuration
const PROGRESS_POLL_INTERVAL = 2000; // ms between status checks
const MAX_RETRY_COUNT = 3;           // Maximum retries for failed operations
const MAX_ERROR_DETAILS_LENGTH = 500; // Max length of error details to display

// State tracking
let pollInterval = null;
let sessionId = null;
let retryCount = 0;
let rebuilding = false;
let lastProgressPercent = 0;

/**
 * Initialize progress tracking for a vector rebuild operation
 * @param {string} newSessionId - Session ID for the rebuild operation
 * @param {boolean} resume - Whether this is resuming a previous operation
 */
function initRebuildProgress(newSessionId, resume = false) {
    // Clear existing interval if any
    stopProgressTracking();
    
    // Set up new tracking
    sessionId = newSessionId;
    rebuilding = true;
    lastProgressPercent = resume ? getLastProgressFromStorage(sessionId) : 0;
    retryCount = 0;
    
    // Initialize UI
    updateProgressUI(lastProgressPercent, 'Starting vector rebuild...');
    
    // Start polling for updates
    pollInterval = setInterval(checkRebuildProgress, PROGRESS_POLL_INTERVAL);
    
    // Log initialization
    console.log(`Started tracking rebuild progress for session ${sessionId}${resume ? ' (resuming)' : ''}`);
    
    // Store session info in localStorage for potential resume
    storeSessionInfo(sessionId, lastProgressPercent);
}

/**
 * Stop tracking progress and clear interval
 */
function stopProgressTracking() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
    rebuilding = false;
}

/**
 * Check the current status of vector rebuild
 */
function checkRebuildProgress() {
    if (!sessionId || !rebuilding) {
        stopProgressTracking();
        return;
    }
    
    // Show spinner if at 0%
    if (lastProgressPercent === 0) {
        showSpinner(true);
    }
    
    // Make API call to check status
    fetch(`/api/process-status?session_id=${sessionId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .then(data => handleProgressUpdate(data))
        .catch(error => handleProgressError(error));
}

/**
 * Handle progress data update from API
 * @param {Object} data - Progress data from API
 */
function handleProgressUpdate(data) {
    // Reset retry counter on successful response
    retryCount = 0;
    
    // Extract progress information
    const status = data.status || 'unknown';
    const progress = data.progress || 0;
    const message = data.message || 'Processing...';
    
    // Update last progress for potential resume
    if (progress > lastProgressPercent) {
        lastProgressPercent = progress;
        storeSessionInfo(sessionId, progress);
    }
    
    // Update UI with progress information
    updateProgressUI(progress, message, status);
    
    // Check if the process is complete or has error
    if (status === 'completed') {
        handleRebuildCompleted(data);
    } else if (status === 'error') {
        handleRebuildError(data);
    }
}

/**
 * Handle error during progress check
 * @param {Error} error - Error object
 */
function handleProgressError(error) {
    console.error('Error checking rebuild progress:', error);
    
    // Increment retry counter
    retryCount++;
    
    if (retryCount <= MAX_RETRY_COUNT) {
        // Continue polling but show warning
        updateProgressUI(
            lastProgressPercent, 
            `Connection error. Retrying... (${retryCount}/${MAX_RETRY_COUNT})`, 
            'warning'
        );
    } else {
        // Too many failures, stop polling and show error
        stopProgressTracking();
        
        updateProgressUI(
            lastProgressPercent,
            'Lost connection to server. The rebuild may still be in progress.',
            'error'
        );
        
        // Offer resume option
        showResumeOption(sessionId);
    }
}

/**
 * Handle successful completion of rebuild
 * @param {Object} data - Completion data
 */
function handleRebuildCompleted(data) {
    stopProgressTracking();
    showSpinner(false);
    
    const duration = data.duration_seconds ? formatDuration(data.duration_seconds) : 'unknown time';
    const bookmarksProcessed = data.processed || 0;
    const successCount = data.success_count || 0;
    const errorCount = data.error_count || 0;
    
    let completionMessage = `Vector rebuild completed in ${duration}.`;
    if (bookmarksProcessed > 0) {
        completionMessage += ` Processed ${bookmarksProcessed} bookmarks (${successCount} successful, ${errorCount} errors).`;
    }
    
    updateProgressUI(100, completionMessage, 'completed');
    
    // Update search UI to indicate vectors are now available
    updateSearchAvailability(true);
    
    // Clear the session info from storage as it completed successfully
    clearSessionInfo(sessionId);
    
    // Show toast notification
    showToast('Vector rebuild completed', 'success');
}

/**
 * Handle error in rebuild process
 * @param {Object} data - Error data
 */
function handleRebuildError(data) {
    stopProgressTracking();
    showSpinner(false);
    
    const errorMessage = data.error || 'Unknown error occurred';
    const progressPercent = data.progress || lastProgressPercent;
    
    // Format error details for display
    let errorDetails = '';
    if (data.details || data.traceback) {
        errorDetails = data.details || data.traceback;
        if (errorDetails.length > MAX_ERROR_DETAILS_LENGTH) {
            errorDetails = errorDetails.substring(0, MAX_ERROR_DETAILS_LENGTH) + '...';
        }
    }
    
    // Update UI with error information
    updateProgressUI(progressPercent, errorMessage, 'error', errorDetails);
    
    // Store progress for potential resume
    storeSessionInfo(sessionId, progressPercent);
    
    // Show resume option
    showResumeOption(sessionId);
    
    // Show toast notification
    showToast('Vector rebuild encountered an error', 'error');
}

/**
 * Show an option to resume the rebuild
 * @param {string} session - Session ID to resume
 */
function showResumeOption(session) {
    const resumeContainer = document.getElementById('rebuild-resume-container');
    if (!resumeContainer) return;
    
    resumeContainer.innerHTML = `
        <div class="alert alert-warning mt-3">
            <p><i class="fas fa-exclamation-triangle"></i> Vector rebuild was interrupted.</p>
            <button class="btn btn-warning btn-sm" onclick="resumeRebuild('${session}')">
                <i class="fas fa-sync"></i> Resume Rebuild
            </button>
        </div>
    `;
    resumeContainer.style.display = 'block';
}

/**
 * Resume an interrupted rebuild process
 * @param {string} session - Session ID to resume
 */
function resumeRebuild(session) {
    // Hide resume option
    const resumeContainer = document.getElementById('rebuild-resume-container');
    if (resumeContainer) {
        resumeContainer.style.display = 'none';
    }
    
    // Show starting message
    updateProgressUI(lastProgressPercent, 'Resuming vector rebuild...', 'processing');
    showSpinner(true);
    
    // Call API to resume rebuild
    fetch('/api/rebuild-vector-store', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: session,
            resume: true
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            // Resume progress tracking
            initRebuildProgress(data.session_id, true);
        } else {
            // Show error
            updateProgressUI(lastProgressPercent, data.error || 'Failed to resume rebuild', 'error');
            showSpinner(false);
        }
    })
    .catch(error => {
        console.error('Error resuming rebuild:', error);
        updateProgressUI(lastProgressPercent, 'Connection error when trying to resume', 'error');
        showSpinner(false);
        
        // Re-show resume option after a failed attempt
        setTimeout(() => showResumeOption(session), 3000);
    });
}

/**
 * Update the progress UI elements
 * @param {number} percent - Progress percentage (0-100)
 * @param {string} message - Status message to display
 * @param {string} status - Status type (processing, completed, error, warning)
 * @param {string} details - Optional details for errors
 */
function updateProgressUI(percent, message, status = 'processing', details = '') {
    // Get UI elements
    const progressBar = document.getElementById('rebuild-progress-bar');
    const progressText = document.getElementById('rebuild-progress-text');
    const detailsContainer = document.getElementById('rebuild-error-details');
    
    if (!progressBar || !progressText) return;
    
    // Update progress bar
    progressBar.style.width = `${percent}%`;
    progressBar.setAttribute('aria-valuenow', percent);
    
    // Clear existing classes
    progressBar.classList.remove('bg-success', 'bg-danger', 'bg-warning', 'bg-info');
    
    // Set appropriate styling based on status
    switch (status) {
        case 'completed':
            progressBar.classList.add('bg-success');
            break;
        case 'error':
            progressBar.classList.add('bg-danger');
            break;
        case 'warning':
            progressBar.classList.add('bg-warning');
            break;
        default:
            progressBar.classList.add('bg-info');
    }
    
    // Update progress text
    progressText.textContent = `${message} (${Math.round(percent)}%)`;
    
    // Show error details if provided
    if (detailsContainer) {
        if (details && status === 'error') {
            detailsContainer.textContent = details;
            detailsContainer.style.display = 'block';
        } else {
            detailsContainer.style.display = 'none';
        }
    }
}

/**
 * Show or hide the loading spinner
 * @param {boolean} show - Whether to show the spinner
 */
function showSpinner(show) {
    const spinner = document.getElementById('rebuild-spinner');
    if (spinner) {
        spinner.style.display = show ? 'inline-block' : 'none';
    }
}

/**
 * Format duration in seconds to a readable string
 * @param {number} seconds - Duration in seconds
 * @returns {string} Formatted duration string
 */
function formatDuration(seconds) {
    if (seconds < 60) {
        return `${Math.round(seconds)} seconds`;
    } else if (seconds < 3600) {
        return `${Math.floor(seconds / 60)} minutes ${Math.round(seconds % 60)} seconds`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours} hours ${minutes} minutes`;
    }
}

/**
 * Update search UI to indicate vector availability
 * @param {boolean} available - Whether vectors are available for search
 */
function updateSearchAvailability(available) {
    const searchForm = document.getElementById('search-form');
    const searchNotice = document.getElementById('search-not-available');
    
    if (searchForm && searchNotice) {
        if (available) {
            searchForm.classList.remove('disabled');
            searchNotice.style.display = 'none';
        } else {
            searchForm.classList.add('disabled');
            searchNotice.style.display = 'block';
        }
    }
}

/**
 * Show a toast notification
 * @param {string} message - Message to display
 * @param {string} type - Type of toast (success, error, warning, info)
 */
function showToast(message, type = 'info') {
    // Check if toast container exists, create if not
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'position-fixed bottom-0 end-0 p-3';
        toastContainer.style.zIndex = '5';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toastId = `toast-${Date.now()}`;
    const iconClass = getIconForToastType(type);
    const bgClass = `bg-${type === 'error' ? 'danger' : type}`;
    
    const toastHtml = `
        <div id="${toastId}" class="toast ${bgClass} text-white" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header ${bgClass} text-white">
                <i class="${iconClass} me-2"></i>
                <strong class="me-auto">Notification</strong>
                <small>${new Date().toLocaleTimeString()}</small>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    // Add toast to container
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    // Initialize and show the toast
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { autohide: true, delay: 5000 });
    toast.show();
    
    // Remove the toast after it's hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

/**
 * Get appropriate icon class for toast type
 * @param {string} type - Toast type
 * @returns {string} Icon class
 */
function getIconForToastType(type) {
    switch (type) {
        case 'success': return 'fas fa-check-circle';
        case 'error': return 'fas fa-exclamation-circle';
        case 'warning': return 'fas fa-exclamation-triangle';
        default: return 'fas fa-info-circle';
    }
}

/**
 * Store session information in localStorage for potential resume
 * @param {string} session - Session ID
 * @param {number} progress - Current progress percentage
 */
function storeSessionInfo(session, progress) {
    try {
        localStorage.setItem('rebuild_session_id', session);
        localStorage.setItem('rebuild_progress', progress.toString());
        localStorage.setItem('rebuild_timestamp', Date.now().toString());
    } catch (e) {
        console.error('Error storing session info in localStorage:', e);
    }
}

/**
 * Get last progress from localStorage
 * @param {string} session - Session ID to validate
 * @returns {number} Last progress percentage or 0
 */
function getLastProgressFromStorage(session) {
    try {
        const storedSession = localStorage.getItem('rebuild_session_id');
        
        // Only use stored progress if session matches
        if (storedSession === session) {
            const progress = parseInt(localStorage.getItem('rebuild_progress') || '0', 10);
            const timestamp = parseInt(localStorage.getItem('rebuild_timestamp') || '0', 10);
            const ageInHours = (Date.now() - timestamp) / (1000 * 60 * 60);
            
            // Only use progress data if less than 24 hours old
            if (ageInHours < 24) {
                return progress;
            }
        }
    } catch (e) {
        console.error('Error retrieving session info from localStorage:', e);
    }
    return 0;
}

/**
 * Clear session information from localStorage
 * @param {string} session - Session ID to clear
 */
function clearSessionInfo(session) {
    try {
        const storedSession = localStorage.getItem('rebuild_session_id');
        if (storedSession === session) {
            localStorage.removeItem('rebuild_session_id');
            localStorage.removeItem('rebuild_progress');
            localStorage.removeItem('rebuild_timestamp');
        }
    } catch (e) {
        console.error('Error clearing session info from localStorage:', e);
    }
}

/**
 * Check for incomplete rebuilds on page load and offer resume
 */
function checkForIncompleteRebuilds() {
    try {
        const session = localStorage.getItem('rebuild_session_id');
        const progress = parseInt(localStorage.getItem('rebuild_progress') || '0', 10);
        const timestamp = parseInt(localStorage.getItem('rebuild_timestamp') || '0', 10);
        const ageInHours = (Date.now() - timestamp) / (1000 * 60 * 60);
        
        // Only offer resume if session exists, progress is incomplete, and data is fresh
        if (session && progress > 0 && progress < 100 && ageInHours < 24) {
            lastProgressPercent = progress;
            setTimeout(() => showResumeOption(session), 1000);
        }
    } catch (e) {
        console.error('Error checking for incomplete rebuilds:', e);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', checkForIncompleteRebuilds);

// Export functions for use in other modules
window.progressTracker = {
    initRebuildProgress,
    stopProgressTracking,
    resumeRebuild,
    updateProgressUI,
    showToast
}; 