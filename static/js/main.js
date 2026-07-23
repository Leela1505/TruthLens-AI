/* TruthLens AI - Main JavaScript Handler */

let isDetectionProcessing = false;

document.addEventListener('DOMContentLoaded', function() {
    
    // Auto dismiss flash alert messages after 5 seconds
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            try {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            } catch (e) {}
        }, 5000);
    });

    // Password Toggle Visibility
    const togglePasswordButtons = document.querySelectorAll('.toggle-password');
    togglePasswordButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const input = document.getElementById(targetId);
            if (input) {
                const isPassword = input.getAttribute('type') === 'password';
                input.setAttribute('type', isPassword ? 'text' : 'password');
                const icon = this.querySelector('i');
                if (icon) {
                    icon.className = isPassword ? 'fas fa-eye-slash' : 'fas fa-eye';
                }
            }
        });
    });

    // Intercept form submission if triggered by Enter key
    const detectForm = document.getElementById('detectForm');
    if (detectForm) {
        detectForm.addEventListener('submit', function(e) {
            e.preventDefault();
            e.stopPropagation();
            runDetectionAsync();
            return false;
        });
    }
});

function clearFormFields() {
    const titleInput = document.getElementById('news_title');
    const textInput = document.getElementById('news_text');
    const resultCard = document.getElementById('resultCard');
    const errorAlert = document.getElementById('errorAlert');
    
    if (titleInput) titleInput.value = '';
    if (textInput) textInput.value = '';
    if (resultCard) resultCard.style.display = 'none';
    if (errorAlert) errorAlert.classList.add('d-none');
}

async function runDetectionAsync() {
    if (isDetectionProcessing) return; // Prevent duplicate requests
    
    const titleInput = document.getElementById('news_title');
    const textInput = document.getElementById('news_text');
    const submitBtn = document.getElementById('btnPredict');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnIcon = document.getElementById('btnIcon');
    
    const resultCard = document.getElementById('resultCard');
    const errorAlert = document.getElementById('errorAlert');
    
    const newsTitle = titleInput ? titleInput.value.trim() : '';
    const newsText = textInput ? textInput.value.trim() : '';
    
    if (!newsText) {
        showError('Please enter a meaningful news article.');
        return;
    }
    
    // Hide previous error alert
    if (errorAlert) errorAlert.classList.add('d-none');
    
    // Set Processing State
    isDetectionProcessing = true;
    if (submitBtn) submitBtn.disabled = true;
    if (btnSpinner) btnSpinner.classList.remove('d-none');
    if (btnIcon) btnIcon.classList.add('d-none');
    if (btnText) btnText.textContent = 'Analyzing Article...';
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                news_title: newsTitle,
                news_text: newsText
            })
        });
        
        const data = await response.json();
        
        if (!response.ok || data.status !== 'success') {
            showError(data.message || 'Please enter a meaningful news article.');
            if (resultCard) resultCard.style.display = 'none';
            return;
        }
        
        // Display Results without clearing input text!
        displayPredictionResult(data);
        
    } catch (err) {
        console.error(err);
        showError('Network error connecting to AI server. Please check your connection.');
        if (resultCard) resultCard.style.display = 'none';
    } finally {
        isDetectionProcessing = false;
        if (submitBtn) submitBtn.disabled = false;
        if (btnSpinner) btnSpinner.classList.add('d-none');
        if (btnIcon) btnIcon.classList.remove('d-none');
        if (btnText) btnText.textContent = 'Analyze Authenticity';
    }
}

function showError(msg) {
    const errorAlert = document.getElementById('errorAlert');
    if (errorAlert) {
        errorAlert.textContent = msg;
        errorAlert.classList.remove('d-none');
    } else {
        alert(msg);
    }
}

function displayPredictionResult(data) {
    const resultCard = document.getElementById('resultCard');
    const badgeElem = document.getElementById('resultBadge');
    const titleElem = document.getElementById('resultTitle');
    const scoreElem = document.getElementById('confidenceScore');
    const progressBar = document.getElementById('confidenceProgressBar');
    const explanationElem = document.getElementById('resultExplanation');
    
    const realKeywordsContainer = document.getElementById('realKeywordsContainer');
    const fakeKeywordsContainer = document.getElementById('fakeKeywordsContainer');
    
    const wordCountElem = document.getElementById('metaWordCount');
    const charCountElem = document.getElementById('metaCharCount');
    const readingTimeElem = document.getElementById('metaReadingTime');
    const modelUsedElem = document.getElementById('metaModelUsed');
    
    if (!resultCard) return;
    
    const isReal = data.prediction === 'REAL';
    const isFake = data.prediction === 'FAKE';
    const isUncertain = data.prediction === 'UNCERTAIN';
    
    // Update Badge & Accent
    if (badgeElem) {
        if (isReal) {
            badgeElem.className = 'badge badge-real fs-6 mb-2';
            badgeElem.textContent = '✓ VERIFIED REAL NEWS';
        } else if (isFake) {
            badgeElem.className = 'badge badge-fake fs-6 mb-2';
            badgeElem.textContent = '⚠ LIKELY FAKE NEWS';
        } else {
            badgeElem.className = 'badge bg-warning text-dark fs-6 mb-2 fw-bold';
            badgeElem.textContent = '❓ UNCERTAIN PREDICTION';
        }
    }
    
    if (titleElem) {
        if (isReal) {
            titleElem.textContent = 'High Confidence Real News Article';
            titleElem.style.color = '#34d399';
        } else if (isFake) {
            titleElem.textContent = 'High Probability Fake News / Disinformation';
            titleElem.style.color = '#f87171';
        } else {
            titleElem.textContent = 'Uncertain Classification (< 70% Certainty)';
            titleElem.style.color = '#fbbf24';
        }
    }
    
    if (scoreElem) {
        scoreElem.textContent = `${data.confidence}%`;
    }
    
    if (progressBar) {
        progressBar.style.width = `${data.confidence}%`;
        if (isReal) {
            progressBar.className = 'progress-bar bg-success confidence-progress-bar';
        } else if (isFake) {
            progressBar.className = 'progress-bar bg-danger confidence-progress-bar';
        } else {
            progressBar.className = 'progress-bar bg-warning confidence-progress-bar';
        }
    }
    
    if (explanationElem) {
        explanationElem.textContent = data.explanation || 'NLP model analyzed semantic patterns and term frequency distributions.';
    }
    
    // Populate Real & Fake Indicators
    if (realKeywordsContainer && data.real_indicators) {
        realKeywordsContainer.innerHTML = '';
        data.real_indicators.forEach(kw => {
            const tag = document.createElement('span');
            tag.className = 'badge bg-success bg-opacity-20 text-success border border-success border-opacity-30 me-2 mb-2 p-2 fw-medium';
            tag.textContent = `+ ${kw}`;
            realKeywordsContainer.appendChild(tag);
        });
    }

    if (fakeKeywordsContainer && data.fake_indicators) {
        fakeKeywordsContainer.innerHTML = '';
        data.fake_indicators.forEach(kw => {
            const tag = document.createElement('span');
            tag.className = 'badge bg-danger bg-opacity-20 text-danger border border-danger border-opacity-30 me-2 mb-2 p-2 fw-medium';
            tag.textContent = `- ${kw}`;
            fakeKeywordsContainer.appendChild(tag);
        });
    }

    // Metadata metrics
    if (wordCountElem) wordCountElem.textContent = data.word_count || 0;
    if (charCountElem) charCountElem.textContent = data.char_count || 0;
    if (readingTimeElem) readingTimeElem.textContent = `${data.reading_time_mins || 1} min`;
    if (modelUsedElem) modelUsedElem.textContent = data.model_used || 'ML Classifier';
    
    resultCard.style.display = 'block';
    
    // Smooth scroll to result
    setTimeout(() => {
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

// History Item Deletion AJAX
async function deleteHistoryItem(predId) {
    if (!confirm('Are you sure you want to delete this prediction entry?')) return;
    
    try {
        const response = await fetch(`/api/history/${predId}/delete`, {
            method: 'DELETE'
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            const row = document.getElementById(`pred-row-${predId}`);
            if (row) {
                row.style.transition = 'all 0.4s ease';
                row.style.opacity = '0';
                setTimeout(() => row.remove(), 400);
            }
        } else {
            alert(data.message || 'Failed to delete entry.');
        }
    } catch (err) {
        console.error(err);
        alert('Network error while deleting entry.');
    }
}
