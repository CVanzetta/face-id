/**
 * Face Recognition Web Interface
 */

class FaceRecognitionApp {
    constructor() {
        this.apiBase = '';
        this.videoStream = null;
        this.isLiveSearchActive = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupRangeInputs();
        this.setupDragDrop();
        this.checkSystemHealth();
        this.loadPersonsList();
    }

    setupEventListeners() {
        // Search
        document.getElementById('searchFile').addEventListener('change', this.handleSearchFile.bind(this));
        document.getElementById('searchBtn').addEventListener('click', this.performSearch.bind(this));

        // Enroll
        document.getElementById('enrollForm').addEventListener('submit', this.handleEnrollForm.bind(this));
        document.getElementById('enrollFiles').addEventListener('change', this.handleEnrollFiles.bind(this));

        // Verify
        document.getElementById('verifyFile1').addEventListener('change', (e) => this.handleVerifyFile(e, 1));
        document.getElementById('verifyFile2').addEventListener('change', (e) => this.handleVerifyFile(e, 2));
        document.getElementById('verifyBtn').addEventListener('click', this.performVerify.bind(this));

        // Live search
        document.getElementById('startCameraBtn').addEventListener('click', this.startCamera.bind(this));
        document.getElementById('stopCameraBtn').addEventListener('click', this.stopCamera.bind(this));

        // Manage
        document.getElementById('refreshPersonsBtn').addEventListener('click', this.loadPersonsList.bind(this));
    }

    setupRangeInputs() {
        // Search top-k
        const topKRange = document.getElementById('searchTopK');
        const topKValue = document.getElementById('searchTopKValue');
        topKRange.addEventListener('input', () => {
            topKValue.textContent = topKRange.value;
        });

        // Search threshold
        const thresholdRange = document.getElementById('searchThreshold');
        const thresholdValue = document.getElementById('searchThresholdValue');
        thresholdRange.addEventListener('input', () => {
            thresholdValue.textContent = thresholdRange.value;
        });
    }

    setupDragDrop() {
        // Setup drag and drop for upload areas
        const uploadAreas = document.querySelectorAll('.upload-area');
        
        uploadAreas.forEach(area => {
            area.addEventListener('dragover', (e) => {
                e.preventDefault();
                area.classList.add('dragover');
            });

            area.addEventListener('dragleave', () => {
                area.classList.remove('dragover');
            });

            area.addEventListener('drop', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    // Determine which input to trigger based on the area
                    if (area.onclick.toString().includes('searchFile')) {
                        document.getElementById('searchFile').files = files;
                        this.handleSearchFile({ target: { files } });
                    } else if (area.onclick.toString().includes('enrollFiles')) {
                        document.getElementById('enrollFiles').files = files;
                        this.handleEnrollFiles({ target: { files } });
                    }
                }
            });
        });
    }

    async checkSystemHealth() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const health = await response.json();
            
            const statusElement = document.getElementById('systemStatus');
            if (health.status === 'healthy') {
                statusElement.innerHTML = '<i class="fas fa-circle text-success"></i> System Online';
            } else {
                statusElement.innerHTML = '<i class="fas fa-circle text-danger"></i> System Offline';
            }
        } catch (error) {
            console.error('Health check failed:', error);
            const statusElement = document.getElementById('systemStatus');
            statusElement.innerHTML = '<i class="fas fa-circle text-danger"></i> System Offline';
        }
    }

    // Search functionality
    handleSearchFile(event) {
        const file = event.target.files[0];
        if (file) {
            document.getElementById('searchBtn').disabled = false;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                const preview = `
                    <div class="mb-3">
                        <img src="${e.target.result}" class="img-fluid rounded" style="max-height: 200px;">
                        <p class="small text-muted mt-1">${file.name}</p>
                    </div>
                `;
                document.getElementById('searchResults').innerHTML = preview + '<p class="text-muted">Click "Search" to find matches</p>';
            };
            reader.readAsDataURL(file);
        }
    }

    async performSearch() {
        const fileInput = document.getElementById('searchFile');
        const file = fileInput.files[0];
        
        if (!file) return;

        const topK = document.getElementById('searchTopK').value;
        const threshold = document.getElementById('searchThreshold').value;

        this.showLoading('Searching for matches...');

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('top_k', topK);
            formData.append('threshold', threshold);

            const response = await fetch(`${this.apiBase}/search`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.displaySearchResults(result);

        } catch (error) {
            console.error('Search failed:', error);
            this.showError('Search failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displaySearchResults(result) {
        const resultsDiv = document.getElementById('searchResults');
        
        if (!result.success) {
            resultsDiv.innerHTML = `<div class="alert alert-danger">${result.message}</div>`;
            return;
        }

        if (result.matches.length === 0) {
            resultsDiv.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i> No matches found above the similarity threshold.
                </div>
            `;
            return;
        }

        let html = `<h6><i class="fas fa-users"></i> Found ${result.matches.length} matches:</h6>`;
        
        result.matches.forEach(match => {
            const similarity = Math.round(match.similarity_score * 100);
            html += `
                <div class="result-card card mb-2">
                    <div class="card-body p-3">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <h6 class="mb-1">${match.name}</h6>
                                <small class="text-muted">ID: ${match.person_id}</small>
                            </div>
                            <span class="badge bg-success">${similarity}%</span>
                        </div>
                        
                        <div class="similarity-bar mt-2 mb-2">
                            <div class="similarity-indicator" style="left: ${similarity}%"></div>
                        </div>
                        
                        <div class="small text-muted">
                            <i class="fas fa-tag"></i> ${match.metadata.source || 'Unknown source'}
                            ${match.metadata.tags && match.metadata.tags.length > 0 ? 
                                `| <i class="fas fa-tags"></i> ${match.metadata.tags.join(', ')}` : ''}
                        </div>
                    </div>
                </div>
            `;
        });

        resultsDiv.innerHTML = html;
    }

    // Enrollment functionality
    handleEnrollFiles(event) {
        const files = Array.from(event.target.files);
        const previewDiv = document.getElementById('enrollPreview');
        
        if (files.length === 0) return;

        let html = '<h6>Selected Images:</h6><div class="row">';
        files.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                html += `
                    <div class="col-4 mb-2">
                        <img src="${e.target.result}" class="img-fluid rounded" style="height: 80px; object-fit: cover;">
                    </div>
                `;
                if (index === files.length - 1) {
                    html += '</div>';
                    previewDiv.innerHTML = html;
                }
            };
            reader.readAsDataURL(file);
        });
    }

    async handleEnrollForm(event) {
        event.preventDefault();
        
        const formData = new FormData();
        formData.append('name', document.getElementById('enrollName').value);
        formData.append('source', document.getElementById('enrollSource').value);
        formData.append('tags', document.getElementById('enrollTags').value);
        formData.append('notes', document.getElementById('enrollNotes').value);
        formData.append('consent_type', 'explicit');

        const files = document.getElementById('enrollFiles').files;
        for (let file of files) {
            formData.append('files', file);
        }

        this.showLoading('Enrolling person...');

        try {
            const response = await fetch(`${this.apiBase}/enroll`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                this.showSuccess(`Successfully enrolled ${result.faces_enrolled} faces for ${document.getElementById('enrollName').value}`);
                document.getElementById('enrollForm').reset();
                document.getElementById('enrollPreview').innerHTML = '';
                this.loadPersonsList(); // Refresh persons list
            } else {
                this.showError(result.message);
            }

        } catch (error) {
            console.error('Enrollment failed:', error);
            this.showError('Enrollment failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    // Verification functionality
    handleVerifyFile(event, imageNumber) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const preview = `
                    <img src="${e.target.result}" class="img-fluid rounded mt-2" style="max-height: 150px;">
                    <p class="small text-muted mt-1">${file.name}</p>
                `;
                document.getElementById(`verifyPreview${imageNumber}`).innerHTML = preview;
            };
            reader.readAsDataURL(file);

            // Enable verify button if both images are selected
            const file1 = document.getElementById('verifyFile1').files[0];
            const file2 = document.getElementById('verifyFile2').files[0];
            document.getElementById('verifyBtn').disabled = !(file1 && file2);
        }
    }

    async performVerify() {
        const file1 = document.getElementById('verifyFile1').files[0];
        const file2 = document.getElementById('verifyFile2').files[0];

        if (!file1 || !file2) return;

        this.showLoading('Verifying faces...');

        try {
            const formData = new FormData();
            formData.append('file1', file1);
            formData.append('file2', file2);

            const response = await fetch(`${this.apiBase}/verify`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.displayVerifyResults(result);

        } catch (error) {
            console.error('Verification failed:', error);
            this.showError('Verification failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayVerifyResults(result) {
        const resultsDiv = document.getElementById('verifyResults');
        
        if (!result.success) {
            resultsDiv.innerHTML = `<div class="alert alert-danger">${result.message}</div>`;
            return;
        }

        const similarity = Math.round(result.similarity_score * 100);
        const matchClass = result.is_match ? 'success' : 'danger';
        const matchIcon = result.is_match ? 'check-circle' : 'times-circle';
        const matchText = result.is_match ? 'MATCH' : 'NO MATCH';

        const html = `
            <div class="alert alert-${matchClass} text-center">
                <i class="fas fa-${matchIcon} fa-3x mb-3"></i>
                <h4>${matchText}</h4>
                <p class="mb-0">Similarity: ${similarity}% | Confidence: ${Math.round(result.confidence * 100)}%</p>
                <small>Faces detected: Image 1 (${result.faces_detected.image1}), Image 2 (${result.faces_detected.image2})</small>
            </div>
        `;

        resultsDiv.innerHTML = html;
    }

    // Live search functionality
    async startCamera() {
        try {
            this.videoStream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 640, 
                    height: 480,
                    facingMode: 'user'
                } 
            });
            
            const video = document.getElementById('videoElement');
            video.srcObject = this.videoStream;

            document.getElementById('startCameraBtn').style.display = 'none';
            document.getElementById('stopCameraBtn').style.display = 'inline-block';

            this.isLiveSearchActive = true;
            this.startLiveProcessing();

        } catch (error) {
            console.error('Camera access failed:', error);
            this.showError('Camera access denied or not available');
        }
    }

    stopCamera() {
        if (this.videoStream) {
            this.videoStream.getTracks().forEach(track => track.stop());
            this.videoStream = null;
        }

        document.getElementById('startCameraBtn').style.display = 'inline-block';
        document.getElementById('stopCameraBtn').style.display = 'none';

        this.isLiveSearchActive = false;
        
        // Clear detections
        document.getElementById('liveDetections').innerHTML = '<p class="text-muted">Camera stopped</p>';
    }

    startLiveProcessing() {
        // This would need WebRTC or WebSocket for real-time processing
        // For now, just show a placeholder
        document.getElementById('liveDetections').innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i> 
                Live processing requires WebSocket connection for real-time face detection.
                This is a placeholder for the live functionality.
            </div>
        `;
    }

    // Persons management
    async loadPersonsList() {
        try {
            const response = await fetch(`${this.apiBase}/persons`);
            const persons = await response.json();
            this.displayPersonsList(persons);

        } catch (error) {
            console.error('Failed to load persons:', error);
            document.getElementById('personsTableBody').innerHTML = `
                <tr><td colspan="6" class="text-center text-danger">Failed to load persons</td></tr>
            `;
        }
    }

    displayPersonsList(persons) {
        const tbody = document.getElementById('personsTableBody');
        
        if (persons.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center">No persons enrolled</td></tr>';
            return;
        }

        let html = '';
        persons.forEach(person => {
            const enrollDate = new Date(person.enrollment_date).toLocaleDateString();
            const tags = person.tags.length > 0 ? person.tags.join(', ') : '-';
            
            html += `
                <tr>
                    <td><strong>${person.name}</strong></td>
                    <td><span class="badge bg-secondary">${person.source}</span></td>
                    <td>${person.faces_count}</td>
                    <td>${enrollDate}</td>
                    <td><small>${tags}</small></td>
                    <td>
                        <button class="btn btn-sm btn-outline-danger" onclick="app.deletePerson('${person.person_id}', '${person.name}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                </tr>
            `;
        });

        tbody.innerHTML = html;
    }

    async deletePerson(personId, name) {
        if (!confirm(`Are you sure you want to delete ${name}? This action cannot be undone.`)) {
            return;
        }

        this.showLoading('Deleting person...');

        try {
            const response = await fetch(`${this.apiBase}/person/${personId}`, {
                method: 'DELETE'
            });

            const result = await response.json();
            
            if (result.success) {
                this.showSuccess(`Successfully deleted ${name}`);
                this.loadPersonsList();
            } else {
                this.showError(result.message);
            }

        } catch (error) {
            console.error('Deletion failed:', error);
            this.showError('Deletion failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    // Utility functions
    showLoading(text = 'Processing...') {
        document.getElementById('loadingText').textContent = text;
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
    }

    hideLoading() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) modal.hide();
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showError(message) {
        this.showToast(message, 'danger');
    }

    showToast(message, type = 'info') {
        // Create toast element
        const toastHtml = `
            <div class="toast align-items-center text-white bg-${type} border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">${message}</div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;

        // Add to page
        let toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toastContainer';
            toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
            toastContainer.style.zIndex = '9999';
            document.body.appendChild(toastContainer);
        }

        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        
        // Show toast
        const toastElement = toastContainer.lastElementChild;
        const toast = new bootstrap.Toast(toastElement);
        toast.show();

        // Remove after hiding
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new FaceRecognitionApp();
});
