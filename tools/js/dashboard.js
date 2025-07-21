document.addEventListener('DOMContentLoaded', () => {
    // --- SHARED ELEMENTS & FUNCTIONS ---
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toast-message');

    function showToast(message, type = 'success') {
        let bgColor = 'bg-green-500';
        if (type === 'error') bgColor = 'bg-red-500';
        else if (type === 'info') bgColor = 'bg-blue-500';

        toastMessage.textContent = message;
        toast.className = `fixed bottom-5 right-5 text-white py-2 px-4 rounded-lg shadow-lg transition-opacity duration-300 ${bgColor}`;
        toast.classList.remove('hidden');
        setTimeout(() => { toast.classList.add('hidden'); }, 4000);
    }

    // --- TAB SWITCHING LOGIC ---
    const tabCuration = document.getElementById('tab-curation');
    const tabRunner = document.getElementById('tab-runner');
    const viewCuration = document.getElementById('view-curation');
    const viewRunner = document.getElementById('view-runner');
    const tabs = [tabCuration, tabRunner];
    const views = [viewCuration, viewRunner];

    function switchTab(activeTab) {
        tabs.forEach(tab => {
            const view = document.getElementById(tab.id.replace('tab-', 'view-'));
            if (tab === activeTab) {
                tab.classList.add('border-blue-500', 'text-white');
                tab.classList.remove('border-transparent', 'text-gray-400');
                view.classList.remove('hidden');
            } else {
                tab.classList.remove('border-blue-500', 'text-white');
                tab.classList.add('border-transparent', 'text-gray-400');
                view.classList.add('hidden');
            }
        });
    }
    tabCuration.addEventListener('click', () => switchTab(tabCuration));
    tabRunner.addEventListener('click', () => switchTab(tabRunner));


    // --- DATA CURATION LOGIC ---
    const dropZone = document.getElementById('drop-zone');
    const csvFileInput = document.getElementById('csv-file-input');
    const uploadSection = document.getElementById('upload-section');
    const dashboardSection = document.getElementById('dashboard-section');
    const fileNameDisplay = document.getElementById('file-name');
    const tableInfoDisplay = document.getElementById('table-info');
    const sampleViewBody = document.getElementById('sample-view-body');
    const exportBtn = document.getElementById('export-csv-btn');
    const mediaModal = document.getElementById('media-modal');
    const modalContent = document.getElementById('modal-content');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const paginationInfo = document.getElementById('pagination-info');
    const prevPageBtn = document.getElementById('prev-page-btn');
    const nextPageBtn = document.getElementById('next-page-btn');

    let tableData = [], tableHeaders = [], currentFileName = '', currentPage = 1;
    const rowsPerPage = 1;

    function handleCsvFile(file = null) {
        if (file && file.type === 'text/csv') {
            currentFileName = file.name;
            const reader = new FileReader();
            reader.onload = (e) => {
                parseAndDisplayCSV(e.target.result);
                uploadSection.classList.add('hidden');
                dashboardSection.classList.remove('hidden');
            };
            reader.readAsText(file);
        } else {
            showToast('Please upload a valid .csv file.', 'error');
        }
    }

    function parseCsvRow(row) {
        const result = []; let current = ''; let inQuote = false;
        for (let i = 0; i < row.length; i++) {
            const char = row[i];
            if (char === '"' && row[i + 1] === '"') { current += '"'; i++; }
            else if (char === '"') { inQuote = !inQuote; }
            else if (char === ',' && !inQuote) { result.push(current); current = ''; }
            else { current += char; }
        }
        result.push(current); return result;
    }

    function parseAndDisplayCSV(csvText) {
        const lines = csvText.trim().split(/\r?\n/);
        if (lines.length < 2) { showToast('CSV is empty or has no data.', 'error'); return; }
        tableHeaders = parseCsvRow(lines[0]);
        tableData = lines.slice(1).map(line => {
            if (line.trim() === '') return null;
            const values = parseCsvRow(line);
            const rowObject = {};
            tableHeaders.forEach((header, index) => { rowObject[header] = values[index] || ''; });
            rowObject['Quality Rating'] = 0;
            return rowObject;
        }).filter(Boolean);
        currentPage = 1;
        renderSampleView();
    }

    function parseMarkdown(text) {
        if (typeof text !== 'string') return '';
        return text.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\*(.*?)\*/g, '<em>$1</em>').replace(/`(.*?)`/g, '<code>$1</code>').replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>').replace(/^\s*# (.*)/gm, '<h1>$1</h1>').replace(/^\s*## (.*)/gm, '<h2>$2</h2>').replace(/^\s*### (.*)/gm, '<h3>$3</h3>').replace(/^\s*-\s(.*)/gm, '<ul><li>$1</li></ul>').replace(/<\/ul>\s*<ul>/gm, '').replace(/\\n/g, '<br>').replace(/\n/g, '<br>');
    }

    function renderSampleView() {
        sampleViewBody.innerHTML = '';
        if (!tableData || tableData.length === 0) return;
        const originalRowIndex = currentPage - 1;
        const rowData = tableData[originalRowIndex];
        if (!rowData) return;

        const sampleContainer = document.createElement('div');
        sampleContainer.className = 'space-y-4';
        tableHeaders.filter(h => h.toLowerCase() !== 'file_type').forEach(header => {
            const fieldContainer = document.createElement('div');
            fieldContainer.className = 'bg-gray-700/50 p-4 rounded-lg';
            const headerDiv = document.createElement('div');
            headerDiv.className = 'flex justify-between items-center mb-2';
            const label = document.createElement('h3');
            label.className = 'text-sm font-semibold text-gray-400 uppercase tracking-wider';
            label.textContent = header.replace(/_/g, ' ');
            headerDiv.appendChild(label);
            if (header === 'source_path') {
                const previewButton = createPreviewButton(rowData);
                if (previewButton) headerDiv.appendChild(previewButton);
            }
            fieldContainer.appendChild(headerDiv);
            const contentDiv = document.createElement('div');
            contentDiv.className = 'text-gray-200 editable-content min-h-[24px]';
            contentDiv.innerHTML = parseMarkdown(rowData[header] || '');
            contentDiv.contentEditable = true;
            contentDiv.dataset.rowIndex = originalRowIndex;
            contentDiv.dataset.header = header;
            contentDiv.addEventListener('blur', (e) => {
                const { rowIndex, header } = e.target.dataset;
                if (tableData[rowIndex][header] !== e.target.textContent) {
                    tableData[rowIndex][header] = e.target.textContent;
                    showToast('Change saved!', 'success');
                }
            });
            fieldContainer.appendChild(contentDiv);
            sampleContainer.appendChild(fieldContainer);
        });
        const ratingContainer = document.createElement('div');
        ratingContainer.className = 'bg-gray-700/50 p-4 rounded-lg flex items-center gap-4';
        const ratingLabel = document.createElement('h3');
        ratingLabel.className = 'text-sm font-semibold text-gray-400 uppercase tracking-wider';
        ratingLabel.textContent = 'Quality Rating';
        ratingContainer.appendChild(ratingLabel);
        ratingContainer.appendChild(createRatingStars(originalRowIndex, rowData['Quality Rating']));
        sampleContainer.appendChild(ratingContainer);
        sampleViewBody.appendChild(sampleContainer);
        fileNameDisplay.textContent = currentFileName;
        tableInfoDisplay.textContent = `${tableData.length} total rows`;
        renderPaginationControls();
    }

    function renderPaginationControls() {
        const totalPages = Math.ceil(tableData.length / rowsPerPage);
        paginationInfo.textContent = `Showing ${currentPage} of ${tableData.length} results`;
        prevPageBtn.disabled = currentPage === 1;
        nextPageBtn.disabled = currentPage === totalPages;
    }

    function createPreviewButton(rowData) {
        const fileType = rowData.file_type ? rowData.file_type.toLowerCase() : '';
        const mediaPath = rowData.source_path;
        let mediaType = '';

        if (fileType === 'image') {
            mediaType = 'image';
        } else if (['video', 'mer', 'au'].includes(fileType)) {
            mediaType = 'video';
        } else if (fileType === 'audio') {
            mediaType = 'audio';
        }

        if (mediaPath && mediaType) {
            const button = document.createElement('button');
            button.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M10 12a2 2 0 100-4 2 2 0 000 4z" /><path fill-rule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.022 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clip-rule="evenodd" /></svg>`;
            button.className = 'text-blue-400 hover:text-blue-300';
            button.onclick = () => openPreviewModal(mediaPath, mediaType);
            return button;
        }
        return null;
    }

    function openPreviewModal(path, type) {
        modalContent.innerHTML = '';
        let mediaElement;
        const fullSrcPath = path; // The path from the CSV is now the full path

        if (type === 'image') {
            mediaElement = document.createElement('img');
            mediaElement.className = 'max-w-full max-h-[80vh] mx-auto rounded';
            mediaElement.src = fullSrcPath;
            mediaElement.onerror = () => {
                mediaElement.alt = `Failed to load image: ${fullSrcPath}`;
                showToast(`Failed to load image: ${fullSrcPath}`, 'error');
            };
        } else if (type === 'video') {
            mediaElement = document.createElement('video');
            mediaElement.className = 'max-w-full max-h-[80vh] mx-auto rounded';
            mediaElement.controls = true;
            const sourceElement = document.createElement('source');
            sourceElement.src = fullSrcPath;
            sourceElement.onerror = () => {
                showToast(`Failed to load video: ${fullSrcPath}`, 'error');
            };
            mediaElement.appendChild(sourceElement);
        } else if (type === 'audio') {
            mediaElement = document.createElement('audio');
            mediaElement.className = 'w-full mx-auto';
            mediaElement.controls = true;
            const sourceElement = document.createElement('source');
            sourceElement.src = fullSrcPath;
            sourceElement.onerror = () => {
                showToast(`Failed to load audio: ${fullSrcPath}`, 'error');
            };
            mediaElement.appendChild(sourceElement);
        }

        if (mediaElement) {
            modalContent.appendChild(mediaElement);
            mediaModal.classList.remove('hidden');
        } else {
            showToast('Could not create media element for this file type.', 'error');
        }
    }

    function createRatingStars(rowIndex, currentRating) {
        const container = document.createElement('div');
        container.className = 'star-rating flex items-center';
        for (let i = 1; i <= 5; i++) {
            const star = document.createElement('span');
            star.className = 'star text-2xl';
            star.innerHTML = i <= currentRating ? '&#9733;' : '&#9734;';
            star.style.color = i <= currentRating ? '#f59e0b' : '#6b7280';
            star.dataset.rowIndex = rowIndex;
            star.dataset.ratingValue = i;
            star.addEventListener('click', (e) => {
                const { rowIndex, ratingValue } = e.currentTarget.dataset;
                tableData[rowIndex]['Quality Rating'] = parseInt(ratingValue, 10);
                renderSampleView();
                showToast(`Rated ${ratingValue} stars.`, 'success');
            });
            container.appendChild(star);
        }
        return container;
    }

    function exportToCSV() {
        const headersToExport = [...tableHeaders.filter(h => h.toLowerCase() !== 'file_type'), 'Quality Rating'];
        const csvRows = [headersToExport.join(',')];
        tableData.forEach(row => {
            const values = headersToExport.map(header => {
                let value = row[header] !== undefined ? String(row[header]) : '';
                if (value.includes(',') || value.includes('"') || value.includes('\n')) {
                    value = `"${value.replace(/"/g, '""')}"`;
                }
                return value;
            });
            csvRows.push(values.join(','));
        });
        const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `edited_${currentFileName}`;
        link.click();
        URL.revokeObjectURL(link.href);
        showToast('Exported as CSV', 'success');
    }

    dropZone.addEventListener('click', () => csvFileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('border-blue-500'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-blue-500'));
    dropZone.addEventListener('drop', (e) => { e.preventDefault(); if (e.dataTransfer.files.length) handleCsvFile(e.dataTransfer.files[0]); });
    csvFileInput.addEventListener('change', (e) => { if (e.target.files.length) handleCsvFile(e.target.files[0]); });
    exportBtn.addEventListener('click', exportToCSV);
    modalCloseBtn.addEventListener('click', () => mediaModal.classList.add('hidden'));
    prevPageBtn.addEventListener('click', () => { if (currentPage > 1) { currentPage--; renderSampleView(); } });
    nextPageBtn.addEventListener('click', () => { if (currentPage < Math.ceil(tableData.length / rowsPerPage)) { currentPage++; renderSampleView(); } });

    // --- PROMPT & RUN LOGIC ---
    const savePromptsBtn = document.getElementById('save-prompts-btn');
    const promptsContainer = document.getElementById('prompts-container');
    const generateCmdBtn = document.getElementById('generate-cmd-btn');
    const commandOutputContainer = document.getElementById('command-output-container');
    const commandOutput = document.getElementById('command-output');
    const copyCmdBtn = document.getElementById('copy-cmd-btn');
    const modelProviderSelect = document.getElementById('model-provider');
    const modelOptionPanels = document.querySelectorAll('.model-options-panel');
    const promptsFileSelect = document.getElementById('prompts-file-select');
    const runConfirmationSection = document.getElementById('run-confirmation-section');
    const confirmRunBtn = document.getElementById('confirm-run-btn');
    const cancelRunBtn = document.getElementById('cancel-run-btn');
    const stopRunBtn = document.getElementById('stop-run-btn');
    const runStatusSection = document.getElementById('run-status-section');

    function updateModelOptions(provider) {
        modelOptionPanels.forEach(panel => {
            if (panel.id === `${provider}-options`) {
                panel.classList.remove('hidden');
            } else {
                panel.classList.add('hidden');
            }
        });
    }

    modelProviderSelect.addEventListener('change', (e) => {
        updateModelOptions(e.target.value);
    });

    // Set initial state
    updateModelOptions(modelProviderSelect.value);

    function renderPrompts(prompts) {
        promptsContainer.innerHTML = '';
        const fragment = document.createDocumentFragment();
        function createPromptFields(obj, path = '') {
            for (const key in obj) {
                const currentPath = path ? `${path}.${key}` : key;
                if (typeof obj[key] === 'string') {
                    const wrapper = document.createElement('div');
                    wrapper.className = 'p-3 bg-gray-700/50 rounded-lg';
                    const label = document.createElement('label');
                    label.htmlFor = currentPath;
                    label.className = 'block text-xs font-medium text-gray-400 mb-1';
                    label.textContent = currentPath;
                    const textarea = document.createElement('textarea');
                    textarea.id = currentPath;
                    textarea.dataset.path = currentPath;
                    textarea.className = 'block w-full bg-gray-600 border border-gray-500 rounded-md py-2 px-3 text-gray-200 h-32 resize-y';
                    textarea.value = obj[key];
                    wrapper.append(label, textarea);
                    fragment.appendChild(wrapper);
                } else if (typeof obj[key] === 'object' && obj[key] !== null) {
                    createPromptFields(obj[key], currentPath);
                }
            }
        }
        createPromptFields(prompts);
        promptsContainer.appendChild(fragment);
    }

    async function saveEditedPrompts() {
        const newPrompts = {};
        promptsContainer.querySelectorAll('textarea[data-path]').forEach(textarea => {
            const path = textarea.dataset.path.split('.');
            let current = newPrompts;
            path.forEach((key, index) => {
                if (index === path.length - 1) { current[key] = textarea.value; }
                else { current[key] = current[key] || {}; current = current[key]; }
            });
        });

        try {
            const response = await fetch('/save-prompt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newPrompts)
            });
            const result = await response.json();
            if (result.success) {
                showToast(`Saved as ${result.filename}`, 'success');
                await loadPromptVersions(); // Refresh the dropdown
            } else {
                throw new Error(result.error || 'Failed to save prompts.');
            }
        } catch (error) {
            showToast(`Error saving prompts: ${error.message}`, 'error');
        }
    }

    function generateCommand() {
        const args = [];
        const inputPath = document.getElementById('input-path').value.trim();
        const outputDir = document.getElementById('output-dir').value.trim();
        if (!inputPath || !outputDir) {
            showToast('Input Path and Output Directory are required.', 'error');
            return;
        }
        args.push(`"${inputPath}"`, `"${outputDir}"`);

        const selectedPromptFile = promptsFileSelect.value;
        if (!selectedPromptFile) {
            showToast('Please select a prompts file.', 'error');
            return;
        }

        const currentProvider = document.getElementById('model-provider').value;

        const options = {
            '--type': document.getElementById('processing-type').value,
            '--task': document.getElementById('task-type').value,
            '--prompts-file': `utils/prompts/${selectedPromptFile}`,
            '--threshold': document.getElementById('threshold-input').value.trim(),
            '--peak_dis': document.getElementById('peak-dis-input').value.trim(),
            '--concurrency': document.getElementById('concurrency-input').value.trim(),
        };

        if (currentProvider === 'chatgpt') {
            const chatgptModel = document.getElementById('chatgpt-model').value.trim();
            if (chatgptModel) {
                options['--chatgpt-model'] = chatgptModel;
            }
        } else if (currentProvider === 'ollama') {
            const ollamaVisionModel = document.getElementById('ollama-vision-model').value.trim();
            const ollamaTextModel = document.getElementById('ollama-text-model').value.trim();
            if (ollamaVisionModel) {
                options['--ollama-vision-model'] = ollamaVisionModel;
            }
            if (ollamaTextModel) {
                options['--ollama-text-model'] = ollamaTextModel;
            }
        } else if (currentProvider === 'huggingface') {
            const huggingfaceModel = document.getElementById('huggingface-model').value.trim();
            if (huggingfaceModel) {
                options['--huggingface-model'] = huggingfaceModel;
            }
        }

        const defaults = {
            '--type': 'MER',
            '--task': 'MERR',
            '--threshold': '0.8',
            '--peak_dis': '15',
            '--concurrency': '4',
        };

        for (const [key, value] of Object.entries(options)) {
            if (value) {
                if (defaults[key] === undefined || value !== defaults[key]) {
                    args.push(`${key} "${value}"`);
                }
            }
        }

        if (document.getElementById('silent-mode').checked) args.push('--silent');
        if (document.getElementById('use-cache').checked) args.push('--cache');

        const command = `python main.py ${args.join(' ')}`;
        commandOutput.textContent = command;
        commandOutputContainer.classList.remove('hidden');
        runConfirmationSection.classList.remove('hidden'); // Show confirmation buttons
        runStatusSection.classList.add('hidden'); // Hide previous results
        showToast('Command generated! Please review and confirm to run.', 'info');
    }

    function executeCommand() {
        const command = commandOutput.textContent;
        if (!command) {
            showToast('No command to run.', 'error');
            return;
        }

        // Update button states
        confirmRunBtn.classList.add('hidden');
        cancelRunBtn.classList.add('hidden');
        stopRunBtn.classList.remove('hidden');

        runConfirmationSection.classList.remove('hidden'); // Keep visible for stop button
        runStatusSection.classList.remove('hidden'); // Show status area
        runStatusSection.innerHTML = `
                    <div class="bg-gray-900 border border-gray-600 rounded-lg p-4 mb-4">
                        <div class="flex items-center gap-3 text-gray-300 mb-3">
                            <svg class="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <span>Executing command... Please wait.</span>
                        </div>
                        <div id="command-live-output" class="bg-black text-green-300 p-3 rounded-md text-xs whitespace-pre-wrap font-mono max-h-96 overflow-y-auto"></div>
                    </div>`;

        const liveOutput = document.getElementById('command-live-output');

        fetch('/run-command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: command })
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                function readStream() {
                    return reader.read().then(({ done, value }) => {
                        if (done) {
                            // Command completed
                            resetButtonStates();
                            showCommandCompleted();
                            return;
                        }

                        const chunk = decoder.decode(value, { stream: true });

                        // Check for special error markers
                        if (chunk.includes('COMMAND_FAILED_WITH_CODE_')) {
                            const errorCode = chunk.match(/COMMAND_FAILED_WITH_CODE_(\d+)/)?.[1];
                            resetButtonStates();
                            showCommandFailed(`Command failed with exit code ${errorCode}`);
                            return;
                        } else if (chunk.includes('COMMAND_EXCEPTION_')) {
                            const error = chunk.replace('COMMAND_EXCEPTION_', '');
                            resetButtonStates();
                            showCommandFailed(`Command exception: ${error}`);
                            return;
                        }

                        // Append normal output
                        liveOutput.textContent += chunk;
                        liveOutput.scrollTop = liveOutput.scrollHeight;

                        return readStream();
                    });
                }

                return readStream();
            })
            .catch(error => {
                resetButtonStates();
                showCommandFailed(`Request failed: ${error.message}`);
            });
    }

    function stopCommand() {
        fetch('/stop-command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showToast('Command terminated successfully.', 'success');
                    resetButtonStates();
                    showCommandStopped();
                } else {
                    showToast(`Failed to stop command: ${data.error}`, 'error');
                }
            })
            .catch(error => {
                showToast(`Error stopping command: ${error.message}`, 'error');
            });
    }

    function resetButtonStates() {
        confirmRunBtn.classList.remove('hidden');
        cancelRunBtn.classList.remove('hidden');
        stopRunBtn.classList.add('hidden');
    }

    function showCommandCompleted() {
        runStatusSection.innerHTML = `
                <div class="bg-green-900/50 border border-green-500/50 text-white p-4 rounded-lg text-center">
                    <p class="font-semibold mb-3">✅ Command completed successfully!</p>
                    <p class="text-gray-300 mb-4">Would you like to export the results?</p>
                    <div class="flex justify-center gap-4">
                        <button id="export-results-btn" class="px-6 py-2 bg-blue-600 hover:bg-blue-500 font-semibold rounded-lg shadow transition-colors">Export Results</button>
                        <button id="dismiss-result-btn" class="px-6 py-2 bg-gray-600 hover:bg-gray-500 font-semibold rounded-lg shadow transition-colors">Dismiss</button>
                    </div>
                </div>`;

        document.getElementById('export-results-btn').addEventListener('click', () => {
            exportResults();
        });
        document.getElementById('dismiss-result-btn').addEventListener('click', () => {
            runStatusSection.classList.add('hidden');
            runConfirmationSection.classList.add('hidden');
        });
    }

    function showCommandFailed(errorMessage) {
        runStatusSection.innerHTML = `
                <div class="bg-red-900/50 border border-red-500/50 text-white p-4 rounded-lg">
                    <p class="font-semibold mb-2">❌ Command failed to execute.</p>
                    <p class="text-sm text-gray-300 mb-3">Reason:</p>
                    <pre class="bg-gray-900 text-red-300 p-3 rounded-md text-xs whitespace-pre-wrap font-mono">${errorMessage}</pre>
                    <div class="text-center mt-4">
                        <button id="dismiss-result-btn" class="px-6 py-2 bg-gray-600 hover:bg-gray-500 font-semibold rounded-lg shadow transition-colors">Dismiss</button>
                    </div>
                </div>`;

        document.getElementById('dismiss-result-btn').addEventListener('click', () => {
            runStatusSection.classList.add('hidden');
            runConfirmationSection.classList.add('hidden');
        });
    }

    function showCommandStopped() {
        runStatusSection.innerHTML = `
                <div class="bg-orange-900/50 border border-orange-500/50 text-white p-4 rounded-lg text-center">
                    <p class="font-semibold mb-3">⏹️ Command was terminated.</p>
                    <p class="text-gray-300 mb-4">The command execution was stopped by user request.</p>
                    <div class="text-center">
                        <button id="dismiss-result-btn" class="px-6 py-2 bg-gray-600 hover:bg-gray-500 font-semibold rounded-lg shadow transition-colors">Dismiss</button>
                    </div>
                </div>`;

        document.getElementById('dismiss-result-btn').addEventListener('click', () => {
            runStatusSection.classList.add('hidden');
            runConfirmationSection.classList.add('hidden');
        });
    }

    // --- PROMPT LOADING LOGIC ---
    function renderPromptUploadFallback() {
        promptsContainer.innerHTML = `
                <div class="text-center text-gray-400 p-8 bg-gray-800 rounded-lg">
                    <p class="mb-4">Could not automatically load <strong>prompts.json</strong>.</p>
                    <p class="mb-4">This can happen when running the HTML file directly from your computer due to browser security policies (CORS).</p>
                    <label class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold rounded-lg shadow transition-colors cursor-pointer">
                        Upload prompts.json
                        <input type="file" id="manual-prompt-upload" class="hidden" accept=".json">
                    </label>
                </div>
            `;
        document.getElementById('manual-prompt-upload').addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file && file.type === 'application/json') {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const prompts = JSON.parse(e.target.result);
                        renderPrompts(prompts);
                        showToast('Prompts loaded successfully!', 'success');
                    } catch (parseError) {
                        showToast('Error parsing the uploaded JSON file.', 'error');
                        console.error('JSON Parse Error:', parseError);
                    }
                };
                reader.readAsText(file);
            } else {
                showToast('Please select a valid .json file.', 'error');
            }
        });
    }

    async function loadPromptForEditing(filename) {
        if (!filename) {
            promptsContainer.innerHTML = '<p class="text-gray-400 text-center">No prompt file selected or found.</p>';
            return;
        }
        try {
            const response = await fetch(`prompts/${filename}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const promptsFromFile = await response.json();
            renderPrompts(promptsFromFile);
            showToast(`Loaded ${filename} for editing.`, 'info');
        } catch (error) {
            console.error(`Could not fetch ${filename}:`, error);
            showToast(`Could not load ${filename}.`, 'error');
            renderPromptUploadFallback();
        }
    }

    async function loadPromptVersions() {
        try {
            const response = await fetch('/list-prompts');
            const filenames = await response.json();
            promptsFileSelect.innerHTML = ''; // Clear existing options
            if (filenames.length === 0) {
                promptsFileSelect.innerHTML = '<option value="">No saved prompt versions found</option>';
                loadPromptForEditing(null); // Clear editor
            } else {
                filenames.forEach(name => {
                    const option = document.createElement('option');
                    option.value = name;
                    option.textContent = name;
                    promptsFileSelect.appendChild(option);
                });
                // Automatically load the first (latest) prompt for editing
                if (filenames[0]) {
                    await loadPromptForEditing(filenames[0]);
                }
            }
        } catch (error) {
            console.error('Could not fetch prompt versions:', error);
            showToast('Could not load prompt versions.', 'error');
        }
    }

    // Initial load
    loadPromptVersions();

    savePromptsBtn.addEventListener('click', saveEditedPrompts);
    generateCmdBtn.addEventListener('click', generateCommand);
    copyCmdBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(commandOutput.textContent).then(() => showToast('Command copied!', 'success'));
    });
    confirmRunBtn.addEventListener('click', executeCommand);
    stopRunBtn.addEventListener('click', stopCommand);
    cancelRunBtn.addEventListener('click', () => {
        runConfirmationSection.classList.add('hidden');
        showToast('Run cancelled.', 'info');
    });
    promptsFileSelect.addEventListener('change', (e) => {
        loadPromptForEditing(e.target.value);
    });

    function exportResults() {
        // Get export parameters from the form
        const outputFolder = document.getElementById('output-dir').value.trim() || 'output_temp';
        const processingType = document.getElementById('processing-type').value || 'MER';
        const exportPath = './'; // Default export path

        // Update UI to show export is in progress
        const exportBtn = document.getElementById('export-results-btn');
        const originalText = exportBtn.textContent;
        exportBtn.textContent = 'Exporting...';
        exportBtn.disabled = true;

        // Call backend export endpoint
        fetch('/export-results', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                output_folder: outputFolder,
                file_type: processingType.toLowerCase(),
                export_path: exportPath
            })
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                // Create export status display
                const exportStatusDiv = document.createElement('div');
                exportStatusDiv.className = 'mt-4 bg-gray-900 border border-gray-600 rounded-lg p-4';
                exportStatusDiv.innerHTML = `
                <div class="flex items-center gap-3 text-gray-300 mb-3">
                    <svg class="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Exporting results...</span>
                </div>
                <div id="export-live-output" class="bg-black text-green-300 p-3 rounded-md text-xs whitespace-pre-wrap font-mono max-h-96 overflow-y-auto"></div>
            `;
                runStatusSection.appendChild(exportStatusDiv);

                const liveOutput = document.getElementById('export-live-output');

                function readStream() {
                    return reader.read().then(({ done, value }) => {
                        if (done) {
                            // Export completed - reset UI
                            exportBtn.textContent = originalText;
                            exportBtn.disabled = false;
                            // Update the spinner section to show completion
                            const spinnerDiv = exportStatusDiv.querySelector('.flex.items-center.gap-3');
                            spinnerDiv.innerHTML = `
                                <svg class="h-5 w-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                                </svg>
                                <span class="text-green-400">Export completed successfully!</span>
                            `;
                            // Add auto-switch functionality
                            setTimeout(() => {
                                switchToDataCurationAndLoadCSV();
                            }, 1000);
                            return;
                        }

                        const chunk = decoder.decode(value, { stream: true });

                        // Check for special markers
                        if (chunk.includes('EXPORT_COMPLETED_SUCCESSFULLY')) {
                            showToast('Results exported successfully!', 'success');
                            // Continue processing to handle the stream completion
                        } else if (chunk.includes('EXPORT_FAILED_WITH_CODE_')) {
                            const errorCode = chunk.match(/EXPORT_FAILED_WITH_CODE_(\d+)/)?.[1];
                            exportBtn.textContent = originalText;
                            exportBtn.disabled = false;
                            showToast(`Export failed with code ${errorCode}`, 'error');
                            return;
                        } else if (chunk.includes('EXPORT_EXCEPTION_')) {
                            const error = chunk.replace('EXPORT_EXCEPTION_', '');
                            exportBtn.textContent = originalText;
                            exportBtn.disabled = false;
                            showToast(`Export error: ${error}`, 'error');
                            return;
                        }

                        // Append normal output (only if it's not just the success marker)
                        if (!chunk.includes('EXPORT_COMPLETED_SUCCESSFULLY')) {
                            liveOutput.textContent += chunk;
                            liveOutput.scrollTop = liveOutput.scrollHeight;
                        }

                        return readStream();
                    });
                }

                return readStream();
            })
            .catch(error => {
                exportBtn.textContent = originalText;
                exportBtn.disabled = false;
                showToast(`Export failed: ${error.message}`, 'error');
            });
    }

    // Add new function to handle auto-switch and CSV loading
    function switchToDataCurationAndLoadCSV() {
        try {
            // Switch to Data Curation tab
            switchTab(tabCuration);

            // Get the output directory and file type from the form
            const processingType = document.getElementById('processing-type').value || 'MER';

            // Construct the expected CSV file path to match the actual export output
            // The export script generates files like: {file_type}_export_data.csv
            const csvFileName = `${processingType.toLowerCase()}_export_data.csv`;
            const csvPath = `./${csvFileName}`; // Files are exported to the current directory

            // Show loading message
            showToast('Automatically loading the exported CSV file...', 'info');

            // Try to load the CSV file
            fetch(csvPath)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.text();
                })
                .then(csvText => {
                    // Set the filename for display
                    currentFileName = csvFileName;

                    // Parse and display the CSV
                    parseAndDisplayCSV(csvText);

                    // Show the dashboard section and hide upload section
                    uploadSection.classList.add('hidden');
                    dashboardSection.classList.remove('hidden');

                    showToast('CSV file has been automatically loaded into the Data Curation page!', 'success');
                })
                .catch(error => {
                    console.error('Failed to auto-load CSV:', error);
                    showToast('Unable to auto-load the CSV file, please upload it manually.', 'error');
                });

        } catch (error) {
            console.error('Error during auto-switch:', error);
            showToast('An error occurred during the auto-switch process.', 'error');
        }
    }
});
