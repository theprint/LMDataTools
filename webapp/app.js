let currentTool = null;
                    let activeWebSocket = null;
            
                    const startJobBtn = document.getElementById('start-job-btn');
                    // Settings Popover Logic
                    const settingsBtn = document.getElementById('settings-btn');
                    const settingsPopover = document.getElementById('settings-popover');
                    const closePopoverBtn = document.getElementById('close-popover-btn');
                    const saveSettingsBtn = document.getElementById('save-settings-btn');
                    const llmProviderSelect = document.getElementById('llm_provider');
                    const baseUrlInput = document.getElementById('base_url'); // Get base_url input
                    const apiKeyInput = document.getElementById('api_key'); // Get api_key input
                    const llmModelSelect = document.getElementById('llm_model');
                    const llmModelTextInput = document.getElementById('llm_model_text');
                    const llmModelGroup = document.getElementById('llm-model-group');
                    const huggingFaceApiKeyInput = document.getElementById('hugging_face_api_key'); // Get hugging_face_api_key input

                    if (startJobBtn) {
                        startJobBtn.style.display = 'none'; // Hide by default
                        startJobBtn.addEventListener('click', startJob);
                    }
            
                    // Store all settings, including API keys per provider
                    let allSettings = JSON.parse(localStorage.getItem('allSettings')) || {
                        providers: {
                            openai: { base_url: 'https://api.openai.com/v1', api_key: '', llm_model: '' },
                            gemini: { base_url: 'https://generativelanguage.googleapis.com/v1beta/openai/', api_key: '', llm_model: '' },
                            openrouter: { base_url: 'https://openrouter.ai/api/v1', api_key: '', llm_model: '' },
                            other: { base_url: 'http://host.docker.internal:11434', api_key: '', llm_model: '' }
                        },
                        currentProvider: 'openai'
                    };
            
                    function loadSettings() {
                        populateCustomProviders(); // Restore any custom providers into the dropdown
                        const currentProviderId = allSettings.currentProvider;
                        const providerSettings = allSettings.providers[currentProviderId];
            
                        llmProviderSelect.value = currentProviderId;
                        baseUrlInput.value = providerSettings.base_url;
                        apiKeyInput.value = providerSettings.api_key;
                        llmModelSelect.value = providerSettings.llm_model;
                        llmModelTextInput.value = providerSettings.llm_model;
                        huggingFaceApiKeyInput.value = allSettings.hugging_face_api_key || ''; // Hugging Face API key is global
                        
                        updateBaseUrlAndApiKeyFields(); // Update fields based on current provider
                        updateModelDropdown(); // Populate model dropdown
                    }
            
                    function saveSettings() {
                        const currentProviderId = llmProviderSelect.value;
                        allSettings.currentProvider = currentProviderId;
            
                        // Save current provider's settings (preserve display_name for custom providers)
                        const existingProvider = allSettings.providers[currentProviderId] || {};
                        allSettings.providers[currentProviderId] = {
                            ...existingProvider,
                            base_url: baseUrlInput.value.trim(),
                            api_key: apiKeyInput.value.trim(),
                            llm_model: llmModelSelect.style.display !== 'none' ? llmModelSelect.value : llmModelTextInput.value
                        };
                        // Save global Hugging Face API key
                        allSettings.hugging_face_api_key = huggingFaceApiKeyInput.value;
            
                        localStorage.setItem('allSettings', JSON.stringify(allSettings));
                        alert('Settings saved!');
                        settingsPopover.style.display = 'none';
                    }
            
                    function updateBaseUrlAndApiKeyFields() {
                        const currentProviderId = llmProviderSelect.value;
                        const providerSettings = allSettings.providers[currentProviderId];
            
                        // Update base_url and api_key fields
                        baseUrlInput.value = providerSettings.base_url;
                        apiKeyInput.value = providerSettings.api_key;
                        llmModelSelect.value = providerSettings.llm_model;
                        llmModelTextInput.value = providerSettings.llm_model;
            
                        // Show/hide base_url group based on provider
                        if (currentProviderId === 'openai' || currentProviderId === 'gemini' || currentProviderId === 'openrouter') {
                            baseUrlInput.setAttribute('readonly', true);
                            baseUrlInput.style.backgroundColor = '#1a1f3a';
                        } else {
                            baseUrlInput.removeAttribute('readonly');
                            baseUrlInput.style.backgroundColor = '#1e2742';
                        }
                    }
            
                    // Built-in provider IDs — custom providers are everything else
                    const builtinProviders = new Set(['openai', 'gemini', 'openrouter', 'other']);

                    function populateCustomProviders() {
                        // Remove any stale custom options first
                        Array.from(llmProviderSelect.options).forEach(opt => {
                            if (!builtinProviders.has(opt.value)) llmProviderSelect.removeChild(opt);
                        });
                        // Re-add all custom providers stored in settings
                        Object.keys(allSettings.providers).forEach(key => {
                            if (!builtinProviders.has(key)) {
                                const opt = document.createElement('option');
                                opt.value = key;
                                opt.textContent = allSettings.providers[key].display_name || key;
                                llmProviderSelect.appendChild(opt);
                            }
                        });
                    }

                    // Add-provider form elements
                    const addProviderBtn = document.getElementById('add-provider-btn');
                    const addProviderForm = document.getElementById('add-provider-form');
                    const confirmAddProviderBtn = document.getElementById('confirm-add-provider-btn');
                    const cancelAddProviderBtn = document.getElementById('cancel-add-provider-btn');
                    const newProviderNameInput = document.getElementById('new_provider_name');
                    const newProviderBaseUrlInput = document.getElementById('new_provider_base_url');
                    const addProviderError = document.getElementById('add-provider-error');

                    addProviderBtn.addEventListener('click', () => {
                        const isOpen = addProviderForm.style.display !== 'none';
                        addProviderForm.style.display = isOpen ? 'none' : 'block';
                        if (!isOpen) {
                            newProviderNameInput.value = '';
                            newProviderBaseUrlInput.value = '';
                            addProviderError.style.display = 'none';
                            newProviderNameInput.focus();
                        }
                    });

                    cancelAddProviderBtn.addEventListener('click', () => {
                        addProviderForm.style.display = 'none';
                    });

                    confirmAddProviderBtn.addEventListener('click', () => {
                        const displayName = newProviderNameInput.value.trim();
                        const baseUrl = newProviderBaseUrlInput.value.trim();

                        if (!displayName) {
                            addProviderError.textContent = 'Provider name is required.';
                            addProviderError.style.display = 'block';
                            return;
                        }

                        // Derive a stable key from the display name
                        const providerId = displayName.toLowerCase()
                            .replace(/[^a-z0-9]+/g, '_')
                            .replace(/^_+|_+$/g, '') || 'custom';

                        if (allSettings.providers[providerId]) {
                            addProviderError.textContent = `A provider with ID "${providerId}" already exists. Choose a different name.`;
                            addProviderError.style.display = 'block';
                            return;
                        }

                        // Store the new provider (same structure as built-ins, plus display_name)
                        allSettings.providers[providerId] = {
                            display_name: displayName,
                            base_url: baseUrl || 'http://localhost:1234/v1',
                            api_key: '',
                            llm_model: ''
                        };
                        allSettings.currentProvider = providerId;
                        localStorage.setItem('allSettings', JSON.stringify(allSettings));

                        // Add to dropdown and switch to it
                        const opt = document.createElement('option');
                        opt.value = providerId;
                        opt.textContent = displayName;
                        llmProviderSelect.appendChild(opt);
                        llmProviderSelect.value = providerId;

                        addProviderForm.style.display = 'none';
                        updateBaseUrlAndApiKeyFields();
                        updateModelDropdown();
                    });

                    llmProviderSelect.addEventListener('change', () => {
                        // Immediately persist the new provider selection so getConfig()
                        // always reads the correct provider, even if Save Settings isn't clicked.
                        allSettings.currentProvider = llmProviderSelect.value;
                        localStorage.setItem('allSettings', JSON.stringify(allSettings));
                        updateBaseUrlAndApiKeyFields();
                        updateModelDropdown(); // Update model dropdown when provider changes
                    });
            
                    baseUrlInput.addEventListener('change', updateModelDropdown); // Update model dropdown when base URL changes
            
                    settingsBtn.addEventListener('click', () => {
                        loadSettings(); // Load settings when opening the popover
                        loadGlobalPrefs();
                        loadPersonaList();
                        settingsPopover.style.display = 'flex';
                    });
            
                    closePopoverBtn.addEventListener('click', () => {
                        settingsPopover.style.display = 'none';
                    });
            
                    saveSettingsBtn.addEventListener('click', saveSettings);
            
                    window.addEventListener('click', (event) => {
                        if (event.target === settingsPopover) {
                            settingsPopover.style.display = 'none';
                        }
                    });
            
                                async function fetchModels(baseUrl, apiKey) {
                                    try {
                                        const provider = llmProviderSelect.value;
                                        const url = `/api/llm/models?provider=${encodeURIComponent(provider)}&base_url=${encodeURIComponent(baseUrl)}&api_key=${encodeURIComponent(apiKey || '')}`;
                                        const response = await fetch(url);
                                        if (!response.ok) { 
                                            const errorData = await response.json();
                                            // Use the detailed error from the backend if available
                                            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
                                        } 
                                        const data = await response.json();
                                        return data.models;
                                    } catch (error) {
                                        console.error('Error fetching models:', error);
                                        // Return the error message to be displayed
                                        return { error: error.message };
                                    }
                                }            
                    async function updateModelDropdown() {
                        const currentProviderId = llmProviderSelect.value;
                        const providerSettings = allSettings.providers[currentProviderId];
                        const baseUrl = providerSettings.base_url;
                        const apiKey = providerSettings.api_key;
            
                        const modelErrorDiv = document.getElementById('model-loading-error');
                        modelErrorDiv.style.display = 'none'; // Hide error initially

                        llmModelSelect.innerHTML = '<option value="">Loading models...</option>';
                        llmModelSelect.style.display = 'block';
                        llmModelTextInput.style.display = 'none';
            
                        const result = await fetchModels(baseUrl, apiKey);
            
                        if (result && !result.error && result.length > 0) {
                            const models = result;
                            llmModelSelect.innerHTML = '';
                            models.forEach(model => {
                                const option = document.createElement('option');
                                option.value = model;
                                option.textContent = model;
                                llmModelSelect.appendChild(option);
                            });
                            // Set selected model if previously saved
                            if (providerSettings.llm_model) {
                                llmModelSelect.value = providerSettings.llm_model;
                            }
                        } else {
                            // If there was an error, display it
                            if (result && result.error) {
                                modelErrorDiv.textContent = `Could not load models. Reason: ${result.error}`;
                                modelErrorDiv.style.display = 'block';
                            }
                            // Fallback to text input
                            llmModelSelect.style.display = 'none';
                            llmModelTextInput.style.display = 'block';
                            llmModelTextInput.value = providerSettings.llm_model || '';
                        }
                    }        
        // Tool button click handlers
        document.querySelectorAll('.tool-btn:not(#settings-btn)').forEach(btn => {
            btn.addEventListener('click', (event) => {
                // Update active state
                document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                startJobBtn.style.display = 'inline-block';

                // Get tool info
                const toolId = btn.dataset.tool;
                const toolName = btn.textContent;
                
                currentTool = {
                    id: toolId,
                    name: toolName
                };
                
                // Show config for selected tool
                showToolConfig(currentTool);
                validateForm();
            });
        });

        // Function to show the overview page
        async function showOverviewPage() {
            const configContent = document.getElementById('config-content');
            configContent.className = 'overview-container'; // Use a specific class for styling
            configContent.innerHTML = `
                <div class="tool-card-grid">
                    <div class="tool-card" data-tool="databird">
                        <h3>DataBird</h3>
                        <p>Generates procedural Q&A datasets from topics, using auto-generated or custom perspectives to create diverse conversational data.</p>
                    </div>
                    <div class="tool-card" data-tool="datapersona">
                        <h3>DataPersona</h3>
                        <p>Rewrites existing QA/Alpaca data sets from a unique perspective by infusing them with a selected persona.</p>
                    </div>
                    <div class="tool-card" data-tool="dataqa">
                        <h3>DataQA</h3>
                        <p>RAG-based tool that scrapes web content and transforms it into rated Q&A pairs.</p>
                    </div>
                    <div class="tool-card" data-tool="datawriter">
                        <h3>DataWriter</h3>
                        <p>Generates a mix of documents from a weighted list of topics, allowing control over document count and generation temperature.</p>
                    </div>
                    <div class="tool-card" data-tool="dataconvo">
                        <h3>DataConvo</h3>
                        <p>Expands single-turn conversations into multi-turn dialogues, with adjustable conversation lengths.</p>
                    </div>
                    <div class="tool-card" data-tool="datathink">
                        <h3>DataThink</h3>
                        <p>Enhances existing datasets by generating reasoning steps before responses, creating more thoughtful and detailed answers.</p>
                    </div>
                    <div class="tool-card" data-tool="datamix">
                        <h3>DataMix</h3>
                        <p>Mixes and samples from multiple Hugging Face datasets to create new, customized datasets based on specified criteria.</p>
                    </div>
                    <div class="tool-card" data-tool="reformat">
                        <h3>Reformat</h3>
                        <p>Converts a JSON dataset to a different standard format like Alpaca, ShareGPT, or simple Q&A pairs.</p>
                    </div>
                </div>
            `;

            // Add click listeners to the cards
            configContent.querySelectorAll('.tool-card').forEach(card => {
                card.addEventListener('click', () => {
                    const toolId = card.dataset.tool;
                    // Find the corresponding button and click it
                    const toolButton = document.querySelector(`.tool-btn[data-tool="${toolId}"]`);
                    if (toolButton) {
                        toolButton.click();
                    }
                });
            });

            currentTool = null;
            startJobBtn.style.display = 'none';
            document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
        }

        // Make app title clickable to return to overview
        document.getElementById('app-header').querySelector('h1').addEventListener('click', showOverviewPage);
        
        // Show configuration for selected tool
        async function showToolConfig(tool, skipRestore = false) {
            const configContent = document.getElementById('config-content');
            configContent.className = ''; // Remove empty-state class
            
            // Detach existing event listeners to avoid multiple triggers
            const newConfigContent = configContent.cloneNode(false);
            configContent.parentNode.replaceChild(newConfigContent, configContent);

            let formHTML = `<h2>Configure ${tool.name}</h2>`;
            
            // Common persona selector for relevant tools
            switch(tool.id) {
                case 'datapersona':
                    formHTML += `<div id="datapersona-form-content">Loading configuration...</div>`;
                    newConfigContent.innerHTML = formHTML;
                    buildDataPersonaForm();
                    break;
                
                case 'databird':
                    formHTML += `
                        <div class="form-group">
                            <label>Dataset Name</label>
                            <input type="text" id="dataset_name" value="my-dataset">
                        </div>
                        <div class="form-group">
                            <label>Topics (one per line)</label>
                            <textarea id="topics" rows="5" placeholder="sourdough starters&#10;baking at home&#10;ingredient ratios"></textarea>
                        </div>
                        <div class="form-group">
                            <div class="checkbox-group">
                                <input type="checkbox" id="full_auto" checked onchange="toggleDataBirdPerspectives()">
                                <label>Auto-generate perspectives</label>
                            </div>
                        </div>

                        <!-- Manual perspectives group (hidden when auto is enabled) -->
                        <div class="form-group" id="databird_manual_perspectives_group" style="display: none;">
                            <label>Manual Perspectives (one per line)</label>
                            <textarea id="manual_perspectives" rows="4" placeholder="a student who is researching the topic&#10;a home baker trying to improve their sourdough&#10;a curious beginner learning about the subject"></textarea>
                            <p style="color: #8a93a2; font-size: 12px; margin-top: 5px;">Enter one perspective per line as a plain description.</p>
                        </div>

                        <div class="form-group">
                            <label>Dataset Size</label>
                            <select id="dataset_size">
                                <option value="small">Small</option>
                                <option value="medium">Medium</option>
                                <option value="large">Large</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Clean Score Threshold</label>
                            <input type="number" step="0.01" id="clean_score" value="0.76">
                        </div>
                        <div class="form-group">
                            <div class="checkbox-group">
                                <input type="checkbox" id="include_reasoning">
                                <label>Include reasoning</label>
                            </div>
                        </div>
                    `;
                    formHTML += await buildPersonaSelector();
                    newConfigContent.innerHTML = formHTML;

                    // Toggle handler for DataBird manual perspectives
                    window.toggleDataBirdPerspectives = () => {
                        const autoCheckbox = document.getElementById('full_auto');
                        const manualGroup = document.getElementById('databird_manual_perspectives_group');
                        if (!manualGroup || !autoCheckbox) return;
                        manualGroup.style.display = autoCheckbox.checked ? 'none' : 'block';
                    };
                    // Ensure initial visibility matches the checkbox state
                    window.toggleDataBirdPerspectives();
                    break;
                
                case 'datawriter':
                // Default values matching the backend
                const defaultMin = 200;
                const defaultMax = 10000;
                const defaultTemp = 0.8;

                formHTML += `
                    <div class="form-group">
                        <label>Dataset Name</label>
                        <input type="text" id="dataset_name" value="my-writer-dataset">
                    </div>
                    <div class="form-group">
                        <label for="document_count">Document Count</label>
                        <input type="number" id="document_count" value="500">
                    </div>
                    <div class="form-group">
                        <label for="post-length-slider">Post Length: <span id="post-length-value">${defaultMin} - ${defaultMax}</span> tokens</label>
                        <div id="post-length-slider" style="margin-top: 10px;"></div>
                        <input type="hidden" id="min_tokens" value="${defaultMin}">
                        <input type="hidden" id="max_tokens" value="${defaultMax}">
                    </div>
                    <div class="form-group">
                        <label for="temperature">Temperature: <span id="temperature-value">${defaultTemp}</span></label>
                        <div class="slider-group">
                            <input type="range" id="temperature" min="0" max="1" step="0.01" value="${defaultTemp}" oninput="document.getElementById('temperature-value').textContent = this.value">
                            <div class="slider-labels">
                                <span>Deterministic</span>
                                <span>Creative</span>
                            </div>
                        </div>
                    </div>
                    <div class="form-group">
                        <div class="checkbox-group">
                            <input type="checkbox" id="add_summary">
                            <label for="add_summary">Add Summary</label>
                        </div>
                        <p style="color: #8a93a2; font-size: 12px; margin-top: 5px;">If checked, a summary will be generated for each document.</p>
                    </div>
                `;
                newConfigContent.innerHTML = formHTML;

                // Initialize noUiSlider for post length
                const postLengthSlider = document.getElementById('post-length-slider');
                const minTokensInput = document.getElementById('min_tokens');
                const maxTokensInput = document.getElementById('max_tokens');
                const postLengthValue = document.getElementById('post-length-value');

                noUiSlider.create(postLengthSlider, {
                    start: [defaultMin, defaultMax],
                    connect: true,
                    step: 100,
                    range: {
                        'min': 100,
                        'max': 20000
                    },
                    behaviour: 'drag',
                    format: {
                        to: function (value) {
                            return Math.round(value);
                        },
                        from: function (value) {
                            return Math.round(value);
                        }
                    }
                });

                // Use 'update' only for display updates (no validation)
                postLengthSlider.noUiSlider.on('update', function (values, handle) {
                    const [min, max] = values;
                    minTokensInput.value = min;
                    maxTokensInput.value = max;
                    postLengthValue.textContent = `${min} - ${max}`;
                });

                // Use 'change' for validation (only fires when user releases handle)
                postLengthSlider.noUiSlider.on('change', function (values, handle) {
                    validateForm();
                });
                
                // Initial validation
                validateForm();
                break;
                
                case 'dataqa':
                    formHTML += `
                        <div class="form-group">
                            <label>Dataset Name</label>
                            <input type="text" id="dataset_name" value="my-qa-dataset">
                        </div>
                        <div class="form-group">
                            <label>Source URLs (one per line)</label>
                            <textarea id="sources" rows="3" placeholder="https://example.com/page1 \nhttps://example.com/page2"></textarea>
                        </div>
                        <div class="form-group">
                            <label>OR Upload Local Files</label>
                            <input type="file" id="files" multiple accept=".txt,.html,.htm,.md">
                            <p style="color: #8a93a2; font-size: 12px; margin-top: 5px;">
                                Upload local files here instead of using file paths. You can select multiple files.
                            </p>
                        </div>
                        <div class="form-group">
                            <div class="checkbox-group">
                                <input type="checkbox" id="auto_perspectives" checked onchange="toggleDataQAPerspectives()">
                                <label>Auto-generate perspectives</label>
                            </div>
                        </div>
                        <div class="form-group" id="manual_perspectives_group" style="display: none;">
                            <label>Manual Perspectives (one per line)</label>
                            <textarea id="manual_perspectives" rows="4" placeholder="a student who is researching the topic&#10;an expert looking for advanced insights&#10;a curious beginner learning about the subject"></textarea>
                            <p style="color: #8a93a2; font-size: 12px; margin-top: 5px;">Enter one perspective per line as a plain description.</p>
                        </div>
                        <div class="form-group">
                            <label>Confidence Threshold</label>
                            <input type="number" step="0.01" id="confidence_threshold" value="0.68">
                        </div>
                    `;
                    formHTML += await buildPersonaSelector();
                    newConfigContent.innerHTML = formHTML;

                    const fileInput = newConfigContent.querySelector('#files');
                    if (fileInput) {
                        fileInput.addEventListener('change', validateForm);
}

                    // Add the toggle function to the window scope
                    window.toggleDataQAPerspectives = () => {
                        const autoPerspectivesCheckbox = document.getElementById('auto_perspectives');
                        const manualPerspectivesGroup = document.getElementById('manual_perspectives_group');
                        if (autoPerspectivesCheckbox.checked) {
                            manualPerspectivesGroup.style.display = 'none';
                        } else {
                            manualPerspectivesGroup.style.display = 'block';
                        }
                    };
                    break;
                
                case 'datamix':
                    // Default values for sliders
                    const defaultMinInstruction = 10;
                    const defaultMaxInstruction = 4000;
                    const defaultMinOutput = 10;
                    const defaultMaxOutput = 4000;

                    formHTML += `
                        <div class="form-group">
                            <label>Dataset Name</label>
                            <input type="text" id="dataset_name" value="mixed-dataset">
                        </div>
                        <div class="form-group">
                            <label>Total Samples</label>
                            <input type="number" id="total_samples" value="10000">
                        </div>
                        <div class="form-group">
                            <label>Random Seed (for reproducibility)</label>
                            <input type="number" id="seed" value="310576">
                        </div>
                        
                        <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(0, 212, 255, 0.1);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <h3 style="color: #00d4ff; font-size: 16px; margin: 0;">Dataset Sources</h3>
                                <button type="button" class="btn-small" style="background: #00d4ff; color: #0a0e27;" onclick="addDatasetRow()">+ Add Dataset</button>
                            </div>
                            
                            <div id="weight-total-display" style="padding: 10px; background: #1a1f3a; border-radius: 6px; margin-bottom: 15px; text-align: center; font-size: 14px; color: #8a93a2;">
                                Total Weight: <span id="weight-total" style="color: #00d4ff; font-weight: 600;">0.00</span> / 1.00
                            </div>
                            
                            <div id="datasets-container">
                                <!-- Dataset rows will be added here -->
                            </div>
                        </div>
                        
                        <div class="form-group" style="margin-top: 20px;">
                            <label>Quality Filters</label>
                            
                            <div style="margin-top: 15px;">
                                <label for="instruction-length-slider">Instruction Length: <span id="instruction-length-value">${defaultMinInstruction} - ${defaultMaxInstruction}</span></label>
                                <div id="instruction-length-slider" style="margin-top: 10px;"></div>
                                <input type="hidden" id="min_instruction_length" value="${defaultMinInstruction}">
                                <input type="hidden" id="max_instruction_length" value="${defaultMaxInstruction}">
                            </div>
                            
                            <div style="margin-top: 25px;">
                                <label for="output-length-slider">Output Length: <span id="output-length-value">${defaultMinOutput} - ${defaultMaxOutput}</span></label>
                                <div id="output-length-slider" style="margin-top: 10px;"></div>
                                <input type="hidden" id="min_output_length" value="${defaultMinOutput}">
                                <input type="hidden" id="max_output_length" value="${defaultMaxOutput}">
                            </div>
                        </div>
                    `;
                    newConfigContent.innerHTML = formHTML;
                    
                    // Initialize instruction length slider
                    const instructionSlider = document.getElementById('instruction-length-slider');
                    const minInstructionInput = document.getElementById('min_instruction_length');
                    const maxInstructionInput = document.getElementById('max_instruction_length');
                    const instructionValue = document.getElementById('instruction-length-value');

                    noUiSlider.create(instructionSlider, {
                        start: [defaultMinInstruction, defaultMaxInstruction],
                        connect: true,
                        step: 10,
                        range: {
                            'min': 10,
                            'max': 10000
                        },
                        behaviour: 'drag',
                        format: {
                            to: function (value) {
                                return Math.round(value);
                            },
                            from: function (value) {
                                return Math.round(value);
                            }
                        }
                    });

                    instructionSlider.noUiSlider.on('update', function (values, handle) {
                        const [min, max] = values;
                        minInstructionInput.value = min;
                        maxInstructionInput.value = max;
                        instructionValue.textContent = `${min} - ${max}`;
                    });

                    instructionSlider.noUiSlider.on('change', function (values, handle) {
                        validateForm();
                    });

                    // Initialize output length slider
                    const outputSlider = document.getElementById('output-length-slider');
                    const minOutputInput = document.getElementById('min_output_length');
                    const maxOutputInput = document.getElementById('max_output_length');
                    const outputValue = document.getElementById('output-length-value');

                    noUiSlider.create(outputSlider, {
                        start: [defaultMinOutput, defaultMaxOutput],
                        connect: true,
                        step: 10,
                        range: {
                            'min': 10,
                            'max': 10000
                        },
                        behaviour: 'drag',
                        format: {
                            to: function (value) {
                                return Math.round(value);
                            },
                            from: function (value) {
                                return Math.round(value);
                            }
                        }
                    });

                    outputSlider.noUiSlider.on('update', function (values, handle) {
                        const [min, max] = values;
                        minOutputInput.value = min;
                        maxOutputInput.value = max;
                        outputValue.textContent = `${min} - ${max}`;
                    });

                    outputSlider.noUiSlider.on('change', function (values, handle) {
                        validateForm();
                    });
                    
                    // Initialize with one dataset row
                    window.datasetRowCounter = 0;
                    addDatasetRow();
                    
                    validateForm();
                    break;
                case 'dataconvo':
                    formHTML += `
                        <div class="form-group">
                            <label>Dataset Name</label>
                            <input type="text" id="dataset_name" value="my-convo-dataset">
                        </div>
                        <div class="form-group">
                            <label>Source File (.json or .jsonl)</label>
                            <input type="file" id="file" accept=".json,.jsonl">
                        </div>
                        <div class="form-group">
                            <label>Conversation Length Distribution</label>
                            <p style="font-size: 13px; color: #8a93a2; margin-bottom: 10px;">Adjust the balance of generated conversation lengths.</p>
                            <div class="slider-group">
                                <input type="range" id="round_weights_slider" min="0" max="100" value="50">
                                <div class="slider-labels">
                                    <span>Shorter (1-2 rounds)</span>
                                    <span>Balanced Mix</span>
                                    <span>Longer (2-3 rounds)</span>
                                </div>
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Save Interval</label>
                            <input type="number" id="save_interval" value="100">
                        </div>
                    `;
                    formHTML += await buildPersonaSelector();
                    newConfigContent.innerHTML = formHTML;
                    // Add event listener for the slider to update a hidden input or a variable
                    const slider = document.getElementById('round_weights_slider');
                    slider.addEventListener('input', () => {
                        // The value will be read directly when getConfig() is called.
                        // No need for a hidden input.
                    });
                    break;

                case 'reformat':
                    formHTML += `
                        <div class="form-group">
                            <label>Dataset Name</label>
                            <input type="text" id="dataset_name" value="reformatted-dataset">
                        </div>
                        <div class="form-group">
                            <label>Source File (.json or .jsonl)</label>
                            <input type="file" id="file" accept=".json,.jsonl">
                        </div>
                        <div class="form-group">
                            <label>Target Format</label>
                            <select id="target_format">
                                <option value="alpaca">Alpaca</option>
                                <option value="sharegpt">ShareGPT</option>
                                <option value="qa">Q&A (question/answer)</option>
                            </select>
                        </div>
                    `;
                    newConfigContent.innerHTML = formHTML;
                    break;

                case 'datathink':
                    formHTML += `
                        <div class="form-group">
                            <label>Dataset Name</label>
                            <input type="text" id="dataset_name" value="my-thinking-dataset">
                        </div>
                        <div class="form-group">
                            <label>Mode</label>
                            <select id="think_mode">
                                <option value="insert_reasoning" selected>Insert Reasoning — keep original answer</option>
                                <option value="generate_new">Generate New — questions only, build from scratch</option>
                            </select>
                            <p style="color: #8a93a2; font-size: 12px; margin-top: 5px;">
                                <strong>Insert Reasoning:</strong> adds a &lt;think&gt; block to each entry; original answer is kept unchanged.<br>
                                <strong>Generate New:</strong> upload questions only — reasoning and answer are generated from scratch.
                            </p>
                        </div>
                        <div class="form-group">
                            <label>Source File (.json or .jsonl)</label>
                            <input type="file" id="file" accept=".json,.jsonl">
                            <p style="color: #8a93a2; font-size: 12px; margin-top: 5px;">Supports Alpaca, ShareGPT, and Q&A formats</p>
                        </div>
                        <div class="form-group">
                            <label>Reasoning Depth</label>
                            <select id="reasoning_level">
                                <option value="low">Low - Brief approach outline</option>
                                <option value="medium" selected>Medium - Approach + challenges</option>
                                <option value="high">High - Full analysis with edge cases</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="thinking_temperature">Thinking Temperature: <span id="thinking-temp-value">0.7</span></label>
                            <div class="slider-group">
                                <input type="range" id="thinking_temperature" min="0" max="1" step="0.01" value="0.7" oninput="document.getElementById('thinking-temp-value').textContent = this.value">
                                <div class="slider-labels">
                                    <span>Focused</span>
                                    <span>Creative</span>
                                </div>
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="response_temperature">Response Temperature: <span id="response-temp-value">0.7</span></label>
                            <div class="slider-group">
                                <input type="range" id="response_temperature" min="0" max="1" step="0.01" value="0.7" oninput="document.getElementById('response-temp-value').textContent = this.value">
                                <div class="slider-labels">
                                    <span>Precise</span>
                                    <span>Expressive</span>
                                </div>
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Save Interval</label>
                            <input type="number" id="save_interval" value="50">
                        </div>
                    `;
                    formHTML += await buildPersonaSelector();
                    newConfigContent.innerHTML = formHTML;
                    break;
            }

            // Add event listeners for validation and form-state persistence
            newConfigContent.querySelectorAll('input, textarea, select').forEach(input => {
                input.addEventListener('input', () => { validateForm(); saveFormState(tool.id); });
                input.addEventListener('change', () => { validateForm(); saveFormState(tool.id); });
            });
            // Also attach to persona selector if it exists
            const personaSelector = newConfigContent.querySelector('#use_persona');
            if (personaSelector) {
                personaSelector.addEventListener('change', () => { validateForm(); saveFormState(tool.id); });
            }
            const postLengthSlider = newConfigContent.querySelector('#post-length-slider');
            if (postLengthSlider && postLengthSlider.noUiSlider) {
                postLengthSlider.noUiSlider.on('change', () => { validateForm(); saveFormState(tool.id); });
            }

            // Restore persisted state + add Reset button after form DOM is fully ready.
            // datapersona builds async and handles its own restore inside buildDataPersonaForm().
            if (tool.id !== 'datapersona') {
                setTimeout(() => {
                    if (!skipRestore) restoreFormState(tool.id);
                    addFormFooter(tool.id);
                    validateForm();
                }, 0);
            }
        }
        
        // Build DataPersona form dynamically
        async function buildDataPersonaForm() {
            try {
                const [defaultsRes, personasRes] = await Promise.all([
                    fetch('/api/defaults/datapersona'),
                    fetch('/api/personas')
                ]);

                if (!defaultsRes.ok || !personasRes.ok) throw new Error('Failed to fetch form data');

                const defaults = await defaultsRes.json();
                const personasData = await personasRes.json();
                
                const personaOptions = personasData.personas.map(p =>
                    `<option value="${p}" ${p === defaults.PERSONA ? 'selected' : ''}>${p}</option>`
                ).join('');

                const defaultPersonaDescription = 'Loading description...';

                const formContent = ` 
                    <div class="form-group">
                        <label>Dataset Name</label>
                        <input type="text" id="dataset_name" value="${defaults.DATASET_NAME || 'my-persona-dataset'}">
                    </div>
                    <div class="form-group">
                        <label>Persona</label>
                        <div id="persona-description" style="color: #8a93a2; font-size: 13px; margin-bottom: 10px; padding: 10px; border: 1px solid rgba(0, 212, 255, 0.1); border-radius: 4px; background: #1a1f3a;">
                            ${defaultPersonaDescription}
                        </div>
                        <select id="persona">${personaOptions}</select>
                    </div>
                    <div class="form-group">
                        <label>Source Files (.json)</label>
                        <input type="file" id="files" multiple accept=".json">
                    </div>
                    <div class="form-group">
                        <label>Number of Replies</label>
                        <div class="checkbox-group" style="gap: 20px;">
                            <div class="checkbox-group">
                                <input type="radio" id="num_replies_1" name="num_replies" value="1" checked>
                                <label for="num_replies_1">1</label>
                            </div>
                            <div class="checkbox-group">
                                <input type="radio" id="num_replies_2" name="num_replies" value="2">
                                <label for="num_replies_2">2</label>
                            </div>
                        </div>
                    </div>                    
                    <div class="form-group">
                        <label>Save Interval</label>
                        <input type="number" id="save_interval" value="${defaults.SAVE_INTERVAL}">
                    </div>
                    <div class="form-group">
                        <div class="checkbox-group">
                            <input type="checkbox" id="export_alpaca" ${defaults.EXPORT_ALPACA ? 'checked' : ''}>
                            <label>Export Alpaca format (from best replies)</label>
                        </div>
                    </div>
                `;
                const formContainer = document.getElementById('datapersona-form-content');
                formContainer.innerHTML = formContent;

                // Add event listener to update description on change
                const personaSelect = formContainer.querySelector('#persona');
                const descriptionDiv = formContainer.querySelector('#persona-description');
                async function updateDataPersonaDescription() {
                    const selectedPersona = personaSelect.value;
                    if (!selectedPersona) {
                        descriptionDiv.textContent = 'Select a persona to see its description.';
                        return;
                    }
                    try {
                        const response = await fetch(`/api/persona/${encodeURIComponent(selectedPersona)}`);
                        if (!response.ok) throw new Error('Could not load description.');
                        const data = await response.json();
                        descriptionDiv.textContent = data.description || 'No description available.';
                    } catch (error) {
                        descriptionDiv.textContent = error.message;
                    }
                }

                personaSelect.addEventListener('change', updateDataPersonaDescription);

                // Initial load of the description for the default persona
                updateDataPersonaDescription();
                
                // Add event listeners after form is built
                formContainer.querySelectorAll('input, textarea, select').forEach(input => {
                    input.addEventListener('input', () => { validateForm(); saveFormState('datapersona'); });
                    input.addEventListener('change', () => { validateForm(); saveFormState('datapersona'); });
                });

                // Restore persisted state and add Reset button
                restoreFormState('datapersona', formContainer);
                addFormFooter('datapersona');
                validateForm(); // Initial validation
            } catch (error) {
                document.getElementById('datapersona-form-content').innerHTML = 
                    `<div class="error-message">Error loading configuration: ${error.message}</div>`;
            }
        }

        // Build a generic persona selector component
        async function buildPersonaSelector() {
            try {
                const response = await fetch('/api/personas');
                if (!response.ok) throw new Error('Failed to fetch personas');
                const data = await response.json();
                const personas = data.personas; // This is now an array of strings

                const personaOptions = personas.map(p => `<option value="${p}">${p}</option>`).join('');

                return `
                    <div class="form-group" style="margin-top: 25px; padding-top: 20px; border-top: 1px solid rgba(0, 212, 255, 0.1);">
                        <div class="checkbox-group">
                            <input type="checkbox" id="use_persona" onchange="togglePersonaDropdown(this.checked)">
                            <label for="use_persona">Use Persona</label>
                        </div>
                    </div>
                    <div id="persona-selection-group" style="display: none;">
                        <div class="form-group">
                            <label for="persona_name">Choose Persona</label>
                            <select id="persona_name" onchange="updateGenericPersonaDescription()">${personaOptions}</select>
                        </div>
                    </div>
                `;
            } catch (error) {
                console.error("Error building persona selector:", error);
                return `<div class="error-message">Could not load personas.</div>`;
            }
        }

        // Function to toggle the persona dropdown
        function togglePersonaDropdown(enabled) {
            const group = document.getElementById('persona-selection-group');
            const dropdown = document.getElementById('persona_name');
            if (group && dropdown) {
                if (enabled) {
                    group.style.display = 'block';
                    dropdown.disabled = false;
                } else {
                    group.style.display = 'none';
                    dropdown.disabled = true;
                }
                // Also update the description when the dropdown is toggled
                if (enabled) {
                    updateGenericPersonaDescription();
                }
            }
        }

        // Function to update the description for the generic persona selector
        async function updateGenericPersonaDescription() {
            const select = document.getElementById('persona_name');
            const descriptionDiv = document.getElementById('generic-persona-description');
            if (!select || !descriptionDiv) return;

            const personaName = select.value;
            if (!personaName) {
                descriptionDiv.textContent = 'Select a persona to see its description.';
                return;
            }

            try {
                const response = await fetch(`/api/persona/${encodeURIComponent(personaName)}`);
                if (!response.ok) throw new Error('Could not load description.');
                const data = await response.json();
                descriptionDiv.textContent = data.description || 'No description available.';
            } catch (error) {
                descriptionDiv.textContent = error.message;
            }
        }

        // ── Form-state persistence ────────────────────────────────────────────

        function saveFormState(toolId) {
            const container = document.getElementById('config-content');
            if (!container || !toolId) return;
            const state = {};

            // Standard inputs (skip file and hidden)
            container.querySelectorAll(
                'input:not([type="file"]):not([type="hidden"]), textarea, select'
            ).forEach(el => {
                const key = el.id || el.name;
                if (!key) return;
                if (el.type === 'radio') {
                    if (el.checked) state['__radio__' + el.name] = el.value;
                } else if (el.type === 'checkbox') {
                    state[key] = el.checked;
                } else {
                    state[key] = el.value;
                }
            });

            // noUiSlider positions (save by element ID)
            container.querySelectorAll('[id$="-slider"]').forEach(el => {
                if (el.noUiSlider) state['__slider__' + el.id] = el.noUiSlider.get();
            });

            // DataMix dataset rows
            const rows = [];
            container.querySelectorAll('.dataset-row').forEach(row => {
                rows.push({
                    name:   row.querySelector('.dataset-name')?.value   || '',
                    weight: row.querySelector('.dataset-weight')?.value || '',
                    subset: row.querySelector('.dataset-subset')?.value || '',
                    format: row.querySelector('.dataset-format')?.value || ''
                });
            });
            if (rows.length) state['__datamix_rows__'] = rows;

            localStorage.setItem('formState_' + toolId, JSON.stringify(state));
        }

        function restoreFormState(toolId, container) {
            const saved = localStorage.getItem('formState_' + toolId);
            if (!saved) return;
            let state;
            try { state = JSON.parse(saved); } catch (e) { return; }

            container = container || document.getElementById('config-content');
            if (!container) return;

            // Radio buttons
            container.querySelectorAll('input[type="radio"]').forEach(el => {
                const savedVal = state['__radio__' + el.name];
                if (savedVal !== undefined) el.checked = (el.value === savedVal);
            });

            // All other inputs (text, number, range, checkbox, textarea, select)
            container.querySelectorAll(
                'input:not([type="file"]):not([type="hidden"]):not([type="radio"]), textarea, select'
            ).forEach(el => {
                const key = el.id || el.name;
                if (!key || state[key] === undefined) return;
                if (el.type === 'checkbox') {
                    el.checked = Boolean(state[key]);
                } else {
                    el.value = state[key];
                }
            });

            // Trigger oninput display-update handlers on range inputs (e.g. temp sliders)
            container.querySelectorAll('input[type="range"]').forEach(el => {
                el.dispatchEvent(new Event('input'));
            });

            // noUiSliders
            container.querySelectorAll('[id$="-slider"]').forEach(el => {
                const saved = state['__slider__' + el.id];
                if (el.noUiSlider && saved !== undefined) el.noUiSlider.set(saved);
            });

            // DataMix dataset rows
            if (state['__datamix_rows__'] && typeof window.addDatasetRow === 'function') {
                const rowContainer = document.getElementById('datasets-container');
                if (rowContainer) {
                    rowContainer.innerHTML = '';
                    window.datasetRowCounter = 0;
                    state['__datamix_rows__'].forEach(rowData => {
                        window.addDatasetRow();
                        const allRows = rowContainer.querySelectorAll('.dataset-row');
                        const last = allRows[allRows.length - 1];
                        if (last) {
                            const n = last.querySelector('.dataset-name');   if (n) n.value = rowData.name;
                            const w = last.querySelector('.dataset-weight'); if (w) w.value = rowData.weight;
                            const s = last.querySelector('.dataset-subset'); if (s) s.value = rowData.subset;
                            const f = last.querySelector('.dataset-format'); if (f) f.value = rowData.format;
                        }
                    });
                    if (typeof window.updateWeightTotal === 'function') window.updateWeightTotal();
                }
            }

            // Re-trigger conditional visibility toggles
            if (typeof window.toggleDataBirdPerspectives === 'function') window.toggleDataBirdPerspectives();
            if (typeof window.toggleDataQAPerspectives   === 'function') window.toggleDataQAPerspectives();
            if (typeof window.togglePersonaDropdown === 'function') {
                const cb = container.querySelector('#use_persona');
                if (cb) window.togglePersonaDropdown(cb.checked);
            }
        }

        function addFormFooter(toolId) {
            // Append to .config-section (the grid container) so the footer
            // sits in the second grid row — outside the scroll area entirely.
            const section = document.querySelector('.config-section');
            if (!section) return;
            section.querySelector(':scope > .form-footer')?.remove();
            const footer = document.createElement('div');
            footer.className = 'form-footer';
            const btn = document.createElement('button');
            btn.className = 'btn-reset';
            btn.textContent = 'Reset';
            btn.title = 'Reset all fields to their default values';
            btn.addEventListener('click', () => {
                localStorage.removeItem('formState_' + toolId);
                showToolConfig(currentTool, true);
            });
            footer.appendChild(btn);
            section.appendChild(footer);
        }

        // ── End form-state persistence ────────────────────────────────────────

        // Validate form and enable/disable start button
        function validateForm() {
            if (!currentTool) {
                startJobBtn.disabled = true; // Still disable, but it's hidden by showOverviewPage
                return;
            }

            const config = getConfig(true); // silent=true: no console output during routine validation
            let isValid = true;

            switch(currentTool.id) {
                case 'databird':
                    isValid = config.dataset_name && config.topics && config.topics.length > 0;
                    break;
                case 'datapersona':
                    const dp_files = document.getElementById('files')?.files;
                    isValid = config.dataset_name && dp_files && dp_files.length > 0;
                    break;
                case 'dataqa':
                    const files = document.getElementById('files')?.files;
                    const hasFiles = files && files.length > 0;
                    const hasSources = config.sources && config.sources.length > 0;
                    
                    isValid = config.dataset_name && (hasSources || hasFiles);
                    
                    if (!config.auto_perspectives) {
                        isValid = isValid && config.manual_perspectives && config.manual_perspectives.length > 0;
                    }
                    break;
                case 'datawriter':
                    // Validate min < max for datawriter (parse as numbers!)
                    const minTokens = parseFloat(config.min_tokens);
                    const maxTokens = parseFloat(config.max_tokens);
                    if (isNaN(minTokens) || isNaN(maxTokens) || minTokens >= maxTokens) {
                        isValid = false;
                    }
                    isValid = isValid && config.dataset_name && config.document_count > 0;
                    break;
                case 'datamix':
                    isValid = config.dataset_name && config.total_samples > 0;
                    break;
                case 'dataconvo':
                    const convoFile = document.getElementById('file').files;
                    isValid = config.dataset_name && convoFile && convoFile.length > 0;
                    break;
                case 'reformat':
                    const reformatFile = document.getElementById('file').files;
                    isValid = config.dataset_name && reformatFile && reformatFile.length > 0;
                    break;
                case 'datathink':
                    const thinkFile = document.getElementById('file').files;
                    isValid = config.dataset_name && thinkFile && thinkFile.length > 0;
                    break;
                default:
                    isValid = false;
            }

            startJobBtn.disabled = !isValid;
        }

        // Add these functions right after the validateForm() function (around line 800):

        // DataMix helper functions
        window.addDatasetRow = function() {
            const container = document.getElementById('datasets-container');
            if (!container) return;
            
            const rowId = window.datasetRowCounter++;
            
            const rowHTML = `
                <div class="dataset-row" id="dataset-row-${rowId}" style="background: #1e2742; border-radius: 8px; padding: 15px; margin-bottom: 12px; border: 2px solid rgba(0, 212, 255, 0.1); transition: all 0.2s;">
                    <div style="display: grid; grid-template-columns: 2fr 1fr 1.5fr 1fr auto; gap: 10px; align-items: end;">
                        <div>
                            <label style="display: block; margin-bottom: 6px; font-weight: 500; color: #c4cdd5; font-size: 12px;">Dataset Name</label>
                            <input type="text" class="dataset-name" placeholder="username/dataset-name" style="width: 100%; padding: 10px 12px; border: 2px solid rgba(0, 212, 255, 0.2); border-radius: 6px; font-size: 14px; transition: all 0.2s; font-family: inherit; background: #252d47; color: #e4e7eb;">
                        </div>
                        <div>
                            <label style="display: block; margin-bottom: 6px; font-weight: 500; color: #c4cdd5; font-size: 12px;">Weight</label>
                            <input type="number" class="dataset-weight" step="0.01" min="0" max="1" value="0.10" onchange="updateWeightTotal()" style="width: 100%; padding: 10px 12px; border: 2px solid rgba(0, 212, 255, 0.2); border-radius: 6px; font-size: 14px; transition: all 0.2s; font-family: inherit; background: #252d47; color: #e4e7eb;">
                        </div>
                        <div>
                            <label style="display: block; margin-bottom: 6px; font-weight: 500; color: #c4cdd5; font-size: 12px;">Subset (optional)</label>
                            <input type="text" class="dataset-subset" placeholder="train, test, etc." style="width: 100%; padding: 10px 12px; border: 2px solid rgba(0, 212, 255, 0.2); border-radius: 6px; font-size: 14px; transition: all 0.2s; font-family: inherit; background: #252d47; color: #e4e7eb;">
                        </div>
                        <div>
                            <label style="display: block; margin-bottom: 6px; font-weight: 500; color: #c4cdd5; font-size: 12px;">Format</label>
                            <select class="dataset-format" style="width: 100%; padding: 10px 12px; border: 2px solid rgba(0, 212, 255, 0.2); border-radius: 6px; font-size: 14px; transition: all 0.2s; font-family: inherit; background: #252d47; color: #e4e7eb; cursor: pointer;">
                                <option value="auto">Auto-detect</option>
                                <optgroup label="Common">
                                    <option value="alpaca">Alpaca (instruction / input / output)</option>
                                    <option value="sharegpt">ShareGPT (conversations)</option>
                                    <option value="qa">Q&amp;A (question / answer)</option>
                                </optgroup>
                                <optgroup label="Instruction variants">
                                    <option value="instruction_response">instruction / response</option>
                                    <option value="instruction_output">instruction / output</option>
                                    <option value="instr_chosen_resp">instruction / chosen_response</option>
                                    <option value="instr_demonstration">instruction / demonstration</option>
                                    <option value="info_summary">instruction / info / summary</option>
                                </optgroup>
                                <optgroup label="Capitalised keys">
                                    <option value="cap_instruction_response">INSTRUCTION / RESPONSE</option>
                                    <option value="cap_context_response">Context / Response</option>
                                    <option value="cap_human_assistant">Human / Assistant</option>
                                </optgroup>
                                <optgroup label="Problem variants">
                                    <option value="problem_answer">problem / answer</option>
                                    <option value="problem_description_response">problem-description / response</option>
                                    <option value="problem_gold_standard">problem / gold_standard_solution</option>
                                </optgroup>
                                <optgroup label="Prompt variants">
                                    <option value="prompt_response">prompt / response</option>
                                    <option value="prompt_chosen">prompt / chosen</option>
                                    <option value="prompt_question">prompt / question</option>
                                </optgroup>
                                <optgroup label="Query / Question variants">
                                    <option value="query_answer">query / answer</option>
                                    <option value="question_response">question / response</option>
                                    <option value="question_cot">question / cot</option>
                                    <option value="question_solution">question / choices / solution</option>
                                </optgroup>
                                <optgroup label="Generic">
                                    <option value="input_output">input / output</option>
                                </optgroup>
                            </select>
                        </div>
                        <div>
                            <button type="button" class="btn-small btn-delete" onclick="removeDatasetRow(${rowId})" style="margin: 0;">Remove</button>
                        </div>
                    </div>
                </div>
            `;
            
            container.insertAdjacentHTML('beforeend', rowHTML);
            
            // Add focus styles to the newly added inputs
            const newRow = document.getElementById(`dataset-row-${rowId}`);
            newRow.querySelectorAll('input, select').forEach(input => {
                input.addEventListener('focus', function() {
                    this.style.borderColor = '#00d4ff';
                    this.style.background = '#2a3447';
                    this.style.boxShadow = '0 0 0 3px rgba(0, 212, 255, 0.1)';
                });
                input.addEventListener('blur', function() {
                    this.style.borderColor = 'rgba(0, 212, 255, 0.2)';
                    this.style.background = '#252d47';
                    this.style.boxShadow = 'none';
                });
                input.addEventListener('input', validateForm);
                input.addEventListener('change', validateForm);
            });
            
            // Add hover effect to row
            newRow.addEventListener('mouseenter', function() {
                this.style.borderColor = 'rgba(0, 212, 255, 0.3)';
                this.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.3)';
            });
            newRow.addEventListener('mouseleave', function() {
                this.style.borderColor = 'rgba(0, 212, 255, 0.1)';
                this.style.boxShadow = 'none';
            });
            
            updateWeightTotal();
            validateForm();
        };

        window.removeDatasetRow = function(rowId) {
            const row = document.getElementById(`dataset-row-${rowId}`);
            if (row) {
                row.remove();
                updateWeightTotal();
                validateForm();
            }
        };

        window.updateWeightTotal = function() {
            const weights = Array.from(document.querySelectorAll('.dataset-weight'))
                .map(input => parseFloat(input.value) || 0);
            
            const total = weights.reduce((sum, w) => sum + w, 0);
            const totalDisplay = document.getElementById('weight-total');
            
            if (totalDisplay) {
                totalDisplay.textContent = total.toFixed(2);
                
                // Color code based on whether total is close to 1.0
                if (Math.abs(total - 1.0) < 0.01) {
                    totalDisplay.style.color = '#00ff88'; // Green
                } else if (Math.abs(total - 1.0) < 0.1) {
                    totalDisplay.style.color = '#ffd93d'; // Yellow
                } else {
                    totalDisplay.style.color = '#ff4757'; // Red
                }
            }
        };
        
        // Collect config from form
        function getConfig(silent = false) {
            // Helper for DataConvo slider
            function getRoundWeights(sliderValue) {
                const value = parseInt(sliderValue, 10);
                let r1, r2, r3;
                if (value <= 50) {
                    // From 100/0/0 to 25/50/25
                    const p = value / 50;
                    r1 = 100 - 75 * p;
                    r2 = 50 * p;
                    r3 = 25 * p;
                } else {
                    // From 25/50/25 to 0/0/100
                    const p = (value - 50) / 50;
                    r1 = 25 - 25 * p;
                    r2 = 50 - 50 * p;
                    r3 = 25 + 75 * p;
                }
                return { "rounds_1": Math.round(r1), "rounds_2": Math.round(r2), "rounds_3": Math.round(r3) };
            }
            
            const config = {};
            const configSection = document.querySelector('.config-section');
            
            configSection.querySelectorAll('input, textarea, select').forEach(input => {
                const id = input.id;
                if (!id) return;
                
                if (input.type === 'checkbox') {
                    config[id] = input.checked;
                } else if (input.type === 'number') {
                    config[id] = parseFloat(input.value);
                } else if (id === 'dataset_size') {
                    const size = input.value;
                    config['dataset_size'] = size;
                    config['extended_auto'] = (size === 'medium' || size === 'large');
                    config['extra_extended_auto'] = (size === 'large');
                } else if (id === 'manual_perspectives') {
                    // Parse plain-sentence perspectives, one per line.
                    // This MUST come before the generic TEXTAREA handler.
                    config[id] = input.value.split('\n').filter(line => line.trim() !== '').map(line => line.trim());
                } else if (input.tagName === 'TEXTAREA') {
                    config[id] = input.value.split('\n').filter(line => line.trim());
                } else if (id === 'round_weights_slider') {
                    config['round_weights'] = getRoundWeights(input.value);
                } else {
                    config[id] = input.value;
                }
            });

            // Handle generic persona selector
            const usePersonaCheckbox = configSection.querySelector('#use_persona');
            if (usePersonaCheckbox) {
                config['use_persona'] = usePersonaCheckbox.checked;
                config['persona_name'] = configSection.querySelector('#persona_name').value;
            }

            // Special handling for DataMix - collect dataset sources
            if (currentTool && currentTool.id === 'datamix') {
                const datasetRows = document.querySelectorAll('.dataset-row');
                const dataset_sources = [];
                
                datasetRows.forEach(row => {
                    const name = row.querySelector('.dataset-name').value.trim();
                    const weight = parseFloat(row.querySelector('.dataset-weight').value) || 0;
                    const subset = row.querySelector('.dataset-subset').value.trim() || null;
                    const format = row.querySelector('.dataset-format').value;
                    
                    if (name && weight > 0) {
                        dataset_sources.push({
                            name: name,
                            weight: weight,
                            subset: subset,
                            format: format === 'auto' ? null : format
                        });
                    }
                });
                
                config['dataset_sources'] = dataset_sources;
            }

            // Add LLM settings — read from localStorage (written by saveSettings())
            const settings = JSON.parse(localStorage.getItem('allSettings')) || {
                providers: {
                    openai: { base_url: 'https://api.openai.com/v1', api_key: '', llm_model: ''},
                    gemini: { base_url: 'https://generativelanguage.googleapis.com/v1beta/openai/', api_key: '', llm_model: ''},
                    openrouter: { base_url: 'https://openrouter.ai/api/v1', api_key: '', llm_model: ''},
                    other: { base_url: 'http://localhost:11434', api_key: '', llm_model: ''},
                },
                currentProvider: 'openai'
            };
            const currentProviderId = settings.currentProvider || 'openai';
            const providerSettings = (settings.providers || {})[currentProviderId] || {};
            const apiKey = (providerSettings.api_key || '').trim();
            const baseUrl = (providerSettings.base_url || '').trim();

            config['llm_settings'] = {
                llm_provider: currentProviderId,
                base_url: baseUrl,
                api_key: apiKey,
                llm_model: providerSettings.llm_model || '',
                hugging_face_api_key: settings.hugging_face_api_key || ''
            };

            // ── Diagnostic: open DevTools (F12) → Console to see this before each job ──
            if (!silent) {
                const _ks = apiKey.length >= 12
                    ? (apiKey.slice(0, 4) + '...' + apiKey.slice(-8))
                    : (apiKey.length >= 4 ? ('...' + apiKey.slice(-4)) : (apiKey || '(empty — check Settings!)'));
                console.log(
                    `%c[LMDataTools] LLM → provider="${currentProviderId}"  api_key=${_ks}  model="${providerSettings.llm_model || 'not set'}"  base_url="${providerSettings.base_url || 'not set'}"`,
                    'color:#4fc; font-weight:bold'
                );
                if (!apiKey) {
                    console.warn('[LMDataTools] api_key is EMPTY — job will fall back to server .env key. Open Settings, select the correct provider, enter your key, and click Save Settings.');
                }
            }

            return config;
        }
        // Start a job
        async function startJob() {
            if (!currentTool) {
                alert('Please select a tool first');
                return;
            }
            
            try {
                let response;
                const config = getConfig();

                if (currentTool.id === 'datapersona') {
                    const formData = new FormData();
                    const files = document.getElementById('files').files;

                    if (files.length === 0) {
                        alert('Please select at least one file to upload.');
                        return;
                    }

                    for (const file of files) {
                        formData.append('files', file);
                    }

                    const numReplies = document.querySelector('input[name="num_replies"]:checked').value;

                    // Append other form fields from config
                    formData.append('persona', config.persona);
                    formData.append('generate_reply_1', numReplies >= 1);
                    formData.append('generate_reply_2', numReplies == 2);
                    formData.append('save_interval', config.save_interval);
                    formData.append('export_alpaca', config.export_alpaca);
                    formData.append('dataset_name', config.dataset_name);
                    formData.append('llm_settings', JSON.stringify(config.llm_settings));

                    response = await fetch(`/api/jobs/datapersona`, {
                        method: 'POST',
                        body: formData
                        // No 'Content-Type' header for multipart/form-data, browser sets it
                    });
                } else if (currentTool.id === 'dataqa') {
                    const formData = new FormData();
                    const files = document.getElementById('files')?.files || [];

                    // Add files if any
                    for (const file of files) {
                        formData.append('files', file);
                    }

                    // Append other form fields
                    formData.append('dataset_name', config.dataset_name);
                    formData.append('sources', (config.sources || []).join('\n'));
                    formData.append('auto_perspectives', config.auto_perspectives || true);
                    formData.append('confidence_threshold', config.confidence_threshold || 0.68);
                    formData.append('use_persona', config.use_persona || false);
                    formData.append('persona_name', config.persona_name || '');
                    formData.append('manual_perspectives', config.manual_perspectives ? JSON.stringify(config.manual_perspectives) : '');
                    formData.append('llm_settings', JSON.stringify(config.llm_settings));

                    response = await fetch(`/api/jobs/dataqa`, {
                        method: 'POST',
                        body: formData
                    });
                } else if (currentTool.id === 'dataconvo') {
                    const formData = new FormData();
                    const file = document.getElementById('file').files[0];

                    if (!file) {
                        alert('Please select a file to upload.');
                        return;
                    }
                    formData.append('file', file);
                    formData.append('dataset_name', config.dataset_name);
                    formData.append('save_interval', config.save_interval);
                    formData.append('round_weights', JSON.stringify(config.round_weights));
                    formData.append('use_persona', config.use_persona);
                    formData.append('persona_name', config.persona_name || '');
                    formData.append('llm_settings', JSON.stringify(config.llm_settings));

                    response = await fetch(`/api/jobs/dataconvo`, {
                        method: 'POST',
                        body: formData
                    });
                } else if (currentTool.id === 'reformat') {
                    const formData = new FormData();
                    const file = document.getElementById('file').files[0];

                    if (!file) {
                        alert('Please select a file to upload.');
                        return;
                    }
                    formData.append('file', file);
                    formData.append('dataset_name', config.dataset_name);
                    formData.append('target_format', config.target_format);
                    formData.append('llm_settings', JSON.stringify(config.llm_settings));

                    response = await fetch(`/api/jobs/reformat`, {
                        method: 'POST',
                        body: formData
                    });
                } else if (currentTool.id === 'datathink') {
                    const formData = new FormData();
                    const file = document.getElementById('file').files[0];

                    if (!file) {
                        alert('Please select a file to upload.');
                        return;
                    }
                    formData.append('file', file);
                    formData.append('dataset_name', config.dataset_name);
                    formData.append('save_interval', config.save_interval);
                    formData.append('thinking_temperature', config.thinking_temperature);
                    formData.append('response_temperature', config.response_temperature);
                    formData.append('reasoning_level', config.reasoning_level);
                    formData.append('think_mode', config.think_mode || 'insert_reasoning');
                    formData.append('use_persona', config.use_persona);
                    formData.append('persona_name', config.persona_name || '');
                    formData.append('llm_settings', JSON.stringify(config.llm_settings));

                    response = await fetch(`/api/jobs/datathink`, {
                        method: 'POST',
                        body: formData
                    });
                } else {
                    // Diagnostic: confirm exactly what api_key is in the payload before sending
                    const _ls = config.llm_settings || {};
                    const _ak = _ls.api_key || '';
                    console.log(
                        `%c[LMDataTools] JSON payload llm_settings: provider="${_ls.llm_provider}"  api_key_len=${_ak.length}  api_key=${_ak.length >= 12 ? (_ak.slice(0,4)+'...'+_ak.slice(-8)) : (_ak || '(EMPTY)')}`,
                        'color:#f90; font-weight:bold'
                    );
                    response = await fetch(`/api/jobs/${currentTool.id}`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(config)
                    });
                }

                if (!response.ok) {
                    let errorMessage = `HTTP error! status: ${response.status}`;
                    try {
                        const errorData = await response.json();
                        console.error('Full error response:', errorData);
                        
                        // Handle Pydantic validation errors
                        if (errorData.detail) {
                            if (Array.isArray(errorData.detail)) {
                                // Pydantic validation errors are arrays
                                errorMessage = errorData.detail.map(err => {
                                    const location = err.loc ? err.loc.join('.') : 'unknown';
                                    return `${location}: ${err.msg}`;
                                }).join('\n');
                            } else if (typeof errorData.detail === 'string') {
                                errorMessage = errorData.detail;
                            } else {
                                errorMessage = JSON.stringify(errorData.detail, null, 2);
                            }
                        } else {
                            errorMessage = JSON.stringify(errorData, null, 2);
                        }
                    } catch (e) {
                        console.error('Failed to parse error:', e);
                        try {
                            const errorText = await response.text();
                            errorMessage = errorText || errorMessage;
                        } catch (e2) {
                            console.error('Could not get error text');
                        }
                    }
                    throw new Error(errorMessage);
                }
                
                const data = await response.json();
                monitorJob(data.job_id);
                loadJobs();
            } catch (error) {
                console.error('Error:', error);
                alert(`Error starting job: ${error.message}`);
            }
        }
        
        // Monitor job via WebSocket
        function monitorJob(jobId) {
            // Close existing WebSocket if any
            if (activeWebSocket) {
                activeWebSocket.close();
            }
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            activeWebSocket = new WebSocket(`${protocol}//${window.location.host}/ws/${jobId}`);
            
            activeWebSocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateJobDisplay(data);

                // Close WebSocket when job is done
                if (data.status === 'completed' || data.status === 'failed') {
                    setTimeout(() => {
                        if (activeWebSocket) {
                            activeWebSocket.close();
                            activeWebSocket = null;
                        }
                    }, 1000);
                }
            };

            activeWebSocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            activeWebSocket.onclose = () => {
                activeWebSocket = null;
            };
        }
        
        // Update job display
        function updateJobDisplay(job) {
            const jobsList = document.getElementById('jobs-list');
            let jobDiv = document.getElementById(`job-${job.job_id}`);

            // Remove empty state if present
            const emptyState = jobsList.querySelector('.empty-jobs');
            if (emptyState) {
                emptyState.remove();
            }

            if (!jobDiv) {
                jobDiv = document.createElement('div');
                jobDiv.id = `job-${job.job_id}`;
                jobDiv.dataset.createdAt = job.created_at;

                // Insert at correct position (newest first)
                const existingJobs = Array.from(jobsList.children);
                const insertBefore = existingJobs.find(el => el.dataset.createdAt < job.created_at);
                jobsList.insertBefore(jobDiv, insertBefore || null);
            }

            jobDiv.className = `job-item status-${job.status}`;

            const statusClass = job.status === 'running'    ? 'running'   :
                              job.status === 'completed'  ? 'completed' :
                              job.status === 'failed'     ? 'failed'    :
                              job.status === 'cancelled'  ? 'failed'    : '';

            // Build the token-usage badge (shown on completed jobs when available)
            let tokenBadge = '';
            if (job.token_usage && job.token_usage.total_tokens > 0) {
                const total = job.token_usage.total_tokens.toLocaleString();
                tokenBadge = `<span class="token-badge" title="Prompt: ${job.token_usage.prompt_tokens.toLocaleString()} / Completion: ${job.token_usage.completion_tokens.toLocaleString()}">⚡ ${total} tokens</span>`;
            }

            let actionsHTML = '';
            let errorHTML = '';
            if (job.status === 'completed') {
                actionsHTML = `
                    <button class="btn-small btn-download" onclick="downloadJob('${job.job_id}')">Download</button>
                `;
            }
            if (job.status === 'running' || job.status === 'starting') {
                actionsHTML = `
                    <button class="btn-small btn-cancel" onclick="cancelJob('${job.job_id}')">Cancel</button>
                `;
            }
            if (job.status === 'failed' || job.status === 'cancelled') {
                actionsHTML = `
                    <button class="btn-small btn-resume" onclick="resumeJob('${job.job_id}')">Resume</button>
                `;
                const msg = job.status === 'cancelled'
                    ? 'Job was cancelled. Resume will continue from the last checkpoint.'
                    : 'Job failed. Resume will continue from the last checkpoint.';
                errorHTML = `<div class="error-message">${msg}</div>`;
            }

            jobDiv.innerHTML = `
                <div class="job-header">
                    <span class="job-title">${job.dataset_name} (${job.tool})</span>
                    <div class="job-header-right">
                        ${tokenBadge}
                        <span class="job-status ${statusClass}">${job.status}</span>
                    </div>
                </div>
                <div class="job-id">${job.job_id}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${job.progress || 0}%"></div>
                </div>
                <div class="progress-text">${job.progress || 0}% complete</div>
                <div class="job-actions">
                    ${errorHTML}
                    <div class="job-actions-buttons">
                        ${actionsHTML}
                        <button class="btn-small btn-delete" onclick="deleteJob('${job.job_id}')">Delete</button>
                    </div>
                </div>
            `;
        }
        
        // Download job output
        async function downloadJob(jobId) {
            window.location.href = `/api/jobs/${jobId}/download`;
        }

        // Cancel a running job
        async function cancelJob(jobId) {
            if (!confirm('Cancel this job? Partial output will be preserved and you can resume later.')) return;
            try {
                const response = await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to cancel job');
                }
                loadJobs();
            } catch (error) {
                alert(`Error cancelling job: ${error.message}`);
                console.error('Cancel error:', error);
            }
        }

        // Resume a failed job from its last checkpoint
        async function resumeJob(jobId) {
            const settings = JSON.parse(localStorage.getItem('allSettings')) || {};
            const currentProviderId = settings.currentProvider || 'openai';
            const providerSettings  = (settings.providers || {})[currentProviderId] || {};

            const resumeApiKey = (providerSettings.api_key || '').trim();
            const llm_settings = {
                llm_provider:         currentProviderId,
                base_url:             (providerSettings.base_url  || '').trim(),
                api_key:              resumeApiKey,
                llm_model:            providerSettings.llm_model || '',
                hugging_face_api_key: settings.hugging_face_api_key || ''
            };

            // ── Diagnostic ──
            const _rks = resumeApiKey.length >= 12
                ? (resumeApiKey.slice(0, 4) + '...' + resumeApiKey.slice(-8))
                : (resumeApiKey.length >= 4 ? ('...' + resumeApiKey.slice(-4)) : (resumeApiKey || '(empty — check Settings!)'));
            console.log(`%c[LMDataTools] RESUME LLM → provider="${currentProviderId}"  api_key=${_rks}`, 'color:#4fc; font-weight:bold');
            if (!resumeApiKey) {
                console.warn('[LMDataTools] api_key is EMPTY for resume — will fall back to server .env key.');
            }

            try {
                const response = await fetch(`/api/jobs/${jobId}/resume`, {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify({ llm_settings })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to resume job');
                }

                const data = await response.json();
                monitorJob(data.job_id);
                loadJobs();
            } catch (error) {
                alert(`Error resuming job: ${error.message}`);
                console.error('Resume error:', error);
            }
        }

        // Recreate job
        async function recreateJob(jobId) {
        try {
            const response = await fetch(`/api/jobs/${jobId}`);
            const job = await response.json();
            
            if (!job.config) {
                alert('This job does not have saved configuration data and cannot be recreated.');
                return;
            }
            
            // Switch to the appropriate tool
            const toolSelect = document.getElementById('tool-select');
            toolSelect.value = job.tool;
            toolSelect.dispatchEvent(new Event('change'));
            
            // Wait for form to render by checking if dataset_name field exists
            let attempts = 0;
            const maxAttempts = 20;
            while (attempts < maxAttempts) {
                const testField = document.querySelector('input[name="dataset_name"]');
                if (testField) break;
                await new Promise(resolve => setTimeout(resolve, 50));
                attempts++;
            }
            
            // Populate common fields
            const datasetNameField = document.querySelector('input[name="dataset_name"]');
            if (datasetNameField && job.config.dataset_name) {
                datasetNameField.value = job.config.dataset_name + '-recreated';
            }
            
            // Tool-specific field population
            switch(job.tool) {
                case 'datapersona':
                    populateDataPersonaForm(job.config);
                    break;
                case 'databird':
                    populateDataBirdForm(job.config);
                    break;
                case 'datawriter':
                    populateDataWriterForm(job.config);
                    break;
                case 'dataqa':
                    populateDataQAForm(job.config);
                    break;
                case 'datathink':
                    populateDataThinkForm(job.config);
                    break;
                case 'dataconvo':
                    populateDataConvoForm(job.config);
                    break;
                case 'reformat':
                    populateReformatForm(job.config);
                    break;
                case 'datamix':
                    populateDataMixForm(job.config);
                    break;
            }
            
            // Populate LLM settings if present
            if (job.config.llm_settings) {
                populateLLMSettings(job.config.llm_settings);
            }
            
            // Show notification about file uploads
            if (job.config.uploaded_filenames && job.config.uploaded_filenames.length > 0) {
                alert(`Note: This job originally used the following file(s):\n${job.config.uploaded_filenames.join('\n')}\n\nYou will need to re-upload these files.`);
            }
            
            // Scroll to top of form
            window.scrollTo({ top: 0, behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error recreating job:', error);
            alert('Failed to recreate job configuration.');
        }
    }

    function populateDataPersonaForm(config) {
        const personaSelect = document.querySelector('select[name="persona"]');
        if (personaSelect && config.persona) {
            personaSelect.value = config.persona;
        }
        
        const reply1Checkbox = document.querySelector('input[name="generate_reply_1"]');
        if (reply1Checkbox) reply1Checkbox.checked = config.generate_reply_1 || false;
        
        const reply2Checkbox = document.querySelector('input[name="generate_reply_2"]');
        if (reply2Checkbox) reply2Checkbox.checked = config.generate_reply_2 || false;
        
        const exportAlpacaCheckbox = document.querySelector('input[name="export_alpaca"]');
        if (exportAlpacaCheckbox) exportAlpacaCheckbox.checked = config.export_alpaca || false;
        
        const saveIntervalInput = document.querySelector('input[name="save_interval"]');
        if (saveIntervalInput && config.save_interval) {
            saveIntervalInput.value = config.save_interval;
        }
    }

    function populateDataBirdForm(config) {
        const topicsTextarea = document.querySelector('textarea[name="topics"]');
        if (topicsTextarea && config.topics) {
            topicsTextarea.value = config.topics.join('\n');
        }
        
        const fullAutoCheckbox = document.querySelector('input[name="full_auto"]');
        if (fullAutoCheckbox) fullAutoCheckbox.checked = config.full_auto !== false;
        
        const datasetSizeSelect = document.querySelector('select[name="dataset_size"]');
        if (datasetSizeSelect && config.dataset_size) {
            datasetSizeSelect.value = config.dataset_size;
        }
        
        const cleanScoreInput = document.querySelector('input[name="clean_score"]');
        if (cleanScoreInput && config.clean_score !== undefined) {
            cleanScoreInput.value = config.clean_score;
        }
        
        if (config.manual_perspectives && !config.full_auto) {
            const manualPerspectivesTextarea = document.querySelector('textarea[name="manual_perspectives"]');
            if (manualPerspectivesTextarea) {
                const persp = config.manual_perspectives;
                // Each item is a plain string. Handle legacy array-of-arrays gracefully.
                manualPerspectivesTextarea.value = Array.isArray(persp)
                    ? persp.map(p => Array.isArray(p) ? p.join(' ') : String(p)).join('\n')
                    : String(persp);
                // Ensure the group is visible since full_auto is false
                const group = document.getElementById('databird_manual_perspectives_group');
                if (group) group.style.display = 'block';
            }
        }
    }

    function populateDataWriterForm(config) {
        const docCountInput = document.querySelector('input[name="document_count"]');
        if (docCountInput && config.document_count) {
            docCountInput.value = config.document_count;
        }
        
        const tempInput = document.querySelector('input[name="temperature"]');
        if (tempInput && config.temperature !== undefined) {
            tempInput.value = config.temperature;
        }
    }

    function populateDataQAForm(config) {
        const sourcesTextarea = document.querySelector('textarea[name="sources"]');
        if (sourcesTextarea && config.sources) {
            // Filter out file paths (they start with 'import/')
            const urlSources = config.sources.filter(s => !s.startsWith('import/'));
            sourcesTextarea.value = urlSources.join('\n');
        }
        
        const autoPerspecCheckbox = document.querySelector('input[name="auto_perspectives"]');
        if (autoPerspecCheckbox) autoPerspecCheckbox.checked = config.auto_perspectives !== false;
        
        const confidenceInput = document.querySelector('input[name="confidence_threshold"]');
        if (confidenceInput && config.confidence_threshold !== undefined) {
            confidenceInput.value = config.confidence_threshold;
        }
        
        const usePersonaCheckbox = document.querySelector('input[name="use_persona"]');
        if (usePersonaCheckbox) usePersonaCheckbox.checked = config.use_persona || false;

        if (config.use_persona && config.persona_name) {
            const personaSelect = document.querySelector('select[name="persona_name"]');
            if (personaSelect) personaSelect.value = config.persona_name;
        }

        if (config.manual_perspectives && config.auto_perspectives === false) {
            const group = document.getElementById('manual_perspectives_group');
            const manualPerspectivesTextarea = group ? group.querySelector('textarea') : null;
            if (manualPerspectivesTextarea) {
                const persp = config.manual_perspectives;
                // Each item is a plain string. Handle legacy array-of-arrays gracefully.
                manualPerspectivesTextarea.value = Array.isArray(persp)
                    ? persp.map(p => Array.isArray(p) ? p.join(' ') : String(p)).join('\n')
                    : String(persp);
                if (group) group.style.display = 'block';
            }
        }
    }

    function populateDataThinkForm(config) {
        const saveIntervalInput = document.getElementById('save_interval');
        if (saveIntervalInput && config.save_interval) {
            saveIntervalInput.value = config.save_interval;
        }
        const thinkModeSelect = document.getElementById('think_mode');
        if (thinkModeSelect && config.think_mode) {
            thinkModeSelect.value = config.think_mode;
        }
        
        const thinkingTempInput = document.querySelector('input[name="thinking_temperature"]');
        if (thinkingTempInput && config.thinking_temperature !== undefined) {
            thinkingTempInput.value = config.thinking_temperature;
        }
        
        const responseTempInput = document.querySelector('input[name="response_temperature"]');
        if (responseTempInput && config.response_temperature !== undefined) {
            responseTempInput.value = config.response_temperature;
        }
        
        const usePersonaCheckbox = document.querySelector('input[name="use_persona"]');
        if (usePersonaCheckbox) usePersonaCheckbox.checked = config.use_persona || false;
        
        if (config.use_persona && config.persona_name) {
            const personaSelect = document.querySelector('select[name="persona_name"]');
            if (personaSelect) personaSelect.value = config.persona_name;
        }
    }

    function populateDataConvoForm(config) {
        const saveIntervalInput = document.querySelector('input[name="save_interval"]');
        if (saveIntervalInput && config.save_interval) {
            saveIntervalInput.value = config.save_interval;
        }
        
        if (config.round_weights) {
            const roundWeights = typeof config.round_weights === 'string' ? 
                JSON.parse(config.round_weights) : config.round_weights;
            
            const rounds1Input = document.querySelector('input[name="rounds_1"]');
            const rounds2Input = document.querySelector('input[name="rounds_2"]');
            const rounds3Input = document.querySelector('input[name="rounds_3"]');
            
            if (rounds1Input) rounds1Input.value = roundWeights.rounds_1 || 25;
            if (rounds2Input) rounds2Input.value = roundWeights.rounds_2 || 50;
            if (rounds3Input) rounds3Input.value = roundWeights.rounds_3 || 25;
        }
        
        const usePersonaCheckbox = document.querySelector('input[name="use_persona"]');
        if (usePersonaCheckbox) usePersonaCheckbox.checked = config.use_persona || false;
        
        if (config.use_persona && config.persona_name) {
            const personaSelect = document.querySelector('select[name="persona_name"]');
            if (personaSelect) personaSelect.value = config.persona_name;
        }
    }

    function populateReformatForm(config) {
        const formatSelect = document.querySelector('select[name="target_format"]');
        if (formatSelect && config.target_format) {
            formatSelect.value = config.target_format;
        }
    }

    function populateDataMixForm(config) {
        const totalSamplesInput = document.querySelector('input[name="total_samples"]');
        if (totalSamplesInput && config.total_samples) {
            totalSamplesInput.value = config.total_samples;
        }
        
        const seedInput = document.querySelector('input[name="seed"]');
        if (seedInput && config.seed) {
            seedInput.value = config.seed;
        }
        
        // Dataset sources is more complex - would need the form structure
        // This is a placeholder for when DataMix form is implemented
    }

    function populateLLMSettings(llmSettings) {
        const providerSelect = document.getElementById('llm-provider');
        if (providerSelect && llmSettings.llm_provider) {
            providerSelect.value = llmSettings.llm_provider;
            providerSelect.dispatchEvent(new Event('change'));
        }
        
        // Wait for provider change to process
        setTimeout(() => {
            const baseUrlInput = document.getElementById('llm-base-url');
            if (baseUrlInput && llmSettings.base_url) {
                baseUrlInput.value = llmSettings.base_url;
            }
            
            const modelInput = document.getElementById('llm-model');
            if (modelInput && llmSettings.llm_model) {
                modelInput.value = llmSettings.llm_model;
            }
        }, 100);
    }
        
        
        // Delete job
        async function deleteJob(jobId) {
            if (!confirm('Are you sure you want to delete this job?')) return;
            
            try {
                const response = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to delete job');
                }
                
                const jobDiv = document.getElementById(`job-${jobId}`);
                if (jobDiv) {
                    jobDiv.remove();
                }
                
                // Add empty state if no jobs left
                const jobsList = document.getElementById('jobs-list');
                if (jobsList.children.length === 0) {
                    jobsList.innerHTML = '<div class="empty-jobs">No jobs yet</div>';
                }
            } catch (error) {
                alert(`Error deleting job: ${error.message}`);
                console.error('Error:', error);
            }
        }
        
        // Load jobs on startup
        async function loadJobs() {
            try {
                const response = await fetch('/api/jobs');
                const data = await response.json();

                const jobsList = document.getElementById('jobs-list');

                if (data.jobs.length === 0) {
                    jobsList.innerHTML = '<div class="empty-jobs">No jobs yet</div>';
                    return;
                }

                // Remove empty state if present
                const emptyState = jobsList.querySelector('.empty-jobs');
                if (emptyState) {
                    emptyState.remove();
                }

                // Simple: just update all jobs every time
                data.jobs.forEach(updateJobDisplay);

            } catch (error) {
                console.error('Error loading jobs:', error);
            }
        }

        // ── Settings Tabs ────────────────────────────────────────────────────
        document.querySelectorAll('.settings-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.settings-panel').forEach(p => p.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');
            });
        });

        // ── Global Preferences ───────────────────────────────────────────────
        async function loadGlobalPrefs() {
            try {
                const resp = await fetch('/api/settings/global/prefs');
                if (!resp.ok) return;
                const prefs = await resp.json();

                // Populate default_persona dropdown with persona names
                const personaResp = await fetch('/api/personas');
                if (personaResp.ok) {
                    const data = await personaResp.json();
                    const sel = document.getElementById('pref_default_persona');
                    sel.innerHTML = '<option value="">(none)</option>';
                    (data.personas || []).forEach(name => {
                        const opt = document.createElement('option');
                        opt.value = name;
                        opt.textContent = name;
                        sel.appendChild(opt);
                    });
                }

                document.getElementById('pref_output_format').value     = prefs.preferred_output_format || 'alpaca';
                document.getElementById('pref_default_persona').value   = prefs.default_persona || '';
                document.getElementById('pref_include_reasoning').checked = !!prefs.include_reasoning_output;
                const tempField = document.getElementById('pref_default_temperature');
                tempField.value = prefs.default_temperature ?? 0.7;
                document.getElementById('pref_temp_display').textContent = parseFloat(tempField.value).toFixed(2);
                document.getElementById('pref_save_interval').value     = prefs.default_save_interval ?? 250;
            } catch (e) {
                console.warn('Could not load global prefs:', e);
            }
        }

        async function saveGlobalPrefs() {
            const payload = {
                preferred_output_format:  document.getElementById('pref_output_format').value,
                default_persona:          document.getElementById('pref_default_persona').value,
                include_reasoning_output: document.getElementById('pref_include_reasoning').checked,
                default_temperature:      parseFloat(document.getElementById('pref_default_temperature').value),
                default_save_interval:    parseInt(document.getElementById('pref_save_interval').value, 10)
            };
            try {
                const resp = await fetch('/api/settings/global/prefs', {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify(payload)
                });
                if (!resp.ok) throw new Error('Save failed');
                alert('Preferences saved!');
            } catch (e) {
                alert(`Error saving preferences: ${e.message}`);
            }
        }

        document.getElementById('save-prefs-btn').addEventListener('click', saveGlobalPrefs);

        // ── Persona Management ───────────────────────────────────────────────
        let _editingPersonaName = null; // null = creating new

        async function loadPersonaList() {
            const listDiv = document.getElementById('persona-list');
            listDiv.innerHTML = '<p style="color:#8899aa;font-size:13px;">Loading…</p>';
            try {
                const resp = await fetch('/api/personas?full=true');
                if (!resp.ok) throw new Error('Failed to load personas');
                const data = await resp.json();
                const personas = data.personas;

                if (!personas || personas.length === 0) {
                    listDiv.innerHTML = '<p style="color:#8899aa;font-size:13px;">No personas found.</p>';
                    return;
                }

                listDiv.innerHTML = personas.map(p => `
                    <div class="persona-row">
                        <span class="persona-name">${p.name || p}</span>
                        <button class="btn-persona-edit"
                            data-name="${p.name || p}"
                            data-desc="${encodeURIComponent(p.description || '')}">Edit</button>
                        <button class="btn-persona-del" data-name="${p.name || p}">Delete</button>
                    </div>
                `).join('');

                listDiv.querySelectorAll('.btn-persona-edit').forEach(btn => {
                    btn.addEventListener('click', () => {
                        _editingPersonaName = btn.dataset.name;
                        document.getElementById('persona-edit-original-name').value = btn.dataset.name;
                        document.getElementById('persona-edit-name').value           = btn.dataset.name;
                        document.getElementById('persona-edit-desc').value           = decodeURIComponent(btn.dataset.desc);
                        document.getElementById('persona-editor').style.display     = 'block';
                    });
                });

                listDiv.querySelectorAll('.btn-persona-del').forEach(btn => {
                    btn.addEventListener('click', async () => {
                        if (!confirm(`Delete persona "${btn.dataset.name}"?`)) return;
                        try {
                            const r = await fetch(`/api/personas/${encodeURIComponent(btn.dataset.name)}`, { method: 'DELETE' });
                            if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Delete failed'); }
                            loadPersonaList();
                        } catch (e) {
                            alert(`Error: ${e.message}`);
                        }
                    });
                });
            } catch (e) {
                listDiv.innerHTML = `<p style="color:#ff4757;font-size:13px;">Error: ${e.message}</p>`;
            }
        }

        document.getElementById('add-persona-btn').addEventListener('click', () => {
            _editingPersonaName = null;
            document.getElementById('persona-edit-original-name').value = '';
            document.getElementById('persona-edit-name').value          = '';
            document.getElementById('persona-edit-desc').value          = '';
            document.getElementById('persona-editor').style.display     = 'block';
        });

        document.getElementById('persona-cancel-btn').addEventListener('click', () => {
            document.getElementById('persona-editor').style.display = 'none';
        });

        document.getElementById('persona-save-btn').addEventListener('click', async () => {
            const name = document.getElementById('persona-edit-name').value.trim();
            const desc = document.getElementById('persona-edit-desc').value.trim();
            if (!name || !desc) { alert('Name and description are required.'); return; }
            try {
                let resp;
                if (_editingPersonaName) {
                    const payload = { description: desc };
                    if (name !== _editingPersonaName) payload.new_name = name;
                    resp = await fetch(`/api/personas/${encodeURIComponent(_editingPersonaName)}`, {
                        method:  'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body:    JSON.stringify(payload)
                    });
                } else {
                    resp = await fetch('/api/personas', {
                        method:  'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body:    JSON.stringify({ name, description: desc })
                    });
                }
                if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || 'Save failed'); }
                document.getElementById('persona-editor').style.display = 'none';
                loadPersonaList();
            } catch (e) {
                alert(`Error saving persona: ${e.message}`);
            }
        });

        // ── Initialize ───────────────────────────────────────────────────────
        loadSettings();
        loadJobs();
        showOverviewPage(); // Show overview page on initial load
        setInterval(loadJobs, 5000); // Refresh every 5 seconds

        // Clear Failed Jobs functionality
        const clearFailedJobsBtn = document.getElementById('clear-failed-jobs-btn');
        if (clearFailedJobsBtn) {
            clearFailedJobsBtn.addEventListener('click', async () => {
                if (!confirm('Are you sure you want to delete all failed jobs?')) {
                    return;
                }
                try {
                    const response = await fetch('/api/jobs/clear_failed', { method: 'DELETE' });
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Failed to clear failed jobs');
                    }
                    alert('Failed jobs cleared successfully!');
                    loadJobs();
                } catch (error) {
                    alert(`Error clearing failed jobs: ${error.message}`);
                    console.error('Error:', error);
                }
            });
        }
