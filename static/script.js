// ==========================================================================
// Application States & Globals
// ==========================================================================

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordTimerInterval = null;
let recordSeconds = 0;

// Custom Audio Player State
const audioEl = document.getElementById("answerAudio");
let isAudioPlaying = false;

// DOM Selectors
const recordButton = document.getElementById("recordButton");
const recordStatus = document.getElementById("recordStatus");
const recordDuration = document.getElementById("recordDuration");
const recordingWaveform = document.getElementById("recordingWaveform");

const textQueryForm = document.getElementById("textQueryForm");
const textQuestionInput = document.getElementById("textQuestionInput");
const exampleChips = document.querySelectorAll(".chip");

const pipelineContainer = document.getElementById("pipelineContainer");
const errorConsole = document.getElementById("errorConsole");
const errorMessage = document.getElementById("errorMessage");
const resultsDashboard = document.getElementById("resultsDashboard");

// Result panel elements
const resultTranscript = document.getElementById("resultTranscript");
const resultAnswer = document.getElementById("resultAnswer");
const audioOutputCard = document.getElementById("audioOutputCard");
const sourcesContainer = document.getElementById("sourcesContainer");
const performanceContainer = document.getElementById("performanceContainer");

// Custom Audio Player DOM Elements
const customPlayPauseBtn = document.getElementById("customPlayPauseBtn");
const progressContainer = document.getElementById("progressContainer");
const progressBar = document.getElementById("progressBar");
const currentTimeLabel = document.getElementById("currentTime");
const durationTimeLabel = document.getElementById("durationTime");
const muteBtn = document.getElementById("muteBtn");
const volumeSlider = document.getElementById("volumeSlider");


// ==========================================================================
// Microphone & Recording Actions
// ==========================================================================

recordButton.addEventListener("click", async () => {

    if (!isRecording) {

        await startRecording();

    } else {

        stopRecording();

    }

});


async function startRecording() {

    try {

        // Reset states
        audioChunks = [];
        recordSeconds = 0;
        recordDuration.textContent = "00:00";

        // Hide previous panels
        hidePanel(resultsDashboard);
        hidePanel(errorConsole);
        hidePanel(pipelineContainer);

        // Request microphone permission
        const stream =
            await navigator.mediaDevices.getUserMedia({
                audio: true
            });

        // Handle supported MIME types
        let options = {};

        if (MediaRecorder.isTypeSupported("audio/webm")) {

            options = {
                mimeType: "audio/webm"
            };

        } else if (
            MediaRecorder.isTypeSupported("audio/ogg")
        ) {

            options = {
                mimeType: "audio/ogg"
            };

        } else if (
            MediaRecorder.isTypeSupported("audio/wav")
        ) {

            options = {
                mimeType: "audio/wav"
            };

        }

        mediaRecorder =
            new MediaRecorder(
                stream,
                options
            );

        mediaRecorder.ondataavailable =
            (event) => {

                if (
                    event.data &&
                    event.data.size > 0
                ) {

                    audioChunks.push(
                        event.data
                    );

                }

            };


        mediaRecorder.onstop =
            async () => {

                // Clean up microphone tracks
                stream
                    .getTracks()
                    .forEach(
                        track =>
                            track.stop()
                    );

                const mimeType =
                    mediaRecorder.mimeType ||
                    "audio/webm";

                const extension =
                    mimeType.includes("ogg")
                        ? "ogg"
                        : mimeType.includes("wav")
                            ? "wav"
                            : "webm";

                const audioBlob =
                    new Blob(
                        audioChunks,
                        {
                            type: mimeType
                        }
                    );

                // Send recording to backend
                await sendAudio(
                    audioBlob,
                    `recording.${extension}`
                );

            };


        // Start MediaRecorder
        mediaRecorder.start();

        isRecording = true;

        // Toggle UI indicators
        recordButton.classList.add(
            "recording"
        );

        recordButton
            .querySelector(".mic-icon")
            .classList.add("hidden");

        recordButton
            .querySelector(".stop-icon")
            .classList.remove("hidden");

        recordStatus.textContent =
            "Listening...";

        showPanel(recordDuration);
        showPanel(recordingWaveform);


        // Start duration counter
        recordTimerInterval =
            setInterval(() => {

                recordSeconds++;

                const mins =
                    String(
                        Math.floor(
                            recordSeconds / 60
                        )
                    ).padStart(2, "0");

                const secs =
                    String(
                        recordSeconds % 60
                    ).padStart(2, "0");

                recordDuration.textContent =
                    `${mins}:${secs}`;

            }, 1000);


    } catch (error) {

        console.error(
            "Recording start failed:",
            error
        );

        showError(
            "Microphone access was denied or is unavailable. Please check your browser permission settings."
        );

        resetRecordingState();

    }

}


function stopRecording() {

    if (
        mediaRecorder &&
        mediaRecorder.state !== "inactive"
    ) {

        mediaRecorder.stop();

    }

    resetRecordingState();

    recordStatus.textContent =
        "Uploading audio...";

}


function resetRecordingState() {

    isRecording = false;

    if (recordTimerInterval) {

        clearInterval(
            recordTimerInterval
        );

        recordTimerInterval = null;

    }

    recordButton.classList.remove(
        "recording"
    );

    recordButton
        .querySelector(".mic-icon")
        .classList.remove("hidden");

    recordButton
        .querySelector(".stop-icon")
        .classList.add("hidden");

    hidePanel(recordDuration);
    hidePanel(recordingWaveform);

}


// ==========================================================================
// RAG Pipeline Visualization Helper
// ==========================================================================

const stepIds = [
    "speech",
    "stt",
    "retrieval",
    "rerank",
    "gemini",
    "tts"
];

let pipelineSimTimer = null;


function initializePipelineUI(isVoice) {

    showPanel(
        pipelineContainer
    );

    // Clear timing timers if active
    if (pipelineSimTimer) {

        clearInterval(
            pipelineSimTimer
        );

    }

    // Reset all steps
    stepIds.forEach(id => {

        const stepEl =
            document.getElementById(
                `step-${id}`
            );

        if (!stepEl) return;

        stepEl.className =
            "pipeline-step pending";

        const statusEl =
            stepEl.querySelector(
                ".step-status"
            );

        if (statusEl) {

            statusEl.textContent =
                "Waiting";

        }

        // Show/hide nodes
        const connectorEl =
            stepEl.nextElementSibling;

        if (
            !isVoice &&
            (
                id === "speech" ||
                id === "stt" ||
                id === "tts"
            )
        ) {

            stepEl.classList.add(
                "hidden"
            );

            if (
                connectorEl &&
                connectorEl.classList.contains(
                    "pipeline-connector"
                )
            ) {

                connectorEl.classList.add(
                    "hidden"
                );

            }

        } else {

            stepEl.classList.remove(
                "hidden"
            );

            if (
                connectorEl &&
                connectorEl.classList.contains(
                    "pipeline-connector"
                )
            ) {

                connectorEl.classList.remove(
                    "hidden"
                );

            }

        }

    });


    // Simulate progress sequence
    let currentStepIndex = 0;

    const activeSteps =
        isVoice
            ? stepIds
            : [
                "retrieval",
                "rerank",
                "gemini"
            ];


    function advanceSimulation() {

        if (currentStepIndex > 0) {

            const prevId =
                activeSteps[
                    currentStepIndex - 1
                ];

            setStepState(
                prevId,
                "completed",
                "Processing Complete"
            );

        }


        if (
            currentStepIndex <
            activeSteps.length
        ) {

            const currId =
                activeSteps[
                    currentStepIndex
                ];

            setStepState(
                currId,
                "processing",
                "Processing..."
            );

            currentStepIndex++;

        } else {

            currentStepIndex = 0;

        }

    }


    advanceSimulation();

    pipelineSimTimer =
        setInterval(
            advanceSimulation,
            800
        );

}


function setStepState(
    stepId,
    state,
    statusText
) {

    const el =
        document.getElementById(
            `step-${stepId}`
        );

    if (!el) return;

    el.className =
        `pipeline-step ${state}`;

    const statusEl =
        el.querySelector(
            ".step-status"
        );

    if (statusEl) {

        statusEl.textContent =
            statusText;

    }

}


function finalizePipelineUI(
    timingData,
    isVoice
) {

    if (pipelineSimTimer) {

        clearInterval(
            pipelineSimTimer
        );

        pipelineSimTimer = null;

    }

    const activeSteps =
        isVoice
            ? stepIds
            : [
                "retrieval",
                "rerank",
                "gemini"
            ];


    activeSteps.forEach(id => {

        let latencyLabel = "";

        if (id === "speech") {

            latencyLabel = "Recorded";

        } else if (
            id === "stt" &&
            timingData.stt_ms !== undefined
        ) {

            latencyLabel =
                `${(
                    timingData.stt_ms / 1000
                ).toFixed(2)}s`;

        } else if (
            id === "retrieval" &&
            timingData.retrieval_ms !== undefined
        ) {

            latencyLabel =
                `${(
                    timingData.retrieval_ms / 1000
                ).toFixed(2)}s`;

        } else if (
            id === "rerank" &&
            timingData.reranking_ms !== undefined
        ) {

            latencyLabel =
                `${(
                    timingData.reranking_ms / 1000
                ).toFixed(2)}s`;

        } else if (
            id === "gemini" &&
            timingData.generation_ms !== undefined
        ) {

            latencyLabel =
                `${(
                    timingData.generation_ms / 1000
                ).toFixed(2)}s`;

        } else if (
            id === "tts" &&
            timingData.tts_ms !== undefined
        ) {

            latencyLabel =
                `${(
                    timingData.tts_ms / 1000
                ).toFixed(2)}s`;

        }

        setStepState(
            id,
            "completed",
            latencyLabel || "Done"
        );

    });

}


// ==========================================================================
// API Handlers
// ==========================================================================

async function sendAudio(
    audioBlob,
    filename
) {

    hidePanel(resultsDashboard);
    hidePanel(errorConsole);

    initializePipelineUI(true);

    recordStatus.textContent =
        "Processing pipeline...";

    const formData =
        new FormData();

    formData.append(
        "file",
        audioBlob,
        filename
    );


    try {

        const response =
            await fetch(
                "/voice-ask",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Voice RAG processing encountered an error."
            );

        }

        finalizePipelineUI(
            data.timing,
            true
        );

        renderResponse(
            data,
            true
        );


    } catch (error) {

        console.error(
            "Audio query failed:",
            error
        );

        showError(
            error.message ||
            "Something went wrong while processing your audio question."
        );

        hidePanel(
            pipelineContainer
        );

    } finally {

        recordStatus.textContent =
            "Tap to speak";

    }

}


// ==========================================================================
// Text Query Submission
// ==========================================================================

textQueryForm.addEventListener(
    "submit",
    async (e) => {

        e.preventDefault();

        const query =
            textQuestionInput.value.trim();

        if (!query) return;

        hidePanel(resultsDashboard);
        hidePanel(errorConsole);

        initializePipelineUI(false);


        try {

            const response =
                await fetch(
                    "/ask",
                    {
                        method: "POST",
                        headers: {
                            "Content-Type":
                                "application/json"
                        },
                        body: JSON.stringify({
                            question: query
                        })
                    }
                );

            const data =
                await response.json();

            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Text question processing failed."
                );

            }

            finalizePipelineUI(
                data.timing,
                false
            );

            renderResponse(
                data,
                false
            );


        } catch (error) {

            console.error(
                "Text query failed:",
                error
            );

            showError(
                error.message ||
                "Failed to process question. Please try again."
            );

            hidePanel(
                pipelineContainer
            );

        }

    }
);


// ==========================================================================
// Example Chips
// ==========================================================================

exampleChips.forEach(chip => {

    chip.addEventListener(
        "click",
        () => {

            textQuestionInput.value =
                chip.textContent;

            textQuestionInput.focus();

        }
    );

});


// ==========================================================================
// Response Renderer
// ==========================================================================

function renderResponse(
    data,
    isVoice
) {

    // ==================================================
    // Transcript / Question
    // ==================================================

    resultTranscript.textContent =
        data.transcript ||
        data.question ||
        textQuestionInput.value;


    // ==================================================
    // Answer
    // ==================================================

    resultAnswer.textContent =
        data.answer ||
        "No answer was returned.";


    // ==================================================
    // Audio Player
    // ==================================================

    if (
        isVoice &&
        data.audio_url
    ) {

        showPanel(
            audioOutputCard
        );

        const audioSrcUrl =
            `${data.audio_url}?t=${Date.now()}`;

        audioEl.src =
            audioSrcUrl;

        audioEl.load();

        resetAudioPlayerUI();

    } else {

        hidePanel(
            audioOutputCard
        );

        audioEl.src = "";

    }


    // ==================================================
    // Reference Sources
    // ==================================================

    sourcesContainer.innerHTML = "";


    if (
        data.sources &&
        data.sources.length > 0
    ) {

        data.sources.forEach(
            (src, idx) => {

                const card =
                    document.createElement(
                        "div"
                    );

                card.className =
                    "source-item";


                // --------------------------------------------------
                // SAFE SCORE HANDLING
                // --------------------------------------------------

                const faissScore =
                    (
                        src.score !== null &&
                        src.score !== undefined &&
                        !Number.isNaN(
                            Number(src.score)
                        )
                    )
                        ? Number(
                            src.score
                        ).toFixed(4)
                        : "N/A";


                const rerankerScore =
                    (
                        src.reranker_score !== null &&
                        src.reranker_score !== undefined &&
                        !Number.isNaN(
                            Number(
                                src.reranker_score
                            )
                        )
                    )
                        ? Number(
                            src.reranker_score
                        ).toFixed(4)
                        : "N/A";


                const chunkId =
                    src.chunk_id !== undefined &&
                    src.chunk_id !== null
                        ? src.chunk_id
                        : "N/A";


                const sourceUrl =
                    src.url ||
                    "#";


                // --------------------------------------------------
                // Source Card HTML
                // --------------------------------------------------

                card.innerHTML = `

                    <div class="source-title-row">

                        <span class="source-number-badge">
                            Source ${idx + 1}
                        </span>

                        <span class="chunk-badge">
                            Chunk ID: ${chunkId}
                        </span>

                    </div>


                    <div class="source-metrics">

                        <div class="source-stat">

                            <span class="source-stat-label">
                                FAISS Score
                            </span>

                            <span class="source-stat-value">
                                ${faissScore}
                            </span>

                        </div>


                        <div class="source-stat">

                            <span class="source-stat-label">
                                Reranker Score
                            </span>

                            <span class="source-stat-value">
                                ${rerankerScore}
                            </span>

                        </div>

                    </div>


                    <a
                        href="${sourceUrl}"
                        target="_blank"
                        rel="noopener noreferrer"
                        class="source-link-btn"
                    >

                        View Source Documents

                        <svg
                            viewBox="0 0 24 24"
                        >
                            <path
                                fill="currentColor"
                                d="M19 19H5V5h7V3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.11 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z"
                            />
                        </svg>

                    </a>

                `;


                sourcesContainer.appendChild(
                    card
                );

            }
        );


    } else {

        sourcesContainer.innerHTML = `
            <p
                style="
                    color: var(--text-muted);
                    font-size: 0.95rem;
                "
            >
                No grounded search matches are available
                for this question.
            </p>
        `;

    }


    // ==================================================
    // Render Latencies & Stats
    // ==================================================

    performanceContainer.innerHTML = "";

    const timing =
        data.timing || {};


    const statsList = [];


    if (
        isVoice &&
        timing.stt_ms !== undefined
    ) {

        statsList.push({
            label:
                "STT Transcription",
            val:
                timing.stt_ms
        });

    }


    if (
        timing.retrieval_ms !== undefined
    ) {

        statsList.push({
            label:
                "FAISS Retrieval",
            val:
                timing.retrieval_ms
        });

    }


    if (
        timing.reranking_ms !== undefined
    ) {

        statsList.push({
            label:
                "Cross-Reranking",
            val:
                timing.reranking_ms
        });

    }


    if (
        timing.generation_ms !== undefined
    ) {

        statsList.push({
            label:
                "Gemini Synthesis",
            val:
                timing.generation_ms
        });

    }


    if (
        isVoice &&
        timing.tts_ms !== undefined
    ) {

        statsList.push({
            label:
                "TTS Voice",
            val:
                timing.tts_ms
        });

    }


    if (
        timing.total_ms !== undefined
    ) {

        statsList.push({
            label:
                "End-To-End Latency",
            val:
                timing.total_ms,
            total:
                true
        });

    }


    statsList.forEach(
        stat => {

            const item =
                document.createElement(
                    "div"
                );

            item.className =
                `stat-item ${
                    stat.total
                        ? "total-stat"
                        : ""
                }`;


            const secondsValue =
                Number(stat.val) >= 0
                    ? (
                        Number(stat.val) /
                        1000
                    ).toFixed(2)
                    : "0.00";


            item.innerHTML = `

                <span class="stat-value">
                    ${secondsValue}s
                </span>

                <span class="stat-label">
                    ${stat.label}
                </span>

            `;


            performanceContainer.appendChild(
                item
            );

        }
    );


    // Reveal dashboard
    showPanel(
        resultsDashboard
    );

}


// ==========================================================================
// Custom Audio Player
// ==========================================================================

customPlayPauseBtn.addEventListener(
    "click",
    () => {

        if (!audioEl.src) return;

        if (isAudioPlaying) {

            audioEl.pause();

        } else {

            audioEl
                .play()
                .catch(
                    e =>
                        console.error(
                            "Playback interrupted:",
                            e
                        )
                );

        }

    }
);


audioEl.addEventListener(
    "play",
    () => {

        isAudioPlaying = true;

        customPlayPauseBtn
            .querySelector(".play-icon")
            .classList.add("hidden");

        customPlayPauseBtn
            .querySelector(".pause-icon")
            .classList.remove("hidden");

    }
);


audioEl.addEventListener(
    "pause",
    () => {

        isAudioPlaying = false;

        customPlayPauseBtn
            .querySelector(".play-icon")
            .classList.remove("hidden");

        customPlayPauseBtn
            .querySelector(".pause-icon")
            .classList.add("hidden");

    }
);


audioEl.addEventListener(
    "timeupdate",
    () => {

        if (!audioEl.duration) return;

        const progressPercent =
            (
                audioEl.currentTime /
                audioEl.duration
            ) * 100;

        progressBar.style.width =
            `${progressPercent}%`;

        currentTimeLabel.textContent =
            formatTime(
                audioEl.currentTime
            );

    }
);


audioEl.addEventListener(
    "loadedmetadata",
    () => {

        durationTimeLabel.textContent =
            formatTime(
                audioEl.duration
            );

    }
);


audioEl.addEventListener(
    "ended",
    () => {

        isAudioPlaying = false;

        progressBar.style.width =
            "0%";

        currentTimeLabel.textContent =
            "0:00";

        customPlayPauseBtn
            .querySelector(".play-icon")
            .classList.remove("hidden");

        customPlayPauseBtn
            .querySelector(".pause-icon")
            .classList.add("hidden");

    }
);


// ==========================================================================
// Audio Timeline
// ==========================================================================

progressContainer.addEventListener(
    "click",
    (e) => {

        if (!audioEl.duration) return;

        const clickX =
            e.offsetX;

        const width =
            progressContainer.offsetWidth;

        const clickedFraction =
            clickX / width;

        audioEl.currentTime =
            clickedFraction *
            audioEl.duration;

    }
);


// ==========================================================================
// Audio Mute Toggle
// ==========================================================================

muteBtn.addEventListener(
    "click",
    () => {

        audioEl.muted =
            !audioEl.muted;

        if (audioEl.muted) {

            muteBtn
                .querySelector(
                    ".volume-high-icon"
                )
                .classList.add("hidden");

            muteBtn
                .querySelector(
                    ".volume-muted-icon"
                )
                .classList.remove("hidden");

        } else {

            muteBtn
                .querySelector(
                    ".volume-high-icon"
                )
                .classList.remove("hidden");

            muteBtn
                .querySelector(
                    ".volume-muted-icon"
                )
                .classList.add("hidden");

        }

    }
);


// ==========================================================================
// Volume Slider
// ==========================================================================

volumeSlider.addEventListener(
    "input",
    (e) => {

        const vol =
            parseFloat(
                e.target.value
            );

        audioEl.volume =
            vol;


        if (vol === 0) {

            audioEl.muted = true;

            muteBtn
                .querySelector(
                    ".volume-high-icon"
                )
                .classList.add("hidden");

            muteBtn
                .querySelector(
                    ".volume-muted-icon"
                )
                .classList.remove("hidden");

        } else {

            audioEl.muted = false;

            muteBtn
                .querySelector(
                    ".volume-high-icon"
                )
                .classList.remove("hidden");

            muteBtn
                .querySelector(
                    ".volume-muted-icon"
                )
                .classList.add("hidden");

        }

    }
);


// ==========================================================================
// Audio Player Reset
// ==========================================================================

function resetAudioPlayerUI() {

    isAudioPlaying = false;

    progressBar.style.width =
        "0%";

    currentTimeLabel.textContent =
        "0:00";

    durationTimeLabel.textContent =
        "0:00";

    customPlayPauseBtn
        .querySelector(".play-icon")
        .classList.remove("hidden");

    customPlayPauseBtn
        .querySelector(".pause-icon")
        .classList.add("hidden");

}


// ==========================================================================
// Format Time
// ==========================================================================

function formatTime(seconds) {

    if (
        isNaN(seconds) ||
        seconds === Infinity
    ) {

        return "0:00";

    }

    const minutes =
        Math.floor(
            seconds / 60
        );

    const secs =
        Math.floor(
            seconds % 60
        );

    return `${minutes}:${String(secs).padStart(2, "0")}`;

}


// ==========================================================================
// Helper Utility Panels
// ==========================================================================

function showPanel(element) {

    if (!element) return;

    element.classList.remove(
        "hidden"
    );

}


function hidePanel(element) {

    if (!element) return;

    element.classList.add(
        "hidden"
    );

}


function showError(messageText) {

    errorMessage.textContent =
        messageText;

    showPanel(
        errorConsole
    );

    errorConsole.scrollIntoView({
        behavior: "smooth"
    });

}
