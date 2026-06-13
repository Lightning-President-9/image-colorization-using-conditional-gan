// ==========================================================
// ELEMENTS
// ==========================================================

const uploadForm = document.getElementById("uploadForm");

const imageInput = document.getElementById("imageInput");

const dropZone = document.getElementById("dropZone");

const previewImage =
    document.getElementById("previewImage");

const modelType =
    document.getElementById("modelType");

const generalContainer =
    document.getElementById("generalContainer");

const categoryContainer =
    document.getElementById("categoryContainer");

const generalModel =
    document.getElementById("generalModel");

const categoryModel =
    document.getElementById("categoryModel");

const loadingOverlay =
    document.getElementById("loadingOverlay");

const loadingText =
    document.getElementById("loadingText");

const resultSection =
    document.getElementById("resultSection");

const emptyState =
    document.getElementById("emptyState");

const infoCard =
    document.getElementById("infoCard");

const infoModel =
    document.getElementById("infoModel");

const infoLoss =
    document.getElementById("infoLoss");

const infoTime =
    document.getElementById("infoTime");

const originalImg =
    document.getElementById("originalImg");

const grayscaleImg =
    document.getElementById("grayscaleImg");

const generatedImg =
    document.getElementById("generatedImg");

const comparisonImg =
    document.getElementById("comparisonImg");

const downloadOriginal =
    document.getElementById("downloadOriginal");

const downloadGenerated =
    document.getElementById("downloadGenerated");

const downloadComparison =
    document.getElementById("downloadComparison");

const downloadZipBtn =
    document.getElementById("downloadZip");

const themeToggle =
    document.getElementById("themeToggle");

const toast =
    document.getElementById("toast");

const toastMessage =
    document.getElementById("toastMessage");

// ==========================================================
// GLOBALS
// ==========================================================

let currentResult = null;

// ==========================================================
// TOAST
// ==========================================================

function showToast(message) {

    toastMessage.innerText = message;

    toast.classList.remove("hidden");

    setTimeout(() => {

        toast.classList.add("hidden");

    }, 3000);
}

// ==========================================================
// LOADING
// ==========================================================

function showLoading() {

    loadingOverlay.classList.remove("hidden");

    const messages = [
        "Checking image...",
        "Converting to grayscale...",
        "Loading TensorFlow model...",
        "Running inference...",
        "Generating result..."
    ];

    let index = 0;

    loadingText.innerText = messages[0];

    const interval = setInterval(() => {

        index++;

        if (index < messages.length) {

            loadingText.innerText =
                messages[index];

        }

    }, 1200);

    loadingOverlay.dataset.interval =
        interval;
}

function hideLoading() {

    loadingOverlay.classList.add("hidden");

    const interval =
        loadingOverlay.dataset.interval;

    if (interval) {

        clearInterval(interval);
    }
}

// ==========================================================
// MODEL SWITCHING
// ==========================================================

modelType.addEventListener(
    "change",
    () => {

        if (modelType.value === "general") {

            generalContainer.classList.remove(
                "hidden"
            );

            categoryContainer.classList.add(
                "hidden"
            );

        } else {

            categoryContainer.classList.remove(
                "hidden"
            );

            generalContainer.classList.add(
                "hidden"
            );
        }
    }
);

// ==========================================================
// DRAG DROP
// ==========================================================

dropZone.addEventListener("click", () => {

    imageInput.click();
});

dropZone.addEventListener(
    "dragover",
    (e) => {

        e.preventDefault();

        dropZone.classList.add("dragover");
    }
);

dropZone.addEventListener(
    "dragleave",
    () => {

        dropZone.classList.remove("dragover");
    }
);

dropZone.addEventListener(
    "drop",
    (e) => {

        e.preventDefault();

        dropZone.classList.remove(
            "dragover"
        );

        const files =
            e.dataTransfer.files;

        if (files.length > 0) {

            imageInput.files = files;

            previewSelectedImage(
                files[0]
            );
        }
    }
);

// ==========================================================
// PREVIEW
// ==========================================================

imageInput.addEventListener(
    "change",
    () => {

        if (
            imageInput.files &&
            imageInput.files[0]
        ) {

            previewSelectedImage(
                imageInput.files[0]
            );
        }
    }
);

function previewSelectedImage(file) {

    const reader =
        new FileReader();

    reader.onload = function (e) {

        previewImage.src =
            e.target.result;

        previewImage.classList.remove(
            "hidden"
        );
    };

    reader.readAsDataURL(file);
}

// ==========================================================
// FORM SUBMIT
// ==========================================================

uploadForm.addEventListener(
    "submit",
    async (e) => {

        e.preventDefault();

        if (
            !imageInput.files ||
            imageInput.files.length === 0
        ) {

            showToast(
                "Please select an image."
            );

            return;
        }

        showLoading();

        const formData =
            new FormData();

        formData.append(
            "image",
            imageInput.files[0]
        );

        formData.append(
            "model_type",
            modelType.value
        );

        const selectedModel =
            modelType.value === "general"
                ? generalModel.value
                : categoryModel.value;

        formData.append(
            "selected_model",
            selectedModel
        );

        formData.append(
            "loss_version",
            document.getElementById(
                "lossVersion"
            ).value
        );

        try {

            const response =
                await fetch(
                    "/generate",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            const data =
                await response.json();

            hideLoading();

            if (!data.success) {

                showToast(
                    data.error
                );

                return;
            }

            currentResult = data;

            displayResults(data);

            showToast(
                "Image generated successfully."
            );

        } catch (error) {

            hideLoading();

            console.error(error);

            showToast(
                "Server error occurred."
            );
        }
    }
);

// ==========================================================
// DISPLAY RESULTS
// ==========================================================

function displayResults(data) {

    emptyState.classList.add(
        "hidden"
    );

    resultSection.classList.remove(
        "hidden"
    );

    infoCard.classList.remove(
        "hidden"
    );

    infoModel.innerText =
        data.model;

    infoLoss.innerText =
        data.loss;

    infoTime.innerText =
        data.inference_time + " sec";

    originalImg.src =
        data.original;

    grayscaleImg.src =
        data.grayscale;

    generatedImg.src =
        data.generated;

    comparisonImg.src =
        data.comparison;

    downloadOriginal.href =
        data.original;

    downloadGenerated.href =
        data.generated;

    downloadComparison.href =
        data.comparison;
}

// ==========================================================
// DOWNLOAD ZIP
// ==========================================================

downloadZipBtn.addEventListener(
    "click",
    async () => {

        if (!currentResult) {

            showToast(
                "Generate image first."
            );

            return;
        }

        const formData =
            new FormData();

        formData.append(
            "original",
            currentResult.original
        );

        formData.append(
            "grayscale",
            currentResult.grayscale
        );

        formData.append(
            "generated",
            currentResult.generated
        );

        formData.append(
            "comparison",
            currentResult.comparison
        );

        const response =
            await fetch(
                "/download_zip",
                {
                    method: "POST",
                    body: formData
                }
            );

        const blob =
            await response.blob();

        const url =
            window.URL.createObjectURL(
                blob
            );

        const a =
            document.createElement("a");

        a.href = url;

        a.download =
            "colorization_results.zip";

        document.body.appendChild(a);

        a.click();

        a.remove();

        window.URL.revokeObjectURL(
            url
        );
    }
);

// ==========================================================
// DARK MODE
// ==========================================================

const savedTheme =
    localStorage.getItem(
        "theme"
    );

if (savedTheme === "dark") {

    document.body.classList.add(
        "dark"
    );

    themeToggle.innerHTML = "☀️";
}

themeToggle.addEventListener(
    "click",
    () => {

        document.body.classList.toggle(
            "dark"
        );

        if (
            document.body.classList.contains(
                "dark"
            )
        ) {

            localStorage.setItem(
                "theme",
                "dark"
            );

            themeToggle.innerHTML =
                "☀️";

        } else {

            localStorage.setItem(
                "theme",
                "light"
            );

            themeToggle.innerHTML =
                "🌙";
        }
    }
);

// ==========================================================
// INITIALIZE
// ==========================================================

window.addEventListener(
    "load",
    () => {

        console.log(
            "AI Image Colorization Ready"
        );
    }
);