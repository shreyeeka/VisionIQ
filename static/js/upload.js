const uploadForm = document.getElementById("uploadForm");
const dropzone = document.getElementById("dropzone");
const dropzoneContent = document.getElementById("dropzoneContent");
const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const removeBtn = document.getElementById("removeBtn");
const browseBtn = document.getElementById("browseBtn");
const loadingState = document.getElementById("loadingState");
const analyzeBtn = document.getElementById("analyzeBtn");

const showPreview = (hasFile) => {
  if (dropzoneContent) dropzoneContent.classList.toggle("hidden", hasFile);
  if (removeBtn) removeBtn.classList.toggle("hidden", !hasFile);
};

const setPreview = (file) => {
  if (!file || !file.type.startsWith("image/")) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target?.result;
    previewImage.classList.remove("hidden");
    showPreview(true);
  };
  reader.readAsDataURL(file);
};

if (browseBtn && imageInput) {
  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    imageInput.click();
  });
}

if (imageInput) {
  imageInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    setPreview(file);
  });
}

if (removeBtn && imageInput && previewImage) {
  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    imageInput.value = "";
    previewImage.src = "";
    previewImage.classList.add("hidden");
    showPreview(false);
  });
}

if (dropzone && imageInput) {
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("highlight");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("highlight");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer?.files;
    if (files && files.length) {
      imageInput.files = files;
      setPreview(files[0]);
    }
  });

  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("#browseBtn") || e.target.closest("#removeBtn")) return;
    imageInput.click();
  });
}

if (uploadForm) {
  uploadForm.addEventListener("submit", (event) => {
    const hasImage = imageInput?.files && imageInput.files.length > 0;
    if (!hasImage) {
      event.preventDefault();
      dropzone?.classList.add("highlight");
      setTimeout(() => dropzone?.classList.remove("highlight"), 600);
      return;
    }

    if (loadingState && analyzeBtn) {
      loadingState.classList.remove("hidden");
      analyzeBtn.disabled = true;
      analyzeBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Analyzing…';
    }
  });
}

if (previewImage?.src && previewImage.src !== window.location.href) {
  showPreview(true);
}
