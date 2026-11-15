const subjectInput = document.getElementById("subjectInput");
const backgroundInput = document.getElementById("backgroundInput");
const subjectDrop = document.getElementById("subjectDrop");
const backgroundDrop = document.getElementById("backgroundDrop");
const removeBtn = document.getElementById("removeBtn");
const downloadBtn = document.getElementById("downloadBtn");
const resetBtn = document.getElementById("resetBtn");
const canvas = document.getElementById("previewCanvas");
const ctx = canvas.getContext("2d");
const emptyState = document.getElementById("emptyState");
const statusBox = document.getElementById("status");
const previewState = document.getElementById("previewState");
const methodSelect = document.getElementById("methodSelect");
const chromaOptions = document.getElementById("chromaOptions");
const methodHelp = document.getElementById("methodHelp");

let originalFile = null;
let subjectImage = null;
let backgroundImage = null;
let backgroundColor = "#e9edf5";
let subjectX = 0.5;
let subjectY = 0.52;
let dragging = false;
let lastPointer = null;

const controls = ["scale", "brightness", "contrast", "saturation", "bgBlur", "feather", "keyTolerance"];
const defaults = {scale:100, brightness:100, contrast:100, saturation:100, bgBlur:0, feather:1.2, keyTolerance:75};

function updateMethodOptions() {
  const method = methodSelect.value;
  chromaOptions.hidden = method !== "chroma";
  const labels = {
    ai: ["✦ Tách nền bằng AI", "AI tự nhận diện chủ thể trong ảnh."],
    grabcut: ["✦ Tách nền + GrabCut", "GrabCut dùng mặt nạ AI làm điểm khởi đầu để tinh chỉnh; xử lý lâu hơn."],
    chroma: ["✦ Xóa phông Chroma key", "Phù hợp nhất với phông xanh hoặc nền một màu đồng nhất."]
  };
  removeBtn.textContent = labels[method][0];
  methodHelp.textContent = labels[method][1];
}

function setStatus(message, type="") {
  statusBox.textContent = message;
  statusBox.className = "status " + type;
}
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}
function updateOutputs() {
  for (const id of controls) {
    const el = document.getElementById(id);
    const out = document.getElementById(id + "Value");
    if (!el || !out) continue;
    let suffix = "%";
    if (id === "bgBlur") suffix = " px";
    if (id === "feather") suffix = "";
    if (id === "keyTolerance") suffix = "";
    out.value = el.value + suffix;
    out.textContent = el.value + suffix;
  }
}
function drawCover(image, blur=0) {
  const cw = canvas.width, ch = canvas.height;
  const scale = Math.max(cw / image.width, ch / image.height);
  const w = image.width * scale, h = image.height * scale;
  ctx.save();
  if (blur > 0) ctx.filter = `blur(${blur}px)`;
  ctx.drawImage(image, (cw-w)/2, (ch-h)/2, w, h);
  ctx.restore();
}
function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (backgroundImage) drawCover(backgroundImage, Number(document.getElementById("bgBlur").value));
  else {
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  if (subjectImage) {
    const scale = Number(document.getElementById("scale").value) / 100;
    const maxW = canvas.width * 0.72;
    const maxH = canvas.height * 0.88;
    const fit = Math.min(maxW / subjectImage.width, maxH / subjectImage.height);
    const w = subjectImage.width * fit * scale;
    const h = subjectImage.height * fit * scale;
    const x = subjectX * canvas.width - w / 2;
    const y = subjectY * canvas.height - h / 2;
    ctx.save();
    ctx.filter = `brightness(${document.getElementById("brightness").value}%) contrast(${document.getElementById("contrast").value}%) saturate(${document.getElementById("saturation").value}%)`;
    ctx.drawImage(subjectImage, x, y, w, h);
    ctx.restore();
  }
  emptyState.style.display = subjectImage ? "none" : "flex";
  downloadBtn.disabled = !subjectImage;
}
async function setSubjectFile(file) {
  if (!file) return;
  if (!["image/png","image/jpeg","image/webp"].includes(file.type)) {
    setStatus("Định dạng không được hỗ trợ. Hãy chọn PNG, JPG hoặc WEBP.", "error");
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    setStatus("Ảnh vượt quá giới hạn 12 MB.", "error");
    return;
  }
  originalFile = file;
  subjectImage = null;
  document.getElementById("subjectName").textContent = file.name;
  previewState.textContent = "Chờ tách nền";
  render();
  setStatus("Đã tải ảnh. Chọn phương pháp rồi nhấn nút xử lý ảnh.");
}
async function setBackgroundFile(file) {
  if (!file) return;
  try {
    backgroundImage = await loadImage(URL.createObjectURL(file));
    backgroundColor = null;
    document.getElementById("backgroundName").textContent = file.name;
    document.querySelectorAll(".swatch").forEach(s => s.classList.remove("selected"));
    render();
  } catch {
    setStatus("Không thể mở ảnh nền này.", "error");
  }
}
subjectInput.addEventListener("change", e => setSubjectFile(e.target.files[0]));
backgroundInput.addEventListener("change", e => setBackgroundFile(e.target.files[0]));

function setupDrop(zone, handler) {
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", e => {
    e.preventDefault(); zone.classList.remove("drag");
    handler(e.dataTransfer.files[0]);
  });
}
setupDrop(subjectDrop, setSubjectFile);
setupDrop(backgroundDrop, setBackgroundFile);

document.querySelectorAll(".swatch").forEach(button => {
  button.addEventListener("click", () => {
    backgroundColor = button.dataset.color;
    backgroundImage = null;
    document.getElementById("backgroundName").textContent = "Nền màu " + button.title;
    document.querySelectorAll(".swatch").forEach(s => s.classList.remove("selected"));
    button.classList.add("selected");
    render();
  });
});

controls.forEach(id => document.getElementById(id).addEventListener("input", () => {
  updateOutputs();
  render();
}));
methodSelect.addEventListener("change", updateMethodOptions);

removeBtn.addEventListener("click", async () => {
  if (!originalFile) {
    setStatus("Vui lòng tải ảnh chủ thể trước.", "error");
    return;
  }
  removeBtn.disabled = true;
  removeBtn.textContent = "Đang xử lý...";
  const method = methodSelect.value;
  setStatus(method === "chroma"
    ? "Đang tạo mặt nạ từ màu phông đã chọn..."
    : "Đang phân tích ảnh và tách nền. Lần chạy đầu có thể mất thêm thời gian.");
  const form = new FormData();
  form.append("image", originalFile);
  form.append("feather", document.getElementById("feather").value);
  form.append("method", method);
  form.append("key_color", document.getElementById("keyColor").value);
  form.append("key_tolerance", document.getElementById("keyTolerance").value);
  try {
    const response = await fetch("/api/remove", {method:"POST", body:form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Có lỗi xảy ra khi xử lý ảnh.");
    subjectImage = await loadImage(data.image_url + "?t=" + Date.now());
    subjectX = 0.5; subjectY = 0.52;
    previewState.textContent = "Đã tách nền";
    setStatus("Xử lý thành công. Bạn có thể chọn nền và tinh chỉnh ảnh.", "ok");
    render();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    removeBtn.disabled = false;
    updateMethodOptions();
  }
});

resetBtn.addEventListener("click", () => {
  for (const [id, value] of Object.entries(defaults)) document.getElementById(id).value = value;
  subjectX = 0.5; subjectY = 0.52;
  updateOutputs(); render();
});

downloadBtn.addEventListener("click", () => {
  if (!subjectImage) return;
  const link = document.createElement("a");
  link.download = "background-studio-result.png";
  link.href = canvas.toDataURL("image/png");
  link.click();
});

// Cho phép kéo chủ thể trong vùng xem trước.
function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) / rect.width,
    y: (event.clientY - rect.top) / rect.height
  };
}
canvas.addEventListener("pointerdown", event => {
  if (!subjectImage) return;
  dragging = true;
  lastPointer = pointerPosition(event);
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", event => {
  if (!dragging || !lastPointer) return;
  const current = pointerPosition(event);
  subjectX += current.x - lastPointer.x;
  subjectY += current.y - lastPointer.y;
  lastPointer = current;
  render();
});
canvas.addEventListener("pointerup", () => { dragging = false; lastPointer = null; });
canvas.addEventListener("pointercancel", () => { dragging = false; lastPointer = null; });

updateOutputs();
updateMethodOptions();
render();
