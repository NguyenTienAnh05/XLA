// ============================================================================
// 1. THAM CHIẾU PHẦN TỬ DOM
// ============================================================================
const subjectInput = document.getElementById("subjectInput");
const backgroundInput = document.getElementById("backgroundInput");
const subjectDrop = document.getElementById("subjectDrop");
const backgroundDrop = document.getElementById("backgroundDrop");

const removeBtn = document.getElementById("removeBtn");
const downloadBtn = document.getElementById("downloadBtn");
const resetBtn = document.getElementById("resetBtn");

// Canvas xem trước kết quả sau khi thay nền
const canvas = document.getElementById("previewCanvas");
const ctx = canvas.getContext("2d");

// Canvas chọn đối tượng tương tác (vẽ điểm, khung chữ nhật, polygon, cọ)
const segmentCanvas = document.getElementById("segmentCanvas");
const segmentCtx = segmentCanvas.getContext("2d");

const segmentEditor = document.getElementById("segmentEditor");
const segmentEmpty = document.getElementById("segmentEmpty");
const segmentState = document.getElementById("segmentState");
const promptSummary = document.getElementById("promptSummary");

const finishPolygonBtn = document.getElementById("finishPolygonBtn");
const undoSegmentBtn = document.getElementById("undoSegmentBtn");
const clearSegmentBtn = document.getElementById("clearSegmentBtn");
const applySegmentationBtn = document.getElementById("applySegmentationBtn");

const emptyState = document.getElementById("emptyState");
const statusBox = document.getElementById("status");
const previewState = document.getElementById("previewState");

const methodSelect = document.getElementById("methodSelect");
const chromaOptions = document.getElementById("chromaOptions");
const methodHelp = document.getElementById("methodHelp");

// ============================================================================
// 2. TRẠNG THÁI ỨNG DỤNG (STATE)
// ============================================================================
let originalFile = null;       // Tệp ảnh gốc người dùng tải lên
let originalImage = null;      // Đối tượng Image của ảnh gốc để vẽ lên segmentCanvas
let subjectImage = null;       // Ảnh chủ thể RGBA đã tách nền để ghép lên previewCanvas
let backgroundImage = null;    // Ảnh nền mới (nếu người dùng tải ảnh nền)
let backgroundColor = "#e9edf5"; // Màu nền đơn sắc hiện tại

// Tọa độ tâm chủ thể trên previewCanvas theo tỷ lệ chuẩn hóa [0..1]
let subjectX = 0.5;
let subjectY = 0.52;

// Trạng thái kéo-thả di chuyển chủ thể trên previewCanvas
let dragging = false;
let lastPointer = null;

// Trạng thái công cụ phân đoạn tương tác và dữ liệu prompt gửi lên server
let currentTool = "point_fg";
let prompts = {
    points: [],
    rect: null,
    polygon: null
};

let polygonDraft = [];         // Các đỉnh tạm thời khi đang vẽ polygon chưa khép kín
let currentRect = null;        // Khung chữ nhật đang được kéo vẽ
let brushDrawing = false;      // Cờ báo đang tô bằng công cụ cọ
let lastBrushPoint = null;     // Điểm cọ cuối cùng để nội suy các điểm đều nhau

// Kết quả phân đoạn và ngăn xếp lịch sử hoàn tác (Undo)
let maskImage = null;
let segmentationImageUrl = null;
let segmentationResult = null;
let segmentationHistory = [];
let segmentationBusy = false;
let segmentRequestId = 0;      // Bộ đếm request giúp bỏ qua kết quả trả về muộn (race condition)

// Danh sách ID các thanh trượt điều khiển và giá trị mặc định tương ứng
const controls = [
    "scale",
    "brightness",
    "contrast",
    "saturation",
    "bgBlur",
    "feather",
    "keyTolerance"
];

const defaults = {
    scale: 100,
    brightness: 100,
    contrast: 100,
    saturation: 100,
    bgBlur: 0,
    feather: 1.2,
    keyTolerance: 75
};

// ============================================================================
// 3. CÁC HÀM TIỆN ÍCH & HIỂN THỊ XEM TRƯỚC (LIVE PREVIEW)
// ============================================================================

/** Hiển thị thông báo trạng thái kèm kiểu (mặc định, 'ok', 'error'). */
function setStatus(message, type = "") {
    statusBox.textContent = message;
    statusBox.className = "status " + type;
}

/** Tải ảnh bất đồng bộ từ URL hoặc Object URL và trả về Promise<HTMLImageElement>. */
function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();

        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

/** Đồng bộ giá trị hiển thị (output) cạnh các thanh trượt điều khiển. */
function updateOutputs() {
    for (const id of controls) {
        const element = document.getElementById(id);
        const output = document.getElementById(id + "Value");

        if (!element || !output) {
            continue;
        }

        let suffix = "%";

        if (id === "bgBlur") {
            suffix = " px";
        }

        if (id === "feather" || id === "keyTolerance") {
            suffix = "";
        }

        output.value = element.value + suffix;
        output.textContent = element.value + suffix;
    }
}

/** Cập nhật giao diện tùy chọn (Chroma Key, nhãn nút, mô tả) theo thuật toán đang chọn. */
function updateMethodOptions() {
    const method = methodSelect.value;

    chromaOptions.hidden = method !== "chroma";
    segmentEditor.hidden = false;

    const labels = {
        interactive: ["Phân đoạn bằng SAM 2", "SAM 2 nhận diện đối tượng theo điểm, khung hoặc polygon."],
        ai: ["Tách nền bằng U²-Net", "Có thể dùng điểm và vùng chọn để hiệu chỉnh mask AI."],
        grabcut: ["Phân đoạn bằng GrabCut", "Dùng seed foreground/background và ROI để tinh chỉnh mask."],
        chroma: ["Xóa phông Chroma Key", "Dùng màu phông kết hợp vùng chọn để hạn chế xóa nhầm."],
        watershed: ["Phân đoạn Watershed", "Dùng kết quả Watershed và các điểm/vùng chọn để hiệu chỉnh."],
        kmeans: ["Phân đoạn K-means", "Phân cụm màu, sau đó áp dụng các điểm và vùng chọn."],
        otsu: ["Phân đoạn Otsu", "Ngưỡng Otsu kết hợp các điểm và vùng chọn."],
        adaptive: ["Adaptive Threshold", "Ngưỡng cục bộ kết hợp các điểm và vùng chọn."],
        contour: ["Phân đoạn Contour", "Tìm đường bao và giới hạn theo lựa chọn của bạn."]
    };

    removeBtn.textContent = labels[method][0];
    methodHelp.textContent = labels[method][1];

    renderSegmentEditor();
}

/** Vẽ ảnh nền phủ kín canvas xem trước theo tỷ lệ cover, có hỗ trợ làm mờ (blur). */
function drawCover(image, blur = 0) {
    const cw = canvas.width;
    const ch = canvas.height;

    const scale = Math.max(
        cw / image.width,
        ch / image.height
    );

    const width = image.width * scale;
    const height = image.height * scale;

    ctx.save();

    if (blur > 0) {
        ctx.filter = `blur(${blur}px)`;
    }

    ctx.drawImage(
        image,
        (cw - width) / 2,
        (ch - height) / 2,
        width,
        height
    );

    ctx.restore();
}

/** Vẽ lại toàn bộ canvas xem trước: lớp nền (ảnh hoặc màu) + chủ thể đã tách nền kèm bộ lọc màu. */
function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (backgroundImage) {
        drawCover(
            backgroundImage,
            Number(document.getElementById("bgBlur").value)
        );
    } else {
        ctx.fillStyle = backgroundColor || "#e9edf5";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    if (subjectImage) {
        const scaleValue =
            Number(document.getElementById("scale").value) / 100;

        const maxWidth = canvas.width * 0.72;
        const maxHeight = canvas.height * 0.88;

        const fit = Math.min(
            maxWidth / subjectImage.width,
            maxHeight / subjectImage.height
        );

        const width = subjectImage.width * fit * scaleValue;
        const height = subjectImage.height * fit * scaleValue;

        const x = subjectX * canvas.width - width / 2;
        const y = subjectY * canvas.height - height / 2;

        ctx.save();

        ctx.filter = [
            `brightness(${document.getElementById("brightness").value}%)`,
            `contrast(${document.getElementById("contrast").value}%)`,
            `saturate(${document.getElementById("saturation").value}%)`
        ].join(" ");

        ctx.drawImage(
            subjectImage,
            x,
            y,
            width,
            height
        );

        ctx.restore();
    }

    emptyState.style.display = subjectImage ? "none" : "flex";
    downloadBtn.disabled = !subjectImage;
}

// ============================================================================
// 4. QUẢN LÝ TRẠNG THÁI & VẼ TRÊN CANVAS PHÂN ĐOẠN TƯƠNG TÁC
// ============================================================================

/** Đặt lại toàn bộ điểm, vùng chọn, mask và lịch sử hoàn tác về trạng thái ban đầu. */
function resetSegmentation() {
    prompts = {
        points: [],
        rect: null,
        polygon: null
    };

    polygonDraft = [];
    currentRect = null;
    maskImage = null;
    segmentationResult = null;
    segmentationImageUrl = null;

    segmentationHistory = [];

    applySegmentationBtn.disabled = true;
    promptSummary.textContent = "Chưa có điểm hoặc vùng chọn.";

    renderSegmentEditor();
}

/** Lưu bản sao trạng thái prompts hiện tại vào ngăn xếp lịch sử (giới hạn tối đa 50 bước). */
function saveSegmentationHistory() {
    segmentationHistory.push(
        JSON.parse(JSON.stringify(prompts))
    );

    if (segmentationHistory.length > 50) {
        segmentationHistory.shift();
    }
}

/** Cập nhật dòng tóm tắt số lượng điểm giữ/xóa, khung chữ nhật và polygon đang có. */
function updatePromptSummary() {
    const positive = prompts.points.filter(
        point => point.label === 1
    ).length;

    const negative = prompts.points.filter(
        point => point.label === 0
    ).length;

    const rectText = prompts.rect ? "Có khung chữ nhật" : "";
    const polygonText = prompts.polygon ? "Có polygon" : "";

    const parts = [];

    if (positive) {
        parts.push(`${positive} điểm giữ`);
    }

    if (negative) {
        parts.push(`${negative} điểm xóa`);
    }

    if (rectText) {
        parts.push(rectText);
    }

    if (polygonText) {
        parts.push(polygonText);
    }

    promptSummary.textContent =
        parts.length > 0
            ? parts.join(" · ")
            : "Chưa có điểm hoặc vùng chọn.";
}

/** Tính khung hiển thị thực tế của ảnh gốc bên trong segmentCanvas (do object-fit: contain). */
function getImageDisplayRect() {
    if (!originalImage) {
        return null;
    }

    const rect = segmentCanvas.getBoundingClientRect();

    const scale = Math.min(
        rect.width / originalImage.width,
        rect.height / originalImage.height
    );

    const width = originalImage.width * scale;
    const height = originalImage.height * scale;

    return {
        left: rect.left + (rect.width - width) / 2,
        top: rect.top + (rect.height - height) / 2,
        width,
        height,
        scale
    };
}

/** Quy đổi tọa độ con trỏ chuột trên màn hình sang tọa độ pixel của ảnh gốc. */
function pointerToImage(event) {
    const display = getImageDisplayRect();

    if (!display) {
        return null;
    }

    const x = (event.clientX - display.left) / display.scale;
    const y = (event.clientY - display.top) / display.scale;

    if (
        x < 0 ||
        y < 0 ||
        x >= originalImage.width ||
        y >= originalImage.height
    ) {
        return null;
    }

    return {
        x: Math.round(x),
        y: Math.round(y)
    };
}

/** Vẽ lớp phủ mask màu xanh bán trong suốt lên segmentCanvas bằng kỹ thuật destination-in. */
function drawMaskOverlay() {
    if (!maskImage || !originalImage) {
        return;
    }

    const overlay = document.createElement("canvas");

    overlay.width = segmentCanvas.width;
    overlay.height = segmentCanvas.height;

    const overlayCtx = overlay.getContext("2d");

    overlayCtx.fillStyle = "rgba(35, 190, 115, 0.48)";
    overlayCtx.fillRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    overlayCtx.globalCompositeOperation = "destination-in";

    overlayCtx.drawImage(
        maskImage,
        0,
        0,
        overlay.width,
        overlay.height
    );

    segmentCtx.save();
    segmentCtx.drawImage(overlay, 0, 0);
    segmentCtx.restore();
}

/** Vẽ các điểm giữ (xanh lá) và điểm xóa (đỏ) kèm viền trắng lên segmentCanvas. */
function drawPoints() {
    const radius = Math.max(
        7,
        Math.min(segmentCanvas.width, segmentCanvas.height) / 110
    );

    for (const point of prompts.points) {
        segmentCtx.beginPath();

        segmentCtx.arc(
            point.x,
            point.y,
            radius,
            0,
            Math.PI * 2
        );

        segmentCtx.fillStyle =
            point.label === 1 ? "#19c37d" : "#ed4b5c";

        segmentCtx.fill();

        segmentCtx.lineWidth = Math.max(2, radius * 0.25);
        segmentCtx.strokeStyle = "#ffffff";
        segmentCtx.stroke();
    }
}

/** Vẽ khung chữ nhật nét đứt màu xanh dương biểu thị vùng chọn ROI. */
function drawRect(rect) {
    if (!rect) {
        return;
    }

    const x = Math.min(rect.x1, rect.x2);
    const y = Math.min(rect.y1, rect.y2);
    const width = Math.abs(rect.x2 - rect.x1);
    const height = Math.abs(rect.y2 - rect.y1);

    segmentCtx.save();

    segmentCtx.strokeStyle = "#4263eb";
    segmentCtx.lineWidth = Math.max(
        3,
        segmentCanvas.width / 600
    );
    segmentCtx.setLineDash([12, 8]);

    segmentCtx.strokeRect(
        x,
        y,
        width,
        height
    );

    segmentCtx.fillStyle = "rgba(66, 99, 235, 0.08)";
    segmentCtx.fillRect(x, y, width, height);

    segmentCtx.restore();
}

/** Vẽ đường đa giác (polygon) màu cam; nếu đã khép kín (closed=true) thì tô phủ nhẹ bên trong. */
function drawPolygon(points, closed = false) {
    if (!points || points.length === 0) {
        return;
    }

    segmentCtx.save();

    segmentCtx.beginPath();
    segmentCtx.moveTo(points[0].x, points[0].y);

    for (let index = 1; index < points.length; index++) {
        segmentCtx.lineTo(
            points[index].x,
            points[index].y
        );
    }

    if (closed) {
        segmentCtx.closePath();
    }

    segmentCtx.strokeStyle = "#f0a020";
    segmentCtx.lineWidth = Math.max(
        3,
        segmentCanvas.width / 600
    );
    segmentCtx.stroke();

    if (closed) {
        segmentCtx.fillStyle = "rgba(240, 160, 32, 0.12)";
        segmentCtx.fill();
    }

    segmentCtx.restore();
}

/** Vẽ lại toàn bộ trình chỉnh sửa phân đoạn: ảnh gốc -> mask -> khung -> polygon -> các điểm. */
function renderSegmentEditor() {
    if (!originalImage) {
        segmentEmpty.style.display = "flex";
        segmentCtx.clearRect(
            0,
            0,
            segmentCanvas.width,
            segmentCanvas.height
        );
        return;
    }

    segmentEmpty.style.display = "none";

    segmentCtx.clearRect(
        0,
        0,
        segmentCanvas.width,
        segmentCanvas.height
    );

    segmentCtx.drawImage(
        originalImage,
        0,
        0,
        segmentCanvas.width,
        segmentCanvas.height
    );

    drawMaskOverlay();

    drawRect(prompts.rect);
    drawRect(currentRect);

    if (prompts.polygon) {
        drawPolygon(prompts.polygon, true);
    }

    if (polygonDraft.length > 0) {
        drawPolygon(polygonDraft, false);
    }

    drawPoints();

    updatePromptSummary();
}

// ============================================================================
// 5. CÔNG CỤ CHỌN VÙNG & GỌI API PHÂN ĐOẠN (/api/segment)
// ============================================================================

/** Kích hoạt công cụ chọn vùng tương ứng và hiển thị hướng dẫn nhanh. */
function setActiveTool(tool) {
    currentTool = tool;

    document.querySelectorAll(".tool-button").forEach(button => {
        button.classList.toggle(
            "active",
            button.dataset.tool === tool
        );
    });

    const labels = {
        point_fg: "Click vào vùng thuộc đối tượng cần giữ.",
        point_bg: "Click vào vùng nền cần loại bỏ.",
        rect: "Kéo chuột để khoanh hình chữ nhật quanh đối tượng.",
        polygon: "Click lần lượt các điểm để tạo polygon.",
        brush_fg: "Tô lên vùng cần giữ.",
        brush_bg: "Tô lên vùng cần xóa."
    };

    setStatus(labels[tool] || "Chọn công cụ.");
}

/** Thêm một điểm giữ (label=1) hoặc điểm xóa (label=0) và chạy lại phân đoạn ngay. */
function addPoint(point, label) {
    if (!point) {
        return;
    }

    saveSegmentationHistory();

    prompts.points.push({
        x: point.x,
        y: point.y,
        label
    });

    runSegmentation();
}

/** Lấy mẫu các điểm dọc theo nét kéo cọ (nội suy tuyến tính để tránh quá thưa hoặc quá dày). */
function addBrushPoint(point, label) {
    if (!point) {
        return;
    }

    if (!lastBrushPoint) {
        prompts.points.push({
            x: point.x,
            y: point.y,
            label
        });

        lastBrushPoint = point;
        return;
    }

    const dx = point.x - lastBrushPoint.x;
    const dy = point.y - lastBrushPoint.y;

    const distance = Math.sqrt(dx * dx + dy * dy);

    const step = Math.max(
        10,
        Math.min(originalImage.width, originalImage.height) / 100
    );

    if (distance < step) {
        return;
    }

    const count = Math.ceil(distance / step);

    for (let index = 1; index <= count; index++) {
        const ratio = index / count;

        prompts.points.push({
            x: Math.round(lastBrushPoint.x + dx * ratio),
            y: Math.round(lastBrushPoint.y + dy * ratio),
            label
        });
    }

    lastBrushPoint = point;
}

/** Khép kín đa giác nháp (cần ít nhất 3 đỉnh) và gửi yêu cầu phân đoạn. */
function finishPolygon() {
    if (polygonDraft.length < 3) {
        setStatus("Polygon cần ít nhất 3 điểm.", "error");
        return;
    }

    saveSegmentationHistory();

    prompts.polygon = [...polygonDraft];
    polygonDraft = [];

    renderSegmentEditor();
    runSegmentation();
}

/** Hoàn tác bước gần nhất (xóa đỉnh polygon nháp trước, sau đó mới khôi phục từ lịch sử). */
function undoSegmentation() {
    if (polygonDraft.length > 0) {
        polygonDraft.pop();
        renderSegmentEditor();
        return;
    }

    const previous = segmentationHistory.pop();

    if (!previous) {
        setStatus("Không còn thao tác để hoàn tác.");
        return;
    }

    prompts = previous;

    maskImage = null;
    segmentationResult = null;
    segmentationImageUrl = null;

    applySegmentationBtn.disabled = true;

    renderSegmentEditor();

    if (prompts.points.length || prompts.rect) {
        runSegmentation();
    }
}

/** Gửi ảnh gốc, prompts và tham số thuật toán lên /api/segment rồi hiển thị mask trả về. */
async function runSegmentation() {
    if (!originalFile) {
        setStatus("Vui lòng tải ảnh chủ thể trước.", "error");
        return null;
    }

    if (
        prompts.points.length === 0 &&
        !prompts.rect &&
        !prompts.polygon
    ) {
        setStatus(
            "Hãy thêm ít nhất một điểm hoặc một khung chọn.",
            "error"
        );
        return null;
    }

    if (prompts.points.length > 512) {
        setStatus(
            "Đã vượt quá giới hạn 512 điểm. Hãy xóa bớt điểm.",
            "error"
        );
        return null;
    }

    const requestId = ++segmentRequestId;

    segmentationBusy = true;
    segmentState.textContent = "Đang phân đoạn";
    applySegmentationBtn.disabled = true;

    setStatus(
        `Đang phân đoạn bằng ${methodSelect.options[methodSelect.selectedIndex].text}. Vui lòng chờ...`
    );

    const form = new FormData();

    form.append("image", originalFile);
    form.append("feather", document.getElementById("feather").value);
    form.append("prompts", JSON.stringify(prompts));
    form.append("method", methodSelect.value);
    form.append("key_color", document.getElementById("keyColor").value);
    form.append("key_tolerance", document.getElementById("keyTolerance").value);

    try {
        const response = await fetch("/api/segment", {
            method: "POST",
            body: form
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Không thể phân đoạn ảnh."
            );
        }

        if (requestId !== segmentRequestId) {
            return null;
        }

        const [loadedMask, loadedSubject] = await Promise.all([
            loadImage(data.mask_url + "?t=" + Date.now()),
            loadImage(data.image_url + "?t=" + Date.now())
        ]);

        if (requestId !== segmentRequestId) {
            return null;
        }

        maskImage = loadedMask;
        segmentationResult = loadedSubject;
        segmentationImageUrl = data.image_url;

        segmentState.textContent = "Đã tạo mask";
        segmentState.className = "pill success";

        applySegmentationBtn.disabled = false;

        setStatus(
            "Đã tạo mask. Bạn có thể thêm điểm để sửa vùng chọn hoặc dùng kết quả.",
            "ok"
        );

        renderSegmentEditor();

        return data;

    } catch (error) {
        if (requestId === segmentRequestId) {
            segmentState.textContent = "Có lỗi";
            setStatus(error.message, "error");
        }

        return null;

    } finally {
        if (requestId === segmentRequestId) {
            segmentationBusy = false;
        }
    }
}

/** Đưa ảnh chủ thể RGBA vừa phân đoạn sang khung xem trước ghép nền (Live Preview). */
function applySegmentation() {
    if (!segmentationResult) {
        setStatus("Chưa có kết quả phân đoạn để áp dụng.", "error");
        return;
    }

    subjectImage = segmentationResult;

    subjectX = 0.5;
    subjectY = 0.52;

    previewState.textContent = "Đã tách nền";
    previewState.className = "pill success";

    setStatus(
        "Đã áp dụng kết quả SAM 2. Bạn có thể thay nền và tải ảnh.",
        "ok"
    );

    render();
}

// ============================================================================
// 6. TẢI TỆP ẢNH CHỦ THỂ / ẢNH NỀN & HỖ TRỢ KÉO-THẢ (DRAG & DROP)
// ============================================================================

/** Kiểm tra định dạng/dung lượng ảnh chủ thể, nạp vào segmentCanvas và đặt lại trạng thái. */
async function setSubjectFile(file) {
    if (!file) {
        return;
    }

    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
        setStatus(
            "Định dạng không được hỗ trợ. Hãy chọn PNG, JPG hoặc WEBP.",
            "error"
        );
        return;
    }

    if (file.size > 12 * 1024 * 1024) {
        setStatus("Ảnh vượt quá giới hạn 12 MB.", "error");
        return;
    }

    try {
        originalFile = file;
        subjectImage = null;

        const objectUrl = URL.createObjectURL(file);
        originalImage = await loadImage(objectUrl);

        segmentCanvas.width = originalImage.naturalWidth;
        segmentCanvas.height = originalImage.naturalHeight;

        resetSegmentation();

        document.getElementById("subjectName").textContent = file.name;

        previewState.textContent = "Chờ tách nền";
        previewState.className = "pill";

        segmentState.textContent = "Đã tải ảnh";
        segmentState.className = "pill";

        render();
        renderSegmentEditor();

        setStatus(
            "Đã tải ảnh. Chọn công cụ và click vào đối tượng cần tách."
        );

        URL.revokeObjectURL(objectUrl);

    } catch (error) {
        setStatus("Không thể mở ảnh này.", "error");
    }
}

/** Nạp ảnh nền mới do người dùng chọn và bỏ chọn các ô màu nền đơn sắc. */
async function setBackgroundFile(file) {
    if (!file) {
        return;
    }

    try {
        const objectUrl = URL.createObjectURL(file);

        backgroundImage = await loadImage(objectUrl);
        backgroundColor = null;

        document.getElementById("backgroundName").textContent = file.name;

        document.querySelectorAll(".swatch").forEach(button => {
            button.classList.remove("selected");
        });

        render();
        URL.revokeObjectURL(objectUrl);

    } catch (error) {
        setStatus("Không thể mở ảnh nền này.", "error");
    }
}

/** Gắn sự kiện kéo-thả tệp (dragover, dragleave, drop) cho một vùng upload. */
function setupDrop(zone, handler) {
    zone.addEventListener("dragover", event => {
        event.preventDefault();
        zone.classList.add("drag");
    });

    zone.addEventListener("dragleave", () => {
        zone.classList.remove("drag");
    });

    zone.addEventListener("drop", event => {
        event.preventDefault();
        zone.classList.remove("drag");

        handler(event.dataTransfer.files[0]);
    });
}

// ============================================================================
// 7. ĐĂNG KÝ SỰ KIỆN GIAO DIỆN (EVENT LISTENERS)
// ============================================================================

// Chọn tệp qua input file và kích hoạt kéo-thả cho ảnh chủ thể & ảnh nền
subjectInput.addEventListener("change", event => {
    setSubjectFile(event.target.files[0]);
});

backgroundInput.addEventListener("change", event => {
    setBackgroundFile(event.target.files[0]);
});

setupDrop(subjectDrop, setSubjectFile);
setupDrop(backgroundDrop, setBackgroundFile);

// Chọn màu nền đơn sắc nhanh từ các ô màu (swatch)
document.querySelectorAll(".swatch").forEach(button => {
    button.addEventListener("click", () => {
        backgroundColor = button.dataset.color;
        backgroundImage = null;

        document.getElementById("backgroundName").textContent =
            "Nền màu " + button.title;

        document.querySelectorAll(".swatch").forEach(item => {
            item.classList.remove("selected");
        });

        button.classList.add("selected");
        render();
    });
});

// Lắng nghe thay đổi trên các thanh trượt hậu kỳ và tham số tách nền
controls.forEach(id => {
    const element = document.getElementById(id);

    element.addEventListener("input", () => {
        updateOutputs();
        render();

        if (id === "feather" && segmentationResult) {
            segmentationResult = null;
            maskImage = null;
            applySegmentationBtn.disabled = true;
            setStatus(
                "Bạn đã đổi độ mềm viền. Hãy phân đoạn lại để áp dụng giá trị mới."
            );
        }
    });
});

// Khi đổi thuật toán tách nền, xóa kết quả mask cũ để tránh nhầm lẫn
methodSelect.addEventListener("change", () => {
    segmentationResult = null;
    maskImage = null;
    segmentationImageUrl = null;
    applySegmentationBtn.disabled = true;
    updateMethodOptions();
});

document.querySelectorAll(".tool-button").forEach(button => {
    button.addEventListener("click", () => {
        setActiveTool(button.dataset.tool);
    });
});

finishPolygonBtn.addEventListener("click", finishPolygon);
undoSegmentBtn.addEventListener("click", undoSegmentation);

clearSegmentBtn.addEventListener("click", () => {
    segmentRequestId++;
    resetSegmentation();

    segmentState.textContent = "Đã xóa lựa chọn";
    setStatus("Đã xóa toàn bộ điểm và vùng chọn.");
});

applySegmentationBtn.addEventListener("click", applySegmentation);

// Sự kiện con trỏ trên segmentCanvas: chấm điểm, kéo khung chữ nhật, vẽ polygon, tô cọ
segmentCanvas.addEventListener("pointerdown", event => {
    if (!originalImage || segmentationBusy) {
        return;
    }

    const point = pointerToImage(event);

    if (!point) {
        return;
    }

    if (currentTool === "point_fg") {
        addPoint(point, 1);
        return;
    }

    if (currentTool === "point_bg") {
        addPoint(point, 0);
        return;
    }

    if (currentTool === "rect") {
        currentRect = {
            x1: point.x,
            y1: point.y,
            x2: point.x,
            y2: point.y
        };

        segmentCanvas.setPointerCapture(event.pointerId);
        renderSegmentEditor();
        return;
    }

    if (currentTool === "polygon") {
        polygonDraft.push(point);
        renderSegmentEditor();
        return;
    }

    if (
        currentTool === "brush_fg" ||
        currentTool === "brush_bg"
    ) {
        saveSegmentationHistory();

        brushDrawing = true;
        lastBrushPoint = null;

        const label = currentTool === "brush_fg" ? 1 : 0;

        addBrushPoint(point, label);

        segmentCanvas.setPointerCapture(event.pointerId);
        renderSegmentEditor();
    }
});

segmentCanvas.addEventListener("pointermove", event => {
    if (!originalImage) {
        return;
    }

    const point = pointerToImage(event);

    if (!point) {
        return;
    }

    if (currentTool === "rect" && currentRect) {
        currentRect.x2 = point.x;
        currentRect.y2 = point.y;

        renderSegmentEditor();
        return;
    }

    if (brushDrawing) {
        const label = currentTool === "brush_fg" ? 1 : 0;

        addBrushPoint(point, label);
        renderSegmentEditor();
    }
});

/** Kết thúc thao tác kéo khung chữ nhật và chạy phân đoạn nếu khung đủ lớn (>= 5 px). */
function finishRectangle() {
    if (!currentRect) {
        return;
    }

    const rect = currentRect;
    currentRect = null;

    if (
        Math.abs(rect.x2 - rect.x1) < 5 ||
        Math.abs(rect.y2 - rect.y1) < 5
    ) {
        renderSegmentEditor();
        return;
    }

    saveSegmentationHistory();

    prompts.rect = rect;

    renderSegmentEditor();
    runSegmentation();
}

/** Kết thúc nét vẽ cọ, giới hạn tối đa 512 điểm và gửi phân đoạn. */
function finishBrush() {
    if (!brushDrawing) {
        return;
    }

    brushDrawing = false;
    lastBrushPoint = null;

    if (prompts.points.length > 512) {
        prompts.points = prompts.points.slice(-512);
    }

    renderSegmentEditor();
    runSegmentation();
}

segmentCanvas.addEventListener("pointerup", () => {
    if (currentTool === "rect") {
        finishRectangle();
    }

    if (brushDrawing) {
        finishBrush();
    }
});

segmentCanvas.addEventListener("pointercancel", () => {
    currentRect = null;

    if (brushDrawing) {
        finishBrush();
    }
});

// Nút xử lý chính: gọi phân đoạn tương tác nếu có prompt/chọn SAM 2, ngược lại gọi /api/remove tự động
removeBtn.addEventListener("click", async () => {
    if (!originalFile) {
        setStatus("Vui lòng tải ảnh chủ thể trước.", "error");
        return;
    }

    const method = methodSelect.value;

    removeBtn.disabled = true;

    try {
        const hasPrompts = prompts.points.length > 0 || prompts.rect || prompts.polygon;
        if (method === "interactive" || hasPrompts) {
            const result = await runSegmentation();
            if (!result) return;
            applySegmentation();
            return;
        }

        setStatus(
            method === "chroma"
                ? "Đang tạo mặt nạ từ màu phông..."
                : "Đang phân tích ảnh và tách nền. Lần chạy đầu có thể mất thêm thời gian."
        );

        const form = new FormData();

        form.append("image", originalFile);
        form.append("feather", document.getElementById("feather").value);
        form.append("method", method);
        form.append("key_color", document.getElementById("keyColor").value);
        form.append("key_tolerance", document.getElementById("keyTolerance").value);
        if (prompts.points.length || prompts.rect || prompts.polygon) {
            form.append("prompts", JSON.stringify(prompts));
        }

        const response = await fetch("/api/remove", {
            method: "POST",
            body: form
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Có lỗi xảy ra khi xử lý ảnh."
            );
        }

        subjectImage = await loadImage(
            data.image_url + "?t=" + Date.now()
        );

        subjectX = 0.5;
        subjectY = 0.52;

        previewState.textContent = "Đã tách nền";
        previewState.className = "pill success";

        setStatus(
            "Xử lý thành công. Bạn có thể chọn nền và tinh chỉnh ảnh.",
            "ok"
        );

        render();

    } catch (error) {
        setStatus(error.message, "error");

    } finally {
        removeBtn.disabled = false;
    }
});

// Đặt lại toàn bộ thông số hậu kỳ và đưa chủ thể về vị trí giữa ảnh
resetBtn.addEventListener("click", () => {
    for (const [id, value] of Object.entries(defaults)) {
        document.getElementById(id).value = value;
    }

    subjectX = 0.5;
    subjectY = 0.52;

    updateOutputs();
    render();
});

// Xuất nội dung previewCanvas hiện tại ra tệp ảnh PNG để người dùng tải về
downloadBtn.addEventListener("click", () => {
    if (!subjectImage) {
        return;
    }

    const link = document.createElement("a");

    link.download = "background-studio-result.png";
    link.href = canvas.toDataURL("image/png");

    link.click();
});

/** Quy đổi vị trí con trỏ trên previewCanvas sang tọa độ chuẩn hóa [0..1] để kéo di chuyển chủ thể. */
function previewPointerPosition(event) {
    const rect = canvas.getBoundingClientRect();

    return {
        x: (event.clientX - rect.left) / rect.width,
        y: (event.clientY - rect.top) / rect.height
    };
}

canvas.addEventListener("pointerdown", event => {
    if (!subjectImage) {
        return;
    }

    dragging = true;
    lastPointer = previewPointerPosition(event);

    canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", event => {
    if (!dragging || !lastPointer) {
        return;
    }

    const current = previewPointerPosition(event);

    subjectX += current.x - lastPointer.x;
    subjectY += current.y - lastPointer.y;

    lastPointer = current;

    render();
});

canvas.addEventListener("pointerup", () => {
    dragging = false;
    lastPointer = null;
});

canvas.addEventListener("pointercancel", () => {
    dragging = false;
    lastPointer = null;
});

// Khởi tạo giao diện ban đầu khi trang vừa tải xong
updateOutputs();
updateMethodOptions();
render();