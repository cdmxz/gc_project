/* 垃圾分类前端交互：文件校验 -> 发送到 Flask 后端 -> 渲染真实推理结果 */
(() => {
  "use strict";

  const MAX_MB = 8;
  const ALLOWED = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
  const CLASS_ZH = {
    cardboard: "纸板", glass: "玻璃", metal: "金属",
    paper: "纸张", plastic: "塑料", trash: "其他垃圾",
  };

  const $ = (id) => document.getElementById(id);
  const dropzone = $("dropzone");
  const fileInput = $("fileInput");
  const preview = $("preview");
  const emptyState = $("emptyState");
  const dropHint = $("dropHint");
  const fileMeta = $("fileMeta");
  const errorBox = $("errorBox");
  const predictBtn = $("predictBtn");
  const resultIdle = $("resultIdle");
  const resultLoading = $("resultLoading");
  const resultBox = $("resultBox");

  let currentFile = null;
  let previewUrl = null;
  let modelInfo = null;
  let cameraStream = null;
  let busy = false;

  /* ---------------- 工具 ---------------- */
  const show = (el) => el.classList.remove("hidden");
  const hide = (el) => el.classList.add("hidden");
  const fmtKB = (bytes) => (bytes / 1024 > 1024
    ? (bytes / 1024 / 1024).toFixed(2) + " MB"
    : (bytes / 1024).toFixed(0) + " KB");

  function showError(msg) {
    errorBox.textContent = msg;
    show(errorBox);
  }
  function clearError() {
    errorBox.textContent = "";
    hide(errorBox);
  }

  /* ---------------- 模型信息 ---------------- */
  async function loadModelInfo() {
    const dot = $("statusDot");
    const text = $("statusText");
    try {
      const res = await fetch("/api/model-info");
      if (!res.ok) throw new Error("HTTP " + res.status);
      modelInfo = await res.json();
      dot.className = "dot dot-ok";
      text.textContent = "服务正常 · 模型已加载";
      $("infoModel").textContent = modelInfo.model_name.toUpperCase() + "（ImageNet 预训练微调）";
      $("infoParams").textContent = modelInfo.num_params.toLocaleString() + " 个参数";
      $("infoSize").textContent = modelInfo.img_size + " × " + modelInfo.img_size;
      $("infoClasses").textContent = modelInfo.num_classes + " 类";
      $("infoDevice").textContent = modelInfo.device.toUpperCase();
      $("infoLoaded").textContent = modelInfo.loaded_at;
      $("modelTag").textContent = "模型 " + modelInfo.model_name + " · v" + modelInfo.version;
      $("deviceTag").textContent = "设备 " + modelInfo.device.toUpperCase();
      $("modelReadyTag").textContent = "模型就绪 · " + modelInfo.checkpoint;
      $("versionTag").textContent = "v" + modelInfo.version + " · torch " + modelInfo.torch_version;
      renderSamples();
    } catch (err) {
      dot.className = "dot dot-bad";
      text.textContent = "无法连接后端服务";
      $("modelReadyTag").textContent = "模型未就绪";
      showError("无法连接后端推理服务，请确认已运行 python src/app.py（" + err.message + "）");
    }
  }

  async function renderSamples() {
    const row = $("sampleRow");
    row.innerHTML = "";
    let samples = [];
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      const seen = new Set();
      samples = [];
      for (const s of data.samples || []) {
        if (seen.has(s.class_name)) continue;   // 每个类别只展示一张，保持单行
        seen.add(s.class_name);
        samples.push([s.file, s.class_zh]);
      }
    } catch (e) {
      return;
    }
    samples.forEach(([file, label]) => {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = "/samples/" + encodeURIComponent(file);
      img.alt = label + "样例";
      img.loading = "lazy";
      img.onerror = () => { fig.style.display = "none"; };
      const cap = document.createElement("figcaption");
      cap.textContent = label;
      fig.appendChild(img);
      fig.appendChild(cap);
      fig.addEventListener("click", () => {
        fetch("/samples/" + encodeURIComponent(file))
          .then((r) => r.blob())
          .then((blob) => {
            const f = new File([blob], file, { type: blob.type || "image/jpeg" });
            acceptFile(f);
            runPredict();
          })
          .catch(() => showError("样例图片加载失败"));
      });
      row.appendChild(fig);
    });
  }

  /* ---------------- 文件处理 ---------------- */
  function validate(file) {
    if (!file) return "未选择文件";
    if (!ALLOWED.includes(file.type) && !/\.(jpe?g|png|bmp|webp)$/i.test(file.name)) {
      return "文件类型不支持，请上传 JPG / PNG / BMP / WEBP 格式的图片";
    }
    if (file.size === 0) return "文件为空，请重新选择";
    if (file.size > MAX_MB * 1024 * 1024) {
      return "图片大小 " + fmtKB(file.size) + " 超过 " + MAX_MB + "MB 限制，请压缩后重试";
    }
    return null;
  }

  function acceptFile(file) {
    clearError();
    const err = validate(file);
    if (err) {
      showError(err);
      return false;
    }
    currentFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    hide(emptyState);
    show(preview);
    fileMeta.textContent = `已选择：${file.name} · ${fmtKB(file.size)} · ${file.type || "未知类型"}`;
    predictBtn.disabled = false;
    resetResult();
    return true;
  }

  function resetResult() {
    show(resultIdle);
    hide(resultLoading);
    hide(resultBox);
    $("latency").textContent = "推理耗时 —";
  }

  function resetAll() {
    currentFile = null;
    if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
    preview.removeAttribute("src");
    hide(preview);
    show(emptyState);
    fileMeta.textContent = "尚未选择图片";
    predictBtn.disabled = true;
    fileInput.value = "";
    clearError();
    resetResult();
  }

  /* ---------------- 预测 ---------------- */
  async function runPredict() {
    if (busy) return;
    const err = validate(currentFile);
    if (err) { showError(err); return; }

    busy = true;
    clearError();
    hide(resultIdle);
    hide(resultBox);
    show(resultLoading);
    predictBtn.disabled = true;
    predictBtn.textContent = "识别中…";

    const form = new FormData();
    form.append("file", currentFile, currentFile.name);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "服务端返回错误 " + res.status);
      renderResult(data);
    } catch (e) {
      hide(resultLoading);
      show(resultIdle);
      showError("预测失败：" + e.message);
    } finally {
      busy = false;
      predictBtn.disabled = false;
      predictBtn.textContent = "开始识别";
    }
  }

  function renderResult(data) {
    hide(resultLoading);
    hide(resultIdle);
    show(resultBox);

    const zh = data.top1.class_zh || CLASS_ZH[data.top1.class_name] || data.top1.class_name;
    $("top1Name").textContent = zh;
    $("top1En").textContent = data.top1.class_name;
    $("top1Conf").textContent = (data.top1.probability * 100).toFixed(2) + "%";
    $("adviceText").textContent = data.advice || "—";
    $("latency").textContent = "推理耗时 " + data.inference_ms.toFixed(1) + " ms";
    $("modelTag").textContent = "模型 " + data.model_name;
    $("deviceTag").textContent = "设备 " + data.device.toUpperCase();

    const list = $("topkList");
    list.innerHTML = "";
    data.topk.forEach((item, i) => {
      const li = document.createElement("li");
      li.className = "topk-item";
      const label = `${i + 1}. ${item.class_zh} <span style="color:#93a39b;font-size:12px">${item.class_name}</span>`;
      li.innerHTML = `
        <span class="topk-name">${label}</span>
        <span class="topk-bar"><span class="topk-fill" style="width:${Math.max(1.5, item.probability * 100)}%"></span></span>
        <span class="topk-pct">${(item.probability * 100).toFixed(2)}%</span>`;
      list.appendChild(li);
    });

    if (data.top1.probability < 0.6) {
      $("adviceText").textContent = "置信度偏低，建议人工复核。 " + (data.advice || "");
    }
  }

  /* ---------------- 事件绑定 ---------------- */
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) acceptFile(e.target.files[0]);
  });

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
      show(dropHint);
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      hide(dropHint);
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) acceptFile(file);
  });

  // 支持 Ctrl+V 粘贴截图
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) acceptFile(item.getAsFile());
  });

  $("pickBtn").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  $("resetBtn").addEventListener("click", resetAll);
  $("predictBtn").addEventListener("click", runPredict);
  $("againBtn").addEventListener("click", () => {
    if (currentFile) runPredict();
    else fileInput.click();
  });

  /* ---------------- 摄像头 ---------------- */
  $("cameraBtn").addEventListener("click", async () => {
    clearError();
    if (!navigator.mediaDevices?.getUserMedia) {
      showError("当前浏览器不支持摄像头采集，请改用「选择本地图片」");
      return;
    }
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      $("video").srcObject = cameraStream;
      show($("cameraModal"));
    } catch (err) {
      showError("无法访问摄像头（" + err.message + "），请检查授权或改用本地上传");
    }
  });

  function closeCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      cameraStream = null;
    }
    hide($("cameraModal"));
  }
  $("closeCameraBtn").addEventListener("click", closeCamera);

  $("captureBtn").addEventListener("click", () => {
    const video = $("video");
    if (!video.videoWidth) { showError("摄像头尚未就绪，请稍后重试"); return; }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
      closeCamera();
      if (acceptFile(file)) runPredict();
    }, "image/jpeg", 0.92);
  });

  /* ---------------- 初始化 ---------------- */
  loadModelInfo();
})();
