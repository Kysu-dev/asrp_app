const btnRecord     = document.getElementById("btn-record");
const btnText       = document.getElementById("btn-text");
const avatarRing    = document.getElementById("avatar-ring");
const micStatus     = document.getElementById("mic-status");
const resultEmpty   = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const resultDigit   = document.getElementById("result-digit");
const resultConf    = document.getElementById("result-confidence");
const resultBars    = document.getElementById("result-bars");
const statusMsg     = document.getElementById("status-msg");
const timerRing     = document.getElementById("timer-ring");
const timerText     = document.getElementById("timer-text");
const timerArc      = document.getElementById("timer-arc");
const historyList   = document.getElementById("history-list");
const btnHistory    = document.getElementById("btn-history");
const historyPanel  = document.getElementById("history-panel");
const uploadArea    = document.getElementById("upload-area");
const fileInput     = document.getElementById("file-input");
const fileInfo      = document.getElementById("file-info");
const fileName      = document.getElementById("file-name");
const btnUpload     = document.getElementById("btn-upload");

const RECORD_MS = 2000;

let mediaRecorder, streamGlobal;
let isRecording  = false;
let history      = [];
let selectedFile = null;
let activeTab    = "mic";


function setStatus(msg, type = "info") {
  statusMsg.textContent  = msg;
  statusMsg.className    = `status-msg ${type}`;
  statusMsg.style.display = "block";
}
function clearStatus() { statusMsg.style.display = "none"; }


function switchTab(tab) {
  activeTab = tab;
  document.getElementById("tab-mic").classList.toggle("active",  tab === "mic");
  document.getElementById("tab-file").classList.toggle("active", tab === "file");
  document.getElementById("pane-mic").style.display  = tab === "mic"  ? "flex" : "none";
  document.getElementById("pane-file").style.display = tab === "file" ? "flex" : "none";
  document.getElementById("pane-mic").style.flexDirection  = "column";
  document.getElementById("pane-file").style.flexDirection = "column";
  clearStatus();
}


let timerInterval = null;

function startTimer(durationMs) {
  avatarRing.style.display = "none";
  timerRing.style.display  = "flex";

  const steps     = durationMs / 100;
  let   elapsed   = 0;
  const circumference = 100;

  timerArc.style.strokeDashoffset = "0";

  timerInterval = setInterval(() => {
    elapsed++;
    const remaining  = Math.max(0, durationMs - elapsed * 100);
    const pct        = elapsed / steps;
    const offset     = pct * circumference;

    timerArc.style.strokeDashoffset = offset;
    timerText.textContent = (remaining / 1000).toFixed(1) + "s";

    if (elapsed >= steps) clearInterval(timerInterval);
  }, 100);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerRing.style.display  = "none";
  avatarRing.style.display = "flex";
}


function setRecordingUi(recording) {
  btnRecord.classList.toggle("recording", recording);
  avatarRing.classList.toggle("recording", recording);
  // Remove btnRecord.disabled = recording; so it can detect mouseup
  btnText.textContent = recording ? "Lepas untuk Selesai" : "Tekan & Tahan untuk Bicara";
  micStatus.textContent = recording
    ? "Ucapkan angka dengan jelas..."
    : "Tekan dan Tahan tombol, lalu ucapkan angka";
}


function renderResult(data) {
  resultEmpty.style.display   = "none";
  resultContent.style.display = "flex";

  resultDigit.textContent     = data.prediction;
  resultConf.textContent      = `Confidence: ${data.confidence}%`;
  
  // ── FITUR TAMBAHAN: Tampilkan Visualisasi MFCC ──
  const imgElem = document.getElementById("mfcc-image");
  if (data.mfcc_image) {
    imgElem.src = data.mfcc_image;
    imgElem.style.display = "block";
  } else {
    imgElem.style.display = "none";
  }
  
  // ── FITUR TAMBAHAN: Integrasi ASR -> TTS ──
  speakResult(data.prediction);

  resultBars.innerHTML = "";
  data.top5.forEach((item, idx) => {
    const div = document.createElement("div");
    div.className = "bar-item";
    div.innerHTML = `
      <div class="bar-header">
        <span class="bar-label">Angka ${item.label}</span>
        <span class="bar-pct">${item.probability}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill rank-${idx + 1}" style="width:0%"></div>
      </div>`;
    resultBars.appendChild(div);

    setTimeout(() => {
      div.querySelector(".bar-fill").style.width = `${item.probability}%`;
    }, 50 + idx * 60);
  });

  addHistory(data);
}


// ── FITUR TAMBAHAN: Fungsi Text to Speech ──
function speakResult(digit) {
  // Deteksi radio button gender
  const genderRadios = document.getElementsByName("tts-gender");
  let isMale = false;
  for (let r of genderRadios) {
    if (r.checked && r.value.includes("B")) {
      isMale = true;
      break;
    }
  }

  const gender = isMale ? "male" : "female";
  const audioUrl = `/static/audio/${digit}_${gender}.mp3`;
  
  const audio = new Audio(audioUrl);
  audio.play().catch(e => console.log("Gagal memutar TTS:", e));
}

function addHistory(data) {
  history.unshift(data);
  if (history.length > 8) history.pop();

  btnHistory.style.display = "block";
  renderHistory();
}

function renderHistory() {
  historyList.innerHTML = "";
  history.forEach(item => {
    const chip = document.createElement("div");
    chip.className = "history-chip";
    chip.innerHTML = `
      <span class="chip-digit">${item.prediction}</span>
      <span class="chip-conf">${item.confidence}%</span>`;
    historyList.appendChild(chip);
  });
}

function toggleHistory() {
  const visible = historyPanel.style.display !== "none";
  historyPanel.style.display = visible ? "none" : "block";
  btnHistory.textContent = visible ? "Riwayat" : "Tutup";
}


function writeString(view, offset, text) {
  for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
}

function write16bit(view, offset, input) {
  const s = Math.max(-1, Math.min(1, input));
  view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
}

function bufferToWav(buf) {
  const ch = buf.numberOfChannels, sr = buf.sampleRate, n = buf.length;
  const ba = ch * 2, ds = n * ba;
  const ab = new ArrayBuffer(44 + ds);
  const v  = new DataView(ab);
  writeString(v, 0, "RIFF"); v.setUint32(4, 36 + ds, true);
  writeString(v, 8, "WAVE"); writeString(v, 12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, ch, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * ba, true); v.setUint16(32, ba, true);
  v.setUint16(34, 16, true); writeString(v, 36, "data");
  v.setUint32(40, ds, true);
  let off = 44;
  for (let i = 0; i < n; i++)
    for (let c = 0; c < ch; c++) { write16bit(v, off, buf.getChannelData(c)[i]); off += 2; }
  return new Blob([ab], { type: "audio/wav" });
}

async function blobToWav(blob) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const ab  = await blob.arrayBuffer();
    const buf = await ctx.decodeAudioData(ab);
    return bufferToWav(buf);
  } finally { ctx.close(); }
}


async function sendAudio(blob, isFile = false, filename = "audio.wav") {
  setStatus("⏳ Memproses audio...", "loading");

  const wavBlob = isFile ? blob : await blobToWav(blob);
  const form    = new FormData();
  form.append("audio", wavBlob, filename);

  const res  = await fetch("/api/predict", { method: "POST", body: form });
  const data = await res.json();

  if (!res.ok || !data.success) throw new Error(data.message || "Prediksi gagal");

  renderResult(data);
  setStatus(`✓ Terdeteksi: Angka ${data.prediction} (${data.confidence}%)`, "info");
}


async function startRecording() {
  if (isRecording) return;
  clearStatus();

  try {
    streamGlobal = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
        sampleRate: { ideal: 16000 }
      }
    });
  } catch {
    setStatus("Mic tidak bisa diakses. Cek izin browser.", "error");
    return;
  }

  isRecording = true;
  setRecordingUi(true);
  
  // Custom timer UI without 2s limit
  avatarRing.style.display = "none";
  timerRing.style.display  = "flex";
  timerArc.style.strokeDashoffset = "0";
  timerText.textContent = "REC";

  const chunks = [];
  try {
    mediaRecorder = new MediaRecorder(streamGlobal, { mimeType: "audio/webm;codecs=opus" });
  } catch {
    mediaRecorder = new MediaRecorder(streamGlobal);
  }

  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };

  mediaRecorder.onstop = async () => {
    streamGlobal.getTracks().forEach(t => t.stop());
    streamGlobal = null;
    isRecording  = false;
    
    timerRing.style.display  = "none";
    avatarRing.style.display = "flex";
    setRecordingUi(false);

    try {
      const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
      await sendAudio(blob, false, "audio.wav");
    } catch (e) {
      setStatus("❌ " + e.message, "error");
    }
  };

  mediaRecorder.start();
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
}

if (btnRecord) {
  btnRecord.addEventListener("mousedown", startRecording);
  btnRecord.addEventListener("mouseup", stopRecording);
  btnRecord.addEventListener("mouseleave", stopRecording);
  
  btnRecord.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
  btnRecord.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });
  btnRecord.addEventListener("touchcancel", (e) => { e.preventDefault(); stopRecording(); });
}


if (uploadArea) {
  uploadArea.addEventListener("click", () => fileInput.click());

  uploadArea.addEventListener("dragover", e => {
    e.preventDefault();
    uploadArea.classList.add("drag-over");
  });

  uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));

  uploadArea.addEventListener("drop", e => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });
}

function setFile(f) {
  selectedFile = f;
  fileName.textContent        = f.name;
  fileInfo.style.display      = "flex";
  btnUpload.style.display     = "flex";
  uploadArea.style.display    = "none";
  clearStatus();
}

function clearFile() {
  selectedFile                = null;
  fileInfo.style.display      = "none";
  btnUpload.style.display     = "none";
  uploadArea.style.display    = "flex";
  fileInput.value             = "";
  clearStatus();
}

async function submitFile() {
  if (!selectedFile) return;
  clearStatus();
  btnUpload.disabled = true;

  try {
    await sendAudio(selectedFile, true, selectedFile.name || "audio.wav");
  } catch (e) {
    setStatus("❌ " + e.message, "error");
  } finally {
    btnUpload.disabled = false;
  }
}
