/**
 * ASR frontend
 */

const btnRecord = document.getElementById("btn-record");
const btnText = document.getElementById("btn-text");
const avatarRing = document.getElementById("avatar-ring");
const micStatus = document.getElementById("mic-status");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const resultDigit = document.getElementById("result-digit");
const resultConfidence = document.getElementById("result-confidence");
const resultBars = document.getElementById("result-bars");
const statusMsg = document.getElementById("status-msg");

const RECORD_DURATION = 2000;

let mediaRecorder;
let streamGlobal;
let isRecording = false;

function setStatus(message, type = "info") {
  statusMsg.textContent = message;
  statusMsg.className = `status-msg ${type}`;
  statusMsg.style.display = "block";
}

function clearStatus() {
  statusMsg.textContent = "";
  statusMsg.style.display = "none";
}

function setRecordingUi(recording) {
  btnRecord.classList.toggle("recording", recording);
  avatarRing.classList.toggle("recording", recording);
  btnRecord.disabled = recording;
  btnText.textContent = recording ? "Merekam..." : "Prediksi Suara";
  micStatus.textContent = recording ? "Dengarkan angka selama 2 detik" : "Tekan tombol untuk merekam";
}

function renderResult(data) {
  resultEmpty.style.display = "none";
  resultContent.style.display = "flex";
  resultDigit.textContent = data.prediction;
  resultConfidence.textContent = `Confidence ${data.confidence}%`;

  resultBars.innerHTML = "";
  data.top5.forEach((item, index) => {
    const bar = document.createElement("div");
    bar.className = "bar-item";
    bar.innerHTML = `
      <div class="bar-header">
        <span class="bar-label">${item.label}</span>
        <span class="bar-pct">${item.probability}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill rank-${index + 1}" style="width: ${item.probability}%"></div>
      </div>
    `;
    resultBars.appendChild(bar);
  });
}

function writeString(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function write16BitPcmSample(view, offset, input) {
  const sample = Math.max(-1, Math.min(1, input));
  view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
}

function audioBufferToWav(buffer) {
  const channelCount = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const samples = buffer.length;
  const blockAlign = channelCount * 2;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples * blockAlign;
  const wavBuffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(wavBuffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples; i += 1) {
    for (let channel = 0; channel < channelCount; channel += 1) {
      const sample = buffer.getChannelData(channel)[i];
      write16BitPcmSample(view, offset, sample);
      offset += 2;
    }
  }

  return new Blob([wavBuffer], { type: "audio/wav" });
}

async function recordedBlobToWav(blob) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContextClass();
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return audioBufferToWav(audioBuffer);
  } finally {
    audioContext.close();
  }
}

async function sendAudio(blob) {
  setStatus("Memproses audio...", "info");

  const wavBlob = await recordedBlobToWav(blob);
  const form = new FormData();
  form.append("audio", wavBlob, "audio.wav");

  const res = await fetch("/api/predict", {
    method: "POST",
    body: form
  });

  const data = await res.json();
  if (!res.ok || !data.success) {
    throw new Error(data.message || "Prediksi gagal");
  }

  renderResult(data);
  setStatus("Prediksi berhasil", "info");
}

btnRecord.addEventListener("click", async () => {
  if (isRecording) return;

  clearStatus();

  try {
    streamGlobal = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    setStatus("Mic tidak bisa diakses", "error");
    return;
  }

  isRecording = true;
  setRecordingUi(true);
  const chunks = [];

  try {
    mediaRecorder = new MediaRecorder(streamGlobal, {
      mimeType: "audio/webm;codecs=opus"
    });
  } catch (e) {
    mediaRecorder = new MediaRecorder(streamGlobal);
  }

  mediaRecorder.ondataavailable = e => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  mediaRecorder.onstop = async () => {
    streamGlobal.getTracks().forEach(track => track.stop());
    streamGlobal = null;
    isRecording = false;
    setRecordingUi(false);

    try {
      const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
      await sendAudio(blob);
    } catch (e) {
      setStatus(e.message, "error");
    }
  };

  mediaRecorder.start();

  setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  }, RECORD_DURATION);
});
