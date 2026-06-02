/**
 * ASR FIXED FRONTEND
 */

const btnRecord     = document.getElementById("btn-record");
const btnText       = document.getElementById("btn-text");
const btnIconWrap   = document.getElementById("btn-icon");
const avatarRing    = document.getElementById("avatar-ring");
const micStatus     = document.getElementById("mic-status");

const RECORD_DURATION = 2000;

let mediaRecorder;
let streamGlobal;
let isRecording = false;
let countdown;

// ── RECORD ───────────────────────────────
btnRecord.addEventListener("click", async () => {
  if (isRecording) return;

  try {
    streamGlobal = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    alert("Mic tidak bisa diakses");
    return;
  }

  isRecording = true;
  const chunks = [];

  const options = {
    mimeType: "audio/webm;codecs=opus"
  };

  mediaRecorder = new MediaRecorder(streamGlobal, options);

  mediaRecorder.ondataavailable = e => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  mediaRecorder.onstop = async () => {
    streamGlobal.getTracks().forEach(t => t.stop());
    isRecording = false;

    // ⚠️ FIX: jangan paksa type webm di blob
    const blob = new Blob(chunks);

    const form = new FormData();
    form.append("audio", blob, "audio.webm");

    const res = await fetch("/api/predict", {
      method: "POST",
      body: form
    });

    const data = await res.json();
    console.log(data);
  };

  mediaRecorder.start();

  setTimeout(() => {
    mediaRecorder.stop();
  }, RECORD_DURATION);
});