const state = {
  coords: null, // { latitude, longitude, accuracy }
};

const locDiv   = document.getElementById("locStatus");
const resultEl = document.getElementById("result");
//const debugEl  = document.getElementById("debug");

function showLocStatus(msg) {
  locDiv.textContent = msg;
}

function tryBrowserLocation() {
  if (!navigator.geolocation) {
    showLocStatus("❌ Geolocation not supported by this browser.");
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      state.coords = {
        latitude: pos.coords.latitude,
        longitude: pos.coords.longitude,
        accuracy: pos.coords.accuracy,
      };
      const acc = Math.round(state.coords.accuracy || 0);
      showLocStatus(`✅ Location: ${state.coords.latitude}, ${state.coords.longitude} (±${acc}m)`);

      // If accuracy is poor (>1000m), hint user to try mobile/Wi-Fi
      if (acc > 1000) {
        locDiv.innerHTML +=
          "<br>ℹ️ Accuracy is low. For better accuracy, open this page on a phone with GPS or connect via Wi-Fi.";
      }
    },
    (err) => {
      showLocStatus("❌ " + err.message + " — falling back to IP location…");
      ipFallback();
    },
    { enableHighAccuracy: true, timeout: 10000 }
  );
}

async function ipFallback() {
  try {
    const r = await fetch("https://ipapi.co/json/");
    const j = await r.json();
    if (j.latitude && j.longitude) {
      state.coords = {
        latitude: j.latitude,
        longitude: j.longitude,
        accuracy: 5000, // rough
      };
      showLocStatus(`🌐 IP location: ${j.city || "Unknown"} (${j.latitude}, ${j.longitude})`);
    } else {
      showLocStatus("⚠️ IP geolocation unavailable.");
    }
  } catch (e) {
    showLocStatus("⚠️ IP fallback failed: " + e.message);
  }
}

async function callBackend(query) {
  resultEl.textContent = "Searching…";
  const payload = {
    query,
    latitude: state.coords?.latitude ?? null,
    longitude: state.coords?.longitude ?? null,
    accuracy: state.coords?.accuracy ?? null,
  };

  const res = await fetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  //debugEl.textContent = JSON.stringify(data.debug || data, null, 2);

  if (data.error) {
    resultEl.textContent = "❌ " + data.error;
    return;
  }

  if (data.mode === "maps") {
    const places = data.places || [];
    if (!places.length) { resultEl.textContent = "No local results."; return; }
    resultEl.innerHTML = "<h3>🍴 Nearby places</h3>" + places.map(p => `
        <div class="place">
          <div><b>${p.title || "Unknown"}</b></div>
          <div>📍 ${p.address || "-"}</div>
          <div>⭐ ${p.rating || "-"} (${p.reviews || "-"} reviews)</div>
          ${p.link ? `<div>🔗 <a href="${p.link}" target="_blank">Link</a></div>` : ""}
        </div>
    `).join("");
  } else if (data.mode === "news") {
    resultEl.innerHTML = "<h3>📰 News Digest</h3><div style='white-space:pre-wrap'>" + (data.digest || "") + "</div>";
  } else {
    resultEl.textContent = "No results.";
  }
}

document.getElementById("go").addEventListener("click", async () => {
  const q = document.getElementById("query").value.trim();
  if (!q) { resultEl.textContent = "Type a query."; return; }
  await callBackend(q);
});

// Start location flow on load
tryBrowserLocation();
