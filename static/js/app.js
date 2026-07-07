document.addEventListener('DOMContentLoaded', () => {
    // ---- Tabs ----
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active', 'border-blue-500', 'text-blue-400'));
            tabBtns.forEach(b => b.classList.add('text-[#94a3b8]'));
            tabContents.forEach(c => c.classList.remove('active', 'block'));
            tabContents.forEach(c => c.classList.add('hidden'));
            
            btn.classList.add('active', 'border-blue-500', 'text-blue-400');
            btn.classList.remove('text-[#94a3b8]');
            document.getElementById(btn.dataset.target).classList.add('active', 'block');
            document.getElementById(btn.dataset.target).classList.remove('hidden');
        });
    });

    // ---- Radar Chart ----
    const ctx = document.getElementById('l1-radar-chart').getContext('2d');
    const radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Phase Jump', 'Jitter', 'Noise Flr', 'Spectral', 'MFCC'],
            datasets: [{
                label: 'Your Audio',
                data: [0, 0, 0, 0, 0],
                backgroundColor: 'rgba(168, 85, 247, 0.4)',
                borderColor: '#a855f7',
                pointBackgroundColor: '#a855f7',
            }]
        },
        options: {
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: '#94a3b8' },
                    ticks: { display: false, min: 0, max: 1 }
                }
            },
            plugins: { legend: { display: false } },
            maintainAspectRatio: false
        }
    });

    // ---- Live Stream ----
    let ws = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let streamTimer = null;
    let rollingResults = [];
    
    const btnStart = document.getElementById('btn-start-stream');
    const btnStop = document.getElementById('btn-stop-stream');
    const statusDiv = document.getElementById('stream-status');
    
    btnStart.addEventListener('click', async () => {
        const userId = document.getElementById('ws-user-id').value;
        if (!userId) { alert("Please enter User ID for Layer 2 Verification"); return; }
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${wsProtocol}//${window.location.host}/api/v1/stream/${userId}`);
            
            ws.onopen = () => {
                statusDiv.innerText = "Connected. Listening...";
                btnStart.classList.add('hidden');
                btnStop.classList.remove('hidden');
                
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = e => {
                    if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                        ws.send(e.data);
                    }
                };
                mediaRecorder.start(1000); // send chunk every 1 second
            };
            
            ws.onmessage = (e) => {
                const res = JSON.parse(e.data);
                updateLiveUI(res);
            };
            
            ws.onclose = () => stopStream();
            
        } catch(err) {
            alert("Microphone error: " + err);
        }
    });
    
    btnStop.addEventListener('click', stopStream);
    
    function stopStream() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(t => t.stop());
        }
        if (ws) ws.close();
        btnStart.classList.remove('hidden');
        btnStop.classList.add('hidden');
        statusDiv.innerText = "Idle";
    }
    
    function updateLiveUI(res) {
        // Update Layer 1
        const sigs = res.layer1.signals;
        if (sigs) {
            document.getElementById('metric-pjr').innerText = sigs.phase_jump_rate ? sigs.phase_jump_rate.toFixed(3) : "0.00";
            document.getElementById('metric-jit').innerText = sigs.jitter ? sigs.jitter.toFixed(4) : "0.00";
            document.getElementById('metric-nf').innerText = sigs.noise_floor ? sigs.noise_floor.toFixed(4) : "0.00";
            document.getElementById('metric-sf').innerText = sigs.spectral_flatness ? sigs.spectral_flatness.toFixed(3) : "0.00";
            document.getElementById('metric-mfcc').innerText = sigs.mfcc_delta_var ? sigs.mfcc_delta_var.toFixed(1) : "0.0";
            
            // Normalize for radar (rough approximations for 0-1 scale)
            radarChart.data.datasets[0].data = [
                Math.min(1, (sigs.phase_jump_rate || 0) / 0.2),
                Math.min(1, (sigs.jitter || 0) / 0.05),
                Math.max(0, 1 - ((sigs.noise_floor || 0) / 0.005)),
                Math.min(1, (sigs.spectral_flatness || 0) / 0.1),
                Math.min(1, Math.abs((sigs.mfcc_delta_var || 0) - 30) / 90)
            ];
            radarChart.update();
        }
        
        // Update Layer 2
        document.getElementById('metric-similarity').innerText = (res.layer2.similarity || 0).toFixed(3);
        document.getElementById('metric-risk-score').innerText = (res.layer2.risk_score || 0).toFixed(1) + "%";
        document.getElementById('metric-risk-level').innerText = res.layer2.risk_level || "Unknown";
        
        // Cumulative Logic (rolling average of past 10 results)
        rollingResults.push(res);
        if (rollingResults.length > 10) rollingResults.shift();
        
        const avgRisk = rollingResults.reduce((sum, r) => sum + (r.layer2.risk_score || 0), 0) / rollingResults.length;
        const verdictEl = document.getElementById('cumulative-verdict');
        verdictEl.innerText = avgRisk > 60 ? "HIGH RISK / FAKE" : (avgRisk > 30 ? "SUSPICIOUS" : "CLEAN");
        verdictEl.className = avgRisk > 60 ? "text-2xl font-bold mt-2 py-2 text-center rounded bg-red-900/40 text-red-400 border border-red-500/50" : 
                             (avgRisk > 30 ? "text-2xl font-bold mt-2 py-2 text-center rounded bg-orange-900/40 text-orange-400 border border-orange-500/50" :
                                             "text-2xl font-bold mt-2 py-2 text-center rounded bg-green-900/40 text-green-400 border border-green-500/50");
    }

    // ---- File Upload ----
    document.getElementById('btn-upload').addEventListener('click', async () => {
        const userId = document.getElementById('upload-user-id').value;
        const fileInput = document.getElementById('upload-file');
        if (!userId || !fileInput.files[0]) { alert("Please provide User ID and a file"); return; }
        
        const btnUpload = document.getElementById('btn-upload');
        const originalText = btnUpload.innerText;
        btnUpload.innerText = "Processing the audio...";
        btnUpload.disabled = true;

        const formData = new FormData();
        formData.append("user_id", userId);
        formData.append("file", fileInput.files[0]);
        // Optional layer1_score could be added here, we pass 0 for now as the backend verify doesn't run L1 automatically.
        // Wait, for full integration the upload endpoint should ideally run L1 too.
        
        try {
            const res = await fetch('/api/v1/verify', { method: 'POST', body: formData });
            const data = await res.json();
            const resDiv = document.getElementById('upload-results');
            resDiv.classList.remove('hidden');
            document.getElementById('upload-json').innerText = JSON.stringify(data, null, 2);
        } catch(e) { 
            alert("Upload error: " + e); 
        } finally {
            btnUpload.innerText = originalText;
            btnUpload.disabled = false;
        }
    });

    // ---- Enrollment ----
    document.getElementById('btn-enroll').addEventListener('click', async () => {
        const name = document.getElementById('enroll-name').value;
        const fileInput = document.getElementById('enroll-file');
        if (!name || !fileInput.files[0]) { alert("Please provide name and an audio file"); return; }
        
        const formData = new FormData();
        formData.append("name", name);
        formData.append("file", fileInput.files[0]);
        
        try {
            const res = await fetch('/api/v1/enroll', { method: 'POST', body: formData });
            const data = await res.json();
            const statusDiv = document.getElementById('enroll-status');
            if (res.ok) {
                statusDiv.innerHTML = `<span class="text-green-400">✅ Enrolled Successfully! User ID: <br><strong class="text-white">${data.user_id}</strong></span>`;
            } else {
                statusDiv.innerHTML = `<span class="text-red-400">❌ Error: ${data.detail}</span>`;
            }
        } catch(e) { alert("Enroll error: " + e); }
    });
});
