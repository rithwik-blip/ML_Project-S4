const { spawn, execSync } = require('child_process');
const path = require('path');

const PORT = 8000;
const URL = `http://127.0.0.1:${PORT}`;
const DOCS_URL = `${URL}/docs`;

// Print clean terminal banner with clickable links
console.log('\n\x1b[36m%s\x1b[0m', '  ==============================================================');
console.log('\x1b[1m\x1b[32m%s\x1b[0m', '   OmniMatch AI — Match Outcome Intelligence Web App');
console.log('\x1b[36m%s\x1b[0m', '  ==============================================================');
console.log('\n  \x1b[32m➜\x1b[0m  \x1b[1mLocal Web App:\x1b[0m  \x1b[36m\x1b[4m' + URL + '\x1b[0m  \x1b[90m(Ctrl+Click to open)\x1b[0m');
console.log('  \x1b[32m➜\x1b[0m  \x1b[1mSwagger Docs:\x1b[0m   \x1b[34m\x1b[4m' + DOCS_URL + '\x1b[0m');
console.log('  \x1b[90m➜  Press Ctrl+C to stop the server\x1b[0m\n');

// Free port 8000 if already in use (prevents exit-code-1 on re-start)
try {
  const result = execSync(
    `netstat -ano | findstr ":${PORT} " | findstr "LISTENING"`,
    { encoding: 'utf8', stdio: ['pipe','pipe','ignore'] }
  );
  const lines = result.trim().split('\n').filter(Boolean);
  lines.forEach(line => {
    const pid = line.trim().split(/\s+/).pop();
    if (pid && /^\d+$/.test(pid) && pid !== '0') {
      try {
        execSync(`taskkill /PID ${pid} /F`, { stdio: 'ignore' });
        console.log(`  \x1b[33m⚠  Freed port ${PORT} (killed PID ${pid})\x1b[0m`);
      } catch (_) {}
    }
  });
} catch (_) {
  // Port was free — nothing to kill
}

const backendDir = path.join(__dirname, 'backend');

// Start uvicorn server in backend folder
const uvicorn = spawn('python', ['-m', 'uvicorn', 'main:app', '--reload', '--host', '127.0.0.1', '--port', String(PORT)], {
  cwd: backendDir,
  stdio: 'inherit'
});

uvicorn.on('close', (code) => {
  // Treat SIGINT/normal shutdown (code 0 or null) as success
  process.exit(code && code !== 0 && code !== null ? code : 0);
});

process.on('SIGINT', () => {
  uvicorn.kill('SIGINT');
  process.exit(0);
});

