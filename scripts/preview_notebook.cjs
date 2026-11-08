// Capture de contrôle via une instance Chrome headless dédiée au projet.
const fs = require('node:fs');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {pathToFileURL} = require('node:url');

async function main() {
  const root = path.resolve(__dirname, '..');
  const qa = path.join(root, '.cache', 'qa_notebook');
  const profile = path.join(qa, 'chrome-profile-' + process.pid);
  fs.mkdirSync(qa, {recursive: true});
  const chrome = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', [
    '--headless=new', '--disable-gpu', '--disable-background-networking',
    '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0',
    '--user-data-dir=' + profile, 'about:blank'
  ], {windowsHide: true, stdio: 'ignore'});
  let ws;
  try {
    let port;
    const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
    for (let i = 0; i < 100; i++) {
      try { port = fs.readFileSync(path.join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0]; break; }
      catch { await delay(200); }
    }
    if (!port) throw new Error('Chrome ne publie pas son port de débogage.');
    const tabs = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const target = tabs.find(t => t.type === 'page');
    ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
    let id = 0;
    const pending = new Map();
    ws.onmessage = event => {
      const data = JSON.parse(event.data);
      if (data.id && pending.has(data.id)) {
        const {resolve, reject} = pending.get(data.id);
        pending.delete(data.id);
        data.error ? reject(data.error) : resolve(data.result);
      }
    };
    const rpc = (method, params = {}) => new Promise((resolve, reject) => {
      const n = ++id; pending.set(n, {resolve, reject}); ws.send(JSON.stringify({id:n, method, params}));
    });
    await rpc('Page.enable');
    await rpc('Emulation.setDeviceMetricsOverride', {width:1360, height:1000, deviceScaleFactor:1, mobile:false});
    await rpc('Page.navigate', {url:pathToFileURL(path.join(root, 'TP_RAG_multimodal_agentique.html')).href});
    await delay(4000);
    async function capture(nom, expression) {
      if (expression) await rpc('Runtime.evaluate', {expression});
      await delay(500);
      const screenshot = await rpc('Page.captureScreenshot', {format:'png', captureBeyondViewport:false});
      fs.writeFileSync(path.join(qa, nom + '.png'), Buffer.from(screenshot.data, 'base64'));
    }
    await capture('01-introduction', 'window.scrollTo(0,0)');
    await capture('02-extraction', `Array.from(document.querySelectorAll('h2')).find(e=>e.textContent.includes('Étape 1')).scrollIntoView()`);
    await capture('03-recherche', `Array.from(document.querySelectorAll('h2')).find(e=>e.textContent.includes('Étape 4')).scrollIntoView()`);
    await capture('04-tests', `Array.from(document.querySelectorAll('h2')).find(e=>e.textContent.includes('Étape 7')).scrollIntoView()`);
    await capture('05-bilan', 'window.scrollTo(0,document.body.scrollHeight)');
    const check = await rpc('Runtime.evaluate', {returnByValue:true, expression:`JSON.stringify({
      titre: document.querySelector('h1').textContent,
      images: document.images.length,
      imagesManquantes: Array.from(document.images).filter(i=>!i.complete||!i.naturalWidth).length,
      sortiesStderr: document.querySelectorAll('.jp-OutputArea-output[data-mime-type="application/vnd.jupyter.stderr"]').length,
      largeurPage: document.documentElement.scrollWidth,
      largeurFenetre: innerWidth
    })`});
    console.log(check.result.value);
  } finally {
    if (ws) ws.close();
    chrome.kill();
  }
}
main().catch(e => { console.error(e); process.exit(1); });
