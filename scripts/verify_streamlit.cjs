// Vérification réelle du navigateur sur l'application locale (Chrome dédié).
const fs = require('node:fs');
const path = require('node:path');
const {spawn} = require('node:child_process');

async function main() {
  const root = path.resolve(__dirname, '..');
  const qa = path.join(root, '.cache', 'qa_streamlit');
  const profile = path.join(qa, 'chrome-' + process.pid);
  fs.mkdirSync(qa, {recursive:true});
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
  const health = await (await fetch('http://127.0.0.1:8502/_stcore/health')).text();
  console.log('Serveur :', health);
  const chrome = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', [
    '--headless=new', '--disable-gpu', '--disable-background-networking', '--no-first-run',
    '--no-default-browser-check', '--remote-debugging-port=0', '--user-data-dir=' + profile,
    'about:blank'
  ], {windowsHide:true, stdio:'ignore'});
  let ws;
  try {
    let port;
    for(let i=0;i<100;i++) {
      try { port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0]; break; }
      catch { await delay(200); }
    }
    if(!port) throw Error('Port Chrome absent');
    const tabs=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
    await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
    let id=0;
    const pending=new Map();
    ws.onmessage=e=>{const d=JSON.parse(e.data);if(pending.has(d.id)){
      const p=pending.get(d.id);pending.delete(d.id);d.error?p.reject(d.error):p.resolve(d.result)}};
    const rpc=(method,params={})=>new Promise((resolve,reject)=>{
      const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}))});
    const evaluate=async expression=>{
      const result=await rpc('Runtime.evaluate',{expression,returnByValue:true});
      if(result.exceptionDetails)throw Error(result.exceptionDetails.text);
      return result.result.value;
    };
    async function waitFor(expression,timeout=30000){
      const end=Date.now()+timeout;
      while(Date.now()<end){if(await evaluate(expression))return;await delay(500)}
      throw Error('Attente dépassée : '+expression);
    }
    async function capture(name){
      await delay(500);
      const img=await rpc('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
      fs.writeFileSync(path.join(qa,name+'.png'),Buffer.from(img.data,'base64'));
    }
    const clickButton=label=>evaluate(`Array.from(document.querySelectorAll('button')).find(e=>e.textContent.trim()===${JSON.stringify(label)}).click()`);
    await rpc('Page.enable');
    await rpc('Emulation.setDeviceMetricsOverride',{width:1440,height:1080,deviceScaleFactor:1,mobile:false});
    await rpc('Page.navigate',{url:'http://127.0.0.1:8502'});
    await waitFor(`document.body.innerText.includes('Explore ton PDF autrement.')`);
    await capture('01-accueil');
    console.log('Accueil chargé. Préparation du PDF…');
    await clickButton('Préparer le document');
    await waitFor(`document.body.innerText.includes('Document prêt à interroger')`,180000);
    await capture('02-document-pret');
    console.log('Document prêt. Recherche CLIP réelle…');
    await rpc('DOM.enable');
    const tree=await rpc('DOM.getDocument');
    const textarea=await rpc('DOM.querySelector',{nodeId:tree.root.nodeId,selector:'textarea'});
    await rpc('DOM.focus',{nodeId:textarea.nodeId});
    await rpc('Input.insertText',{text:'Why does the Transformer use multi-head attention?'});
    await delay(400);
    await clickButton('Rechercher dans le PDF');
    await waitFor(`document.body.innerText.includes('Source 1 · Page')`,60000);
    await evaluate(`document.querySelector('[data-testid="stChatMessage"]').scrollIntoView({block:'start'})`);
    await capture('03-recherche-texte');
    const texte=await evaluate('document.body.innerText');
    if(!texte.includes('Passage de texte'))throw Error('Aucun passage affiché');
    if(texte.includes('Traceback')||texte.includes('StreamlitAPIException'))throw Error('Erreur visible');
    console.log('Passages et citations affichés.');
    await evaluate(`Array.from(document.querySelectorAll('[role="tab"]')).find(e=>e.textContent.includes('Figures du document')).click()`);
    await delay(800);
    await evaluate(`document.querySelector('[role="tablist"]').scrollIntoView({block:'start'})`);
    await capture('04-figures');
    const bilan={serveur:health,preparation:true,rechercheClip:true,
      sourcePagePresente:texte.includes('Source 1 · Page'),
      aucuneErreurVisible:!texte.includes('Traceback'),
      largeurPage:await evaluate('document.documentElement.scrollWidth'),
      largeurFenetre:1440};
    fs.writeFileSync(path.join(qa,'verification.json'),JSON.stringify(bilan,null,2));
    console.log(JSON.stringify(bilan));
  } finally {if(ws)ws.close();chrome.kill();}
}
main().catch(e=>{console.error(e);process.exit(1)});
