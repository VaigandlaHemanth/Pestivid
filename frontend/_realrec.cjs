// A real MediaRecorder clip from a fake camera, then the real send path.
// Everything before this used a hand-made 4KB blob, which is not what a phone
// produces and told me nothing about the MIME the recorder actually emits.
const {chromium}=require('playwright');
(async()=>{
const b=await chromium.launch({args:[
  '--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',
  '--allow-file-access-from-files']});
const r=await (await fetch('http://127.0.0.1:3001/api/auth/login',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({email:'demo.farmer@pestivid.sim',password:'password123'})})).json();
const c=await b.newContext({viewport:{width:390,height:844},permissions:['camera','microphone']});
const p=await c.newPage();
await p.addInitScript(([t,u])=>{localStorage.setItem('pv.token',t);localStorage.setItem('pv.user',u)},[r.token,JSON.stringify(r.user)]);
p.on('pageerror',e=>console.log('  PAGEERROR:',String(e).slice(0,200)));
const seen=[];
p.on('response',async res=>{const u=res.url();
  if(/pinata|videos\/(upload-url|confirm-upload)|api\/videos$/.test(u)){
    let t='';try{t=(await res.text()).slice(0,220)}catch(e){t='<unreadable>'}
    seen.push(res.status()+' '+res.request().method()+' '+u.split('?')[0].split('/').slice(-2).join('/')+' -> '+t);}});

await p.goto('http://127.0.0.1:3001/app/record.html',{waitUntil:'load'});
await p.waitForTimeout(2500);
console.log('  --- record page ---');
console.log('  viewfinder video element:', await p.evaluate(()=>{
  const v=document.querySelector('video');
  if(!v) return 'NO <video>';
  const r=v.getBoundingClientRect();
  return `${Math.round(r.width)}x${Math.round(r.height)} readyState=${v.readyState} paused=${v.paused} srcObject=${!!v.srcObject} opacity=${getComputedStyle(v).opacity}`;}));
console.log('  recording already running:', await p.evaluate(()=>document.body.innerText.replace(/\s+/g,' ').match(/0:\d\d of 0:45/)?.[0]||'no timer'));
console.log('  controls:', await p.evaluate(()=>[...document.querySelectorAll('[data-act]')].map(e=>e.textContent.trim().replace(/\s+/g,' ')).filter(Boolean).join(' | ')));

// press Record
await p.evaluate(()=>document.querySelector('[data-shutter]').click());
await p.waitForTimeout(4000);
console.log('  while recording:', await p.evaluate(()=>({
  label: document.querySelector('[data-shutter-label]')?.textContent,
  clock: document.querySelector('[data-bind="clip.elapsed"]')?.textContent,
  size: document.querySelector('[data-bind="clip.size"]')?.textContent,
  bar: document.querySelector('[data-progress]')?.style.transform })));
await p.evaluate(()=>document.querySelector('[data-shutter]').click());
await p.waitForTimeout(4000);
console.log('  after Stop, url:', p.url().split('/app/')[1]);
if(/sent/.test(p.url())){
  console.log('  clip line:', await p.evaluate(()=>document.querySelector('[data-bind="clip.line"]')?.textContent));
  await p.evaluate(()=>document.querySelector('.pk,.pkOn')?.click());
  await p.waitForTimeout(400);
  await p.evaluate(()=>{const l=[...document.querySelectorAll('div')].find(d=>!d.children.length&&d.textContent.trim()==='Keep it and send');l?.parentElement.click();});
  await p.waitForTimeout(22000);
  console.log('  send button now:', await p.evaluate(()=>[...document.querySelectorAll('div')].map(d=>d.textContent.trim()).find(t=>/^(Sent|Sending|Keep it and send)/.test(t))));
  console.log('  message on screen:', await p.evaluate(()=>[...document.querySelectorAll('[role=alert],[role=status]')].map(e=>e.innerText.replace(/\s+/g,' ')).join(' // ')||'none'));
}
console.log('  --- network ---');
seen.forEach(s=>console.log('   ',s));
await b.close()})()
