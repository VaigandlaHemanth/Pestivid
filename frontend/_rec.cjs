const {chromium}=require('playwright');
const OUT=process.argv[2];
(async()=>{const b=await chromium.launch();
const r=await (await fetch('http://127.0.0.1:3001/api/auth/login',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({email:'demo.farmer@pestivid.sim',password:'password123'})})).json();
const c=await b.newContext({viewport:{width:390,height:844},deviceScaleFactor:2});
const p=await c.newPage();
await p.addInitScript(([t,u])=>{localStorage.setItem('pv.token',t);localStorage.setItem('pv.user',u)},[r.token,JSON.stringify(r.user)]);
await p.goto('http://127.0.0.1:3001/app/record.html',{waitUntil:'load'});await p.waitForTimeout(1500);
await p.screenshot({path:OUT+'/record-refuse.png',fullPage:true});
await p.goto('http://127.0.0.1:3001/app/home.html',{waitUntil:'load'});
await p.evaluate(async()=>{const m=await import('./clip.js');await m.putClip({file:new File([new Blob(['x'.repeat(9500000)])],'clip.webm',{type:'video/webm'}),size:9500000,duration:41});});
await p.goto('http://127.0.0.1:3001/app/sent.html',{waitUntil:'load'});await p.waitForTimeout(1600);
await p.screenshot({path:OUT+'/sent-real.png',fullPage:true});
console.log('shots');await b.close()})()
