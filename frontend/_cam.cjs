const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch({args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']});
const t=await (await fetch('http://127.0.0.1:3001/api/auth/login',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({email:'demo.farmer@pestivid.sim',password:'password123'})})).json();
const c=await b.newContext({viewport:{width:1440,height:900},permissions:['camera','microphone'],deviceScaleFactor:2});
const p=await c.newPage();
await p.addInitScript(([a,u])=>{localStorage.setItem('pv.token',a);localStorage.setItem('pv.user',u)},[t.token,JSON.stringify(t.user)]);
await p.goto('http://127.0.0.1:3001/app/record.html',{waitUntil:'load'});await p.waitForTimeout(2600);
await p.screenshot({path:process.argv[2]+'/record-cam.png',fullPage:true});
console.log('shot');await b.close()})()
