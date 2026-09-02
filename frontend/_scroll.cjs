const {chromium}=require('playwright');
const [,,OUT,slug,role,w,h,...ys]=process.argv;
(async()=>{const b=await chromium.launch();
const p=await b.newPage({viewport:{width:+w,height:+h},deviceScaleFactor:2});
if(role&&role!=='none'){const r=await (await fetch('http://127.0.0.1:3001/api/auth/login',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({email:`demo.${role}@pestivid.sim`,password:'password123'})})).json();
await p.addInitScript(([t,u])=>{localStorage.setItem('pv.token',t);localStorage.setItem('pv.user',u)},[r.token,JSON.stringify(r.user)]);}
await p.goto(`http://127.0.0.1:3001/app/${slug}.html`,{waitUntil:'load'});await p.evaluate(()=>document.fonts.ready);await p.waitForTimeout(1200);
for(const y of ys){await p.evaluate(v=>window.scrollTo(0,+v),y);await p.waitForTimeout(700);
  await p.screenshot({path:`${OUT}/${slug}-s${y}.png`});}
console.log('shots',ys.join(','));await b.close()})()
