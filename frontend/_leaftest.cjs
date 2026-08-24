const {chromium}=require('playwright');
const IMG=process.argv[2];
(async()=>{
const b=await chromium.launch();
const t=await (await fetch('http://127.0.0.1:3001/api/auth/login',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({email:'demo.farmer@pestivid.sim',password:'password123'})})).json();
const p=await b.newPage({viewport:{width:390,height:900}});
await p.addInitScript(([a,u])=>{localStorage.setItem('pv.token',a);localStorage.setItem('pv.user',u)},[t.token,JSON.stringify(t.user)]);
p.on('pageerror',e=>console.log('  PAGEERROR:',String(e).slice(0,220)));
p.on('console',m=>{if(m.type()==='error')console.log('  console.error:',m.text().slice(0,180))});
await p.goto('http://127.0.0.1:3001/app/leaf-check.html',{waitUntil:'load'});
await p.waitForTimeout(1600);
console.log('  upload affordance:', await p.evaluate(()=>[...document.querySelectorAll('[data-act]')].map(e=>e.innerText.replace(/\s+/g,' ').slice(0,60)).filter(Boolean).slice(0,4)));
const inp = await p.$('input[type=file]');
if(!inp){console.log('  NO FILE INPUT');await b.close();return}
await inp.setInputFiles(IMG);
// the backbone is ~173 MB from the HF CDN
for (let i=0;i<40;i++){
  await p.waitForTimeout(6000);
  const st = await p.evaluate(()=>{
    const v=document.querySelector('[data-bind="verdict.name"]');
    const steps=[...document.querySelectorAll('div')].map(d=>d.textContent.trim()).filter(t=>/Downloading|Reading your photo|Ready to use offline|did not run/.test(t)).slice(0,2);
    const ready=[...document.querySelectorAll('div')].some(d=>d.textContent.trim()==='Ready to use offline'&&d.previousElementSibling);
    return {verdict:v?v.textContent.trim():'(none)', steps, ms:document.querySelector('[data-bind="shot.where"]')?.textContent||''};
  });
  console.log('  t+'+((i+1)*6)+'s verdict='+st.verdict+' | '+st.ms+' | '+st.steps.join(' ~ ').slice(0,90));
  if (st.ms && /ms/.test(st.ms)) break;
  if (st.steps.some(s=>/did not run/.test(s))) break;
}
console.log('\n  --- screen ---');
console.log((await p.evaluate(()=>document.body.innerText.replace(/\s+/g,' '))).slice(0,700));
await p.screenshot({path:process.argv[3]+'/leaf-result.png',fullPage:true});
await b.close()})()
