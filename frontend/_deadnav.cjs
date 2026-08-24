// Anything that READS like a destination and is not wired.
// click-everything only ever saw elements already carrying data-act, which is
// why it reported "0 dead" on pages whose whole nav bar was inert.
const {chromium}=require('playwright');
const WORDS = ['Browse','Portfolio','Messages','My orders','Buy produce','Lots for sale',
  'What you bought','Sign in','Create an account','Home','Profile','Download','Forgot?'];
const ROLE = {landing:null,signin:null,signup:null,setup:null,invest:'investor',
  portfolio:'investor','confirm-investment':'investor',market:'buyer',orders:'buyer',admin:'admin'};
(async()=>{
const b=await chromium.launch();
const tok={};
const pages=process.argv.slice(2);
let dead=0;
for (const slug of pages){
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  const p=await b.newPage({viewport:{width:1440,height:900}});
  if(role){
    if(!tok[role]) tok[role]=await (await fetch('http://127.0.0.1:3001/api/auth/login',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({email:`demo.${role}@pestivid.sim`,password:'password123'})})).json();
    await p.addInitScript(([t,u])=>{localStorage.setItem('pv.token',t);localStorage.setItem('pv.user',u)},[tok[role].token,JSON.stringify(tok[role].user)]);
  }
  await p.goto('http://127.0.0.1:3001/app/'+slug+'.html',{waitUntil:'load'});
  await p.waitForTimeout(1400);
  const hits = await p.evaluate((WORDS)=>[...document.querySelectorAll('div,span,a,td')]
    .filter(el=>el.children.length===0 && WORDS.includes(el.textContent.trim()))
    .filter(el=>!el.closest('[data-act]') && !el.closest('[data-go]') && !el.closest('a[href]'))
    .map(el=>el.textContent.trim()), WORDS);
  if(hits.length){dead+=hits.length;console.log('  '+slug.padEnd(20)+hits.join(', '));}
  await p.close();
}
console.log('\n  '+dead+' destination(s) that look clickable and are not');
await b.close()})()
