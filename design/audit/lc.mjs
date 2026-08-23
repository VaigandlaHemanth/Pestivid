const hex=h=>({r:parseInt(h.slice(1,3),16),g:parseInt(h.slice(3,5),16),b:parseInt(h.slice(5,7),16)});
const apca=(t,b)=>{const sY=c=>Math.pow(c/255,2.4);const Y=c=>0.2126729*sY(c.r)+0.7151522*sY(c.g)+0.0721750*sY(c.b);
let x=Y(t),y=Y(b);x=x>0.022?x:x+Math.pow(0.022-x,1.414);y=y>0.022?y:y+Math.pow(0.022-y,1.414);
if(Math.abs(y-x)<0.0005)return 0;let s,o;
if(y>x){s=(Math.pow(y,0.56)-Math.pow(x,0.57))*1.14;o=s<0.1?0:s-0.027;}else{s=(Math.pow(y,0.65)-Math.pow(x,0.62))*1.14;o=s>-0.1?0:s+0.027;}
return o*100;};
const rl=c=>{const f=v=>{v/=255;return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4)};return .2126*f(c.r)+.7152*f(c.g)+.0722*f(c.b)};
const wc=(a,b)=>{const l1=rl(a),l2=rl(b);return (Math.max(l1,l2)+.05)/(Math.min(l1,l2)+.05)};
const [bg,...fgs]=process.argv.slice(2);
for(const f of fgs) console.log(`  ${f} on ${bg}   Lc ${apca(hex(f),hex(bg)).toFixed(0).padStart(4)}   wcag ${wc(hex(f),hex(bg)).toFixed(2)}`);
