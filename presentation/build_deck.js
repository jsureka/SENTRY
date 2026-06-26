/* SENTRY — Results & Discussion deck. Light bg, dark text, minimal words. */
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");

// ---------- palette ----------
const C = {
  bg:   "FFFFFF",
  bg2:  "F3F7F9",
  ink:  "16324F",   // deep navy headers
  body: "33414F",
  mute: "6B7785",
  teal: "028090",
  tealD:"01636E",
  tealT:"DCEEF0",
  green:"2E8B57",
  greenT:"E3F1E9",
  red:  "C0392B",
  redT: "F8E6E3",
  gold: "B7791F",
  line: "DBE4E8",
};
const HFONT = "Cambria", BFONT = "Calibri";
const W = 13.333, H = 7.5, M = 0.7;

// ---------- icon rasterizer ----------
async function icon(name, color) {
  const Comp = fa[name];
  if (!Comp) throw new Error("missing icon " + name);
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(Comp, { color: "#" + color, size: "256" })
  );
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}
const ICONS = {};
async function loadIcons() {
  const spec = {
    shield:["FaShieldHalved", C.teal], shieldW:["FaShieldHalved","FFFFFF"],
    bug:["FaBug", C.red], robot:["FaRobot", C.teal], brain:["FaBrain", C.teal],
    check:["FaCircleCheck", C.green], xmark:["FaCircleXmark", C.red],
    bolt:["FaBolt", C.gold], scale:["FaScaleBalanced", C.teal],
    code:["FaCode", C.teal], warn:["FaTriangleExclamation", C.gold],
    glass:["FaMagnifyingGlass", C.teal], users:["FaUserGroup", C.teal],
    gauge:["FaGaugeHigh", C.teal], arrow:["FaArrowRightLong", C.teal],
    lock:["FaLock", C.teal], layer:["FaLayerGroup", C.teal],
    flask:["FaFlask", C.teal], thumb:["FaThumbsUp", C.green],
    book:["FaBookOpen", C.teal], gears:["FaGears", C.teal],
  };
  for (const [k,[n,col]] of Object.entries(spec)) ICONS[k] = await icon(n, col);
}

const shadow = () => ({ type:"outer", color:"9AA9B2", blur:7, offset:3, angle:90, opacity:0.28 });

// ---------- builders ----------
function base(s, { n, kicker, alt=false } = {}) {
  s.background = { color: alt ? C.bg2 : C.bg };
  // motif: shield in teal circle, top-left
  s.addShape("ellipse", { x:M, y:0.42, w:0.42, h:0.42, fill:{color:C.teal} });
  s.addImage({ data:ICONS.shieldW, x:M+0.1, y:0.52, w:0.22, h:0.22 });
  s.addText("SENTRY", { x:M+0.5, y:0.42, w:3, h:0.42, fontFace:HFONT, fontSize:13,
    color:C.ink, bold:true, charSpacing:2, valign:"middle", margin:0 });
  if (kicker) s.addText(kicker.toUpperCase(), { x:W-4.7, y:0.46, w:4, h:0.36, fontFace:BFONT,
    fontSize:11, color:C.teal, bold:true, align:"right", charSpacing:2, valign:"middle", margin:0 });
  if (n) s.addText(String(n), { x:W-0.9, y:H-0.55, w:0.5, h:0.3, fontFace:BFONT,
    fontSize:10, color:C.mute, align:"right", margin:0 });
}
function title(s, t, y=1.35, w=W-2*M, x=M, size=34, color=C.ink) {
  s.addText(t, { x, y, w, h:0.9, fontFace:HFONT, fontSize:size, bold:true, color, margin:0 });
}
function card(s, x, y, w, h, fill=C.bg) {
  s.addShape("roundRect", { x, y, w, h, rectRadius:0.09, fill:{color:fill}, line:{color:C.line,width:1}, shadow:shadow() });
}
function iconCircle(s, x, y, d, data, fill=C.tealT) {
  s.addShape("ellipse", { x, y, w:d, h:d, fill:{color:fill} });
  s.addImage({ data, x:x+d*0.23, y:y+d*0.23, w:d*0.54, h:d*0.54 });
}

const pres = new pptxgen();
pres.defineLayout({ name:"WIDE", width:W, height:H });
pres.layout = "WIDE";
pres.author = "Jitesh Sureka";
pres.title = "SENTRY — Results & Discussion";

// ============ 1. TITLE ============
function s1(){
  const s = pres.addSlide(); s.background={color:C.bg};
  s.addShape("ellipse",{x:W/2-0.95,y:1.35,w:1.9,h:1.9,fill:{color:C.teal},shadow:shadow()});
  s.addImage({data:ICONS.shieldW,x:W/2-0.55,y:1.78,w:1.1,h:1.05});
  s.addText("SENTRY",{x:0,y:3.55,w:W,h:0.95,fontFace:HFONT,fontSize:60,bold:true,color:C.ink,align:"center",charSpacing:4,margin:0});
  s.addText("Making code-analysis AI trustworthy",{x:0,y:4.55,w:W,h:0.5,fontFace:BFONT,fontSize:20,color:C.teal,align:"center",italic:true,margin:0});
  s.addText("A training-free reliability layer for code classifiers",{x:0,y:5.05,w:W,h:0.4,fontFace:BFONT,fontSize:13,color:C.mute,align:"center",margin:0});
  s.addText([
    {text:"Jitesh Sureka",options:{bold:true,color:C.ink}},
    {text:"   (Roll 1988)",options:{color:C.mute}},
    {text:"     •     Supervised by ",options:{color:C.mute}},
    {text:"Dr. Emon Kumar Dey",options:{bold:true,color:C.ink}},
  ],{x:0,y:6.35,w:W,h:0.4,fontFace:BFONT,fontSize:14,align:"center",margin:0});
}

// ============ 2. HOOK ============
function s2(){
  const s = pres.addSlide(); base(s,{n:2,kicker:"The question"});
  title(s,"AI now judges our code.",1.7,11,M,40);
  title(s,"Can we trust how sure it is?",2.6,11,M,40,C.teal);
  iconCircle(s,W-3.3,3.3,2.2,ICONS.robot,C.tealT);
  s.addText("Defect & vulnerability detectors decide what ships.",{x:M,y:4.4,w:7.2,h:0.5,fontFace:BFONT,fontSize:18,color:C.body,margin:0});
  s.addText("Their confidence guides those decisions — so it has to be honest.",{x:M,y:4.95,w:7.2,h:0.6,fontFace:BFONT,fontSize:18,color:C.body,margin:0});
}

// ============ 3. PROBLEM ============
function s3(){
  const s = pres.addSlide(); base(s,{n:3,kicker:"The problem"});
  title(s,"Models are overconfident");
  s.addText("They announce “99% sure” — and are still wrong.",{x:M,y:2.25,w:11,h:0.5,fontFace:BFONT,fontSize:18,color:C.body,margin:0});
  // confidence meter pegged
  card(s,M,3.1,5.2,2.7);
  iconCircle(s,M+0.35,3.45,0.9,ICONS.gauge,C.tealT);
  s.addText("Says: 99% confident",{x:M+1.45,y:3.55,w:3.5,h:0.45,fontFace:BFONT,fontSize:16,bold:true,color:C.ink,margin:0});
  s.addShape("roundRect",{x:M+0.35,y:4.7,w:4.5,h:0.34,rectRadius:0.05,fill:{color:C.line}});
  s.addShape("roundRect",{x:M+0.35,y:4.7,w:4.45,h:0.34,rectRadius:0.05,fill:{color:C.red}});
  s.addText("Actually right far less often",{x:M+0.35,y:5.15,w:4.5,h:0.4,fontFace:BFONT,fontSize:13,color:C.mute,italic:true,margin:0});
  // analogy
  card(s,M+5.7,3.1,5.3,2.7,C.bg2);
  iconCircle(s,M+6.05,3.45,0.9,ICONS.warn,"FBEBD2");
  s.addText("Like an intern who never says “I’m not sure.”",{x:M+7.15,y:3.5,w:3.5,h:1.1,fontFace:BFONT,fontSize:17,bold:true,color:C.ink,valign:"top",margin:0});
  s.addText("Confidence that doesn’t track reality is worse than no confidence at all.",{x:M+6.05,y:4.75,w:4.6,h:0.9,fontFace:BFONT,fontSize:14,color:C.body,margin:0});
}

// ============ 4. WHY IT MATTERS ============
function s4(){
  const s = pres.addSlide(); base(s,{n:4,kicker:"Why it matters",alt:true});
  title(s,"Confident + wrong = risky");
  const items = [
    [ICONS.bug,"Misses real bugs","A defect waved through with high confidence reaches production."],
    [ICONS.lock,"False alarms","Crying wolf on safe code wastes review time and erodes trust."],
    [ICONS.gears,"Automated pipelines","CI/CD acts on the score. A wrong-but-confident score acts too."],
  ];
  let x=M;
  for(const [ic,h,d] of items){
    card(s,x,2.5,3.7,3.4);
    iconCircle(s,x+0.35,2.85,1.0,ic,C.tealT);
    s.addText(h,{x:x+0.35,y:4.0,w:3.0,h:0.5,fontFace:HFONT,fontSize:19,bold:true,color:C.ink,margin:0});
    s.addText(d,{x:x+0.35,y:4.55,w:3.0,h:1.2,fontFace:BFONT,fontSize:14,color:C.body,margin:0});
    x+=3.95;
  }
}

// ============ 5. EXISTING WORK ============
function s5(){
  const s = pres.addSlide(); base(s,{n:5,kicker:"Existing work"});
  title(s,"What people do today");
  const items = [
    [ICONS.robot,"Fine-tuned models","CodeBERT / GraphCodeBERT. Strong accuracy — but overconfident."],
    [ICONS.book,"Large language models","Capable, but costly at scale and still poorly calibrated."],
    [ICONS.layer,"CodeImprove (ICSE’25)","Fixes the INPUT: detect odd code, rewrite it. No retraining."],
  ];
  let x=M;
  for(const [ic,h,d] of items){
    card(s,x,2.4,3.7,3.5);
    iconCircle(s,x+0.35,2.75,1.0,ic,C.tealT);
    s.addText(h,{x:x+0.35,y:3.95,w:3.05,h:0.6,fontFace:HFONT,fontSize:18,bold:true,color:C.ink,margin:0});
    s.addText(d,{x:x+0.35,y:4.6,w:3.05,h:1.2,fontFace:BFONT,fontSize:14,color:C.body,margin:0});
    x+=3.95;
  }
}

// ============ 6. THE GAP ============
function s6(){
  const s = pres.addSlide(); base(s,{n:6,kicker:"The gap",alt:true});
  title(s,"The missing piece");
  iconCircle(s,M,2.7,1.4,ICONS.glass,C.tealT);
  s.addText("No one makes the model’s confidence trustworthy —",
    {x:M,y:4.4,w:11.5,h:0.6,fontFace:HFONT,fontSize:26,bold:true,color:C.ink,margin:0});
  s.addText([
    {text:"without ",options:{color:C.ink}},
    {text:"retraining",options:{color:C.teal,bold:true}},
    {text:" the model.",options:{color:C.ink}},
  ],{x:M,y:5.05,w:11.5,h:0.6,fontFace:HFONT,fontSize:26,bold:true,margin:0});
  s.addText("That is the gap SENTRY fills.",{x:M,y:5.95,w:11,h:0.5,fontFace:BFONT,fontSize:16,color:C.teal,italic:true,margin:0});
}

// ============ 7. OUR IDEA ============
function s7(){
  const s = pres.addSlide(); base(s,{n:7,kicker:"Our idea"});
  s.addShape("ellipse",{x:M,y:1.9,w:1.5,h:1.5,fill:{color:C.teal},shadow:shadow()});
  s.addImage({data:ICONS.shieldW,x:M+0.42,y:2.27,w:0.66,h:0.66});
  title(s,"SENTRY",1.95,6,M+1.9,40);
  s.addText("A reliability layer you bolt onto any trained model.",{x:M+1.9,y:2.85,w:9,h:0.5,fontFace:BFONT,fontSize:18,color:C.body,margin:0});
  const tags=[["No retraining",ICONS.check],["Works on any model",ICONS.check],["Honest confidence",ICONS.check]];
  let x=M;
  for(const [t,ic] of tags){
    card(s,x,4.3,3.7,1.5,C.bg2);
    iconCircle(s,x+0.3,4.62,0.85,ic,C.greenT);
    s.addText(t,{x:x+1.3,y:4.3,w:2.3,h:1.5,fontFace:BFONT,fontSize:16,bold:true,color:C.ink,valign:"middle",margin:0});
    x+=3.95;
  }
}

// ============ 8. ANALOGY ============
function s8(){
  const s = pres.addSlide(); base(s,{n:8,kicker:"In plain words",alt:true});
  title(s,"SENTRY is a guard at the gate");
  iconCircle(s,M,2.7,2.0,ICONS.shield,C.tealT);
  const rows=[
    [ICONS.check,"Confident prediction?","Wave it straight through.",C.greenT],
    [ICONS.glass,"Unsure?","Double-check against memory of similar past code.",C.tealT],
  ];
  let y=2.7;
  for(const [ic,h,d,bg] of rows){
    card(s,M+2.7,y,8.0,1.55);
    iconCircle(s,M+2.95,y+0.32,0.9,ic,bg);
    s.addText(h,{x:M+4.05,y:y+0.2,w:6.4,h:0.5,fontFace:HFONT,fontSize:19,bold:true,color:C.ink,margin:0});
    s.addText(d,{x:M+4.05,y:y+0.72,w:6.4,h:0.7,fontFace:BFONT,fontSize:15,color:C.body,margin:0});
    y+=1.75;
  }
}

// ===== data-flow scene shared geometry (slides 9-12) =====
const SCENE_TITLE = "How SENTRY works";
function sceneBase(n, kicker){
  const s = pres.addSlide(); base(s,{n,kicker});
  title(s,SCENE_TITLE);
  // three fixed stations across the slide
  return s;
}
function station(s,x,label,ic,active){
  const bg = active?C.teal:C.bg2, tc=active?"FFFFFF":C.ink, ring=active?C.teal:C.line;
  s.addShape("roundRect",{x,y:3.0,w:2.5,h:1.9,rectRadius:0.1,fill:{color:bg},line:{color:ring,width:1.25},shadow:shadow()});
  iconCircle(s,x+0.85,3.25,0.8, ic, active?"FFFFFF":C.tealT);
  s.addText(label,{x,y:4.15,w:2.5,h:0.6,fontFace:BFONT,fontSize:13,bold:true,color:tc,align:"center",margin:0});
}
function arrow(s,x){ s.addShape("rightArrow",{x,y:3.78,w:0.7,h:0.34,fill:{color:C.teal}}); }
function codeCard(s,x,y){ // the moving "data"
  s.addShape("roundRect",{x,y,w:1.7,h:1.15,rectRadius:0.08,fill:{color:"FFFFFF"},line:{color:C.teal,width:1.5},shadow:shadow()});
  s.addImage({data:ICONS.code,x:x+0.12,y:y+0.12,w:0.32,h:0.32});
  s.addShape("rect",{x:x+0.2,y:y+0.62,w:1.3,h:0.08,fill:{color:C.line}});
  s.addShape("rect",{x:x+0.2,y:y+0.78,w:1.0,h:0.08,fill:{color:C.line}});
  s.addShape("rect",{x:x+0.2,y:y+0.94,w:1.2,h:0.08,fill:{color:C.line}});
}
const SX1=M+0.4, SX2=W/2-1.25, SX3=W-M-2.9;

// 9: input enters
function s9(){
  const s=sceneBase(9,"Walkthrough  1 / 4");
  station(s,SX1,"Code in",ICONS.code,true);
  station(s,SX2,"Base model",ICONS.robot,false);
  station(s,SX3,"Prediction",ICONS.gauge,false);
  arrow(s,SX1+2.55); arrow(s,SX2+2.55);
  codeCard(s,SX1+0.4,3.35);
  s.addText("A code snippet arrives.",{x:M,y:5.4,w:11,h:0.5,fontFace:BFONT,fontSize:17,color:C.body,margin:0});
}
// 10: model -> confidence meter
function s10(){
  const s=sceneBase(10,"Walkthrough  2 / 4");
  station(s,SX1,"Code in",ICONS.code,false);
  station(s,SX2,"Base model",ICONS.robot,true);
  station(s,SX3,"Prediction",ICONS.gauge,true);
  arrow(s,SX1+2.55); arrow(s,SX2+2.55);
  codeCard(s,SX2+0.4,3.35);
  // confidence meter under prediction
  s.addText("Confidence",{x:SX3,y:5.0,w:2.5,h:0.3,fontFace:BFONT,fontSize:12,color:C.mute,align:"center",margin:0});
  s.addShape("roundRect",{x:SX3+0.25,y:5.32,w:2.0,h:0.26,rectRadius:0.05,fill:{color:C.line}});
  s.addShape("roundRect",{x:SX3+0.25,y:5.32,w:1.5,h:0.26,rectRadius:0.05,fill:{color:C.teal}});
  s.addText("The model predicts and reports how sure it is.",{x:M,y:5.9,w:11,h:0.5,fontFace:BFONT,fontSize:17,color:C.body,margin:0});
}
// 11: the gate
function s11(){
  const s=sceneBase(11,"Walkthrough  3 / 4");
  station(s,SX1,"Confident?",ICONS.glass,true);
  // two paths
  s.addShape("rightArrow",{x:SX1+2.55,y:3.0,w:0.7,h:0.34,fill:{color:C.green}});
  s.addText("Sure → pass",{x:SX1+2.4,y:2.6,w:2.0,h:0.3,fontFace:BFONT,fontSize:11,bold:true,color:C.green,margin:0});
  s.addShape("downArrow",{x:SX1+1.0,y:4.95,w:0.34,h:0.6,fill:{color:C.gold}});
  s.addText("Unsure ↓ check",{x:SX1-0.1,y:5.55,w:2.6,h:0.3,fontFace:BFONT,fontSize:11,bold:true,color:C.gold,align:"center",margin:0});
  // confident path -> output
  station(s,SX3,"Output",ICONS.check,true);
  // memory station bottom-center
  s.addShape("roundRect",{x:SX2-0.1,y:5.85,w:3.0,h:1.25,rectRadius:0.1,fill:{color:C.tealT},line:{color:C.teal,width:1.25},shadow:shadow()});
  iconCircle(s,SX2+0.15,6.05,0.85,ICONS.brain,"FFFFFF");
  s.addText("Memory of similar code",{x:SX2+1.05,y:5.95,w:1.85,h:1.05,fontFace:BFONT,fontSize:12,bold:true,color:C.ink,valign:"middle",margin:0});
  s.addText("The gate: confident cases pass; unsure cases get a second opinion.",{x:M,y:2.25,w:11,h:0.45,fontFace:BFONT,fontSize:16,color:C.body,margin:0});
}
// 12: blend + calibrate -> trustworthy output
function s12(){
  const s=sceneBase(12,"Walkthrough  4 / 4");
  station(s,SX1,"Memory vote",ICONS.users,true);
  station(s,SX2,"Blend + calibrate",ICONS.scale,true);
  station(s,SX3,"Trustworthy output",ICONS.shield,true);
  arrow(s,SX1+2.55); arrow(s,SX2+2.55);
  s.addText("Honest confidence",{x:SX3,y:5.0,w:2.5,h:0.3,fontFace:BFONT,fontSize:12,color:C.green,align:"center",bold:true,margin:0});
  s.addShape("roundRect",{x:SX3+0.25,y:5.32,w:2.0,h:0.26,rectRadius:0.05,fill:{color:C.line}});
  s.addShape("roundRect",{x:SX3+0.25,y:5.32,w:1.18,h:0.26,rectRadius:0.05,fill:{color:C.green}});
  s.addText("Neighbours vote, results blend, confidence is recalibrated.",{x:M,y:5.9,w:11,h:0.5,fontFace:BFONT,fontSize:17,color:C.body,margin:0});
}

// ============ 13. RESEARCH QUESTIONS ============
function s13(){
  const s = pres.addSlide(); base(s,{n:13,kicker:"Research questions",alt:true});
  title(s,"What we asked");
  const qs=[
    ["RQ1","Are models trustworthy out of the box?",ICONS.gauge],
    ["RQ2","Can a training-free layer improve them?",ICONS.shield],
    ["RQ3","When does it work — and when not?",ICONS.scale],
  ];
  let x=M;
  for(const [q,t,ic] of qs){
    card(s,x,2.5,3.7,3.5);
    iconCircle(s,x+0.35,2.85,1.0,ic,C.tealT);
    s.addText(q,{x:x+0.35,y:4.0,w:3,h:0.5,fontFace:HFONT,fontSize:22,bold:true,color:C.teal,margin:0});
    s.addText(t,{x:x+0.35,y:4.55,w:3.05,h:1.3,fontFace:BFONT,fontSize:16,color:C.ink,margin:0});
    x+=3.95;
  }
}

// ============ 14. RQ1 RESULT ============
function s14(){
  const s = pres.addSlide(); base(s,{n:14,kicker:"RQ1 — calibration"});
  title(s,"Overconfidence is real — and fixable");
  s.addText([
    {text:"Accuracy ",options:{bold:true,color:C.ink}},
    {text:"= how often right.   ",options:{color:C.body}},
    {text:"Calibration ",options:{bold:true,color:C.ink}},
    {text:"= does “90% sure” mean right 90% of the time.",options:{color:C.body}},
  ],{x:M,y:2.2,w:11.6,h:0.5,fontFace:BFONT,fontSize:16,margin:0});
  // before / after big stats
  card(s,M,3.1,5.2,3.0,C.redT);
  s.addText("Before",{x:M,y:3.35,w:5.2,h:0.4,fontFace:BFONT,fontSize:15,color:C.red,bold:true,align:"center",margin:0});
  s.addText("0.23",{x:M,y:3.75,w:5.2,h:1.3,fontFace:HFONT,fontSize:72,bold:true,color:C.red,align:"center",margin:0});
  s.addText("miscalibration (ECE) — confidence way off",{x:M,y:5.2,w:5.2,h:0.6,fontFace:BFONT,fontSize:13,color:C.body,align:"center",margin:0});
  s.addImage({data:ICONS.arrow,x:M+5.45,y:4.35,w:0.7,h:0.5});
  card(s,M+6.4,3.1,5.2,3.0,C.greenT);
  s.addText("After SENTRY",{x:M+6.4,y:3.35,w:5.2,h:0.4,fontFace:BFONT,fontSize:15,color:C.green,bold:true,align:"center",margin:0});
  s.addText("0.06",{x:M+6.4,y:3.75,w:5.2,h:1.3,fontFace:HFONT,fontSize:72,bold:true,color:C.green,align:"center",margin:0});
  s.addText("confidence now tracks reality",{x:M+6.4,y:5.2,w:5.2,h:0.6,fontFace:BFONT,fontSize:13,color:C.body,align:"center",margin:0});
  s.addText("Worst-case base (vulnerability, binary). Temperature scaling restores calibration on every task (defect 0.08→0.02); overconfidence in neural nets is well documented (Guo et al., 2017).",
    {x:M,y:6.35,w:11.8,h:0.4,fontFace:BFONT,fontSize:11,italic:true,color:C.mute,margin:0});
}

// ============ 15. RQ2 RESULT ============
function s15(){
  const s = pres.addSlide(); base(s,{n:15,kicker:"RQ2 — accuracy",alt:true});
  title(s,"Defect prediction: a real gain");
  s.addChart(pres.charts.BAR,[{name:"Accuracy",labels:["Base model","SENTRY","CodeImprove\n(base)"],values:[80.6,83.5,81.9]}],{
    x:M,y:2.4,w:6.6,h:3.9,barDir:"col",chartColors:["9DB2BD",C.teal,"9DB2BD"],
    valAxisMinVal:74,valAxisMaxVal:84,showValue:true,dataLabelPosition:"outEnd",dataLabelColor:C.ink,
    dataLabelFontFace:BFONT,dataLabelFontSize:13,dataLabelFormatCode:'0.0"%"',
    catAxisLabelColor:C.body,catAxisLabelFontFace:BFONT,catAxisLabelFontSize:12,
    valAxisHidden:true,valGridLine:{style:"none"},catGridLine:{style:"none"},showLegend:false,
    chartArea:{fill:{color:C.bg2}},
  });
  card(s,M+7.0,2.5,4.6,1.6,C.greenT);
  iconCircle(s,M+7.3,2.85,0.9,ICONS.check,"FFFFFF");
  s.addText("Statistically significant",{x:M+8.35,y:2.65,w:3.0,h:0.5,fontFace:HFONT,fontSize:16,bold:true,color:C.ink,margin:0});
  s.addText("McNemar, p < 1e-15",{x:M+8.35,y:3.2,w:3.0,h:0.5,fontFace:BFONT,fontSize:13,color:C.body,margin:0});
  card(s,M+7.0,4.3,4.6,2.0);
  iconCircle(s,M+7.3,4.62,0.9,ICONS.scale,C.tealT);
  s.addText("Above CodeImprove’s base model (+1.6 pts) — while also fixing confidence, no retraining.",
    {x:M+8.35,y:4.45,w:3.0,h:1.7,fontFace:BFONT,fontSize:14,color:C.body,valign:"top",margin:0});
}

// ============ 16. RQ3 RESULT ============
function s16(){
  const s = pres.addSlide(); base(s,{n:16,kicker:"RQ3 — where it works"});
  title(s,"Honest about the limits");
  card(s,M,2.5,5.5,3.7,C.greenT);
  iconCircle(s,M+0.35,2.85,1.0,ICONS.thumb,"FFFFFF");
  s.addText("Works: defect prediction",{x:M+1.5,y:3.0,w:3.8,h:0.6,fontFace:HFONT,fontSize:19,bold:true,color:C.ink,margin:0});
  s.addText([
    {text:"Embeddings separate the 4 classes (MCC ≈ 0.74).",options:{breakLine:true}},
    {text:"Neighbours are trustworthy — kNN adds accuracy, significantly.",options:{}},
  ],{x:M+0.4,y:4.2,w:4.7,h:1.7,fontFace:BFONT,fontSize:15,color:C.body,margin:0});
  card(s,M+6.0,2.5,5.5,3.7,C.redT);
  iconCircle(s,M+6.35,2.85,1.0,ICONS.xmark,"FFFFFF");
  s.addText("Retrieval can’t: vulnerability",{x:M+7.5,y:3.0,w:3.9,h:0.6,fontFace:HFONT,fontSize:19,bold:true,color:C.ink,margin:0});
  s.addText([
    {text:"Vulnerable vs safe code overlaps (MCC ≈ 0.26, ReVeal).",options:{breakLine:true}},
    {text:"Neighbours are noise — SENTRY falls back to calibration; accuracy preserved.",options:{}},
  ],{x:M+6.4,y:4.2,w:4.7,h:1.7,fontFace:BFONT,fontSize:15,color:C.body,margin:0});
  s.addText("Retrieval helps iff the embedding separates classes — multiclass AND binary clone detection; it fails on vulnerability (unlearnable). The axis is separability, not binary-vs-multiclass.",{x:M,y:6.4,w:11,h:0.4,fontFace:BFONT,fontSize:12,italic:true,color:C.mute,align:"center",margin:0});
}

// ============ 17. COMPARE ============
function s17(){
  const s = pres.addSlide(); base(s,{n:17,kicker:"Compared with prior work",alt:true});
  title(s,"Where SENTRY sits");
  const hdr = (t)=>({text:t,options:{fill:{color:C.teal},color:"FFFFFF",bold:true,fontFace:BFONT,fontSize:13,align:"center",valign:"middle"}});
  const cell=(t,b)=>({text:t,options:{fontFace:BFONT,fontSize:13,color:C.body,align:"center",valign:"middle",bold:!!b}});
  const rows=[
    [hdr("Approach"),hdr("Defect acc."),hdr("Trust / calibration"),hdr("Retraining")],
    [cell("CodeBERT (base)"),cell("~82%"),cell("✗ overconfident"),cell("needed")],
    [cell("GraphCodeBERT (base)"),cell("~81%"),cell("✗ overconfident"),cell("needed")],
    [cell("CodeImprove (ICSE’25)"),cell("~82%"),cell("— not addressed"),cell("none")],
    [cell("Devign SOTA (CodeT5…)"),cell("66–67%*"),cell("— not addressed"),cell("needed")],
    [{text:"SENTRY (ours)",options:{fill:{color:C.tealT},bold:true,color:C.ink,fontFace:BFONT,fontSize:13,align:"center",valign:"middle"}},
     {text:"~84%",options:{fill:{color:C.tealT},bold:true,color:C.ink,fontFace:BFONT,fontSize:13,align:"center",valign:"middle"}},
     {text:"✓ calibrated",options:{fill:{color:C.tealT},bold:true,color:C.green,fontFace:BFONT,fontSize:13,align:"center",valign:"middle"}},
     {text:"none",options:{fill:{color:C.tealT},bold:true,color:C.ink,fontFace:BFONT,fontSize:13,align:"center",valign:"middle"}}],
  ];
  s.addTable(rows,{x:M,y:2.4,w:W-2*M,colW:[3.95,2.6,3.4,1.98],rowH:0.62,border:{pt:1,color:C.line},valign:"middle"});
  s.addText("* vulnerability task. SENTRY adds calibration and reliability-gated accuracy on a comparable base — complementary to post-hoc calibration (incl. kNN-UE, NAACL’25) and to input-side CodeImprove.",
    {x:M,y:6.55,w:11.9,h:0.5,fontFace:BFONT,fontSize:11,italic:true,color:C.mute,margin:0});
}

// ============ 18. STRENGTHS / WEAKNESS / NEXT ============
function s18(){
  const s = pres.addSlide(); base(s,{n:18,kicker:"Discussion"});
  title(s,"Strengths, limits, and next steps");
  const cols=[
    ["Strengths",C.green,C.greenT,ICONS.thumb,["Accuracy + calibration together, no retraining","Significant defect gain (McNemar p=4e-6)","7 datasets × 4 encoders; clone control"]],
    ["Weaknesses",C.red,C.redT,ICONS.warn,["Retrieval can’t help non-separable tasks (vuln)","Separability seen post-hoc, not predicted","Vulnerability = representability ceiling (data)"]],
    ["What’s next",C.teal,C.tealT,ICONS.flask,["Auto-gate from the separability signal","Compose with uncertainty estimators (kNN-UE)","Broaden tasks; combine with CodeImprove"]],
  ];
  let x=M;
  for(const [h,col,bg,ic,items] of cols){
    card(s,x,2.4,3.7,4.0);
    iconCircle(s,x+0.35,2.7,0.85,ic,bg);
    s.addText(h,{x:x+1.3,y:2.7,w:2.3,h:0.85,fontFace:HFONT,fontSize:18,bold:true,color:col,valign:"middle",margin:0});
    s.addText(items.map((t,i)=>({text:t,options:{bullet:{indent:14},breakLine:true,paraSpaceAfter:10}})),
      {x:x+0.4,y:3.75,w:3.0,h:2.5,fontFace:BFONT,fontSize:14,color:C.body,valign:"top",margin:0});
    x+=3.95;
  }
}

// ============ 19. CLOSING ============
function s19(){
  const s = pres.addSlide(); s.background={color:C.bg2};
  s.addShape("ellipse",{x:W/2-0.8,y:1.5,w:1.6,h:1.6,fill:{color:C.teal},shadow:shadow()});
  s.addImage({data:ICONS.shieldW,x:W/2-0.46,y:1.88,w:0.92,h:0.88});
  s.addText("SENTRY",{x:0,y:3.35,w:W,h:0.8,fontFace:HFONT,fontSize:44,bold:true,color:C.ink,align:"center",charSpacing:3,margin:0});
  s.addText("Trustworthy confidence for code models — no retraining.",{x:0,y:4.2,w:W,h:0.5,fontFace:BFONT,fontSize:18,color:C.teal,align:"center",italic:true,margin:0});
  s.addText("github.com/jsureka/SENTRY",{x:0,y:5.0,w:W,h:0.4,fontFace:BFONT,fontSize:14,color:C.body,align:"center",margin:0});
  s.addText([
    {text:"Jitesh Sureka",options:{bold:true,color:C.ink}},
    {text:" (Roll 1988)   •   Supervised by ",options:{color:C.mute}},
    {text:"Dr. Emon Kumar Dey",options:{bold:true,color:C.ink}},
  ],{x:0,y:6.2,w:W,h:0.4,fontFace:BFONT,fontSize:14,align:"center",margin:0});
}

(async()=>{
  await loadIcons();
  [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19].forEach(f=>f());
  await pres.writeFile({ fileName:"/Users/bs01366/SENTRY/presentation/SENTRY_results_discussion.pptx" });
  console.log("deck written");
})().catch(e=>{ console.error(e); process.exit(1); });
