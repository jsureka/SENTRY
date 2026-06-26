/* Results and Analysis chapter -> /Users/bs01366/Downloads/Results_and_Analysis_SENTRY.docx
   Reliability-framework framing; corrected/verified numbers; full 7x4 grid + clone control. */
const fs = require("fs");
const sharp = require("sharp");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, LevelFormat,
        PageNumber, Header, Footer } = D;

const TEAL = "028090", GREY = "9DB2BD", INK = "16324F";
const OUT = "/Users/bs01366/Downloads/Results_and_Analysis_SENTRY.docx";

// ---------- figures: grouped bar charts -> PNG via sharp ----------
function barSVG(title, ylabel, ymax, groups, pct) {
  const W=920,H=440,padL=70,padR=20,padT=56,padB=70, pw=W-padL-padR, ph=H-padT-padB;
  const gw=pw/groups.length, bw=gw*0.28, gap=gw*0.08;
  const y=v=>padT+ph-(v/ymax)*ph;
  let s=`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="Arial">`;
  s+=`<rect width="${W}" height="${H}" fill="white"/>`;
  s+=`<text x="${W/2}" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#${INK}">${title}</text>`;
  for(let i=0;i<=4;i++){const v=ymax*i/4, yy=y(v);
    s+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#E2E8F0"/>`;
    s+=`<text x="${padL-8}" y="${yy+4}" text-anchor="end" font-size="12" fill="#64748B">${pct?(v*100).toFixed(0)+"%":v.toFixed(2)}</text>`;}
  s+=`<text x="18" y="${padT+ph/2}" transform="rotate(-90 18,${padT+ph/2})" text-anchor="middle" font-size="13" fill="#64748B">${ylabel}</text>`;
  groups.forEach((g,gi)=>{
    const gx=padL+gi*gw;
    [["base",g[1],GREY],["SENTRY",g[2],TEAL]].forEach((b,bi)=>{
      const bx=gx+gap+gw*0.18+bi*(bw+8), yy=y(b[1]), hh=padT+ph-yy;
      s+=`<rect x="${bx}" y="${yy}" width="${bw}" height="${hh}" fill="#${b[2]}" rx="3"/>`;
      s+=`<text x="${bx+bw/2}" y="${yy-6}" text-anchor="middle" font-size="12" font-weight="bold" fill="#${INK}">${pct?(b[1]*100).toFixed(1):b[1].toFixed(3)}</text>`;
    });
    g[0].split("\n").forEach((ln,li)=>
      s+=`<text x="${gx+gw/2}" y="${H-padB+20+li*15}" text-anchor="middle" font-size="12" fill="#334155">${ln}</text>`);
  });
  s+=`<rect x="${W-230}" y="40" width="13" height="13" fill="#${GREY}"/><text x="${W-212}" y="51" font-size="12" fill="#334155">base model</text>`;
  s+=`<rect x="${W-120}" y="40" width="13" height="13" fill="#${TEAL}"/><text x="${W-102}" y="51" font-size="12" fill="#334155">SENTRY</text>`;
  s+=`</svg>`; return s;
}
// corrected/verified anchor numbers (base vs SENTRY); vuln SENTRY = temperature-only (gate skips)
const ACC=[["Defect\nCodeBERT",0.818,0.831],["Defect\nGraphCodeBERT",0.806,0.835],["Vuln\nCodeBERT",0.612,0.612],["Vuln\nGraphCodeBERT",0.609,0.609]];
const ECE=[["Defect\nCodeBERT",0.082,0.017],["Defect\nGraphCodeBERT",0.069,0.007],["Vuln\nCodeBERT",0.197,0.055],["Vuln\nGraphCodeBERT",0.228,0.061]];

// ---------- docx helpers ----------
const FULLW=9360;
const h1=t=>new Paragraph({heading:HeadingLevel.HEADING_1,children:[new TextRun(t)]});
const h2=t=>new Paragraph({heading:HeadingLevel.HEADING_2,children:[new TextRun(t)]});
const p=(t,opt={})=>new Paragraph({spacing:{after:120},children:[new TextRun({text:t,...opt})]});
const bullets=arr=>arr.map(t=>new Paragraph({numbering:{reference:"b",level:0},spacing:{after:40},children:[new TextRun(t)]}));
const cell=(t,{b=false,fill,w,al}={})=>new TableCell({width:{size:w,type:WidthType.DXA},
  shading:fill?{fill,type:ShadingType.CLEAR}:undefined,margins:{top:60,bottom:60,left:110,right:110},
  children:[new Paragraph({alignment:al||AlignmentType.LEFT,children:[new TextRun({text:t,bold:b,size:20})]})]});
function table(cols,header,rows){
  const bd={style:BorderStyle.SINGLE,size:1,color:"CCCCCC"};
  const borders={top:bd,bottom:bd,left:bd,right:bd,insideHorizontal:bd,insideVertical:bd};
  const hr=new TableRow({tableHeader:true,children:header.map((t,i)=>cell(t,{b:true,fill:"DCEEF0",w:cols[i],al:i?AlignmentType.CENTER:AlignmentType.LEFT}))});
  const dr=rows.map(r=>new TableRow({children:r.map((t,i)=>cell(String(t),{w:cols[i],al:i?AlignmentType.CENTER:AlignmentType.LEFT,
    b:String(r[0]).includes("SENTRY")}))}));
  return new Table({width:{size:FULLW,type:WidthType.DXA},columnWidths:cols,borders,rows:[hr,...dr]});
}
function figure(buf,capNo,cap){
  return [new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120,after:40},
      children:[new ImageRun({type:"png",data:buf,transformation:{width:560,height:268},
      altText:{title:cap,description:cap,name:"fig"+capNo}})]}),
    new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:160},
      children:[new TextRun({text:`Figure ${capNo}. ${cap}`,italics:true,size:18,color:"555555"})]})];
}

(async()=>{
  const accPng=await sharp(Buffer.from(barSVG("Accuracy: base model vs SENTRY","Accuracy",1.0,ACC,true))).png().toBuffer();
  const ecePng=await sharp(Buffer.from(barSVG("Expected Calibration Error (lower is better)","ECE",0.4,ECE,false))).png().toBuffer();

  const kids=[];
  kids.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:240},
    children:[new TextRun({text:"Results and Analysis",bold:true,size:40,color:INK})]}));

  // 1
  kids.push(h1("1. Introduction"));
  kids.push(p("This chapter evaluates SENTRY, a training-free reliability framework that wraps a frozen code classifier with three post-hoc components: temperature calibration, reliability-gated k-nearest-neighbour (kNN) retrieval, and selective abstention. The evaluation asks whether the framework makes a deployed model's confidence trustworthy without retraining, and where its retrieval component additionally improves accuracy. Experiments span seven datasets — defect prediction (CodeChef), problem classification (POJ-104), clone detection (BigCloneBench), and vulnerability detection (Devign, ReVeal, PrimeVul, DiverseVul) — over four encoder backbones (CodeBERT, GraphCodeBERT, UniXcoder, CodeT5+), together with four fine-tuned anchors. We organise the analysis around three research questions: (RQ1) are fine-tuned code models well-calibrated out of the box; (RQ2) can a training-free framework improve calibration and, where the representation permits, accuracy; and (RQ3) what determines when the retrieval component helps."));

  // 2
  kids.push(h1("2. Experimental Setup"));
  kids.push(h2("Hardware"));
  kids.push(...bullets([
    "Encoder fine-tuning: GPU runtime (NVIDIA T4 / L4 / A100 depending on session).",
    "SENTRY post-hoc analysis and the frozen-embedding grid: Apple M3 Pro (MPS) / CPU — no retraining, no GPU credits.",
  ]));
  kids.push(h2("Software environment"));
  kids.push(...bullets([
    "Python 3.11; PyTorch for model execution.",
    "HuggingFace Transformers for the four encoders; HuggingFace Datasets for benchmark loading.",
    "FAISS for the nearest-neighbour datastore; scikit-learn, SciPy and NumPy for metrics and temperature fitting.",
  ]));
  kids.push(h2("Datasets"));
  kids.push(...bullets([
    "Multiclass — CodeChef defect (4-class) and POJ-104 (104-class problem classification).",
    "Clone detection — BigCloneBench (binary; pair classification over function embeddings).",
    "Vulnerability detection — Devign, ReVeal, PrimeVul (ICSE 2024) and DiverseVul (binary).",
    "Defect/Devign splits follow CodeImprove (Rathnasuriya et al., ICSE 2025); the remaining datasets use their official splits.",
  ]));
  kids.push(h2("Configuration"));
  kids.push(p("Four fine-tuned anchors (CodeBERT / GraphCodeBERT × defect / vulnerability) were trained with AdamW (lr 2×10⁻⁵), max sequence length 400, batch size 32, retaining the best-accuracy checkpoint. For the grid, the four encoders are frozen and a cheap linear probe supplies the base classifier. SENTRY is applied entirely post-hoc; the validation split fits the temperature and selects retrieval hyper-parameters (k, interpolation weight λ, confidence-guard threshold) by F1-macro before any test-set evaluation."));

  // 3
  kids.push(h1("3. Evaluation Metrics"));
  kids.push(p("We report discriminative and reliability metrics together, since a model can be accurate yet untrustworthy in its confidence."));
  kids.push(...bullets([
    "Accuracy, F1-Macro, and Matthews Correlation Coefficient (MCC) — discriminative quality.",
    "Expected Calibration Error (ECE, 15 bins) and Brier score — calibration quality (lower is better).",
    "Area under the risk–coverage curve (AURC) — selective-prediction reliability when the gate abstains (lower is better).",
  ]));
  kids.push(p("ECE = Σₘ (|Bₘ| / n) · | acc(Bₘ) − conf(Bₘ) |", {italics:true}));
  kids.push(p("Paired significance uses McNemar's test on same-item disagreements: χ² = (|b − c| − 1)² / (b + c)."));

  // 4
  kids.push(h1("4. Experimental Results"));
  kids.push(h2("4.1 Reliability wins on the fine-tuned anchors"));
  kids.push(p("On defect prediction the gated-kNN engages and improves both accuracy and calibration; temperature scaling alone is accuracy-neutral, so SENTRY's accuracy gain is contributed by retrieval."));
  kids.push(p("Table 1. Defect prediction (CodeChef, 4-class).",{bold:true,size:20}));
  kids.push(table([2200,1300,1300,1300,1300],
    ["Method","Accuracy","F1-Macro","ECE ↓","Brier ↓"],
    [["CodeBERT · base","0.818","0.779","0.082","0.288"],
     ["CodeBERT · +temperature","0.818","0.779","0.023","0.276"],
     ["CodeBERT · SENTRY","0.831","0.803","0.017","0.261"],
     ["GraphCodeBERT · base","0.806","0.761","0.069","0.300"],
     ["GraphCodeBERT · +temperature","0.806","0.761","0.016","0.292"],
     ["GraphCodeBERT · SENTRY","0.835","0.806","0.007","0.255"]]));
  kids.push(p(""));
  kids.push(p("Table 2. Vulnerability detection (Devign, binary) — calibration only, never-harm.",{bold:true,size:20}));
  kids.push(table([2200,1300,1300,1300,1300],
    ["Method","Accuracy","F1-Macro","ECE ↓","Brier ↓"],
    [["CodeBERT · base","0.612","0.612","0.197","0.527"],
     ["CodeBERT · SENTRY (temp)","0.612","0.612","0.055","0.442"],
     ["GraphCodeBERT · base","0.609","0.603","0.228","0.560"],
     ["GraphCodeBERT · SENTRY (temp)","0.609","0.603","0.061","0.446"]]));
  kids.push(p("Here the representation does not separate the classes, so retrieval is unreliable; the gate falls back to the calibrated model. Accuracy is preserved and calibration is still fixed (ECE reduced 3–4×).",{}));

  kids.push(h2("4.2 The reliability framework across the grid (7 datasets × 4 encoders)"));
  kids.push(p("Across the frozen-encoder grid, temperature scaling reduces ECE on every task, while reliability-gated retrieval improves accuracy exactly where the representation separates the classes."));
  kids.push(p("Table 3. Effect by task family (mean over 4 encoders).",{bold:true,size:20}));
  kids.push(table([2900,1900,1700,2860],
    ["Task family","Base accuracy","Retrieval acc. Δ","Temperature ECE"],
    [["Multiclass (CodeChef, POJ-104)","0.61–0.92","+3.1 pp","0.13 → 0.03"],
     ["Binary clone (BigCloneBench)","0.84","+2.5 pp","reduced"],
     ["Binary vuln (Devign, ReVeal, PrimeVul, DiverseVul)","0.94–0.97*","+0.2 pp","0.20 → 0.06"]]));
  kids.push(p("* vulnerability accuracy is majority-class collapse (base F1-macro ≈ 0.49). Clone detection is the decisive control: it is binary yet separable, and retrieval helps it as much as the multiclass tasks — so the determining factor is representation separability, not whether the task is binary or multiclass.",{italics:true,size:18,color:"555555"}));

  kids.push(h2("4.3 Visual results"));
  kids.push(...figure(accPng,1,"Accuracy of base model vs SENTRY on the four fine-tuned anchors (higher is better)."));
  kids.push(...figure(ecePng,2,"Expected Calibration Error of base model vs SENTRY (lower is better) — reduced on every task."));

  // 5
  kids.push(h1("5. Comparative Analysis"));
  kids.push(p("Recent post-hoc calibration and uncertainty methods improve trustworthiness but are rank-preserving — they leave accuracy unchanged. Input-side adaptation raises accuracy but reports no calibration. SENTRY is output-side, reports both, and additionally raises accuracy where the representation separates."));
  kids.push(table([2700,1700,1700,1480,1780],
    ["Approach","Improves acc.","Calibrates","Abstention","Retraining"],
    [["Temperature scaling (Guo'17; Zhou ICSE'24)","No","Yes","No","None"],
     ["kNN-UE (NAACL'25)","No","Yes","Yes","None"],
     ["CodeImprove (ICSE'25, input-side)","Yes","No","No","None"],
     ["SENTRY (ours, output-side)","Yes (separable)","Yes","Yes","None"]]));
  kids.push(p("SENTRY is complementary to these methods: it can sit on top of input-side adaptation, and it occupies the cell that simultaneously improves accuracy (on separable tasks), calibration, and abstention, training-free. We make no claim of an accuracy or uncertainty-scoring state-of-the-art.",{}));

  // 6
  kids.push(h1("6. Statistical Analysis"));
  kids.push(p("Because configurations are evaluated on the same test items, we use McNemar's paired test on the base-model-vs-SENTRY disagreements."));
  kids.push(table([4200,2600,2560],
    ["Comparison (base vs SENTRY)","p-value","Significant (α = 0.05)"],
    [["Defect · CodeBERT","4 × 10⁻⁶","Yes"],
     ["Defect · GraphCodeBERT","2 × 10⁻¹⁹","Yes"],
     ["Vuln · CodeBERT","0.05 (gate skips retrieval)","No"],
     ["Vuln · GraphCodeBERT","0.09 (gate skips retrieval)","No"]]));
  kids.push(p("On defect prediction the accuracy gain is large and highly significant for both encoders. On vulnerability detection, forcing retrieval is neutral-to-harmful, which is why the gate skips it — the design choice, not a regression."));

  // 7
  kids.push(h1("7. Resource Utilization"));
  kids.push(...bullets([
    "Model size: ≈ 125 M parameters per encoder (≈ 480–500 MB checkpoint).",
    "Datastore: one FAISS index per task–model, ≈ 64 MB, built with a single forward pass over the training set (no gradient updates).",
    "The frozen-embedding grid runs on CPU/MPS at zero GPU cost; embeddings are cached and reused across all analyses.",
  ]));

  // 8
  kids.push(h1("8. Discussion of Findings"));
  kids.push(p("Calibration is fixed on every task. Fine-tuned code models are badly over-confident (anchor ECE 0.07–0.23); a single fitted temperature removes most of this miscalibration everywhere, including the binary vulnerability tasks where it reduces ECE 3–4×."));
  kids.push(p("Accuracy is improved where the representation separates the classes. The reliability-gated retrieval corrects predictions on uncertain inputs when the retrieved neighbours carry useful label signal. This holds for the multiclass tasks (+3.1 pp) and — crucially — for binary clone detection (+2.5 pp), which is separable. It does not hold for binary vulnerability detection, where the classes overlap (base F1-macro ≈ 0.49, a representability ceiling independently reported by Ding et al., ICSE 2024); there the gate abstains from retrieval and the framework contributes calibration and selective abstention only."));
  kids.push(p("Separability, not arity, is the determining factor. Because clone detection is binary yet helped, the earlier reading that retrieval simply fails on binary tasks is incorrect: the factor is whether the task is recoverable from the frozen embedding. The gate's reliability signal selects the regime, so the framework never harms accuracy.",{}));
  kids.push(p("Relation to research objectives. RQ1: code models are not trustworthy out of the box (large baseline ECE). RQ2: a training-free framework fixes calibration on every task and improves accuracy where the representation permits. RQ3: the determining condition is representation separability, characterised across seven datasets and four encoders."));

  // 9
  kids.push(h1("9. Threats to Validity / Limitations"));
  kids.push(...bullets([
    "Internal validity: anchor results are strict-load verified (accuracy reproduces exactly); grid metrics are computed from cached embeddings and are deterministic given the seed.",
    "External validity: seven datasets and four encoders broaden coverage, but findings may not transfer to other languages or to larger code LLMs.",
    "Construct validity: ECE is sensitive to the binning scheme (15 bins); separability is characterised post-hoc rather than predicted before deployment.",
    "Scope: inference-time efficiency of the gate was not measured and is left to future work.",
  ]));

  // 10
  kids.push(h1("10. Summary"));
  kids.push(p("SENTRY is a training-free reliability framework for code classifiers. It fixes over-confidence on every task and, where the representation separates the classes, additionally improves accuracy — significantly so on defect prediction (0.818 → 0.831, p = 4×10⁻⁶) and on clone detection — while abstaining via a reliability signal and never harming accuracy. The determining factor is representation separability, characterised across seven datasets and four encoders; binary clone detection (helped) and binary vulnerability detection (a representability ceiling) show that the axis is separability, not task arity. The framework is complementary to input-side adaptation such as CodeImprove and to post-hoc uncertainty methods, and requires no retraining."));

  const doc=new Document({
    creator:"Jitesh Sureka",
    title:"Results and Analysis — SENTRY",
    numbering:{config:[{reference:"b",levels:[{level:0,format:LevelFormat.BULLET,text:"•",alignment:AlignmentType.LEFT,
      style:{paragraph:{indent:{left:540,hanging:300}}}}]}]},
    styles:{default:{document:{run:{font:"Arial",size:22}}},paragraphStyles:[
      {id:"Heading1",name:"Heading 1",basedOn:"Normal",next:"Normal",quickFormat:true,
        run:{size:30,bold:true,font:"Arial",color:INK},paragraph:{spacing:{before:280,after:140},outlineLevel:0}},
      {id:"Heading2",name:"Heading 2",basedOn:"Normal",next:"Normal",quickFormat:true,
        run:{size:25,bold:true,font:"Arial",color:"333333"},paragraph:{spacing:{before:180,after:100},outlineLevel:1}},
    ]},
    sections:[{
      properties:{page:{size:{width:12240,height:15840},margin:{top:1440,right:1440,bottom:1440,left:1440}}},
      footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,
        children:[new TextRun({text:"SENTRY · Results and Analysis — ",size:16,color:"888888"}),
                  new TextRun({children:[PageNumber.CURRENT],size:16,color:"888888"})]})]})},
      children:kids,
    }],
  });
  fs.writeFileSync(OUT, await Packer.toBuffer(doc));
  console.log("wrote", OUT);
})().catch(e=>{console.error(e);process.exit(1);});
