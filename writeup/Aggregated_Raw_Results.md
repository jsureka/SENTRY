# Aggregated Experimental Results

## Experiment: CI_kNN_defect_codebert_results

### Final Pipeline Metrics
```json
[
  {
    "Method": "CodeBERT (Lu et al. 2021)",
    "Acc": 0.613,
    "F1-M": "--",
    "MCC": "--",
    "ECE": "--",
    "Brier": "--"
  },
  {
    "Method": "GraphCodeBERT (Guo et al. 2021)",
    "Acc": 0.715,
    "F1-M": "--",
    "MCC": "--",
    "ECE": "--",
    "Brier": "--"
  },
  {
    "Method": "B1: Model-Only",
    "Acc": 0.8181549379065641,
    "F1-M": 0.7787314595071271,
    "MCC": 0.7341430386809378,
    "ECE": 0.37583514443521027,
    "Brier": 0.4896665465048345
  },
  {
    "Method": "B3: + Temp Scaling",
    "Acc": 0.8181549379065641,
    "F1-M": 0.7787314595071271,
    "MCC": 0.7341430386809378,
    "ECE": 0.05217108010085328,
    "Brier": 0.288419614967873
  },
  {
    "Method": "B4: + kNN (always-on)",
    "Acc": 0.826729745712596,
    "F1-M": 0.792540200534074,
    "MCC": 0.7476561529987592,
    "ECE": 0.06804055643255463,
    "Brier": 0.27617440800460996
  },
  {
    "Method": "B5: + ProtokNN",
    "Acc": 0.820816085156712,
    "F1-M": 0.7852976361473739,
    "MCC": 0.7411078745010184,
    "ECE": 0.10622410270873243,
    "Brier": 0.31509413403907033
  },
  {
    "Method": "M1: CI-Gated kNN [OURS]",
    "Acc": 0.8248078060319338,
    "F1-M": 0.7881837272466155,
    "MCC": 0.7447293573152783,
    "ECE": 0.035259687734000324,
    "Brier": 0.26773561105738414
  }
]
```

### k/λ Parameter Ablation Results
```json
[
  {
    "k": 4,
    "lambda": 0.1,
    "accuracy": 0.826729745712596,
    "f1_macro": 0.792540200534074,
    "mcc": 0.7476561529987592,
    "ece": 0.06804055643255463,
    "brier": 0.27617440800460996
  },
  {
    "k": 4,
    "lambda": 0.3,
    "accuracy": 0.8261383796570076,
    "f1_macro": 0.7908638239509952,
    "mcc": 0.7467410875933028,
    "ece": 0.046770918173828877,
    "brier": 0.2689165722961347
  },
  {
    "k": 4,
    "lambda": 0.5,
    "accuracy": 0.8245121230041396,
    "f1_macro": 0.7872320475037106,
    "mcc": 0.7441957830686331,
    "ece": 0.03255231403735753,
    "brier": 0.2673610181941866
  },
  {
    "k": 4,
    "lambda": 0.7,
    "accuracy": 0.8217031342400947,
    "f1_macro": 0.7834149283340016,
    "mcc": 0.7396594843104451,
    "ece": 0.04061531925508797,
    "brier": 0.27150774569876573
  },
  {
    "k": 4,
    "lambda": 0.9,
    "accuracy": 0.8200768775872265,
    "f1_macro": 0.7806305350946732,
    "mcc": 0.7370633502729185,
    "ece": 0.061475677205315195,
    "brier": 0.2813567548098721
  },
  {
    "k": 8,
    "lambda": 0.1,
    "accuracy": 0.8218509757539917,
    "f1_macro": 0.7875433416708759,
    "mcc": 0.7406646361936472,
    "ece": 0.06134436901305225,
    "brier": 0.2737099386018907
  },
  {
    "k": 8,
    "lambda": 0.3,
    "accuracy": 0.8228858663512715,
    "f1_macro": 0.7878348098213915,
    "mcc": 0.7419806216699973,
    "ece": 0.04633884114076944,
    "brier": 0.2707790954155958
  },
  {
    "k": 8,
    "lambda": 0.5,
    "accuracy": 0.8225901833234772,
    "f1_macro": 0.7864097537353056,
    "mcc": 0.7412870131999677,
    "ece": 0.034827938125779986,
    "brier": 0.2713909151758798
  },
  {
    "k": 8,
    "lambda": 0.7,
    "accuracy": 0.8218509757539917,
    "f1_macro": 0.7837426329659168,
    "mcc": 0.7398151448914718,
    "ece": 0.03999236163160007,
    "brier": 0.2755453978827429
  },
  {
    "k": 8,
    "lambda": 0.9,
    "accuracy": 0.8196333530455352,
    "f1_macro": 0.7800179918077597,
    "mcc": 0.7364372859830889,
    "ece": 0.057014508671606526,
    "brier": 0.2832425435361849
  },
  {
    "k": 16,
    "lambda": 0.1,
    "accuracy": 0.8231815493790656,
    "f1_macro": 0.7886361284865652,
    "mcc": 0.742684819828367,
    "ece": 0.05867314223895649,
    "brier": 0.2745752131510282
  },
  {
    "k": 16,
    "lambda": 0.3,
    "accuracy": 0.8240685984624483,
    "f1_macro": 0.7886457332377641,
    "mcc": 0.7436307236984262,
    "ece": 0.0450284696472769,
    "brier": 0.27348196071376224
  },
  {
    "k": 16,
    "lambda": 0.5,
    "accuracy": 0.8233293908929628,
    "f1_macro": 0.786176376678766,
    "mcc": 0.7423159656243549,
    "ece": 0.03504575324036519,
    "brier": 0.2747714432332635
  },
  {
    "k": 16,
    "lambda": 0.7,
    "accuracy": 0.8209639266706091,
    "f1_macro": 0.7819480918879677,
    "mcc": 0.738481990601541,
    "ece": 0.04506182401431288,
    "brier": 0.2784436607095319
  },
  {
    "k": 16,
    "lambda": 0.9,
    "accuracy": 0.8190419869899468,
    "f1_macro": 0.7796273233241152,
    "mcc": 0.7355823208201836,
    "ece": 0.05255334754016416,
    "brier": 0.28449861314256747
  },
  {
    "k": 32,
    "lambda": 0.1,
    "accuracy": 0.8218509757539917,
    "f1_macro": 0.7859137199930252,
    "mcc": 0.7407783037931552,
    "ece": 0.059386947373158666,
    "brier": 0.2759110705768353
  },
  {
    "k": 32,
    "lambda": 0.3,
    "accuracy": 0.8222945002956831,
    "f1_macro": 0.7855456360617247,
    "mcc": 0.7411242852212498,
    "ece": 0.046586956652337266,
    "brier": 0.27558310287359306
  },
  {
    "k": 32,
    "lambda": 0.5,
    "accuracy": 0.8219988172678888,
    "f1_macro": 0.7838846315050371,
    "mcc": 0.7402930886224683,
    "ece": 0.0386281807608671,
    "brier": 0.2770309318758274
  },
  {
    "k": 32,
    "lambda": 0.7,
    "accuracy": 0.8209639266706091,
    "f1_macro": 0.7820222279881069,
    "mcc": 0.738543888493305,
    "ece": 0.038899682844638565,
    "brier": 0.28025455758353823
  },
  {
    "k": 32,
    "lambda": 0.9,
    "accuracy": 0.8185984624482555,
    "f1_macro": 0.7792464473902103,
    "mcc": 0.7348678777316454,
    "ece": 0.05015748869963661,
    "brier": 0.2852539799967256
  }
]
```

### OOD Detection Metrics (Mahalanobis, Energy, RMD, Baseline)
```json
{
  "knn_distance": {
    "method": "knn_distance",
    "auc": 0.687655983263844,
    "cvr_at_threshold": 0.32926829268292684,
    "mvr_at_threshold": 0.12865919768702566,
    "threshold": 0.2,
    "cvr_optimal": 0.6317073170731707,
    "mvr_optimal": 0.3534513913986267,
    "optimal_threshold": 0.09561651200056076,
    "n_oos": 1230,
    "n_is": 5534,
    "oos_ratio": 0.18184506209343584
  },
  "disagreement": {
    "method": "disagreement",
    "auc": 0.7667454112199236,
    "cvr_at_threshold": 0.13089430894308943,
    "mvr_at_threshold": 0.023852547885796892,
    "threshold": 0.2,
    "cvr_optimal": 0.7853658536585366,
    "mvr_optimal": 0.3543548970003614,
    "optimal_threshold": 0.07379875962317539,
    "n_oos": 1230,
    "n_is": 5534,
    "oos_ratio": 0.18184506209343584
  },
  "entropy": {
    "method": "entropy",
    "auc": 0.7647835553165796,
    "cvr_at_threshold": 1.0,
    "mvr_at_threshold": 1.0,
    "threshold": 0.2,
    "cvr_optimal": 0.7365853658536585,
    "mvr_optimal": 0.3299602457535237,
    "optimal_threshold": 0.3840712852387412,
    "n_oos": 1230,
    "n_is": 5534,
    "oos_ratio": 0.18184506209343584
  },
  "confidence_delta": {
    "method": "confidence_delta",
    "auc": 0.5617903808239384,
    "cvr_at_threshold": 0.5024390243902439,
    "mvr_at_threshold": 0.31568485724611495,
    "threshold": 0.2,
    "cvr_optimal": 0.36829268292682926,
    "mvr_optimal": 0.15016263100831226,
    "optimal_threshold": 0.2349554543217241,
    "n_oos": 1230,
    "n_is": 5534,
    "oos_ratio": 0.18184506209343584
  },
  "composite": {
    "method": "composite",
    "auc": 0.7763523936287429,
    "cvr_at_threshold": 0.640650406504065,
    "mvr_at_threshold": 0.21666064329598844,
    "threshold": 0.2,
    "cvr_optimal": 0.7170731707317073,
    "mvr_optimal": 0.2900252981568486,
    "optimal_threshold": 0.17999684337848754,
    "n_oos": 1230,
    "n_is": 5534,
    "oos_ratio": 0.18184506209343584
  },
  "energy_score": {
    "method": "energy_score",
    "auc": 0.23462990059969263,
    "cvr_at_threshold": 1.0,
    "mvr_at_threshold": 1.0,
    "threshold": 0.2,
    "cvr_optimal": 0.0008130081300813008,
    "mvr_optimal": 0.0003614022406938923,
    "optimal_threshold": 1.7419469356536865,
    "n_oos": 1230,
    "n_is": 5534,
    "oos_ratio": 0.18184506209343584
  },
  "mahalanobis": {
    "method": "mahalanobis",
    "auc": 0.33673139586473566,
    "cvr_at_threshold": 0.0,
    "mvr_at_threshold": 0.0,
    "threshold": 0.2,
    "cvr_optimal": 0.0,
    "mvr_optimal": 0.0,
    "optimal_threshold": Infinity,
    "n_oos": 1230,
    "n_is": 5534,
    "oos_ratio": 0.18184506209343584
  },
  "relative_mahalanobis": {
    "method": "relative_mahalanobis",
    "auc": 0.3237862908083364,
    "cvr_at_threshold": 0.7032520325203252,
    "mvr_at_threshold": 0.9289844597036502,
    "threshold": 0.2,
    "cvr_optimal": 0.0024390243902439024,
    "mvr_optimal": 0.001264907842428623,
    "optimal_threshold": 14.018295288085938,
    "n_oos": 1230,
    "n_is": 5534,
    "oos_ratio": 0.18184506209343584
  }
}
```

### McNemar Statistical Significance Tests
```json
{
  "B1_model_only_vs_B3_calibrated": {
    "chi2": 0.0,
    "p_value": 1.0,
    "b": 0,
    "c": 0,
    "significant": false
  },
  "B1_model_only_vs_B4_knn": {
    "method_a": "B1_model_only",
    "method_b": "B4_knn",
    "b_a_right_b_wrong": 104,
    "c_a_wrong_b_right": 162,
    "chi2": 12.214285714285714,
    "p_value": 0.0004742496450418354,
    "significant_at_0.05": true,
    "significant_at_0.01": true
  },
  "B1_model_only_vs_B5_protoknn": {
    "method_a": "B1_model_only",
    "method_b": "B5_protoknn",
    "b_a_right_b_wrong": 113,
    "c_a_wrong_b_right": 131,
    "chi2": 1.1844262295081966,
    "p_value": 0.2764567418803161,
    "significant_at_0.05": false,
    "significant_at_0.01": false
  },
  "B1_model_only_vs_M1_gated_knn": {
    "method_a": "B1_model_only",
    "method_b": "M1_gated_knn",
    "b_a_right_b_wrong": 69,
    "c_a_wrong_b_right": 114,
    "chi2": 10.579234972677595,
    "p_value": 0.0011436497674470747,
    "significant_at_0.05": true,
    "significant_at_0.01": true
  }
}
```

## Experiment: CI_kNN_defect_graphcodebert_results

### Final Pipeline Metrics
```json
[
  {
    "Method": "CodeBERT (Lu et al. 2021)",
    "Acc": 0.613,
    "F1-M": "--",
    "MCC": "--",
    "ECE": "--",
    "Brier": "--"
  },
  {
    "Method": "GraphCodeBERT (Guo et al. 2021)",
    "Acc": 0.715,
    "F1-M": "--",
    "MCC": "--",
    "ECE": "--",
    "Brier": "--"
  },
  {
    "Method": "B1: Model-Only",
    "Acc": 0.8344175044352454,
    "F1-M": 0.8007747719535439,
    "MCC": 0.7595821327515,
    "ECE": 0.39139786804571863,
    "Brier": 0.48199283762479445
  },
  {
    "Method": "B3: + Temp Scaling",
    "Acc": 0.8344175044352454,
    "F1-M": 0.8007747719535439,
    "MCC": 0.7595821327515,
    "ECE": 0.0366676150924921,
    "Brier": 0.26637386047285194
  },
  {
    "Method": "B4: + kNN (always-on)",
    "Acc": 0.8366351271437019,
    "F1-M": 0.8034916581540781,
    "MCC": 0.7628778481618301,
    "ECE": 0.0394330630939424,
    "Brier": 0.26568507909293837
  },
  {
    "Method": "B5: + ProtokNN",
    "Acc": 0.8307214665878179,
    "F1-M": 0.7976100226570968,
    "MCC": 0.7567535792527493,
    "ECE": 0.09481896680433477,
    "Brier": 0.29327819987502185
  },
  {
    "Method": "M1: CI-Gated kNN [OURS]",
    "Acc": 0.8366351271437019,
    "F1-M": 0.8034916581540781,
    "MCC": 0.7628778481618301,
    "ECE": 0.038940606106582606,
    "Brier": 0.26555707710726867
  }
]
```

### k/λ Parameter Ablation Results
```json
[
  {
    "k": 4,
    "lambda": 0.1,
    "accuracy": 0.8358959195742165,
    "f1_macro": 0.8035723796952048,
    "mcc": 0.7621877329821778,
    "ece": 0.0705889744421863,
    "brier": 0.26346953751170227
  },
  {
    "k": 4,
    "lambda": 0.3,
    "accuracy": 0.8370786516853933,
    "f1_macro": 0.8049088840716059,
    "mcc": 0.7639153114093161,
    "ece": 0.04934813414041779,
    "brier": 0.2557527555399499
  },
  {
    "k": 4,
    "lambda": 0.5,
    "accuracy": 0.8369308101714962,
    "f1_macro": 0.8039982115400918,
    "mcc": 0.7634923700370588,
    "ece": 0.03403844229855069,
    "brier": 0.25281436618204334
  },
  {
    "k": 4,
    "lambda": 0.7,
    "accuracy": 0.8369308101714962,
    "f1_macro": 0.8034035314659108,
    "mcc": 0.7633330519159095,
    "ece": 0.025703183647611812,
    "brier": 0.2546543694379825
  },
  {
    "k": 4,
    "lambda": 0.9,
    "accuracy": 0.8356002365464222,
    "f1_macro": 0.8022546999303029,
    "mcc": 0.7613938453457976,
    "ece": 0.04490286432058527,
    "brier": 0.26127276530776733
  },
  {
    "k": 8,
    "lambda": 0.1,
    "accuracy": 0.8361916026020106,
    "f1_macro": 0.8041343042245257,
    "mcc": 0.7631411527438577,
    "ece": 0.06161600576628396,
    "brier": 0.2624413688249016
  },
  {
    "k": 8,
    "lambda": 0.3,
    "accuracy": 0.8356002365464222,
    "f1_macro": 0.8036424965780394,
    "mcc": 0.7621106716578202,
    "ece": 0.04816490287005497,
    "brier": 0.25833395856475744
  },
  {
    "k": 8,
    "lambda": 0.5,
    "accuracy": 0.8363394441159078,
    "f1_macro": 0.8037772221914669,
    "mcc": 0.7628534766825629,
    "ece": 0.03514080457701248,
    "brier": 0.2570730039006259
  },
  {
    "k": 8,
    "lambda": 0.7,
    "accuracy": 0.8366351271437019,
    "f1_macro": 0.8039816028951589,
    "mcc": 0.7631082304905663,
    "ece": 0.02602695533291567,
    "brier": 0.2586585048325069
  },
  {
    "k": 8,
    "lambda": 0.9,
    "accuracy": 0.835304553518628,
    "f1_macro": 0.8021797651432891,
    "mcc": 0.760959875776004,
    "ece": 0.03876984128324824,
    "brier": 0.26309046136040043
  },
  {
    "k": 16,
    "lambda": 0.1,
    "accuracy": 0.8332347723240686,
    "f1_macro": 0.7995277847478606,
    "mcc": 0.759054341815608,
    "ece": 0.06468163630322518,
    "brier": 0.26388760447283
  },
  {
    "k": 16,
    "lambda": 0.3,
    "accuracy": 0.8350088704908338,
    "f1_macro": 0.8012251054032258,
    "mcc": 0.7613475759145937,
    "ece": 0.04996398563706665,
    "brier": 0.2610712400318235
  },
  {
    "k": 16,
    "lambda": 0.5,
    "accuracy": 0.8358959195742165,
    "f1_macro": 0.8027983235000333,
    "mcc": 0.762369682073418,
    "ece": 0.03696292707749993,
    "brier": 0.26017994174758535
  },
  {
    "k": 16,
    "lambda": 0.7,
    "accuracy": 0.8356002365464222,
    "f1_macro": 0.80261265736925,
    "mcc": 0.7616088914649526,
    "ece": 0.02680949745621655,
    "brier": 0.2612137096201157
  },
  {
    "k": 16,
    "lambda": 0.9,
    "accuracy": 0.8350088704908338,
    "f1_macro": 0.8017177009582195,
    "mcc": 0.7604613430850641,
    "ece": 0.03660601239743207,
    "brier": 0.26417254364941445
  },
  {
    "k": 32,
    "lambda": 0.1,
    "accuracy": 0.8320520402128918,
    "f1_macro": 0.7974559050943091,
    "mcc": 0.7577118537631137,
    "ece": 0.06653134415217517,
    "brier": 0.2675070577111293
  },
  {
    "k": 32,
    "lambda": 0.3,
    "accuracy": 0.8336782968657599,
    "f1_macro": 0.7986741343999475,
    "mcc": 0.7595917952568445,
    "ece": 0.052729094204742116,
    "brier": 0.2645392545375111
  },
  {
    "k": 32,
    "lambda": 0.5,
    "accuracy": 0.8347131874630396,
    "f1_macro": 0.7996809711494066,
    "mcc": 0.7607351556422192,
    "ece": 0.04302411790658122,
    "brier": 0.2631234408296712
  },
  {
    "k": 32,
    "lambda": 0.7,
    "accuracy": 0.8354523950325251,
    "f1_macro": 0.8014481730924344,
    "mcc": 0.761429476799049,
    "ece": 0.028477545140232683,
    "brier": 0.26325961658760977
  },
  {
    "k": 32,
    "lambda": 0.9,
    "accuracy": 0.8347131874630396,
    "f1_macro": 0.800865213021444,
    "mcc": 0.760080841566889,
    "ece": 0.03318699210952773,
    "brier": 0.2649477818113266
  }
]
```

### OOD Detection Metrics (Mahalanobis, Energy, RMD, Baseline)
```json
{
  "knn_distance": {
    "method": "knn_distance",
    "auc": 0.6971387915105801,
    "cvr_at_threshold": 0.47946428571428573,
    "mvr_at_threshold": 0.20481927710843373,
    "threshold": 0.2,
    "cvr_optimal": 0.6526785714285714,
    "mvr_optimal": 0.3548901488306166,
    "optimal_threshold": 0.12801529467105865,
    "n_oos": 1120,
    "n_is": 5644,
    "oos_ratio": 0.16558249556475457
  },
  "disagreement": {
    "method": "disagreement",
    "auc": 0.7718831470841349,
    "cvr_at_threshold": 0.10803571428571429,
    "mvr_at_threshold": 0.019844082211197732,
    "threshold": 0.2,
    "cvr_optimal": 0.7776785714285714,
    "mvr_optimal": 0.3391211906449327,
    "optimal_threshold": 0.06625228886335521,
    "n_oos": 1120,
    "n_is": 5644,
    "oos_ratio": 0.16558249556475457
  },
  "entropy": {
    "method": "entropy",
    "auc": 0.7653297275235394,
    "cvr_at_threshold": 1.0,
    "mvr_at_threshold": 1.0,
    "threshold": 0.2,
    "cvr_optimal": 0.6991071428571428,
    "mvr_optimal": 0.29659815733522327,
    "optimal_threshold": 0.3655772998895124,
    "n_oos": 1120,
    "n_is": 5644,
    "oos_ratio": 0.16558249556475457
  },
  "confidence_delta": {
    "method": "confidence_delta",
    "auc": 0.6148912087425332,
    "cvr_at_threshold": 0.4642857142857143,
    "mvr_at_threshold": 0.21545003543586108,
    "threshold": 0.2,
    "cvr_optimal": 0.5276785714285714,
    "mvr_optimal": 0.25992204110559886,
    "optimal_threshold": 0.18714293000393994,
    "n_oos": 1120,
    "n_is": 5644,
    "oos_ratio": 0.16558249556475457
  },
  "composite": {
    "method": "composite",
    "auc": 0.7776235983851373,
    "cvr_at_threshold": 0.6553571428571429,
    "mvr_at_threshold": 0.231218993621545,
    "threshold": 0.2,
    "cvr_optimal": 0.6892857142857143,
    "mvr_optimal": 0.25744153082919913,
    "optimal_threshold": 0.19132212214174144,
    "n_oos": 1120,
    "n_is": 5644,
    "oos_ratio": 0.16558249556475457
  },
  "energy_score": {
    "method": "energy_score",
    "auc": 0.234419532120077,
    "cvr_at_threshold": 1.0,
    "mvr_at_threshold": 1.0,
    "threshold": 0.2,
    "cvr_optimal": 1.0,
    "mvr_optimal": 0.9998228206945429,
    "optimal_threshold": 1.6425858736038208,
    "n_oos": 1120,
    "n_is": 5644,
    "oos_ratio": 0.16558249556475457
  },
  "mahalanobis": {
    "method": "mahalanobis",
    "auc": 0.315140999924066,
    "cvr_at_threshold": 0.0,
    "mvr_at_threshold": 0.0,
    "threshold": 0.2,
    "cvr_optimal": 1.0,
    "mvr_optimal": 0.9994684620836286,
    "optimal_threshold": -4009.9912109375,
    "n_oos": 1120,
    "n_is": 5644,
    "oos_ratio": 0.16558249556475457
  },
  "relative_mahalanobis": {
    "method": "relative_mahalanobis",
    "auc": 0.3086900912473423,
    "cvr_at_threshold": 0.6723214285714286,
    "mvr_at_threshold": 0.9227498228206945,
    "threshold": 0.2,
    "cvr_optimal": 0.0017857142857142857,
    "mvr_optimal": 0.00035435861091424523,
    "optimal_threshold": 14.141703605651855,
    "n_oos": 1120,
    "n_is": 5644,
    "oos_ratio": 0.16558249556475457
  }
}
```

### McNemar Statistical Significance Tests
```json
{
  "B1_model_only_vs_B3_calibrated": {
    "chi2": 0.0,
    "p_value": 1.0,
    "b": 0,
    "c": 0,
    "significant": false
  },
  "B1_model_only_vs_B4_knn": {
    "method_a": "B1_model_only",
    "method_b": "B4_knn",
    "b_a_right_b_wrong": 30,
    "c_a_wrong_b_right": 45,
    "chi2": 2.6133333333333333,
    "p_value": 0.10596880912720208,
    "significant_at_0.05": false,
    "significant_at_0.01": false
  },
  "B1_model_only_vs_B5_protoknn": {
    "method_a": "B1_model_only",
    "method_b": "B5_protoknn",
    "b_a_right_b_wrong": 107,
    "c_a_wrong_b_right": 82,
    "chi2": 3.0476190476190474,
    "p_value": 0.0808555983700523,
    "significant_at_0.05": false,
    "significant_at_0.01": false
  },
  "B1_model_only_vs_M1_gated_knn": {
    "method_a": "B1_model_only",
    "method_b": "M1_gated_knn",
    "b_a_right_b_wrong": 30,
    "c_a_wrong_b_right": 45,
    "chi2": 2.6133333333333333,
    "p_value": 0.10596880912720208,
    "significant_at_0.05": false,
    "significant_at_0.01": false
  }
}
```

## Experiment: CI_kNN_vuln_codebert_results

### Final Pipeline Metrics
```json
[
  {
    "Method": "CodeBERT (Lu et al. 2021)",
    "Acc": "--",
    "F1-M": 0.628,
    "MCC": "--",
    "ECE": "--",
    "Brier": "--"
  },
  {
    "Method": "kNN-for-Vuln (EMNLP 2022)",
    "Acc": "--",
    "F1-M": 0.66,
    "MCC": "--",
    "ECE": "--",
    "Brier": "--"
  },
  {
    "Method": "B1: Model-Only",
    "Acc": 0.6116398243045388,
    "F1-M": 0.6011527916808563,
    "MCC": 0.2812547103808644,
    "ECE": 0.05143767200399461,
    "Brier": 0.4615511143913126
  },
  {
    "Method": "B3: + Temp Scaling",
    "Acc": 0.6116398243045388,
    "F1-M": 0.6011527916808563,
    "MCC": 0.2812547103808644,
    "ECE": 0.08873271380113561,
    "Brier": 0.4593597379575855
  },
  {
    "Method": "B4: + kNN (always-on)",
    "Acc": 0.6182284040995608,
    "F1-M": 0.6112125100711066,
    "MCC": 0.28370727449336813,
    "ECE": 0.08573922133385867,
    "Brier": 0.4572849720856621
  },
  {
    "Method": "B5: + ProtokNN",
    "Acc": 0.6138360175695461,
    "F1-M": 0.6054197458239803,
    "MCC": 0.27904429774583844,
    "ECE": 0.07488570019348789,
    "Brier": 0.4543992454901385
  },
  {
    "Method": "M1: CI-Gated kNN [OURS]",
    "Acc": 0.6248169838945827,
    "F1-M": 0.6242619719114926,
    "MCC": 0.2667095397720488,
    "ECE": 0.07743967278427437,
    "Brier": 0.4583368624184163
  }
]
```

### k/λ Parameter Ablation Results
```json
[
  {
    "k": 4,
    "lambda": 0.1,
    "accuracy": 0.612371888726208,
    "f1_macro": 0.6123676820061998,
    "mcc": 0.23222558340658778,
    "ece": 0.17359021789215467,
    "brier": 0.5209495053360779
  },
  {
    "k": 4,
    "lambda": 0.3,
    "accuracy": 0.612371888726208,
    "f1_macro": 0.6123676820061998,
    "mcc": 0.23222558340658778,
    "ece": 0.13258576236415182,
    "brier": 0.4840727390610953
  },
  {
    "k": 4,
    "lambda": 0.5,
    "accuracy": 0.612371888726208,
    "f1_macro": 0.6123676820061998,
    "mcc": 0.23222558340658778,
    "ece": 0.0915813068361489,
    "brier": 0.46044748781581807
  },
  {
    "k": 4,
    "lambda": 0.7,
    "accuracy": 0.6222547584187409,
    "f1_macro": 0.6213163476478134,
    "mcc": 0.26442148014921363,
    "ece": 0.06041525572867055,
    "brier": 0.4500737516002461
  },
  {
    "k": 4,
    "lambda": 0.9,
    "accuracy": 0.6145680819912153,
    "f1_macro": 0.6080992035689614,
    "mcc": 0.27359364197849195,
    "ece": 0.07111019083327896,
    "brier": 0.4529515304143794
  },
  {
    "k": 8,
    "lambda": 0.1,
    "accuracy": 0.6306734992679356,
    "f1_macro": 0.6304096201713616,
    "mcc": 0.26458453887565,
    "ece": 0.13373998815985005,
    "brier": 0.4813338977999359
  },
  {
    "k": 8,
    "lambda": 0.3,
    "accuracy": 0.6306734992679356,
    "f1_macro": 0.6304096201713616,
    "mcc": 0.26458453887565,
    "ece": 0.09863988790495506,
    "brier": 0.45861280194105
  },
  {
    "k": 8,
    "lambda": 0.5,
    "accuracy": 0.6317715959004392,
    "f1_macro": 0.6317713985599867,
    "mcc": 0.27178986681693673,
    "ece": 0.05999883639766381,
    "brier": 0.44608481992630816
  },
  {
    "k": 8,
    "lambda": 0.7,
    "accuracy": 0.6251830161054173,
    "f1_macro": 0.6236209861953319,
    "mcc": 0.2744520788775548,
    "ece": 0.05895683792126637,
    "brier": 0.4437499517557108
  },
  {
    "k": 8,
    "lambda": 0.9,
    "accuracy": 0.6182284040995608,
    "f1_macro": 0.6112125100711066,
    "mcc": 0.28370727449336813,
    "ece": 0.07664831941444397,
    "brier": 0.45160819742925784
  },
  {
    "k": 16,
    "lambda": 0.1,
    "accuracy": 0.633601756954612,
    "f1_macro": 0.6326780661794976,
    "mcc": 0.26658658786257966,
    "ece": 0.117582839404925,
    "brier": 0.4660018133686253
  },
  {
    "k": 16,
    "lambda": 0.3,
    "accuracy": 0.6350658857979502,
    "f1_macro": 0.6343486258828652,
    "mcc": 0.2704384631182593,
    "ece": 0.08591878343097896,
    "brier": 0.44939607535766807
  },
  {
    "k": 16,
    "lambda": 0.5,
    "accuracy": 0.6277452415812591,
    "f1_macro": 0.6276693669501225,
    "mcc": 0.2669049098468515,
    "ece": 0.05756478553434677,
    "brier": 0.4414358923483957
  },
  {
    "k": 16,
    "lambda": 0.7,
    "accuracy": 0.626281112737921,
    "f1_macro": 0.6241649767125195,
    "mcc": 0.27985149267042814,
    "ece": 0.06562086491853326,
    "brier": 0.442121264340808
  },
  {
    "k": 16,
    "lambda": 0.9,
    "accuracy": 0.6160322108345534,
    "f1_macro": 0.608818398339217,
    "mcc": 0.27958575628303545,
    "ece": 0.07991016481204606,
    "brier": 0.45145219133490516
  },
  {
    "k": 32,
    "lambda": 0.1,
    "accuracy": 0.636896046852123,
    "f1_macro": 0.6356775097259526,
    "mcc": 0.27205644770459975,
    "ece": 0.11527461691025048,
    "brier": 0.4590847123915107
  },
  {
    "k": 32,
    "lambda": 0.3,
    "accuracy": 0.6314055636896047,
    "f1_macro": 0.6311125324236933,
    "mcc": 0.265794737645297,
    "ece": 0.08000535692570197,
    "brier": 0.4452558985013911
  },
  {
    "k": 32,
    "lambda": 0.5,
    "accuracy": 0.6299414348462665,
    "f1_macro": 0.6297802783297535,
    "mcc": 0.27284346060967496,
    "ece": 0.05801823017851114,
    "brier": 0.43936418785846043
  },
  {
    "k": 32,
    "lambda": 0.7,
    "accuracy": 0.6226207906295754,
    "f1_macro": 0.6203992295027199,
    "mcc": 0.2727436737939059,
    "ece": 0.07382641488391109,
    "brier": 0.4414095804627188
  },
  {
    "k": 32,
    "lambda": 0.9,
    "accuracy": 0.6142020497803806,
    "f1_macro": 0.6069936033183911,
    "mcc": 0.2755158488796141,
    "ece": 0.07535137381856058,
    "brier": 0.451392076314166
  }
]
```

### OOD Detection Metrics (Mahalanobis, Energy, RMD, Baseline)
```json
{
  "knn_distance": {
    "method": "knn_distance",
    "auc": 0.47139792806375425,
    "cvr_at_threshold": 0.34872761545711595,
    "mvr_at_threshold": 0.412327947336924,
    "threshold": 0.2,
    "cvr_optimal": 0.9802073515551367,
    "mvr_optimal": 0.9575104727707959,
    "optimal_threshold": 0.04147139936685562,
    "n_oos": 1061,
    "n_is": 1671,
    "oos_ratio": 0.3883601756954612
  },
  "disagreement": {
    "method": "disagreement",
    "auc": 0.3893594843792567,
    "cvr_at_threshold": 0.04241281809613572,
    "mvr_at_threshold": 0.054458408138839016,
    "threshold": 0.2,
    "cvr_optimal": 0.007540056550424128,
    "mvr_optimal": 0.003590664272890485,
    "optimal_threshold": 0.2984337087616833,
    "n_oos": 1061,
    "n_is": 1671,
    "oos_ratio": 0.3883601756954612
  },
  "entropy": {
    "method": "entropy",
    "auc": 0.655679211430112,
    "cvr_at_threshold": 1.0,
    "mvr_at_threshold": 1.0,
    "threshold": 0.2,
    "cvr_optimal": 0.764373232799246,
    "mvr_optimal": 0.5122681029323758,
    "optimal_threshold": 0.9280195854110226,
    "n_oos": 1061,
    "n_is": 1671,
    "oos_ratio": 0.3883601756954612
  },
  "confidence_delta": {
    "method": "confidence_delta",
    "auc": 0.3886352599170526,
    "cvr_at_threshold": 0.6098020735155514,
    "mvr_at_threshold": 0.7528426092160383,
    "threshold": 0.2,
    "cvr_optimal": 0.9990574929311969,
    "mvr_optimal": 0.997606223818073,
    "optimal_threshold": 0.0007558917905080964,
    "n_oos": 1061,
    "n_is": 1671,
    "oos_ratio": 0.3883601756954612
  },
  "composite": {
    "method": "composite",
    "auc": 0.3971048506681873,
    "cvr_at_threshold": 1.0,
    "mvr_at_threshold": 1.0,
    "threshold": 0.2,
    "cvr_optimal": 0.00942507068803016,
    "mvr_optimal": 0.00718132854578097,
    "optimal_threshold": 0.5737497014657744,
    "n_oos": 1061,
    "n_is": 1671,
    "oos_ratio": 0.3883601756954612
  },
  "energy_score": {
    "method": "energy_score",
    "auc": 0.3443207885698879,
    "cvr_at_threshold": 1.0,
    "mvr_at_threshold": 1.0,
    "threshold": 0.2,
    "cvr_optimal": 0.998114985862394,
    "mvr_optimal": 0.995212447636146,
    "optimal_threshold": 1.193157434463501,
    "n_oos": 1061,
    "n_is": 1671,
    "oos_ratio": 0.3883601756954612
  },
  "mahalanobis": {
    "method": "mahalanobis",
    "auc": 0.5672970352484106,
    "cvr_at_threshold": 0.0,
    "mvr_at_threshold": 0.0,
    "threshold": 0.2,
    "cvr_optimal": 0.530631479736098,
    "mvr_optimal": 0.40514661879114305,
    "optimal_threshold": -592.6171875,
    "n_oos": 1061,
    "n_is": 1671,
    "oos_ratio": 0.3883601756954612
  },
  "relative_mahalanobis": {
    "method": "relative_mahalanobis",
    "auc": 0.341691526630196,
    "cvr_at_threshold": 0.49670122525918947,
    "mvr_at_threshold": 0.7157390783961699,
    "threshold": 0.2,
    "cvr_optimal": 1.0,
    "mvr_optimal": 0.997606223818073,
    "optimal_threshold": -0.4708878695964813,
    "n_oos": 1061,
    "n_is": 1671,
    "oos_ratio": 0.3883601756954612
  }
}
```

### McNemar Statistical Significance Tests
```json
{
  "B1_model_only_vs_B3_calibrated": {
    "chi2": 0.0,
    "p_value": 1.0,
    "b": 0,
    "c": 0,
    "significant": false
  },
  "B1_model_only_vs_B4_knn": {
    "method_a": "B1_model_only",
    "method_b": "B4_knn",
    "b_a_right_b_wrong": 30,
    "c_a_wrong_b_right": 48,
    "chi2": 3.7051282051282053,
    "p_value": 0.05424550401058548,
    "significant_at_0.05": false,
    "significant_at_0.01": false
  },
  "B1_model_only_vs_B5_protoknn": {
    "method_a": "B1_model_only",
    "method_b": "B5_protoknn",
    "b_a_right_b_wrong": 19,
    "c_a_wrong_b_right": 25,
    "chi2": 0.5681818181818182,
    "p_value": 0.45098231926888255,
    "significant_at_0.05": false,
    "significant_at_0.01": false
  },
  "B1_model_only_vs_M1_gated_knn": {
    "method_a": "B1_model_only",
    "method_b": "M1_gated_knn",
    "b_a_right_b_wrong": 152,
    "c_a_wrong_b_right": 188,
    "chi2": 3.6029411764705883,
    "p_value": 0.05767744381641915,
    "significant_at_0.05": false,
    "significant_at_0.01": false
  }
}
```

## Experiment: CI_kNN_vuln_graphcodebert_results

### Final Pipeline Metrics
```json
[
  {
    "Method": "CodeBERT (Lu et al. 2021)",
    "Acc": "--",
    "F1-M": 0.628,
    "MCC": "--",
    "ECE": "--",
    "Brier": "--"
  },
  {
    "Method": "kNN-for-Vuln (EMNLP 2022)",
    "Acc": "--",
    "F1-M": 0.66,
    "MCC": "--",
    "ECE": "--",
    "Brier": "--"
  },
  {
    "Method": "B1: Model-Only",
    "Acc": 0.6105417276720352,
    "F1-M": 0.6050553386212763,
    "MCC": 0.26112635408329565,
    "ECE": 0.052016824287516486,
    "Brier": 0.4628655687763507
  },
  {
    "Method": "B3: + Temp Scaling",
    "Acc": 0.6105417276720352,
    "F1-M": 0.6050553386212763,
    "MCC": 0.26112635408329565,
    "ECE": 0.07503827088890423,
    "Brier": 0.4604725692125256
  },
  {
    "Method": "B4: + kNN (always-on)",
    "Acc": 0.6149341142020498,
    "F1-M": 0.6126013630089717,
    "MCC": 0.25720270918220617,
    "ECE": 0.07378638339376872,
    "Brier": 0.45989854677581316
  },
  {
    "Method": "B5: + ProtokNN",
    "Acc": 0.6105417276720352,
    "F1-M": 0.6070918028603374,
    "MCC": 0.25302409591430675,
    "ECE": 0.08754624520846889,
    "Brier": 0.4523103995701936
  },
  {
    "Method": "M1: CI-Gated kNN [OURS]",
    "Acc": 0.6145680819912153,
    "F1-M": 0.6125944157658993,
    "MCC": 0.25463090301884755,
    "ECE": 0.07667121502557714,
    "Brier": 0.4604917363040412
  }
]
```

### k/λ Parameter Ablation Results
```json
[
  {
    "k": 4,
    "lambda": 0.1,
    "accuracy": 0.6142020497803806,
    "f1_macro": 0.6141019538218746,
    "mcc": 0.23996282230230137,
    "ece": 0.1817705303423512,
    "brier": 0.5238582187319196
  },
  {
    "k": 4,
    "lambda": 0.3,
    "accuracy": 0.6145680819912153,
    "f1_macro": 0.6144118076361527,
    "mcc": 0.24159456617601713,
    "ece": 0.14077251957319678,
    "brier": 0.488971851542191
  },
  {
    "k": 4,
    "lambda": 0.5,
    "accuracy": 0.6149341142020498,
    "f1_macro": 0.6147604839379877,
    "mcc": 0.24258124878445667,
    "ece": 0.09983955343762783,
    "brier": 0.46597157995047933
  },
  {
    "k": 4,
    "lambda": 0.7,
    "accuracy": 0.6185944363103953,
    "f1_macro": 0.6176469324118428,
    "mcc": 0.25695344485416555,
    "ece": 0.05689324833101965,
    "brier": 0.45485740395678503
  },
  {
    "k": 4,
    "lambda": 0.9,
    "accuracy": 0.6120058565153733,
    "f1_macro": 0.6083486788435364,
    "mcc": 0.25701089852375947,
    "ece": 0.057071182064542626,
    "brier": 0.4556293235611078
  },
  {
    "k": 8,
    "lambda": 0.1,
    "accuracy": 0.6120058565153733,
    "f1_macro": 0.6119457545210372,
    "mcc": 0.23473540937378323,
    "ece": 0.16388536228383627,
    "brier": 0.4957482153271066
  },
  {
    "k": 8,
    "lambda": 0.3,
    "accuracy": 0.6120058565153733,
    "f1_macro": 0.611922665414493,
    "mcc": 0.23520878762234315,
    "ece": 0.12800683410664804,
    "brier": 0.4722080867275468
  },
  {
    "k": 8,
    "lambda": 0.5,
    "accuracy": 0.6127379209370425,
    "f1_macro": 0.6124535889180822,
    "mcc": 0.2394718972693578,
    "ece": 0.09167951177965657,
    "brier": 0.45764001305921725
  },
  {
    "k": 8,
    "lambda": 0.7,
    "accuracy": 0.6134699853587116,
    "f1_macro": 0.6120376511443644,
    "mcc": 0.24940720238947442,
    "ece": 0.08012538535039193,
    "brier": 0.4520439943221179
  },
  {
    "k": 8,
    "lambda": 0.9,
    "accuracy": 0.609809663250366,
    "f1_macro": 0.6057863070898885,
    "mcc": 0.2538462540930578,
    "ece": 0.06030590281251494,
    "brier": 0.4554200305162488
  },
  {
    "k": 16,
    "lambda": 0.1,
    "accuracy": 0.6142020497803806,
    "f1_macro": 0.6141193295725242,
    "mcc": 0.239642922606605,
    "ece": 0.14962048119239346,
    "brier": 0.4840071591790659
  },
  {
    "k": 16,
    "lambda": 0.3,
    "accuracy": 0.6149341142020498,
    "f1_macro": 0.6148249167616852,
    "mcc": 0.2416024307781378,
    "ece": 0.11616158331129466,
    "brier": 0.4657198376849679
  },
  {
    "k": 16,
    "lambda": 0.5,
    "accuracy": 0.6171303074670571,
    "f1_macro": 0.6165746769626439,
    "mcc": 0.2509984772690695,
    "ece": 0.08347134068988087,
    "brier": 0.454893894826667
  },
  {
    "k": 16,
    "lambda": 0.7,
    "accuracy": 0.6174963396778916,
    "f1_macro": 0.6157343081369533,
    "mcc": 0.259607139052065,
    "ece": 0.08073507328414094,
    "brier": 0.4515293306041628
  },
  {
    "k": 16,
    "lambda": 0.9,
    "accuracy": 0.6083455344070278,
    "f1_macro": 0.6043659900161705,
    "mcc": 0.2505456184884861,
    "ece": 0.061895535381228065,
    "brier": 0.45562614501745546
  },
  {
    "k": 32,
    "lambda": 0.1,
    "accuracy": 0.6120058565153733,
    "f1_macro": 0.6118958286762226,
    "mcc": 0.23568728470801925,
    "ece": 0.1459896904032526,
    "brier": 0.47659924443215046
  },
  {
    "k": 32,
    "lambda": 0.3,
    "accuracy": 0.6116398243045388,
    "f1_macro": 0.6112084659695478,
    "mcc": 0.23870105029554636,
    "ece": 0.115486546386695,
    "brier": 0.46136576059744633
  },
  {
    "k": 32,
    "lambda": 0.5,
    "accuracy": 0.6134699853587116,
    "f1_macro": 0.6124812284580605,
    "mcc": 0.24668736376393535,
    "ece": 0.09010612457067319,
    "brier": 0.45278929289436676
  },
  {
    "k": 32,
    "lambda": 0.7,
    "accuracy": 0.6142020497803806,
    "f1_macro": 0.6121233477517787,
    "mcc": 0.25440497231083276,
    "ece": 0.08490667748309925,
    "brier": 0.4508698413229119
  },
  {
    "k": 32,
    "lambda": 0.9,
    "accuracy": 0.6076134699853587,
    "f1_macro": 0.6034480891009102,
    "mcc": 0.24972647761798916,
    "ece": 0.06287039127977646,
    "brier": 0.4556074058830816
  }
]
```

### OOD Detection Metrics (Mahalanobis, Energy, RMD, Baseline)
```json
{
  "knn_distance": {
    "method": "knn_distance",
    "auc": 0.5548811890337354,
    "cvr_at_threshold": 0.9821428571428571,
    "mvr_at_threshold": 0.8908872901678657,
    "threshold": 0.2,
    "cvr_optimal": 0.8872180451127819,
    "mvr_optimal": 0.7697841726618705,
    "optimal_threshold": 0.321641206741333,
    "n_oos": 1064,
    "n_is": 1668,
    "oos_ratio": 0.38945827232796487
  },
  "disagreement": {
    "method": "disagreement",
    "auc": 0.3509045207443068,
    "cvr_at_threshold": 0.005639097744360902,
    "mvr_at_threshold": 0.01498800959232614,
    "threshold": 0.2,
    "cvr_optimal": 0.9924812030075187,
    "mvr_optimal": 0.9850119904076738,
    "optimal_threshold": 1.1629357813847108e-05,
    "n_oos": 1064,
    "n_is": 1668,
    "oos_ratio": 0.38945827232796487
  },
  "entropy": {
    "method": "entropy",
    "auc": 0.646212822974703,
    "cvr_at_threshold": 1.0,
    "mvr_at_threshold": 1.0,
    "threshold": 0.2,
    "cvr_optimal": 0.831766917293233,
    "mvr_optimal": 0.6001199040767387,
    "optimal_threshold": 0.9146165292117076,
    "n_oos": 1064,
    "n_is": 1668,
    "oos_ratio": 0.38945827232796487
  },
  "confidence_delta": {
    "method": "confidence_delta",
    "auc": 0.3547783014190151,
    "cvr_at_threshold": 0.5234962406015038,
    "mvr_at_threshold": 0.7002398081534772,
    "threshold": 0.2,
    "cvr_optimal": 0.9943609022556391,
    "mvr_optimal": 0.9910071942446043,
    "optimal_threshold": 0.004355056053338615,
    "n_oos": 1064,
    "n_is": 1668,
    "oos_ratio": 0.38945827232796487
  },
  "composite": {
    "method": "composite",
    "auc": 0.420502413858387,
    "cvr_at_threshold": 1.0,
    "mvr_at_threshold": 1.0,
    "threshold": 0.2,
    "cvr_optimal": 0.0,
    "mvr_optimal": 0.0,
    "optimal_threshold": Infinity,
    "n_oos": 1064,
    "n_is": 1668,
    "oos_ratio": 0.38945827232796487
  },
  "energy_score": {
    "method": "energy_score",
    "auc": 0.35378717702529705,
    "cvr_at_threshold": 1.0,
    "mvr_at_threshold": 1.0,
    "threshold": 0.2,
    "cvr_optimal": 0.9915413533834586,
    "mvr_optimal": 0.9838129496402878,
    "optimal_threshold": 1.1932041645050049,
    "n_oos": 1064,
    "n_is": 1668,
    "oos_ratio": 0.38945827232796487
  },
  "mahalanobis": {
    "method": "mahalanobis",
    "auc": 0.4765429197994987,
    "cvr_at_threshold": 0.0,
    "mvr_at_threshold": 0.0,
    "threshold": 0.2,
    "cvr_optimal": 0.974624060150376,
    "mvr_optimal": 0.9628297362110312,
    "optimal_threshold": -1497.0511474609375,
    "n_oos": 1064,
    "n_is": 1668,
    "oos_ratio": 0.38945827232796487
  },
  "relative_mahalanobis": {
    "method": "relative_mahalanobis",
    "auc": 0.35467406150267755,
    "cvr_at_threshold": 0.5140977443609023,
    "mvr_at_threshold": 0.7038369304556354,
    "threshold": 0.2,
    "cvr_optimal": 0.9943609022556391,
    "mvr_optimal": 0.9928057553956835,
    "optimal_threshold": -0.5203239917755127,
    "n_oos": 1064,
    "n_is": 1668,
    "oos_ratio": 0.38945827232796487
  }
}
```

### McNemar Statistical Significance Tests
```json
{
  "B1_model_only_vs_B3_calibrated": {
    "chi2": 0.0,
    "p_value": 1.0,
    "b": 0,
    "c": 0,
    "significant": false
  },
  "B1_model_only_vs_B4_knn": {
    "method_a": "B1_model_only",
    "method_b": "B4_knn",
    "b_a_right_b_wrong": 51,
    "c_a_wrong_b_right": 63,
    "chi2": 1.0614035087719298,
    "p_value": 0.3028952953364402,
    "significant_at_0.05": false,
    "significant_at_0.01": false
  },
  "B1_model_only_vs_B5_protoknn": {
    "method_a": "B1_model_only",
    "method_b": "B5_protoknn",
    "b_a_right_b_wrong": 35,
    "c_a_wrong_b_right": 35,
    "chi2": 0.014285714285714285,
    "p_value": 0.9048611294504482,
    "significant_at_0.05": false,
    "significant_at_0.01": false
  },
  "B1_model_only_vs_M1_gated_knn": {
    "method_a": "B1_model_only",
    "method_b": "M1_gated_knn",
    "b_a_right_b_wrong": 61,
    "c_a_wrong_b_right": 72,
    "chi2": 0.7518796992481203,
    "p_value": 0.385881758660254,
    "significant_at_0.05": false,
    "significant_at_0.01": false
  }
}
```

