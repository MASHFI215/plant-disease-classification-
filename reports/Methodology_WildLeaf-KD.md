# Proposed Methodology & Research Pathway
## "Foundation-to-Mobile: Distilling Self-Supervised Vision Foundation Models into Lightweight CNNs for Robust and Explainable In-the-Wild Plant Disease Recognition"

**প্রস্তাবিত পদ্ধতি ও গবেষণার রোডম্যাপ** — Kaggle T4 GPU-র জন্য সাজানো

Working name of the framework: **WildLeaf-KD**
Prepared: 23 Sep 2026 · Builds on the verified literature review (step 1)

---

## 0. The one-paragraph idea / এক অনুচ্ছেদে মূল ধারণা

**English.** The literature shows a clear split: large foundation models (CLIP, DINOv2) are the best on in-the-wild datasets (PlantWild ≈ 76–80%) but cannot run on a phone, while lightweight CNNs (MobileNet, EfficientNet) run on a phone but have never been properly benchmarked on PlantWild and often learn the *background* instead of the disease. WildLeaf-KD uses a frozen foundation model in **two roles**: (1) as a **teacher** that distills its knowledge into a mobile CNN, and (2) as a **leaf-finder** whose attention maps let us build *background-counterfactual* training images. We then prove, with numbers rather than pretty heatmaps, whether the student learned to look at the leaf, and whether it generalizes across datasets — all under a leakage-free, multi-seed protocol that fits in Kaggle's free T4 quota.

**বাংলা।** সাহিত্য পর্যালোচনায় দেখা গেছে: বড় foundation model (CLIP, DINOv2) wild ডেটায় সবচেয়ে ভালো, কিন্তু মোবাইলে চলে না; আর ছোট মডেল (MobileNet, EfficientNet) মোবাইলে চলে, কিন্তু PlantWild-এ কেউ সঠিকভাবে পরীক্ষা করেনি, এবং এরা প্রায়ই রোগ না দেখে **ব্যাকগ্রাউন্ড** দেখে সিদ্ধান্ত নেয়। আমাদের পদ্ধতিতে বড় মডেলটিকে **দুটি কাজে** লাগানো হবে — (১) **শিক্ষক (teacher)** হিসেবে ছোট মডেলকে শেখাবে (knowledge distillation), (২) **পাতা খুঁজে বের করার যন্ত্র** হিসেবে, যার attention map দিয়ে ব্যাকগ্রাউন্ড বদলে নতুন ট্রেনিং ছবি বানানো হবে। তারপর সংখ্যা দিয়ে প্রমাণ করা হবে ছোট মডেলটি সত্যিই পাতার দিকে তাকাচ্ছে কি না এবং অন্য ডেটাসেটে কাজ করে কি না।

---

## 1. Honest novelty positioning / নতুনত্ব কোথায় — সৎ অবস্থান

Individual ingredients already exist. A reviewer will know these papers, so the thesis must say exactly how it differs:

| Closest prior work | What it already did | What it did **not** do (your space) |
|---|---|---|
| AgriKD (arXiv 2605.01355, 2026) | ViT-Base (supervised) → MobileNetV2 multi-level KD, edge deployment | Mostly curated leaf datasets; no PlantDoc/PlantWild; teacher not a self-supervised foundation model; no shortcut analysis |
| TinyCNN (arXiv 2609.20290, 2026) | Grad-CAM "off-leaf ratio" showing PlantVillage models look at background; PV→PlantDoc drop | Uses a segmentation proxy, no ground-truth boxes; diagnosis only, no fix; no PlantWild |
| DC-FEN (Agriculture 16:1790, 2026) | Duplicate-audited splits, 5 seeds, KD ablations | PlantVillage/FGVC8; PlantDoc only as a small "pressure test"; no PlantWild; no foundation teacher |
| El Karch et al. (Front. Plant Sci., 2026) | Frozen DINOv2+DINOv3+CLIP fusion, 80.23% on PlantWild | Not deployable (3× ViT-L); no student model |
| Duhan et al. (2026) | Lightweight models + quantization + Raspberry Pi on PlantDoc | No KD, no PlantWild, augmentation-timing/leakage unclear, qualitative Grad-CAM |
| Kumar et al. (arXiv 2508.10817, 2025) | MobileNet/EfficientNet on merged PD+PV+PW (94.7%) | PlantVillage dominates; no per-dataset wild evaluation |

**Your novelty = the combination that nobody has published (as far as my search found):**

- **N1. Dual-role foundation teacher.** The same frozen DINOv2 (+CLIP) is used both as a distillation teacher *and* as an unsupervised leaf-localizer to generate background-counterfactual training views. No agricultural paper found uses one frozen model for both.
- **N2. Background-Counterfactual Consistency (BCC).** A training objective that forces the student to give the same (teacher-guided) prediction whether the background is real, swapped, or removed. This *fixes* the shortcut problem that TinyCNN only diagnosed.
- **N3. Ground-truth-based quantitative explainability + causal test.** Grad-CAM *energy-in-box* and *pointing game* using PlantDoc's real leaf bounding boxes, plus a **Background Reliance Score** (how much the prediction changes when only the background changes). This tests the research question "does distillation from a foundation model transfer *where to look*, not only *what to predict*?"
- **N4. First leakage-free lightweight benchmark on PlantWild** (official split, 3 seeds, macro-F1, calibration, on-device latency), plus PlantDoc ↔ PlantWild cross-dataset evaluation on shared classes and FieldPlant as a true field test.

> ⚠️ **Novelty check (do this in week 1):** search Google Scholar, arXiv and Semantic Scholar for "background swap augmentation plant disease", "copy-paste leaf augmentation", "DINOv2 distillation agriculture", "PlantWild MobileNet". If something identical appears, we adjust N1/N2 wording before writing. New papers appear monthly in this area.

**বাংলা।** আলাদা আলাদা উপাদান (KD, Grad-CAM, lightweight model) আগেই আছে — reviewer এগুলো জানবে। আমাদের নতুনত্ব হলো **সমন্বয়**, যা এখন পর্যন্ত খুঁজে পাইনি: (N1) একই বড় মডেলকে শিক্ষক ও পাতা-খোঁজার যন্ত্র — দুই কাজে ব্যবহার; (N2) ব্যাকগ্রাউন্ড বদলালেও একই উত্তর দিতে বাধ্য করা (Background-Counterfactual Consistency) — TinyCNN শুধু সমস্যা দেখিয়েছে, আমরা সমাধান করছি; (N3) PlantDoc-এর আসল bounding box দিয়ে Grad-CAM সংখ্যায় মাপা, আর ব্যাকগ্রাউন্ড বদলালে উত্তর কতটা বদলায় তা মাপা; (N4) PlantWild-এ প্রথম leakage-মুক্ত হালকা মডেলের benchmark। প্রথম সপ্তাহে আবার একবার novelty যাচাই করতে হবে।

---

## 2. Research questions & objectives / গবেষণা প্রশ্ন ও উদ্দেশ্য

### Research questions
- **RQ1.** How well do lightweight CNNs perform on PlantDoc and PlantWild under a leakage-free, reproducible protocol, and how far are they from foundation models?
- **RQ2.** Does distillation from a frozen self-supervised foundation model close that gap for a mobile-sized student?
- **RQ3.** Does background-counterfactual consistency reduce shortcut learning (reliance on background) and improve cross-dataset generalization?
- **RQ4.** Does the student learn to *look at the leaf*? (measured with ground-truth boxes)
- **RQ5.** What is the accuracy–efficiency trade-off after INT8 quantization on CPU/mobile?

### Objectives (mapped to gaps from the literature review)

| Objective | Covers gap |
|---|---|
| **O1.** Build an audited, leakage-free protocol for PlantDoc, PlantWild and FieldPlant (duplicate detection, official splits, shared-class mapping, label-noise report). | G2, G7 |
| **O2.** Benchmark 4 lightweight students and 2 foundation-model reference models on both wild datasets with 3 seeds, macro-F1 and calibration. | G1, G6 |
| **O3.** Design and evaluate WildLeaf-KD: foundation-teacher distillation + background-counterfactual consistency. | G5, and the lab-vs-wild gap |
| **O4.** Quantify explainability and shortcut reliance with ground-truth boxes and counterfactual tests. | G4 |
| **O5.** Evaluate cross-dataset generalization PD ↔ PW and → FieldPlant on shared classes. | G3 |
| **O6.** Measure deployability: params, MACs, model size, FP32/INT8 CPU latency (and optionally a phone). | G6 |

**বাংলা।** পাঁচটি গবেষণা প্রশ্ন ও ছয়টি উদ্দেশ্য — প্রতিটি উদ্দেশ্য সাহিত্য পর্যালোচনার কোনো না কোনো গ্যাপ (G1–G7) পূরণ করে। এতে থিসিস ও জার্নাল পেপার দুটোতেই "আমরা কোন গ্যাপ কীভাবে পূরণ করলাম" পরিষ্কারভাবে দেখানো যাবে।

---

## 3. Datasets & how each is used / ডেটাসেট ও তাদের ভূমিকা

| Dataset | Role | Notes |
|---|---|---|
| **PlantWild** (18,542 imgs, 89 classes) | Main benchmark (train/val/test) | Use the **official 70/10/20 split** from the authors (GitHub tqwei05/MVPDR / Hugging Face). Imbalanced (44–589/class) → macro-F1. |
| **PlantDoc** (2,598 imgs, 27 classes) | Second benchmark + **explainability ground truth** (leaf boxes) | Use the **official train/test split** from the authors' GitHub, not a random Kaggle re-split. Take 15% of train as validation (stratified). Both full-image and cropped-leaf settings reported. |
| **PlantVillage** (54k lab imgs) | Optional intermediate pretraining only (ImageNet → PV → target) | Never mixed into wild test sets. |
| **FieldPlant** (5,170 field imgs) | **External test only** on classes shared with training data | True farm photos; strongest generalization evidence. |

**Shared-class mapping.** Build a table of classes that exist in both PlantDoc and PlantWild (and FieldPlant), e.g. same crop + same disease. Map names manually and have your supervisor verify. Cross-dataset tests use only these classes.

**বাংলা।** PlantWild মূল benchmark, PlantDoc দ্বিতীয় benchmark এবং এর bounding box দিয়ে explainability মাপা হবে, PlantVillage শুধু মাঝখানের pretraining-এর জন্য (টেস্টে কখনো নয়), আর FieldPlant শুধু বাইরের টেস্ট সেট হিসেবে। অবশ্যই **লেখকদের দেওয়া অফিসিয়াল split** ব্যবহার করবে — Kaggle-এর র‍্যান্ডম split নয়। দুই ডেটাসেটে মিল থাকা ক্লাসগুলোর একটি তালিকা বানিয়ে সুপারভাইজারকে দিয়ে যাচাই করাবে।

---

## 4. The pipeline / সম্পূর্ণ পাইপলাইন

```
Phase A: Data audit ─► Phase B: Teachers + leaf masks (cached once)
        │                               │
        ▼                               ▼
Phase C: Baselines (students, CE only) ─► Phase D: WildLeaf-KD students
        │                                        │
        └──────────────► Phase E: Evaluation ◄───┘
                   (in-domain, cross-dataset, XAI, shortcut, calibration)
                                   │
                                   ▼
                  Phase F: INT8 compression & latency
```

### Phase A — Data audit & leakage-free protocol (O1)
1. **Exact/near-duplicate detection across splits**: perceptual hash (pHash, Hamming ≤ 6) **plus** cosine similarity of DINOv2 embeddings (> 0.95). Any test image with a near-twin in train is reported; main results on the *deduplicated* test set, original test set reported for comparability.
2. **Composite/collage and off-topic images** (known PlantDoc issue): flag with a simple rule (multiple detected leaves of different classes) + manual review. Report counts; do not silently delete.
3. **Label-noise audit**: train the teacher linear probe with 5-fold cross-validation on train only, then use *confident learning* (library `cleanlab`) to list likely mislabeled training images. Report the list; run one ablation "with vs without suspected-noise removal". Test sets are never edited.
4. **Rule**: resize/normalize → split fixed → *all augmentation only inside the training dataloader* → never before splitting.

### Phase B — Foundation teachers & leaf masks (computed once, cached)
- **Teachers (frozen)**: DINOv2 ViT-B/14 and CLIP ViT-B/16 (base size fits T4 easily). Extract [CLS] features once for all train/val/test images → save to disk.
- **Teacher head**: train a linear (or linear+prototype) classifier on concatenated features (seconds per run). This is the teacher *T*. Also report ViT-L versions as an **upper-bound reference only** (features cached once, no fine-tuning), so you can say how close the student gets to the best frozen model.
- **Leaf masks**: from DINOv2's last-layer CLS→patch attention, build a soft foreground map per image (average heads, threshold at a percentile, small morphological cleaning). Cache as 16×16 maps.
- **Validate the masks on PlantDoc** using its ground-truth leaf boxes (report box-hit rate / IoU). This shows the masks are trustworthy before using them on PlantWild (which has no boxes).

### Phase C — Baseline students (CE only) (O2)
Students (all ImageNet-pretrained from `timm`):
- **MobileNetV3-Large** (~5.4M params) — main student
- **EfficientNet-B0** (~5.3M)
- **ShuffleNetV2 ×1.0** or **MobileNetV4-Conv-Small** (ultra-light)
- **ResNet-50** (~25M) — "not-lightweight" reference

Training recipe (same for all): 224×224, AdamW, lr 1e-3 head / 1e-4 backbone (or single lr with cosine + 3-epoch warmup), weight decay 0.05, 40 epochs, batch 64, AMP fp16, label smoothing 0.1, RandAugment-light + RandomResizedCrop + flip + color jitter, class-balanced sampling *or* logit-adjusted loss (compare both once on PlantWild, keep the better). Early stopping on **validation macro-F1**. 3 seeds.
Optional variant: ImageNet → PlantVillage → target (two-stage transfer) for the main student only.

### Phase D — WildLeaf-KD (O3, the proposed method)

For each training image *x* with label *y* and cached leaf mask *m*:

1. **Views.** Original view *x*; **background-counterfactual view** *x̃ = m·x + (1−m)·b*, where *b* is one of: another random training image's background, heavy Gaussian blur of *x*, or a flat grey (random choice per batch). Applied with probability p (e.g. 0.5).
2. **Teacher target.** Teacher probabilities *p_T(x)* from the frozen DINOv2+CLIP head (computed on the fly in fp16, or cached for the un-augmented image).
3. **Loss.**

```
L = CE_ls(y, s(x))                                   # label-smoothed cross-entropy
  + α · T² · KL( p_T(x) / T  ||  s(x) / T )          # logit distillation
  + β · (1 − cos( P(f_s(x)), f_T(x) ))                # feature distillation (projected GAP → teacher embedding)
  + γ · KL( stopgrad(s(x)) || s(x̃) )                 # background-counterfactual consistency (BCC)
```
Start with α = 0.7, T = 4, β = 0.5, γ = 1.0; tune on **validation** only.
Note: prior work (a DINOv2→small-model tea study) found feature-alignment can *hurt*; so β is ablated and may end at 0 — that is an acceptable, publishable finding.

4. **Why it should work.** KD gives the small model the richer "dark knowledge" of the foundation model; BCC tells it that the disease label must not depend on the background — exactly the failure mode PlantDoc/PlantWild expose.

### Phase E — Evaluation (O2, O4, O5)
- **Classification**: accuracy, **macro-F1**, balanced accuracy, top-5 (PlantWild), per-class F1, confusion matrix.
- **Calibration**: Expected Calibration Error (ECE) before/after temperature scaling — farmers need trustworthy confidence.
- **Explainability (PlantDoc test, ground truth boxes)**:
  - *Energy-in-box* = % of Grad-CAM (and Grad-CAM++) mass inside any leaf box of the true class.
  - *Pointing game* = does the Grad-CAM peak fall inside a true-class box?
- **Shortcut tests (both datasets)**:
  - *Background Reliance Score (BRS)* = accuracy drop / prediction-flip rate when **only the background** is swapped.
  - *Foreground Sufficiency* = accuracy on foreground-only images (background grey).
  - A robust model: small BRS, high foreground sufficiency.
- **Cross-dataset (shared classes)**: train PD → test PW; train PW → test PD; train each → test FieldPlant.
- **Statistics**: 3 seeds → mean ± std; paired bootstrap 95% CI on the test set, and McNemar's test between baseline and WildLeaf-KD for the same seed.

### Phase F — Compression & deployment (O6)
- Export best student to **ONNX**; dynamic/static **INT8 quantization** with ONNX Runtime (and/or TFLite for Android).
- Report params, MACs (`fvcore` or `thop`), file size, **CPU latency at batch size 1** (Kaggle CPU; optional Android phone via TFLite benchmark app), and accuracy/macro-F1 after INT8.
- Compare against teacher (DINOv2+CLIP ViT-B) latency on the same CPU → gives the "×N faster, ×M smaller" headline.

**বাংলা (সংক্ষেপে পুরো পাইপলাইন)।**
- **Phase A:** ডুপ্লিকেট খোঁজা (pHash + DINOv2 embedding), কোলাজ ছবি চিহ্নিত করা, cleanlab দিয়ে সম্ভাব্য ভুল লেবেল তালিকা — কিছুই গোপনে মুছবে না, রিপোর্ট করবে। augmentation শুধু train dataloader-এ।
- **Phase B:** DINOv2 ও CLIP (base সাইজ) একবার চালিয়ে ফিচার সেভ করা, তার উপর ছোট linear head = teacher। DINOv2-এর attention map থেকে পাতার mask বানানো, আর PlantDoc-এর box দিয়ে mask ঠিক আছে কি না যাচাই।
- **Phase C:** MobileNetV3-Large, EfficientNet-B0, ShuffleNetV2/MobileNetV4-small, এবং রেফারেন্স ResNet-50 — একই রেসিপিতে, ৩টি seed।
- **Phase D (আমাদের মূল পদ্ধতি):** ছবির ব্যাকগ্রাউন্ড বদলে নতুন ভিউ বানানো; loss = cross-entropy + teacher থেকে শেখা (KD) + ফিচার মেলানো + "ব্যাকগ্রাউন্ড বদলালেও উত্তর একই রাখো" (BCC)।
- **Phase E:** accuracy, macro-F1, calibration (ECE), Grad-CAM box-এর ভেতরে কতটা, ব্যাকগ্রাউন্ড বদলালে উত্তর কতটা বদলায় (BRS), cross-dataset টেস্ট, পরিসংখ্যানিক পরীক্ষা (bootstrap CI, McNemar)।
- **Phase F:** ONNX + INT8, CPU-তে ১টি ছবির latency, মডেলের সাইজ।

---

## 5. Experiment matrix & ablations / পরীক্ষার তালিকা

**Main table (each on PlantWild and PlantDoc, 3 seeds):**
1. Students, CE only (4 models)
2. Students + standard augmentation + class balancing
3. Students + logit KD only
4. Students + logit KD + feature KD
5. **WildLeaf-KD (KD + BCC)** — MobileNetV3-L and EfficientNet-B0
6. Teachers: DINOv2-B, CLIP-B, fusion (linear probe) — reference; ViT-L fusion — upper bound
7. Published numbers (MVPDR, El Karch) — clearly marked "different protocol"

**Ablations (MobileNetV3-L, PlantWild + PlantDoc):**
- A1. Teacher: DINOv2 vs CLIP vs fusion
- A2. KD terms: α only / +β / +γ (BCC) / all
- A3. BCC background type: swap vs blur vs grey vs mixed
- A4. Mask source: DINOv2 attention vs PlantDoc GT boxes (PlantDoc only — shows how much better "perfect" masks would be)
- A5. Two-stage transfer via PlantVillage: yes/no
- A6. Label-noise removal: yes/no
- A7. Input resolution 224 vs 288 (latency trade-off)
- A8. INT8 vs FP32

**বাংলা।** মূল টেবিলে ধাপে ধাপে দেখানো হবে কোন অংশ যোগ করলে কতটা উন্নতি হয় (baseline → KD → KD+feature → পূর্ণ WildLeaf-KD)। Ablation-এ প্রতিটি উপাদান আলাদাভাবে সরিয়ে/বদলে দেখা হবে — ভালো জার্নালে এটা বাধ্যতামূলক।

---

## 6. Kaggle T4 compute plan / Kaggle T4-এর জন্য বাস্তব পরিকল্পনা

**Constraints (Kaggle docs/community, verify on your account):** weekly GPU quota ≈ 30 h (varies), T4×2 = two 16 GB GPUs, sessions time out after roughly 9–12 h, ~4 CPU cores (dataloading is the usual bottleneck), /kaggle/working is saved on commit.

**Engineering rules:**
1. **Pre-resize once** (short side 256, JPEG q=95) and upload as your own private Kaggle dataset → removes the CPU bottleneck.
2. **Cache everything expensive once**: teacher features, teacher logits on un-augmented images, DINOv2 attention masks (one ~1–2 h job).
3. **Use both T4s in parallel** — run two seeds (or two models) at once as two processes, `CUDA_VISIBLE_DEVICES=0` and `=1`. Nearly doubles experiments per quota hour.
4. AMP fp16, `channels_last`, `torch.backends.cudnn.benchmark=True`, `num_workers=4`, `pin_memory=True`.
5. **Checkpoint every epoch** to /kaggle/working and support resume; use "Save & Run All (Commit)" so runs continue in the background.
6. Log with a CSV (or Weights & Biases free tier); fix seeds; save config YAML with every run.
7. **Measure your first epoch** and update the budget below — numbers below are estimates.

**Estimated budget (to be confirmed after the first run):**

| Block | Runs | Est. GPU-hours |
|---|---|---|
| Phase A–B (dedup embeddings, teacher features, masks; ViT-B + ViT-L features) | 1–3 jobs | 4–6 |
| Baselines: 4 students × 2 datasets × 3 seeds (PlantDoc runs are minutes) | 24 | 12–18 |
| KD variants (rows 3–5): 2 students × 2 datasets × 3 seeds × 3 variants | 36 | 20–30 |
| Ablations A1–A8 (1 seed each, 3 seeds for the key ones) | ~25 | 10–15 |
| Cross-dataset, XAI, calibration, INT8 (inference only) | — | 3–5 |
| **Total** | | **≈ 50–75 GPU-h ≈ 2–3 weeks of quota**, less with both T4s in parallel |

**বাংলা।** Kaggle-এ সপ্তাহে প্রায় ৩০ ঘণ্টা GPU, দুটি T4, আর একটানা ৯–১২ ঘণ্টার সেশন সীমা। তাই: ছবি আগে থেকে ছোট করে নিজের Kaggle dataset বানাও; বড় মডেলের ফিচার ও mask একবারই হিসাব করে সেভ করো; দুটি GPU-তে একসাথে দুটি seed চালাও; প্রতি epoch-এ checkpoint সেভ করো। পুরো পরীক্ষায় আনুমানিক ৫০–৭৫ GPU-ঘণ্টা লাগবে, অর্থাৎ প্রায় ২–৩ সপ্তাহের কোটা — প্রথম epoch চালিয়ে আসল সময় মেপে নিয়ে হিসাব আপডেট করবে।

---

## 7. Timeline (6 months) / সময়সূচি

| Month | Work | Output |
|---|---|---|
| 1 | Novelty re-check; download & pre-resize data; Phase A audit; shared-class mapping; cache teacher features & masks | Data-audit report (a paper section by itself) |
| 2 | Phase C baselines on PD & PW (3 seeds); teacher references | Baseline table |
| 3 | Implement WildLeaf-KD; tune on validation; main runs | Main result table |
| 4 | Ablations; cross-dataset; FieldPlant | Ablation + generalization tables |
| 5 | XAI (energy-in-box, pointing game), BRS, calibration, INT8 & latency | Figures + deployment table |
| 6 | Write paper & thesis; release code + splits + dedup lists on GitHub | Journal submission + thesis |

---

## 8. Expected contributions (how the paper will read) / প্রত্যাশিত অবদান

1. An audited, leakage-free benchmark of lightweight CNNs on PlantWild and PlantDoc, with released splits, duplicate lists and a shared-class map. (G1, G2, G7)
2. **WildLeaf-KD**, a dual-role foundation-teacher framework combining distillation with background-counterfactual consistency. (G5)
3. The first ground-truth-based measurement of where lightweight models look on PlantDoc, and a Background Reliance Score showing whether distillation transfers "where to look". (G4)
4. Cross-dataset (PD ↔ PW → FieldPlant) evidence and INT8 CPU/mobile deployment numbers. (G3, G6)

Even if KD gains turn out small, contributions 1, 3 and 4 still stand — this protects the thesis from a "negative result" risk.

**বাংলা।** চারটি অবদান: (১) leakage-মুক্ত benchmark ও উন্মুক্ত split; (২) WildLeaf-KD পদ্ধতি; (৩) Grad-CAM-কে আসল box দিয়ে মাপা এবং ব্যাকগ্রাউন্ড-নির্ভরতা মাপা; (৪) cross-dataset ও মোবাইলে চালানোর প্রমাণ। KD থেকে উন্নতি কম হলেও বাকি তিনটি অবদান টিকে থাকবে — তাই থিসিস ঝুঁকিমুক্ত।

---

## 9. Target journals / লক্ষ্য জার্নাল

Suggested order (check current scope, APC and indexing yourself before submitting):
- *Computers and Electronics in Agriculture* (Elsevier) — ambitious, strict on novelty and statistics
- *Smart Agricultural Technology* (Elsevier) — good fit for deployment-oriented work
- *Plant Phenomics* — likes rigorous benchmarks
- *Frontiers in Plant Science* (Technical Advances section) — where MVPDR-follow-ups and Salman et al. were published
- *Artificial Intelligence in Agriculture*, *Ecological Informatics* — solid alternatives

What these reviewers will look for: released code and splits, multiple seeds with std, macro-F1, fair comparison statements, ablations, and deployment numbers — all built into this plan.

---

## 10. Risks & mitigation / ঝুঁকি ও সমাধান

| Risk | Mitigation |
|---|---|
| KD gives little gain (seen in some prior work) | Keep strong CE baseline; report honestly; contributions 1, 3, 4 still stand; BCC may help robustness even if accuracy gain is small |
| DINOv2 masks poor on cluttered images | Validate on PlantDoc boxes; use soft masks; mix with blur-background variant; ablation A4 |
| Kaggle quota/session limits | Caching, two-GPU parallel seeds, checkpoint/resume, pre-resized data |
| PlantWild download/label format issues | Use official repo split files; document any missing/corrupt files |
| Shared-class mapping disputes | Supervisor verification; publish the mapping table |
| Someone publishes the same idea | Week-1 and month-4 novelty checks; the audit + XAI parts remain distinct |

---

## 11. Immediate next step / পরবর্তী ধাপ

**Step 3 (next):** Phase A on Kaggle — download PlantDoc (official GitHub split) and PlantWild (official split), pre-resize, run the duplicate audit, and draft the PlantDoc ↔ PlantWild shared-class table. I can write the Kaggle notebook code for this step.

**ধাপ ৩:** Kaggle-এ Phase A — দুটি ডেটাসেট অফিসিয়াল split সহ নামানো, ছবি ছোট করা, ডুপ্লিকেট অডিট, এবং মিল থাকা ক্লাসের তালিকা তৈরি। এই ধাপের Kaggle নোটবুক কোড আমি লিখে দিতে পারি।

---

### Key references for this methodology
- Wei et al. (2024), PlantWild/MVPDR, ACM MM — https://doi.org/10.1145/3664647.3680599
- Singh et al. (2020), PlantDoc, CoDS-COMAD — https://doi.org/10.1145/3371158.3371196
- El Karch et al. (2026), frozen foundation-model fusion, Front. Plant Sci. — https://doi.org/10.3389/fpls.2026.1860665
- AgriKD (2026), arXiv:2605.01355 — ViT→MobileNetV2 distillation
- TinyCNN (2026), arXiv:2609.20290 — off-leaf ratio / shortcut diagnosis
- DC-FEN (2026), Agriculture — https://doi.org/10.3390/agriculture16161790 — duplicate-audited KD evaluation
- Duhan et al. (2026), Discover IoT — https://doi.org/10.1007/s43926-026-00310-0
- Kumar et al. (2025), arXiv:2508.10817 — merged-dataset lightweight benchmark
- Moupojou et al. (2023), FieldPlant, IEEE Access — https://doi.org/10.1109/ACCESS.2023.3263042
- Kaggle GPU quota documentation — https://www.kaggle.com/docs/efficient-gpu-usage
