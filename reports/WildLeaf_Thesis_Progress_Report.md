# WildLeaf-KD — Thesis Progress Report (Notebooks 1–7)

**Project:** Lightweight plant-disease classification on in-the-wild images (PlantDoc + PlantWild) using knowledge distillation from foundation models
**Report date:** 26 September 2026
**Platform:** Kaggle (2 × T4 GPU)

> **Note about numbers / সংখ্যা সম্পর্কে নোট**
> All Step 7 and Step 8 numbers in this report are the **final** ones, from the saved rerun (`wildleaf-step7` dataset) and the Step 8 notebook. For the thesis, still copy exact values from the CSV tables (`figures/table_*.csv`), because they have more decimal places than this report.
>
> এই রিপোর্টের Step 7 আর Step 8-এর সব সংখ্যা **চূড়ান্ত**, save করা নতুন run (`wildleaf-step7`) আর Step 8 notebook থেকে নেওয়া। Thesis লেখার সময় সঠিক মান CSV table (`figures/table_*.csv`) থেকেই নেবে।

---

## 0. The big picture / পুরো কাজের সারসংক্ষেপ

**English.**
Farmers photograph sick plants in real fields: messy backgrounds, bad light, many leaves in one picture. Two public datasets contain exactly this kind of photo: **PlantDoc** (2,576 images, 27 classes) and **PlantWild** (18,542 images, 89 classes). The goal of the thesis is a **small, fast model** (about 4 million parameters, runs on a phone or cheap CPU) that recognises the disease accurately on such photos.

The work so far has three parts:

1. **We checked the datasets first (audit).** We found that the official test sets are "leaky": about **11% of test images have a near-copy (a "twin") in the training set**, and many images carry **conflicting labels**. We built clean, leakage-free versions of both datasets.
2. **We built a strong "teacher".** Two large pretrained vision models (DINOv2 and CLIP) were used as frozen feature extractors, with a simple linear classifier on top. This teacher is accurate, but far too big for a phone.
3. **We trained small "students" and taught them with the teacher (knowledge distillation, KD).** KD improved every small model on both datasets, made their confidence much more trustworthy, and let a 4M-parameter model beat a 24M-parameter ResNet-50.

**বাংলা।**
কৃষকরা মাঠে অসুস্থ গাছের ছবি তোলেন: এলোমেলো পটভূমি, খারাপ আলো, এক ছবিতে অনেক পাতা। এই ধরনের ছবি আছে দুটো public dataset-এ: **PlantDoc** (২,৫৭৬টি ছবি, ২৭টি class) আর **PlantWild** (১৮,৫৪২টি ছবি, ৮৯টি class)। আমাদের লক্ষ্য একটা **ছোট, দ্রুত model** (প্রায় ৪০ লক্ষ parameter, মোবাইল বা সস্তা CPU-তে চলে), যা এই ধরনের ছবিতে সঠিকভাবে রোগ চিনতে পারে।

এখন পর্যন্ত কাজ তিন ভাগে:

1. **আগে dataset পরীক্ষা (audit) করেছি।** দেখা গেছে official test set-এ "leakage" আছে: **প্রায় ১১% test ছবির প্রায়-হুবহু কপি ("যমজ") train set-এ আছে**, আর অনেক ছবির label পরস্পরবিরোধী। দুটো dataset-এরই পরিষ্কার, leakage-মুক্ত সংস্করণ বানিয়েছি।
2. **একটা শক্তিশালী "teacher" বানিয়েছি।** দুটো বড় pretrained model (DINOv2 আর CLIP) থেকে feature নিয়ে তার ওপর একটা সরল linear classifier। Teacher খুব ভালো, কিন্তু মোবাইলের জন্য অনেক বড়।
3. **ছোট "student" model train করে teacher দিয়ে শিখিয়েছি (knowledge distillation, KD)।** KD দুটো dataset-এই প্রতিটা ছোট model-এর ফল ভালো করেছে, তাদের আত্মবিশ্বাসকে অনেক বেশি নির্ভরযোগ্য করেছে, আর ৪০ লক্ষ parameter-এর model দিয়ে ২ কোটি ৪০ লক্ষ parameter-এর ResNet-50-কে হারিয়ে দিয়েছে।

---

## Glossary — easy words / সহজ ভাষায় শব্দকোষ

| Term | English (easy) | বাংলা (সহজ) |
|---|---|---|
| **Macro-F1** | Our main score (0–1). It averages the score of every class equally, so rare classes count as much as common ones. Higher is better. | আমাদের মূল score (০–১)। প্রতিটা class-কে সমান গুরুত্ব দেয়, তাই কম ছবির class-ও সমান গোনা হয়। যত বেশি, তত ভালো। |
| **pp (percentage points)** | Difference between two percentages. 0.66 → 0.71 is +5 pp. | দুটো শতাংশের পার্থক্য। ০.৬৬ থেকে ০.৭১ মানে +৫ pp। |
| **Twin / near-duplicate** | Two images that are almost the same photo (resized, cropped, re-uploaded). | দুটো ছবি যা প্রায় একই ছবি (ছোট-বড় করা, কাটা, আবার upload করা)। |
| **Leakage** | A test image whose twin is in training. The model "has seen the answer", so the score is inflated. | Test ছবির যমজ train-এ থাকা। Model আগেই "উত্তর দেখে ফেলেছে", তাই score কৃত্রিমভাবে বেশি আসে। |
| **Label conflict** | Twin images that carry different class labels, so at least one label is wrong. | যমজ ছবির label আলাদা, মানে অন্তত একটা label ভুল। |
| **P1 / P2 / P3** | P1 = official split (leaky). P2 = our clean split. P3 = clean 5-fold cross-validation (PlantDoc main result). | P1 = official split (leakage আছে)। P2 = আমাদের পরিষ্কার split। P3 = পরিষ্কার ৫-fold (PlantDoc-এর মূল ফল)। |
| **Fold / Seed** | Fold = one of 5 different train/test splits. Seed = a different random start for the same split. Both show how stable a result is. | Fold = ৫টা আলাদা train/test ভাগের একটা। Seed = একই ভাগে আলাদা random শুরু। দুটোই দেখায় ফল কতটা স্থিতিশীল। |
| **Teacher / Student** | Teacher = big accurate model. Student = small fast model that learns from the teacher. | Teacher = বড়, নির্ভুল model। Student = ছোট, দ্রুত model, যে teacher থেকে শেখে। |
| **Knowledge distillation (KD)** | The student learns from the true label **and** from the teacher's "soft" probabilities over all classes. | Student সঠিক label **এবং** teacher-এর সব class-এর সম্ভাবনা (soft label) দুটো থেকেই শেখে। |
| **Linear probe** | A simple classifier trained on top of frozen features. | জমাট (frozen) feature-এর ওপর একটা সরল classifier। |
| **AUROC** | How well a score separates two groups (0.5 = random, 1.0 = perfect). | একটা score দুটো দলকে কতটা ভালো আলাদা করে (০.৫ = এলোমেলো, ১.০ = নিখুঁত)। |
| **ECE (calibration error)** | Gap between how confident a model is and how often it is right. Lower is better. | Model কতটা আত্মবিশ্বাসী আর আসলে কতবার ঠিক, তার ফারাক। যত কম, তত ভালো। |
| **GMACs / latency** | Amount of computation / time for one prediction. Lower means faster and cheaper. | একটা prediction-এর জন্য গণনা / সময়। কম মানে দ্রুত ও সস্তা। |

---

## Notebook 1 — PlantDoc audit (Steps 3.1–3.6)

### What I did / কী করেছি

**English.**
- Loaded all 2,576 PlantDoc images (27 classes; official split 2,340 train / 236 test) and resized them into a processed copy (`proc/`).
- Searched for **near-duplicate images** in three layers: image hashing, DINOv2 embedding similarity, and **ORB keypoint matching** to confirm candidate pairs. Every group of twins got one `dup_group` id.
- Measured **leakage**: how many official test images have a twin in training.
- Found **label conflicts**: twin groups whose images have different labels.
- Built three evaluation protocols:
  - **P1** — the official split (kept only to show how leaky it is),
  - **P2** — a clean train / val / test split,
  - **P3** — clean **5-fold cross-validation**, grouped by `dup_group` so twins never cross between train and test. **P3 is the main PlantDoc result** because the test set is small.
- Checked how sensitive the leak rate is to the duplicate-detection threshold.

**বাংলা।**
- PlantDoc-এর সব ২,৫৭৬টি ছবি (২৭ class; official ভাগ ২,৩৪০ train / ২৩৬ test) নিয়ে একটা প্রক্রিয়াজাত কপি (`proc/`) বানিয়েছি।
- তিন স্তরে **প্রায়-একই ছবি** খুঁজেছি: image hashing, DINOv2 embedding-এর মিল, আর নিশ্চিত করতে **ORB keypoint matching**। প্রতিটা যমজ-দলকে একটা `dup_group` নম্বর দিয়েছি।
- **Leakage** মেপেছি: কতগুলো official test ছবির যমজ train-এ আছে।
- **Label conflict** খুঁজেছি: যে যমজ-দলে ছবিগুলোর label আলাদা।
- তিনটা মূল্যায়ন পদ্ধতি বানিয়েছি: **P1** (official, শুধু leakage দেখাতে), **P2** (পরিষ্কার ভাগ), **P3** (পরিষ্কার ৫-fold, যমজ ছবি কখনো train আর test-এ ভাগ হয় না)। Test set ছোট বলে **P3-ই PlantDoc-এর মূল ফল**।
- Duplicate ধরার threshold বদলালে leak rate কতটা বদলায়, সেটাও দেখেছি।

### Results / ফলাফল

| Item | Value |
|---|---|
| Official test images with a twin in train | **27 / 236 = 11.44%** |
| Range across detection thresholds | 7.2% (strict only) – 13.56% (ORB ≥ 16) |
| Label-conflict images | **94** |
| Clean train+val / clean test | 2,184 / 220 |
| P3 folds | ~1,648 train / 275 val / ~481 test each |

### Is it good? / ফলাফল কেমন?

**English.** Yes — this is a solid, useful finding. For **every** reasonable threshold, more than 1 in 10 official test images is a near-copy of a training image. That means any paper reporting results on the official PlantDoc split is partly measuring memory, not recognition. Our clean protocols remove this problem.

**বাংলা।** হ্যাঁ, এটা একটা শক্ত ও কাজের ফলাফল। যেকোনো যুক্তিসঙ্গত threshold-এ official test-এর ১০টার মধ্যে ১টার বেশি ছবি train ছবির প্রায়-কপি। তাই official PlantDoc split-এ যেসব paper ফল দেখায়, তারা আংশিকভাবে "মুখস্থ" মাপছে, "চেনা" নয়। আমাদের পরিষ্কার protocol এই সমস্যা দূর করে।

---

## Notebook 2 — PlantWild audit + cross-dataset protocol (Steps 4.1–4.9)

### What I did / কী করেছি

**English.**
- Same audit pipeline on **PlantWild v1** (18,542 images, 89 classes; official 13,045 train / 1,820 val / 3,677 test).
- Matched our images to the **expert-corrected PlantWild v2** to see what the experts kept, removed or relabelled.
- Compared the two datasets with each other: found **cross-dataset twins** (same photo in both datasets) and mapped the **27 shared classes**.
- Built **cross-dataset protocols** (train on one dataset, test on the other) and flagged four "organ-shift" classes where the two datasets photograph different plant parts.
- Produced the combined audit summary table.

**বাংলা।**
- **PlantWild v1**-এ (১৮,৫৪২টি ছবি, ৮৯ class; official ১৩,০৪৫ train / ১,৮২০ val / ৩,৬৭৭ test) একই audit চালিয়েছি।
- Expert-সংশোধিত **PlantWild v2**-এর সাথে মিলিয়ে দেখেছি experts কোন ছবি রেখেছেন, বাদ দিয়েছেন বা label বদলেছেন।
- দুটো dataset পরস্পরের সাথে তুলনা করেছি: দুটোতেই থাকা একই ছবি (**cross-dataset twin**) খুঁজেছি আর **২৭টা common class** মিলিয়েছি।
- **Cross-dataset protocol** বানিয়েছি (এক dataset-এ train, অন্যটায় test), আর ৪টা "organ-shift" class চিহ্নিত করেছি, যেখানে দুই dataset গাছের আলাদা অংশের ছবি তোলে।

### Results / ফলাফল

| Item | PlantDoc | PlantWild v1 |
|---|---|---|
| Images / classes | 2,576 / 27 | 18,542 / 89 |
| Official test with a twin in train | **11.44%** | **11.26%** |
| Leak range across thresholds | 7.2–13.56% | 7.15–11.26% |
| Label-conflict images | 94 | **1,130** |
| Clean train / clean test | 2,184 / 220 | 11,508 / 3,438 |
| Images that also appear in the other dataset | 20.0% | 3.1% |

- **Expert v1 → v2** (58 shared disease classes): kept 6,298 · removed 3,938 · relabelled 349 · not judged 7,957.
- **Expert-verified test subset:** 1,232 images, 58 classes (1,151 of them in our clean test).
- **Cross-dataset:** PW→PD trains on 3,727 / val 527 and tests on PlantDoc clean 220 (twin-free 173). PD→PW trains on 1,798 / val 306 and tests on 1,111 twin-free PlantWild images.
- **Organ-shift classes:** Apple Scab Leaf, Apple rust leaf, Tomato leaf late blight, grape leaf black rot.

### Is it good? / ফলাফল কেমন?

**English.** Yes. PlantWild, a much larger and newer dataset, has **the same ~11% leakage problem**, plus over a thousand conflicting labels. **One in five PlantDoc images also appears in PlantWild**, so any "cross-dataset" test that ignores this would also be leaky; ours excludes these twins. The expert v2 data gives us something rare: a ground truth to check our automatic noise detection against (used in Notebook 3).

**বাংলা।** হ্যাঁ। অনেক বড় আর নতুন dataset PlantWild-এও **একই ~১১% leakage সমস্যা**, সাথে হাজারেরও বেশি পরস্পরবিরোধী label। **PlantDoc-এর প্রতি ৫টা ছবির ১টা PlantWild-এও আছে**, তাই এটা না মেনে কেউ cross-dataset পরীক্ষা করলে সেটাও leaky হবে; আমরা এই যমজগুলো বাদ দিয়েছি। Expert v2 data আমাদের একটা বিরল সুযোগ দেয়: আমাদের স্বয়ংক্রিয় ভুল-label শনাক্তকরণ মিলিয়ে দেখার মতো "আসল উত্তর" (Notebook 3-এ ব্যবহার করা হয়েছে)।

---

## Notebook 3 — Foundation-model teachers (Step 5)

### What I did / কী করেছি

**English.**
1. **Extracted features once** for all ~21k images with two frozen foundation models: **DINOv2-Base** (CLS token + mean of patch tokens) and **CLIP ViT-B/16** (LAION-2B weights). Saved as `.npy` files so the big models never need to run again.
2. **Trained linear-probe teachers** on four feature sets (DINOv2-CLS, DINOv2-CLS+mean, CLIP, DINOv2+CLIP) for every protocol (PD P1, P2, P3 × 5 folds; PW P1, P2). The weight decay was chosen **only on validation** — the test set was never used for any choice.
3. **Measured leakage inflation directly:** accuracy on official-test images *with* a twin in train vs *without*.
4. **Out-of-fold teacher + label-noise detection:** every image got a prediction from a teacher that never saw it; then cleanlab's label-quality score was checked against the experts' decisions.
5. **Made all thesis figures** for this step and for the dataset chapter (Notebooks 1–2).

**বাংলা।**
1. ~২১ হাজার ছবির জন্য দুটো frozen foundation model দিয়ে **একবারই feature বের করেছি**: **DINOv2-Base** আর **CLIP ViT-B/16**। `.npy` ফাইলে রেখেছি, যাতে বড় model আর চালাতে না হয়।
2. চারটা feature set আর প্রতিটা protocol-এর জন্য **linear-probe teacher** train করেছি। Weight decay বাছা হয়েছে **শুধু validation দেখে**; কোনো সিদ্ধান্তে test set ব্যবহার হয়নি।
3. **Leakage কতটা score ফোলায়** সরাসরি মেপেছি: যমজ-থাকা আর যমজ-ছাড়া test ছবির accuracy আলাদা করে।
4. **Out-of-fold teacher দিয়ে ভুল label খোঁজা:** প্রতিটা ছবির prediction এসেছে এমন teacher থেকে, যে ছবিটা কখনো দেখেনি; তারপর experts-এর সিদ্ধান্তের সাথে মিলিয়ে দেখেছি।
5. এই ধাপ আর dataset chapter-এর (Notebook 1–2) **সব thesis figure** বানিয়েছি।

### Results / ফলাফল

**Teacher accuracy (test macro-F1, DINOv2+CLIP):**

| Protocol | PlantDoc | PlantWild |
|---|---|---|
| P1 (official, leaky + noisy) | 0.749 | 0.752 |
| P2 (clean) | 0.772 | 0.770 |
| P3 (clean, 5-fold) | **0.775 ± 0.017** | — |

**Leakage inflation (label-conflict images excluded):**

| | Twin-free test | Twin-in-train test | Gap |
|---|---|---|---|
| PlantWild teacher (n = 241 twins) | 0.797 | 0.959 | **+16.2 pp** |
| PlantWild 1-NN "memoriser" | 0.724 | 0.992 | **+26.8 pp** |
| PlantDoc teacher (n = 11 twins) | 0.794 | 1.000 | +20.6 pp |

**Label-noise detection (checked against the experts):**

| Comparison | AUROC |
|---|---|
| Expert **relabelled** vs kept | **0.833** (AP 0.299 vs base rate 0.053) |
| Expert removed vs kept | 0.654 |
| Our label-conflict flag, PlantWild / PlantDoc | 0.757 / 0.769 |

- Teacher confidence in the given label: **0.048** for images the experts later relabelled vs **0.987** for images they kept.
- **30.9%** of the images we flagged as conflicts were later relabelled by experts, vs **0.5%** of the rest (62× more).

### Is it good? / ফলাফল কেমন?

**English.** Very good, and it gives the thesis three strong findings:
1. **Leakage is real and large.** Twinned test images are "solved" 96–100% of the time, 16–27 pp above twin-free ones.
2. **Label noise hides it.** Conflicting test labels pull scores *down*, so on the official split the two effects partly cancel and the headline number hides both problems. That is why cleaning *raised* the teacher's score (0.749 → 0.772).
3. **A frozen foundation teacher finds label errors automatically.** Without any human help, it ranks the experts' later corrections far above the images they kept (AUROC 0.833). "Removed" is weaker (0.654) because experts also removed images for quality reasons, not only for wrong labels.

Caveat: PlantDoc has only 11 twin test images, so its gap is noisy; report it with a confidence interval and lead with PlantWild.

**বাংলা।** খুব ভালো, আর thesis-এর জন্য তিনটা শক্ত ফলাফল দেয়:
1. **Leakage সত্যি এবং বড়।** যমজ-থাকা test ছবি ৯৬–১০০% সময় ঠিক হয়, যমজ-ছাড়া ছবির চেয়ে ১৬–২৭ pp বেশি।
2. **ভুল label এটা ঢেকে রাখে।** পরস্পরবিরোধী label score *কমায়*, তাই official split-এ দুটো প্রভাব একে অপরকে কাটাকাটি করে, আর মূল সংখ্যায় কোনো সমস্যাই ধরা পড়ে না। এজন্যই পরিষ্কার করার পর teacher-এর score *বেড়েছে* (০.৭৪৯ → ০.৭৭২)।
3. **Frozen foundation teacher নিজে থেকেই ভুল label খুঁজে পায়।** কোনো মানুষের সাহায্য ছাড়াই experts যেগুলো পরে ঠিক করেছেন, সেগুলোকে আলাদা করতে পারে (AUROC ০.৮৩৩)।

সতর্কতা: PlantDoc-এ মাত্র ১১টা যমজ test ছবি, তাই সেখানের সংখ্যা অস্থির; confidence interval দিয়ে লিখবে আর PlantWild-কে মূল প্রমাণ হিসেবে দেখাবে।

---

## Notebook 4 — Lightweight student baselines (Step 6)

### What I did / কী করেছি

**English.**
- Wrote one reusable training script (`train_student.py`): full fine-tuning from ImageNet weights with AdamW, warm-up + cosine schedule, label smoothing 0.1 and data augmentation; the best epoch is chosen on **validation macro-F1**, and the test set is scored only once at the end.
- **Made training ~5× faster:** first cached every image as a 256×256 array, then moved all augmentation onto the GPU. A run went from ~11 min to ~1.5–4 min.
- **Chose the learning rate on validation** (1e-3), then used one uniform recipe for all models.
- Trained **4 students**: MobileNetV4-Conv-S (2.5M), MobileNetV3-L (4.2M), EfficientNet-B0 (4.0M) and ResNet-50 (23.6M, heavy reference), on PlantDoc P3 (5 folds) and PlantWild P2, plus P1 runs for the leakage comparison.
- Measured **efficiency** (parameters, GMACs, CPU and GPU latency) and made the Step 6 figures.

**বাংলা।**
- একটা reusable training script (`train_student.py`) লিখেছি: ImageNet weight থেকে পুরো fine-tune, AdamW, warm-up + cosine schedule, label smoothing ০.১ আর augmentation। সেরা epoch বাছা হয় **validation macro-F1** দেখে, test শুধু শেষে একবার মাপা হয়।
- **Training প্রায় ৫ গুণ দ্রুত করেছি:** আগে সব ছবি 256×256 array হিসেবে cache করেছি, তারপর সব augmentation GPU-তে নিয়ে গেছি। একটা run ~১১ মিনিট থেকে ~১.৫–৪ মিনিটে নেমেছে।
- **Validation দেখে learning rate বেছেছি** (1e-3), তারপর সব model-এ একই পদ্ধতি ব্যবহার করেছি।
- **৪টা student** train করেছি PlantDoc P3 (৫ fold) আর PlantWild P2-তে, সাথে leakage তুলনার জন্য P1-ও।
- **দক্ষতা** মেপেছি (parameter, GMACs, CPU/GPU latency) আর Step 6-এর figure বানিয়েছি।

### Results / ফলাফল

**Baselines (test macro-F1):**

| Model | Params | PlantDoc P3 | PlantWild P2 | CPU latency |
|---|---|---|---|---|
| MobileNetV4-S | 2.5 M | 0.520 ± 0.015 | 0.585 | 18 ms |
| MobileNetV3-L | 4.2 M | 0.606 ± 0.028 | 0.660 | 28 ms |
| EfficientNet-B0 | 4.0 M | 0.622 ± 0.032 | 0.665 | 35 ms |
| ResNet-50 (reference) | 23.6 M | 0.663 ± 0.026 | 0.669 | 83 ms |
| *Teacher (DINOv2+CLIP)* | *~170 M* | *0.775* | *0.770* | — |

**Leakage hits small models harder:**

| | Twin-free | Twin-in-train | Gap |
|---|---|---|---|
| PlantWild teacher | 0.797 | 0.959 | +16.2 pp |
| PlantWild EfficientNet-B0 | 0.710 | 0.971 | **+26.1 pp** |
| PlantWild MobileNetV3-L | 0.692 | 0.971 | **+27.9 pp** |

- Removing only the label-conflict test images raises every model's macro-F1 by **+2.3 to +2.9 pp**.

### Is it good? / ফলাফল কেমন?

**English.** These are honest, reasonable baselines, and they set up the main story well:
- All small models are **10–25 pp below the teacher**. Even ResNet-50, six times bigger, is 11 pp behind on PlantDoc. So "just use a bigger CNN" does not solve in-the-wild images — this motivates KD.
- **New finding:** leaky benchmarks **flatter small models the most** (+26–28 pp vs +16 pp for the teacher). Official results therefore overstate exactly the kind of model this thesis is about.
- MobileNetV3-L is the best speed/accuracy balance (0.215 GMACs, 28 ms on CPU).
- Caveat: MobileNetV4-S is weak partly because one uniform training recipe may not suit it; state this honestly.

**বাংলা।** এগুলো সৎ ও যুক্তিসঙ্গত baseline, আর মূল গল্পটা সুন্দরভাবে দাঁড় করায়:
- সব ছোট model teacher-এর চেয়ে **১০–২৫ pp পিছিয়ে**। ছয় গুণ বড় ResNet-50-ও PlantDoc-এ ১১ pp পিছিয়ে। অর্থাৎ শুধু বড় CNN নিলেই সমস্যা মেটে না, এজন্যই KD দরকার।
- **নতুন ফলাফল:** leaky benchmark **ছোট model-কে সবচেয়ে বেশি ফুলিয়ে দেখায়** (teacher-এর +১৬ pp-এর বিপরীতে +২৬–২৮ pp)। তাই official ফল ঠিক আমাদের ধরনের model-কেই সবচেয়ে বেশি বাড়িয়ে দেখায়।
- MobileNetV3-L গতি আর নির্ভুলতার সবচেয়ে ভালো ভারসাম্য দেয় (০.২১৫ GMACs, CPU-তে ২৮ ms)।
- সতর্কতা: MobileNetV4-S দুর্বল হওয়ার একটা কারণ হতে পারে যে একই training পদ্ধতি তার জন্য উপযুক্ত নয়; এটা সৎভাবে লিখবে।

---

## Notebook 5 — WildLeaf-KD: distillation, ablations, analysis (Step 7)

### What I did / কী করেছি

**English.**
1. **Leak-proof teacher targets (7.1).** The Notebook 3 out-of-fold teacher had seen each protocol's *test* images during training, so distilling it would have leaked test labels into the students. We therefore re-made the teacher's soft labels **inside each protocol's training split only** (nested cross-fitting, grouped by `dup_group`), and fitted a temperature on validation to correct over-confidence (T ≈ 1.1 PlantDoc, ≈ 1.7 PlantWild).
2. **KD training + hyperparameter choice (7.2).** Loss = (1 − α)·CE + α·τ²·KL(teacher ‖ student). Tried α ∈ {0.5, 0.9} × τ ∈ {2, 4}; **α = 0.5, τ = 4** was best on validation on *both* datasets.
3. **Main KD grid (7.3).** Three students × 5 PlantDoc folds, and 3 seeds on PlantWild. Every KD run is compared with a baseline run on the **same fold or seed** (paired comparison), with t-tests and Wilcoxon tests.
4. **Background-counterfactual consistency (BCC) — tested and dropped (7.4).** BCC needed leaf masks to swap backgrounds. Two unsupervised masking methods (DINOv2 PCA and DINOv2 attention) both failed: PCA masked out the *lesions* and all non-leaf organs (fruit, stems, panicles), and attention was too scattered. We report this as a negative result and removed BCC from the method.
5. **Ablations (7.5):** DINOv2-only teacher, CLIP-only teacher, in-sample (not cross-fitted) teacher, no temperature calibration, α = 1 (teacher only, no labels).
6. **Analysis (7.6):** expert-verified test subset, per-class gains, calibration (ECE).
7. **Cross-dataset test (7.7)** — PlantWild → PlantDoc and PlantDoc → PlantWild; run in Notebook 6, see below.

**বাংলা।**
1. **Leak-proof teacher target (7.1)।** Notebook 3-এর teacher প্রতিটা protocol-এর *test* ছবিও দেখেছিল, তাই সরাসরি ব্যবহার করলে test-এর তথ্য student-এ চলে যেত। তাই প্রতিটা protocol-এর **শুধু train অংশের ভেতরে** teacher-এর soft label নতুন করে বানিয়েছি (nested cross-fitting), আর validation দিয়ে temperature ঠিক করেছি, যাতে teacher অতিরিক্ত আত্মবিশ্বাসী না হয়।
2. **KD training ও hyperparameter বাছাই (7.2)।** চারটা setting চেষ্টা করেছি; **α = 0.5, τ = 4** দুটো dataset-এই validation-এ সেরা।
3. **মূল KD পরীক্ষা (7.3)।** ৩টা student × PlantDoc-এর ৫ fold, আর PlantWild-এ ৩টা seed। প্রতিটা KD run-কে **একই fold/seed-এর** baseline-এর সাথে জোড়ায় তুলনা করেছি, সাথে statistical test।
4. **BCC পরীক্ষা করে বাদ দিয়েছি (7.4)।** BCC-র জন্য পাতার mask লাগত। দুটো পদ্ধতিই ব্যর্থ: PCA রোগের *দাগ* আর ফল-কাণ্ডের মতো অংশ বাদ দিয়ে দিচ্ছিল, আর attention-এর mask ছিল এলোমেলো। এটা negative result হিসেবে রিপোর্ট করব।
5. **Ablation (7.5):** শুধু DINOv2, শুধু CLIP, cross-fit ছাড়া teacher, calibration ছাড়া, আর label ছাড়া শুধু teacher (α = 1)।
6. **বিশ্লেষণ (7.6):** expert-যাচাই করা test, প্রতিটা class-এর উন্নতি, calibration।
7. **Cross-dataset পরীক্ষা (7.7)** — Notebook 6-এ চালানো হয়েছে, নিচে দেখো।

### Results / ফলাফল

**Main result — baseline vs KD (test macro-F1, paired):**

| Dataset / model | Baseline | KD | Gain (95% CI) | Wins | t-test p |
|---|---|---|---|---|---|
| PlantDoc P3 · MobileNetV3-L | 0.613 | 0.670 | **+5.6 pp** (1.3 to 10.0) | 5/5 | 0.022 |
| PlantDoc P3 · EfficientNet-B0 | 0.623 | 0.662 | **+3.9 pp** (0.6 to 7.1) | 4/5 | 0.030 |
| PlantDoc P3 · MobileNetV4-S | 0.517 | 0.609 | **+9.2 pp** (6.2 to 12.2) | 5/5 | 0.001 |
| PlantWild P2 · MobileNetV3-L | 0.661 | 0.708 | **+4.7 pp** (2.6 to 6.7) | 3/3 | 0.010 |
| PlantWild P2 · EfficientNet-B0 | 0.666 | 0.715 | **+4.9 pp** (2.4 to 7.4) | 3/3 | 0.014 |
| PlantWild P2 · MobileNetV4-S | 0.576 | 0.656 | +8.0 pp | 1/1 | — |

- **KD wins 21 of 22 paired comparisons**, with a mean gain of about **+5.9 pp**. Every per-model t-test has p < 0.05.
- On PlantWild, 4M-parameter students with KD (0.708–0.715) **beat the 24M ResNet-50 baseline (0.669)**.
- Note for the thesis: with 5 folds, the smallest possible two-sided Wilcoxon p-value is 0.0625, so lead with t-tests, 95% CIs and the pooled test.

**Ablations (PlantWild P2, MobileNetV3-L, 2 seeds):**

| Variant | Test macro-F1 | vs full KD |
|---|---|---|
| Baseline (no KD) | 0.658 | −5.1 pp |
| CLIP-only teacher | 0.679 | −3.0 pp |
| α = 1.0 (teacher only) | 0.685 | −2.4 pp |
| DINOv2-only teacher | 0.694 | −1.6 pp |
| No calibration (T = 1) | 0.701 | −0.9 pp |
| In-sample teacher (no cross-fit) | 0.703 | −0.6 pp |
| **Full WildLeaf-KD** | **0.709** | — |

**Expert-verified labels, calibration and per-class effects:**

| | Baseline | KD |
|---|---|---|
| Accuracy on 1,151 expert-verified test images (EffNet-B0 / MBV3) | 0.754 / 0.748 | **0.789 / 0.785** |
| Calibration error, ECE (PlantWild, EffNet-B0 / MBV3) | 0.133 / 0.146 | **0.040 / 0.048** |
| Calibration error, ECE (PlantDoc, EffNet-B0 / MBV3) | 0.168 / 0.159 | **0.054 / 0.055** |

- Classes improved: **79 / 89** on PlantWild, **25 / 27** on PlantDoc. Healthy and disease classes gain about equally (+4.5 to +5.4 pp).
- Rare classes tend to gain the most (first run: +8.3 pp for the smallest third of PlantWild classes vs +3.6 pp for the largest; check `table_kd_per_class.csv` for the final split).
- Weakest area: **corn leaf diseases** (gray leaf spot, northern leaf blight, rust) and **rice blast**, which improve least or drop slightly (−2 to −4 pp). These diseases look very alike, and the teacher likely confuses them too.

### Is it good? / ফলাফল কেমন?

**English.** Yes — this is the thesis's main contribution, and it is strong and honest:
- **Consistent:** KD helps every student on both datasets, across folds and seeds, and the gain is statistically significant.
- **Real, not a label-noise trick:** the gain holds on the experts' verified labels.
- **Practically useful:** KD students are about **3× better calibrated**, so their confidence can be trusted (e.g., an app can say "not sure — ask an expert").
- **Fair about the method:** the ablations show that most of the gain comes from the teacher's "dark knowledge" (which wrong classes look similar), not mainly from correcting noisy labels. Cross-fitting gives only a small gain (+0.6 pp), so we present it as a **leakage safeguard**, not as a big accuracy booster. BCC is reported as a negative result.

**বাংলা।** হ্যাঁ, এটাই thesis-এর মূল অবদান, আর ফলাফল শক্ত ও সৎ:
- **ধারাবাহিক:** KD দুটো dataset-এই প্রতিটা student-এর উন্নতি করেছে, প্রতিটা fold ও seed-এ, আর উন্নতি statistically significant।
- **আসল উন্নতি:** experts-এর যাচাই করা label-এও উন্নতি টিকে থাকে, তাই এটা ভুল-label-এর কারসাজি নয়।
- **বাস্তবে কাজের:** KD student-এর calibration প্রায় **৩ গুণ ভালো**, তাই তাদের আত্মবিশ্বাস বিশ্বাস করা যায় (যেমন app বলতে পারে "নিশ্চিত নই, expert-কে দেখাও")।
- **পদ্ধতি নিয়ে সৎ:** ablation দেখায় উন্নতির বড় অংশ আসে teacher-এর "dark knowledge" থেকে (কোন ভুল class-গুলো দেখতে কাছাকাছি), ভুল label ঠিক করা থেকে নয়। Cross-fitting সামান্য সাহায্য করে (+০.৬ pp), তাই এটাকে **leakage প্রতিরোধের সুরক্ষা** হিসেবে উপস্থাপন করব, বড় উন্নতির উৎস হিসেবে নয়। BCC negative result হিসেবে থাকবে।

---

## Notebook 6 — Cross-dataset generalisation + INT8 deployment (Steps 7.7, 8)

### What I did / কী করেছি

**English.**
1. **Cross-dataset test (7.7).** Trained on one dataset and tested on the other, using the 27 shared classes: **PlantWild → PlantDoc** (tested on the 220 clean PlantDoc images, and on the **173 twin-free** ones) and **PlantDoc → PlantWild** (1,111 twin-free images). Baseline vs KD, MobileNetV3-L and EfficientNet-B0, 2 seeds each. The KD teacher was cross-fitted on the *source* training data only.
2. **INT8 deployment (8.1–8.3).** Exported the seed-0 PlantWild students (baseline and KD) to **ONNX**, quantised them to **INT8** with ONNX Runtime post-training quantisation calibrated on 256 **training** images, and chose the quantisation recipe on the **validation** set. Measured test accuracy, file size and CPU latency (batch 1, 1 and 4 threads).

**বাংলা।**
1. **Cross-dataset পরীক্ষা (7.7)।** এক dataset-এ train করে অন্যটায় test, ২৭টা common class দিয়ে: **PlantWild → PlantDoc** (২২০টা clean ছবি, আর যমজ-মুক্ত **১৭৩টা**) আর **PlantDoc → PlantWild** (যমজ-মুক্ত ১,১১১টা)। Baseline বনাম KD, দুটো model, প্রতিটায় ২টা seed। Teacher শুধু *উৎস* dataset-এর train অংশে cross-fit করা।
2. **INT8 deployment (8.1–8.3)।** PlantWild-এর student model-গুলো **ONNX**-এ export করে **INT8**-এ রূপান্তর করেছি (train set-এর ২৫৬টা ছবি দিয়ে calibrate), আর পদ্ধতি বাছাই করেছি **validation** দেখে। Test accuracy, ফাইলের আকার আর CPU-তে গতি মেপেছি।

### Results / ফলাফল

**Cross-dataset (macro-F1 on twin-free test, mean of 2 seeds):**

| Direction | Model | Baseline | KD | Gain |
|---|---|---|---|---|
| PlantWild → PlantDoc (173) | EfficientNet-B0 | 0.534 | 0.572 | **+3.8 pp** |
| | MobileNetV3-L | 0.559 | 0.592 | **+3.3 pp** |
| | *Teacher* | | *0.698* | |
| PlantDoc → PlantWild (1,111) | EfficientNet-B0 | 0.423 | 0.486 | **+6.2 pp** |
| | MobileNetV3-L | 0.425 | 0.464 | **+3.9 pp** |
| | *Teacher* | | *0.628* | |

- Cross-dataset twins inflate scores too: on all 220 PlantDoc images (including 47 twins also in PlantWild), baseline EfficientNet scores 0.581 vs 0.534 on the twin-free 173 (**+4.7 pp**).
- Organ-shift classes: mixed and based on few images, so inconclusive.

**INT8 deployment (PlantWild P2 test, seed 0; recipe chosen on validation):**

| Model | Method | FP32 F1 | INT8 F1 | Drop | Size FP32 → INT8 | CPU 1-thread (FP32 / INT8) |
|---|---|---|---|---|---|---|
| MobileNetV4-S | baseline | 0.577 | 0.564 | −1.2 pp | 10.4 → 2.8 MB | 4.6 / 6.4 ms |
| MobileNetV4-S | **KD** | **0.656** | **0.633** | −2.3 pp | 10.4 → 2.8 MB | 4.7 / 6.4 ms |
| MobileNetV3-L | baseline | 0.662 | 0.594 | −6.8 pp | 17.3 → 4.6 MB | 9.0 / 13.6 ms |
| MobileNetV3-L | **KD** | **0.708** | **0.661** | −4.7 pp | 17.3 → 4.6 MB | 8.8 / 13.4 ms |
| EfficientNet-B0 | baseline | 0.658 | 0.594 | −6.5 pp | 16.5 → 4.6 MB | 18.8 / 27.6 ms |
| EfficientNet-B0 | **KD** | **0.716** | **0.668** | −4.8 pp | 16.5 → 4.6 MB | 18.8 / 28.0 ms |

- Default INT8 first **collapsed** (EfficientNet to 0.11 F1). The cause: Kaggle's Xeon CPU has **no VNNI**, so ONNX Runtime's 8-bit arithmetic overflowed. **Percentile calibration + `reduce_range`** fixed it.
- Keeping depthwise layers in FP32 (mixed precision) made results much worse in ONNX Runtime; reported as a negative result.
- On this CPU, INT8 is **not faster** than FP32 (no hardware INT8 support). The benefit here is **3.6× smaller files**.

### Is it good? / ফলাফল কেমন?

**English.**
- **Cross-dataset:** yes. Real domain shift is harsh (scores fall to 0.42–0.59), but **KD improves robustness in all 4 cases** (+3.3 to +6.2 pp), most in the hardest direction. This supports the claim that KD transfers some of the teacher's generality. Only 2 seeds per cell, so present it as consistent gains, not as a significance claim.
- **Deployment:** yes, with honest limits. The KD students are **fast on a plain CPU in FP32** (4.7 ms for MobileNetV4-S, 8.8 ms for MobileNetV3-L on one thread). In INT8, **KD students lose less accuracy than baselines**, and an INT8 KD student (0.661–0.668) is **as accurate as an FP32 baseline (0.658–0.662) at 3.6× smaller size**. Post-training INT8 still costs 2–5 pp; quantisation-aware training (QAT) is the natural future work.

**বাংলা।**
- **Cross-dataset:** হ্যাঁ। বাস্তবের domain shift কঠিন (score ০.৪২–০.৫৯-এ নেমে যায়), কিন্তু **চারটা ক্ষেত্রেই KD উন্নতি করেছে** (+৩.৩ থেকে +৬.২ pp), সবচেয়ে কঠিন দিকে সবচেয়ে বেশি। মানে teacher-এর সাধারণীকরণ ক্ষমতার কিছুটা student পায়। প্রতিটায় মাত্র ২টা seed, তাই significance দাবি না করে "ধারাবাহিক উন্নতি" হিসেবে লিখবে।
- **Deployment:** হ্যাঁ, সৎ সীমাবদ্ধতাসহ। KD student সাধারণ CPU-তে FP32-এ খুব দ্রুত (MobileNetV4-S ৪.৭ ms, MobileNetV3-L ৮.৮ ms)। INT8-এ **KD student baseline-এর চেয়ে কম accuracy হারায়**, আর INT8 KD student (০.৬৬১–০.৬৬৮) **FP32 baseline-এর (০.৬৫৮–০.৬৬২) সমান, অথচ ৩.৬ গুণ ছোট**। INT8-এ এখনো ২–৫ pp ক্ষতি হয়; quantisation-aware training (QAT) future work।

---

## Notebook 7 — Explainability: where do the students look? (Step 9)

### What I did / কী করেছি

**English.**
1. **Got ground-truth leaf boxes.** The PlantDoc authors published object-detection labels (boxes around diseased leaves). We matched them to our images by file name: **2,416 of 2,576 images (94%)** matched, with image sizes agreeing 99.3% and classes 99.6%. The boxes were converted to our 256×256 image frame (checked visually in figD15).
2. **Grad-CAM heat maps** for MobileNetV3-L and EfficientNet-B0, baseline vs KD, on all 2,286 clean PlantDoc images with boxes. Each image was explained by the P3 fold model that had it in its **test** set (never seen in training).
3. **Measured localisation:** the share of heat-map energy inside the leaf boxes, "lift" over chance, and the **pointing game** (is the hottest pixel on a leaf?). These were compared against chance and a **centre prior** (a blob in the middle of the image), with a paired Wilcoxon test.

**বাংলা।**
1. **পাতার আসল box জোগাড় করেছি।** PlantDoc-এর লেখকদের প্রকাশিত detection label (রোগাক্রান্ত পাতার চারপাশে বাক্স) file name দিয়ে মিলিয়েছি: **২,৫৭৬-এর মধ্যে ২,৪১৬টি ছবি (৯৪%)**, মাপ ৯৯.৩% আর class ৯৯.৬% মিলেছে।
2. **Grad-CAM heat map** বানিয়েছি baseline আর KD model-এর জন্য, ২,২৮৬টি clean ছবিতে; প্রতিটা ছবি এমন model দিয়ে ব্যাখ্যা করা হয়েছে, যে ছবিটা training-এ দেখেনি।
3. **মেপেছি** heat-এর কত অংশ পাতার box-এর ভেতরে পড়ে, আর সবচেয়ে গরম বিন্দু পাতায় পড়ে কিনা; তুলনা করেছি এলোমেলো map আর মাঝখানের একটা blob-এর সাথে।

### Results / ফলাফল

| Model | Method | Energy in box (all) | Pointing game (all) | Energy in box (small boxes) | Pointing game (small boxes) |
|---|---|---|---|---|---|
| MobileNetV3-L | baseline | 0.670 | 0.786 | 0.344 | 0.521 |
| MobileNetV3-L | **KD** | **0.679** | **0.813** | **0.354** | **0.574** |
| EfficientNet-B0 | baseline | 0.703 | 0.819 | 0.372 | 0.553 |
| EfficientNet-B0 | **KD** | **0.717** | **0.843** | **0.389** | **0.601** |
| *Chance (uniform)* | | *0.588* | | *0.257* | |
| *Centre prior* | | *0.698* | | *0.347* | |

- KD − baseline, energy in box: **+0.8 pp** (MobileNetV3, p = 4 × 10⁻⁴) and **+1.4 pp** (EfficientNet, p = 2 × 10⁻²⁴).
- Small boxes (< 40% of the image, 561 images): pointing-game gains of **+5.4 pp** and **+4.8 pp**.

### Is it good? / ফলাফল কেমন?

**English.** It is an honest, useful result:
- All students focus on the diseased leaves **more than chance**.
- But they are **not much better than a simple centre blob**: in-the-wild photos are strongly centred, so part of the apparent "good attention" is a photographic shortcut. This supports the thesis theme that these benchmarks contain shortcuts.
- **KD makes students look at the diseased leaves a little more, and consistently** (small but highly significant; clearer in the pointing game and on small boxes). Suggested thesis phrasing: *"KD modestly but significantly improves localisation on annotated leaves; it does not transform where the model looks."*
- Limitations: Grad-CAM is coarse (7×7), the boxes are large (about 59% of the image on average), and only PlantDoc has boxes.

**বাংলা।** সৎ ও কাজের ফল:
- সব student এলোমেলোর চেয়ে বেশি রোগাক্রান্ত পাতার দিকে তাকায়।
- কিন্তু মাঝখানের একটা সাধারণ blob-এর চেয়ে খুব বেশি ভালো নয়, কারণ ছবিগুলো সাধারণত পাতা মাঝখানে রেখে তোলা। এটা benchmark-এর আরেকটা shortcut, যা thesis-এর মূল বক্তব্যকে সমর্থন করে।
- **KD মনোযোগ অল্প কিন্তু ধারাবাহিকভাবে রোগাক্রান্ত পাতার দিকে সরায়** (ছোট box-এ সবচেয়ে গরম বিন্দু ৫ pp বেশিবার ঠিক পাতায়)। KD কোথায় তাকায় সেটা পুরো বদলায় না।
- সীমাবদ্ধতা: Grad-CAM মোটা দাগের (7×7), box বড় (গড়ে ছবির ~৫৯%), আর শুধু PlantDoc-এ box আছে।

---

## 6. Overall — are my results good? / সামগ্রিকভাবে ফলাফল কেমন?

**English.** Yes. The thesis now has a clear, well-supported story:

1. **Problem found:** both in-the-wild benchmarks leak (~11% of test images have training twins) and contain many conflicting labels. Leakage inflates scores (most strongly for small models) while noise deflates them, so official numbers hide both problems.
2. **Tool built:** a frozen DINOv2+CLIP teacher that is accurate (~0.77 macro-F1) and can find label errors automatically (AUROC 0.83 against experts).
3. **Solution shown:** leak-proof KD makes 4M-parameter students **+3.9 to +9.2 pp** better, significantly and consistently, **about 3× better calibrated**, more robust across datasets (+3.3 to +6.2 pp), and better than a 6× larger ResNet-50 on PlantWild.
4. **Deployable:** the KD students run in 5–9 ms on one CPU thread, and in INT8 they shrink to 2.8–4.6 MB while staying as accurate as FP32 baselines.

Strengths: careful leakage control everywhere (the test set is never used for choices), paired statistics, expert-label validation, honest negative results (BCC masks, mixed-precision INT8). Limitations to state: PlantDoc is small (hence 5-fold CV); students still trail the teacher by ~5–10 pp; MobileNetV4-S used a recipe not tuned for it; ablations and cross-dataset used 2 seeds; post-training INT8 costs 2–5 pp and gave no speedup on the test CPU.

**বাংলা।** হ্যাঁ। Thesis-এর এখন একটা পরিষ্কার, প্রমাণ-সমৃদ্ধ গল্প আছে:

1. **সমস্যা:** দুটো benchmark-এই leakage (~১১% test ছবির যমজ train-এ) আর অনেক পরস্পরবিরোধী label। Leakage score বাড়ায় (ছোট model-এ সবচেয়ে বেশি), আর ভুল label কমায়, তাই official সংখ্যায় দুটো সমস্যাই লুকিয়ে থাকে।
2. **হাতিয়ার:** frozen DINOv2+CLIP teacher, যা নির্ভুল (~০.৭৭) আর নিজে থেকেই ভুল label খুঁজে পায় (experts-এর বিপরীতে AUROC ০.৮৩)।
3. **সমাধান:** leak-proof KD ৪০ লক্ষ parameter-এর student-কে **+৩.৯ থেকে +৯.২ pp** ভালো করে, ধারাবাহিকভাবে ও significant-ভাবে, **প্রায় ৩ গুণ ভালো calibration** দেয়, অন্য dataset-এও ভালো কাজ করে (+৩.৩ থেকে +৬.২ pp), আর PlantWild-এ ৬ গুণ বড় ResNet-50-কেও হারায়।
4. **বাস্তবে ব্যবহারযোগ্য:** KD student এক CPU thread-এ ৫–৯ ms-এ চলে, আর INT8-এ ২.৮–৪.৬ MB-এ নেমে আসে, তবুও FP32 baseline-এর সমান নির্ভুল।

শক্তি: সব জায়গায় সতর্ক leakage নিয়ন্ত্রণ (কোনো সিদ্ধান্তে test ব্যবহার হয়নি), জোড়া-তুলনার statistics, expert label দিয়ে যাচাই, আর সৎ negative result (BCC mask, mixed-precision INT8)। যে সীমাবদ্ধতাগুলো লিখতে হবে: PlantDoc ছোট (তাই ৫-fold), student এখনো teacher-এর চেয়ে ~৫–১০ pp পিছিয়ে, MobileNetV4-S-এর জন্য পদ্ধতি আলাদা করে tune করা হয়নি, ablation আর cross-dataset-এ ২টা seed, আর INT8-এ ২–৫ pp ক্ষতি হয় এবং এই CPU-তে গতি বাড়েনি।

---

## 7. What is still open / যা এখনো বাকি

| Item | Status |
|---|---|
| Step 7 rerun numbers | Done (this report) |
| Cross-dataset (7.7) | Done |
| Mask figures D11–D14 (BCC negative result) | Done (regenerated in the rerun) |
| INT8 quantisation + CPU latency | Done (post-training; QAT = future work) |
| XAI (Grad-CAM vs PlantDoc leaf boxes) | Done (Notebook 7) |
| Save Step 8 and Step 9 outputs as Kaggle datasets | Step 8 done; Step 9 to do |
| Update the Methodology document (BCC removed, cross-fitted KD, INT8) | To do |
| Thesis writing | After the above |

**বাংলা।** Step 7, cross-dataset, mask figure আর INT8 শেষ। বাকি: Step 8-এর ফাইল dataset হিসেবে save করা, XAI (Grad-CAM দিয়ে model ছবির কোথায় তাকায়), methodology আপডেট আর thesis লেখা।

---

## 8. Figures and tables produced / তৈরি হওয়া figure ও table

All are in `figures/` (PNG 300 dpi + PDF), with captions in `figures/figure_log.csv`.

| Chapter | Figures | Tables |
|---|---|---|
| Dataset (NB 1–2) | figD1–D2 class distribution · figD3 small-image shortcut · figD4 threshold sensitivity · figD5 protocols · figD6 expert vs conflict · figD7 leak twins · figD8 label conflicts · figD9 within-train duplicates · figD10 cross-dataset twins | audit summary, class counts, class mapping, protocols, expert vs conflict |
| Teachers (NB 3) | fig5_1 teacher probes · fig5_2 leakage inflation · fig5_3 noise ROC · fig5_4 confidence by expert decision · fig5_5 PlantDoc teacher confusion matrix | teacher probes, noise detection |
| Students (NB 4) | fig6_1 learning curves · fig6_2 baselines vs teacher · fig6_3 accuracy vs latency · fig6_4 leakage for students | baselines, efficiency, student leakage |
| KD (NB 5) | fig7_1 main KD result · fig7_2 ablations · fig7_3 per-class gain · fig7_4 calibration · figD11–D14 masks (BCC negative result) | KD main, paired units, significance, ablations, expert/calibration, per-class |
| Cross-dataset + deployment (NB 6) | fig7_5 cross-dataset · fig8_1 INT8 accuracy vs size | cross-dataset, cross-dataset runs, INT8 recipe search, INT8 deployment |
| Explainability (NB 7) | figD15 box check · fig9_1 Grad-CAM examples · fig9_2 localisation scores | Grad-CAM localisation, Grad-CAM per image |

---

## 9. Practical lessons / কাজের অভিজ্ঞতা থেকে শিক্ষা

**English.**
- Kaggle deletes `/kaggle/working` when a session stops. Before stopping, either download the zip **and** upload it as a dataset, or use **Save & Run All (Commit)** and wait until the version shows "Running".
- Inside commits, download all pretrained weights once in the main process, then set offline mode before starting parallel training.
- Keep GitHub for code, figures and small CSVs only; large files (`.npy`, `.pt`, `.zip`) stay in Kaggle datasets.

**বাংলা।**
- Session বন্ধ হলে Kaggle `/kaggle/working` মুছে দেয়। বন্ধ করার আগে হয় zip download করে dataset হিসেবে upload করো, নয়তো **Save & Run All (Commit)** দিয়ে version "Running" না দেখা পর্যন্ত অপেক্ষা করো।
- Commit-এর ভেতরে সব pretrained weight আগে main process-এ একবার download করো, তারপর offline mode চালু করে parallel training শুরু করো।
- GitHub-এ শুধু code, figure আর ছোট CSV রাখো; বড় ফাইল (`.npy`, `.pt`, `.zip`) Kaggle dataset-এই থাকবে।
