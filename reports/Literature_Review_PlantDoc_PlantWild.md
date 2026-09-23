# Literature Review (Verified) — Lightweight Transfer-Learning Plant Disease Classification on Real-World Leaf Images (PlantDoc & PlantWild)

**সাহিত্য পর্যালোচনা (যাচাইকৃত) — PlantDoc ও PlantWild ডেটাসেটে হালকা (lightweight) ট্রান্সফার-লার্নিং ভিত্তিক উদ্ভিদ রোগ শ্রেণিবিন্যাস**

Prepared for: Emon, Fahad & Mahir (IIUC CSE) — Supervisor: Mr. Rayhanuzzaman
Scope: every paper cited in the proposal (8) + 6 additional key works. Date of review: 23 Sep 2026.

---

## 0. How this review was done / কীভাবে যাচাই করা হয়েছে

**English.** Every paper was checked against its original source (arXiv, ACM, Frontiers, MDPI, Springer, Nature). For each paper I record (a) what they did, (b) the exact numbers they report, (c) the experimental protocol (split, dataset version, cropped/uncropped), and (d) the gap or weakness. A short "Proposal check" line says whether your proposal describes the paper correctly. Where I could **not** confirm a number from the source, I say so explicitly — do not cite unverified numbers in the thesis.

**বাংলা।** প্রতিটি পেপার মূল উৎস (arXiv, ACM, Frontiers, MDPI, Springer, Nature) থেকে মিলিয়ে দেখা হয়েছে। প্রতিটির জন্য লেখা হয়েছে — (ক) তারা কী করেছে, (খ) তারা ঠিক কোন সংখ্যা রিপোর্ট করেছে, (গ) পরীক্ষার পদ্ধতি (split, ডেটাসেটের কোন ভার্সন, cropped নাকি uncropped), এবং (ঘ) দুর্বলতা/গ্যাপ। "Proposal check" লাইনে বলা আছে তোমাদের প্রপোজালে পেপারটির বর্ণনা ঠিক আছে কি না। যে সংখ্যা মূল উৎস থেকে নিশ্চিত করা যায়নি, সেটা স্পষ্টভাবে বলা হয়েছে — থিসিসে যাচাই না করা সংখ্যা ব্যবহার করবে না।

---

## 1. The core problem: the lab-to-field gap / মূল সমস্যা: ল্যাব থেকে মাঠে পারফরম্যান্স পতন

### 1.1 Mohanty, Hughes & Salathé (2016) — *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science 7:1419

**What they did.** Trained AlexNet/GoogLeNet on PlantVillage (54,306 lab images, 14 crops, 38 classes).
**Result.** 99.35% on a held-out PlantVillage test set, but only **31.4%** when tested on images collected from online sources taken under different conditions.
**Gap.** Proved that very high PlantVillage accuracy does not transfer to real photos; the authors themselves called for more diverse training data.
**Why it matters for you.** This is the foundational citation for your "Motivation" section. It is *the* reason PlantDoc and PlantWild exist.

**বাংলা।** তারা PlantVillage-এর ল্যাবে তোলা ছবিতে CNN ট্রেন করে ৯৯.৩৫% accuracy পেয়েছিল, কিন্তু ইন্টারনেট থেকে নেওয়া বাস্তব ছবিতে টেস্ট করলে accuracy নেমে আসে মাত্র ৩১.৪%-এ। অর্থাৎ ল্যাবের পরিষ্কার ছবিতে ভালো করা মডেল মাঠের ছবিতে ব্যর্থ হয়। তোমাদের থিসিসের "Motivation" অংশের মূল ভিত্তি এই পেপার — PlantDoc ও PlantWild তৈরির কারণই এই সমস্যা।

---

## 2. The two datasets / দুটি ডেটাসেট

### 2.1 PlantDoc — Singh, Jain, Jain, Kayal, Kumawat & Batra (CoDS-COMAD 2020, pp. 249–253; arXiv 1911.10317, 2019)

**What they did.**
- Collected ~20,900 web images (Google Images, Ecosia), filtered with APSNet guidelines, each image checked by two people, removed classes with <50 images → **2,598 images, 13 species, 27 classes (17 disease + 10 healthy)**.
- Drew bounding boxes around every leaf (LabelImg) → also released **Cropped-PlantDoc (C-PD): 9,216 leaf crops**.
- Benchmarked classification (VGG16, InceptionV3, InceptionResNetV2) and detection (Faster R-CNN, MobileNet-SSD). Detection split: 2,360 train / 238 test.

**Key numbers (verified from the paper's tables).**

| Setting | Model | Accuracy |
|---|---|---|
| Uncropped PlantDoc, ImageNet weights only | VGG16 | 13.74% |
| Uncropped, trained on PlantVillage, tested on PlantDoc | VGG16 | 15.08% |
| Uncropped, ImageNet+PlantVillage pretraining, fine-tuned on PlantDoc | VGG16 | 29.73% |
| Cropped (C-PD), ImageNet only | InceptionResNetV2 | 49.04% |
| Cropped, trained on PVD only → tested on C-PD | InceptionResNetV2 | 39.87% |
| Cropped, ImageNet+PVD pretraining → fine-tuned on C-PD | InceptionResNetV2 | **70.53%** (F1 0.70) |
| Detection mAP@0.5 | Faster R-CNN InceptionResNetV2 (COCO) | 38.9 |

**Gaps / weaknesses.**
1. Images were resized to only **100×100** — very low resolution for fine lesion features; modern baselines at 224×224 are not comparable.
2. The authors admit some images may be **mislabeled** (e.g., tomato bacterial spot vs. Septoria look alike) due to limited domain expertise.
3. Small data: many classes have only ~50–100 images.
4. Web images, not true field photos from farmers' phones.

**Proposal check.** Correct (2,598 images, 13 species, up to 17 diseases). Add: *27 classes total*, the *cropped vs uncropped* distinction, and the 70.53% baseline.

**বাংলা ব্যাখ্যা।** PlantDoc হলো প্রথম বড় পাবলিক "in-the-wild" উদ্ভিদ রোগের ডেটাসেট। ইন্টারনেট থেকে ছবি নিয়ে বাছাই করে মোট ২,৫৯৮টি ছবি, ১৩টি প্রজাতি, ২৭টি ক্লাস (১৭টি রোগ + ১০টি সুস্থ) রাখা হয়েছে। প্রতিটি পাতার চারপাশে bounding box আঁকা আছে, তাই পাতাগুলো কেটে আলাদা করে ৯,২১৬টি ছবির "Cropped-PlantDoc" তৈরি হয়েছে।

গুরুত্বপূর্ণ ফলাফল: পুরো ছবি (uncropped) দিয়ে VGG16 পেয়েছে মাত্র ২৯.৭৩%, কিন্তু কাটা পাতার ছবিতে (cropped) InceptionResNetV2 পেয়েছে ৭০.৫৩%। অর্থাৎ **ব্যাকগ্রাউন্ড সরালে accuracy অনেক বাড়ে** — তোমাদের "leaf cropping/background reduction" আইডিয়ার পক্ষে এটা সরাসরি প্রমাণ।

দুর্বলতা: ছবি মাত্র ১০০×১০০ পিক্সেলে ছোট করা হয়েছিল; কিছু ছবির লেবেল ভুল থাকতে পারে বলে লেখকেরা নিজেরাই স্বীকার করেছেন; প্রতি ক্লাসে ছবি কম।

---

### 2.2 PlantWild + MVPDR — Wei, Chen, Huang & Yu (ACM Multimedia 2024; arXiv 2408.03120)

**What they did.**
- Collected >50,000 images from Google, Ecosia and **Baidu** (English + Chinese queries), filtered by 5 annotators; each image cross-checked by ≥2 people and verified by an expert → **18,542 images, 89 classes (56 diseased + 33 healthy)**. Class sizes range from **44 to 589** images (strong imbalance).
- Added multiple **text descriptions per class** (Wikipedia + GPT-3.5) → a multimodal dataset.
- Split: **70% train / 10% val / 20% test**.
- Proposed **MVPDR**: frozen CLIP encoders; per class, K-means clusters of image features become *visual prototypes*; text descriptions become *textual prototypes*; only prototypes are trained.

**Key numbers (Table 1, CLIP-ResNet101 backbone, fully supervised).**

| Method | PlantVillage Acc | PlantDoc Acc | PlantWild Acc | PlantWild Macro-F1 |
|---|---|---|---|---|
| T-CNN (CNN SOTA) | 98.80 | 64.36 | 63.61 | 58.70 |
| DHBP (CNN SOTA) | 98.88 | 65.94 | 65.92 | 59.66 |
| CoOp (CLIP) | 91.00 | 66.73 | 61.16 | 56.26 |
| **MVPDR** | 97.72 | **69.90** | **67.20** | **62.84** |

With a bigger CLIP ViT-L/14 backbone, MVPDR reaches **77.23% (PlantDoc)** and **76.18% (PlantWild)**. Few-shot (16/class) on PlantWild: 51.80%.

**Gaps / weaknesses.**
1. Backbone is **CLIP ViT-L/14 (~300M+ parameters)** for the best result — impossible for a phone.
2. **No lightweight CNN (MobileNet/EfficientNet) baseline** was benchmarked on PlantWild.
3. Accuracy grows with backbone size → open question whether small models can come close.
4. Web images again (not farmer-captured); strong class imbalance (44 vs 589).
5. Explainability shown only as qualitative similarity maps.

**Proposal check.** Correct. Add: 70/10/20 split, the 67.20%/76.18% numbers, and class-size range 44–589.

**বাংলা ব্যাখ্যা।** PlantWild এখন পর্যন্ত সবচেয়ে বড় পাবলিক in-the-wild ডেটাসেট: ১৮,৫৪২টি ছবি, ৮৯টি ক্লাস (৫৬টি রোগ, ৩৩টি সুস্থ)। প্রতিটি ক্লাসে রোগের টেক্সট বর্ণনাও আছে। ক্লাসভেদে ছবির সংখ্যা ৪৪ থেকে ৫৮৯ — অর্থাৎ ভীষণ class imbalance, তাই শুধু accuracy নয়, **macro-F1 রিপোর্ট করা বাধ্যতামূলক**।

তাদের MVPDR মডেল CLIP ব্যবহার করে ছবি ও টেক্সট দুটো থেকে "prototype" বানায়। ছোট CLIP-ResNet101 দিয়ে PlantWild-এ ৬৭.২০%, আর বিশাল ViT-L/14 দিয়ে ৭৬.১৮%। কিন্তু **কোনো MobileNet/EfficientNet-এর মতো হালকা মডেল তারা টেস্ট করেনি** — এটাই তোমাদের থিসিসের জন্য সবচেয়ে বড় সুযোগ।

---

### 2.3 (Related dataset) FieldPlant — Moupojou et al., IEEE Access 11 (2023)

5,170 plantation images (Cameroon), 8,629 annotated leaves, 27 classes, annotated with plant pathologists. Useful as a possible **third external test set** for generalization, and it is truly field-captured (unlike PlantDoc/PlantWild web images).

**বাংলা।** FieldPlant আসল খামার থেকে তোলা ছবি (ওয়েব থেকে নয়), উদ্ভিদ-রোগ বিশেষজ্ঞদের দিয়ে লেবেল করা। তোমরা চাইলে এটিকে তৃতীয় "বাইরের" টেস্ট সেট হিসেবে ব্যবহার করতে পারো।

---

## 3. Method papers (the 8 in the proposal + extras) / পদ্ধতিভিত্তিক পেপারসমূহ

### 3.1 Duhan, Gulia, Gill, Oliveira & Singh (2026) — *Evaluation of lightweight and efficient deep learning models for plant disease classification*, Discover Internet of Things 6:43

**What they did.** Compared ShuffleNetV2×0.5, MobileNetV2, MobileNetV3-Small, and YOLOv8n/YOLOv11n (converted to classifiers) on PlantDoc, Mango and Soybean; with/without augmentation; ablations with 8-bit quantization and structured pruning; deployed TFLite models on **Raspberry Pi 5**; showed Grad-CAM.
**Protocol.** Used a Kaggle PlantDoc version with **2,569 images, 28 classes**; 224×224; 80:10:10 split; minority classes were augmented (Albumentations) **up to the size of the majority class**.
**Results.** MobileNetV2 = **89.86%** on augmented PlantDoc (90.00% after quantization). MobileNetV3-Small rose from 65.68% → 89.23% with augmentation. Quantization barely hurt accuracy; pruning cut accuracy by ≥25% for some models. Raspberry Pi latency: ShuffleNetV2 7 ms, YOLOv8n 28 ms, MobileNetV2 46 ms, MobileNetV3-Small 137 ms.

**Gaps / critique (important).**
1. ⚠️ **Possible data leakage.** The paper balances classes by augmenting minority classes to majority size and reports before/after counts for the whole dataset, but does not clearly state that augmentation was applied **only to the training split after splitting**. If augmented variants of an image land in both train and test, accuracy is inflated. The +20–25 point jumps from simple augmentation on PlantDoc are unusually large and consistent with this risk. **Treat 89.86% with caution and do not claim to "beat" it unless you replicate their protocol.**
2. No PlantWild; no cross-dataset test; no macro-F1 on an official split.
3. Grad-CAM is qualitative only.

**Proposal check.** Number correct. Add the dataset version (2,569/28 classes) and the leakage caveat.

**বাংলা ব্যাখ্যা।** তোমাদের থিসিসের সবচেয়ে কাছের কাজ এটা — হালকা মডেল (MobileNetV2, MobileNetV3-Small, ShuffleNetV2, YOLO-nano) PlantDoc-এ তুলনা, quantization/pruning, আর Raspberry Pi-তে চালানো। MobileNetV2 পেয়েছে ৮৯.৮৬%।

কিন্তু সাবধান: তারা ছোট ক্লাসগুলোকে augmentation দিয়ে বড় ক্লাসের সমান বানিয়েছে, এবং এই augmentation শুধু train সেটে split-এর পরে করা হয়েছিল কি না তা পেপারে পরিষ্কার নয়। যদি split-এর আগে করা হয়, তাহলে একই ছবির পরিবর্তিত কপি train আর test দুই জায়গাতেই থাকবে — ফলে accuracy কৃত্রিমভাবে বেশি দেখাবে (data leakage)। শুধু augmentation দিয়ে ২০-২৫% লাফ অস্বাভাবিক। তাই এই ৮৯.৮৬%-কে "টার্গেট" না ধরে, **সঠিক প্রোটোকলে (আগে split, পরে শুধু train-এ augmentation)** কাজ করাই তোমাদের শক্তি হবে।

---

### 3.2 Krishna, Machado, Otuka, Yahaya, Neves dos Santos & Ihianle (2025) — *Plant Leaf Disease Detection Using Deep Learning: A Multi-Dataset Approach*, J (MDPI) 8(1):4

**What they did.** Fine-tuned EfficientNet-B0/B3, ResNet50, DenseNet201 on PlantDoc and on a web-sourced dataset; added Gaussian-noise augmentation; ran cross-dataset tests.
**Results.** EfficientNet-B3: **73.31%** (train/test PlantDoc); **76.77%** trained on PlantDoc → tested on web data; **80.19%** trained on combined data → tested on web data. Some classes (apple rust, grape leaf) had F1 > 0.90.
**Gaps.** No model-size/latency analysis; web dataset not a public standard benchmark; no PlantWild; weak classes not deeply analyzed.
**Proposal check.** Correct.

**বাংলা।** EfficientNet-B3 দিয়ে PlantDoc-এ ৭৩.৩১%। একাধিক ডেটাসেট একসাথে ট্রেন করলে (combined) ৮০.১৯% — অর্থাৎ **বেশি বৈচিত্র্যময় ডেটা = ভালো generalization**। তবে মডেলের সাইজ বা গতি তারা মাপেনি, আর PlantWild ব্যবহার করেনি।

---

### 3.3 Salman, Muhammad & Han (2025) — *Plant disease classification in the wild using vision transformers and mixture of experts*, Frontiers in Plant Science 16:1522985

**What they did.** ViT-B/16 backbone + Mixture-of-Experts classifier head with a gating network; added Gaussian feature noise, entropy, orthogonal and usage regularization. Measured PlantVillage↔PlantDoc distribution shift with KL divergence. Built balanced PlantVillage subsets (PV_100, PV_200) via t-SNE clustering.
**Important protocol detail.** They **cleaned PlantDoc**: split composite images into separate samples and removed irrelevant images (e.g., fruits) → they call it a "PlantDoc sub-dataset". Their numbers are therefore **not directly comparable** to standard PlantDoc.
**Results.** PlantDoc sub-dataset (80:10:10): **74%** (ViT-Base 69%, InceptionV3 65%, EfficientNet 59%). Cross-domain PV_200 → PlantDoc: **68%** (ViT-Base 48%). PlantVillage: 99.96%.
**Gaps.** Heavy model (ViT-B ~86M params + experts); authors admit MoE adds inference latency unsuitable for mobile; single-label only; no PlantWild.
**Proposal check.** Correct, but add that 74% is on a *cleaned subset*.

**বাংলা।** ViT + Mixture of Experts দিয়ে PlantDoc-এ ৭৪%, আর PlantVillage-এ ট্রেন করে PlantDoc-এ টেস্ট করলে ৬৮%। তবে তারা PlantDoc পরিষ্কার করেছে (একাধিক রোগের কোলাজ ছবি ভাগ করা, অপ্রাসঙ্গিক ছবি বাদ) — তাই এই ৭৪% সাধারণ PlantDoc-এর সাথে সরাসরি তুলনীয় নয়। মডেল ভারী, মোবাইলে চলার মতো নয় — লেখকরাই স্বীকার করেছেন। **শিক্ষা:** PlantDoc-এর লেবেল/কোলাজ সমস্যা বাস্তব, তোমাদের data inspection ধাপে এটা নথিভুক্ত করা উচিত।

---

### 3.4 Zubair, Saleh, Akbari & Al Maadeed (2025) — *A Robust Ensemble Model for Plant Disease Detection Using Deep Learning Architectures*, AgriEngineering 7(5):159

**What they did.** Feature-level ensemble: InceptionResNetV2 + MobileNetV2 + EfficientNetB3 (last 10 blocks fine-tuned), concatenated GAP features → Dense(512) + BN + Dropout. Augmentation applied **only to the training set** (good practice); used provided PlantDoc train/test split with 20% of train as validation.
**Results.** PlantVillage **99.69%**, PlantDoc **60%**, FieldPlant **83%**. Model: **69.6 M parameters, ~265 MB**.
**Gaps.** Strong overfitting on PlantDoc (train ~90% vs val ~65–70%); authors admit they did not beat PlantDoc SOTA; heavy model; only 15 epochs.
**Proposal check.** Correct. Add model size — it is a useful *counter-example* showing that bigger/ensembled ≠ better on PlantDoc.

**বাংলা।** তিনটি বড় মডেল জোড়া লাগিয়েও PlantDoc-এ মাত্র ৬০%, অথচ PlantVillage-এ ৯৯.৬৯%। মডেলের আকার ৬৯.৬ মিলিয়ন প্যারামিটার (~২৬৫ MB)। এটা তোমাদের জন্য খুব ভালো যুক্তি: **"বড় মডেল = ভালো" নয়; সঠিক প্রশিক্ষণ কৌশল দিয়ে ছোট মডেলও ভালো করতে পারে।** একটা ভালো দিক: তারা augmentation শুধু train সেটে করেছে — তোমরাও তাই করবে।

---

### 3.5 Miao, Meng & Zhou (2025) — *SerpensGate-YOLOv8: an enhanced YOLOv8 model for accurate plant disease detection*, Frontiers in Plant Science 15:1514832 (DOI 10.3389/fpls.2024.1514832)

**What they did.** Modified YOLOv8 with Dynamic Snake Convolution in C2f, SPPELAN, and Super Token Attention; trained/tested on PlantDoc (object **detection**).
**Verified results.** Precision **0.719**; mAP@0.5 **+3.3%** over vanilla YOLOv8.
**⚠️ Unverified.** Your proposal states mAP@0.5 = 64.9%. I could **not** confirm 64.9% from the abstract or accessible text. Open the paper's results table and confirm before citing; otherwise cite only "precision 0.719 and +3.3% mAP@0.5 over YOLOv8".
**Gaps.** Detection task (not classification); no efficiency/latency on device; no cross-dataset test.

**বাংলা।** এটা detection পেপার (bounding box খোঁজা), classification নয়। Precision ০.৭১৯ এবং YOLOv8-এর চেয়ে mAP ৩.৩% বেশি — এটুকু নিশ্চিত। প্রপোজালের "mAP 64.9%" সংখ্যাটা আমি মূল উৎস থেকে নিশ্চিত করতে পারিনি — পেপারের টেবিল খুলে মিলিয়ে নিও, নইলে বাদ দাও।

---

### 3.6 Bera, Bhattacharjee & Krejcar (2024) — *PND-Net: plant nutrition deficiency and disease classification using graph convolutional network*, Scientific Reports 14:15537

**What they did.** CNN backbone → multi-scale spatial pyramid pooling of regions → Graph Convolutional Network over region features.
**Results.** PlantDoc **84.30%** (Xception), 81.0% (InceptionV3); also Banana/Coffee deficiency ~90%, Potato 96.18%.
**Model size.** Xception-PND-Net 26–38 M params; the MobileNetV2 variant is 6.75 M (GCN-1024).
**Gaps.** Heavier than plain MobileNet; PlantDoc protocol (cropped vs full, split) must be checked before comparing; no PlantWild; no on-device test.
**Proposal check.** Correct.

**বাংলা।** CNN-এর ফিচারকে ছোট ছোট অঞ্চলে ভাগ করে Graph Neural Network দিয়ে সম্পর্ক শেখানো হয়েছে; PlantDoc-এ ৮৪.৩০%। মূল শিক্ষা: **পাতার আলাদা আলাদা অঞ্চল (region) থেকে ফিচার নিলে রোগের দাগ ভালো ধরা পড়ে।** তবে মডেল তুলনামূলক ভারী।

---

### 3.7 (Extra) Duhan, Gulia, Gill & Narwal (2025) — *RTR_Lite_MobileNetV2*, Current Plant Biology 42:100459

MobileNetV2 + attention modules (SE, ECA, Triplet Attention). Reports **82.00% on PlantDoc**, 99.92% on PlantVillage-type data; deployed on Raspberry Pi 4/5. Shows that **attention-enhanced MobileNet** is a strong lightweight baseline for PlantDoc.

**বাংলা।** MobileNetV2-এ attention মডিউল যোগ করে PlantDoc-এ ৮২%। তোমাদের "improved lightweight model"-এর জন্য এটা একটা বাস্তব রেফারেন্স।

---

### 3.8 (Extra) Kumar, Monga, Brahma, Kalra & Sherif (2025) — *Mobile-Friendly Deep Learning for Plant Disease Detection: A Lightweight CNN Benchmark Across 101 Classes of 33 Crops*, arXiv 2508.10817 (preprint)

Merged **PlantDoc + PlantVillage + PlantWild** into 101 classes and benchmarked MobileNetV2/V3 and EfficientNet-B0/B1; **EfficientNet-B1 = 94.7%**.
**Critique.** Merging lets the huge, easy PlantVillage portion dominate the overall accuracy; no per-source (PlantWild-only / PlantDoc-only) test results reported in the abstract; single training runs; not peer-reviewed. **This is the closest prior work to your idea — you must cite it and clearly differ from it** (evaluate each wild dataset separately, report macro-F1, avoid lab-image inflation).

**বাংলা।** তিনটি ডেটাসেট একসাথে মিশিয়ে হালকা মডেলে ৯৪.৭% পেয়েছে। কিন্তু মিশ্রণে PlantVillage-এর সহজ ছবিই বেশি, তাই এই accuracy বাস্তব মাঠের ছবির পারফরম্যান্স বোঝায় না। এটা এখনও peer-reviewed নয়। **তোমাদের কাজের সবচেয়ে কাছের পেপার এটা — অবশ্যই উল্লেখ করতে হবে এবং দেখাতে হবে তোমরা কোথায় আলাদা** (প্রতিটি wild ডেটাসেট আলাদাভাবে মূল্যায়ন, macro-F1, ল্যাব-ছবি বাদ)।

---

### 3.9 (Extra) El Karch, Natij, Benaly, El Gouri & Mezouari (2026) — *Backbone diversity beats text supervision: frozen multi-foundation model fusion for in-the-wild plant disease recognition*, Frontiers in Plant Science (DOI 10.3389/fpls.2026.1860665)

Frozen DINOv2 + DINOv3 + CLIP (ViT-L) features concatenated → simple linear/prototype heads. **PlantWild: 80.23% ± 0.41 (5 seeds)** — current best reported. Importantly, they reproduced MVPDR under the official split and got **72.27%** vs the published 76.18% → **protocol differences alone shift results by ~4 points.**
**Why it matters.** (1) Sets the current *upper bound* on PlantWild; (2) shows reproducibility problems; (3) these giant frozen models are ideal **teachers for knowledge distillation** into a MobileNet — an open gap.

**বাংলা।** তিনটি বিশাল foundation model (DINOv2, DINOv3, CLIP) একসাথে করে PlantWild-এ ৮০.২৩% — এখন পর্যন্ত সর্বোচ্চ। তারা দেখিয়েছে একই পদ্ধতি ভিন্ন প্রোটোকলে চালালে ফল প্রায় ৪% বদলে যায়। **এই বড় মডেলগুলোকে "শিক্ষক" (teacher) বানিয়ে ছোট MobileNet-কে শেখানো (knowledge distillation)** — এটা এখনও কেউ PlantWild-এ করেনি; তোমাদের জন্য সম্ভাব্য নতুনত্ব।

---

## 4. Master comparison table / সারসংক্ষেপ তুলনা সারণি

| # | Paper (year) | Dataset(s) | Model | Size | Real-world result | Protocol notes |
|---|---|---|---|---|---|---|
| 1 | Mohanty (2016) | PlantVillage → web images | AlexNet/GoogLeNet | medium | 31.4% on web images | shows domain gap |
| 2 | Singh / PlantDoc (2020) | PlantDoc | InceptionResNetV2 | ~55M | 70.53% (cropped), 29.73% uncropped (VGG16) | 100×100 input |
| 3 | Wei / PlantWild (2024) | PlantWild, PlantDoc | MVPDR (CLIP) | RN101 → ViT-L | PW 67.20→76.18; PD 69.90→77.23 | official 70/10/20 split |
| 4 | Duhan (2026) | PlantDoc (28 cls) | MobileNetV2 | ~3.5M | 89.86% (augmented) | ⚠️ leakage risk unclear |
| 5 | Krishna (2025) | PlantDoc + web | EfficientNet-B3 | ~12M | 73.31% PD; 80.19% combined | cross-dataset |
| 6 | Salman (2025) | PlantDoc subset | ViT + MoE | >86M | 74% (cleaned subset); 68% PV→PD | not standard PlantDoc |
| 7 | Zubair (2025) | PV, PD, FieldPlant | 3-CNN ensemble | 69.6M | PD 60%, FieldPlant 83% | overfits |
| 8 | Miao (2025) | PlantDoc (detection) | SerpensGate-YOLOv8 | — | Precision 0.719 | detection, not classification |
| 9 | Bera (2024) | PlantDoc | PND-Net (Xception+GCN) | 26–38M | 84.30% | check crop/split |
| 10 | Duhan (2025) | PlantDoc + others | RTR_Lite_MobileNetV2 | small | 82.00% | attention MobileNet |
| 11 | Kumar (2025, preprint) | PD+PV+PW merged | EfficientNet-B1 | ~7.8M | 94.7% (merged) | PV dominates |
| 12 | El Karch (2026) | PlantWild, PlantDoc | DINOv2+v3+CLIP fusion | ViT-L ×3 | PW 80.23% | reproduces MVPDR at 72.27% |

*Parameter counts for standard backbones (MobileNetV2 ≈3.5M, EfficientNet-B3 ≈12M, ViT-B ≈86M, InceptionResNetV2 ≈55M) are from the original architecture papers, not these studies; verify the exact figure you cite.*

**বাংলা।** সারণি থেকে পরিষ্কার: PlantDoc-এ ফলাফল ৬০% থেকে ৯০% পর্যন্ত ছড়ানো — কিন্তু এগুলো **একে অপরের সাথে তুলনীয় নয়**, কারণ কেউ cropped, কেউ uncropped, কেউ ২৭ ক্লাস, কেউ ২৮ ক্লাস, কেউ পরিষ্কার-করা subset, কেউ হয়তো leakage-সহ augmentation ব্যবহার করেছে। PlantWild-এ সব ভালো ফলাফল এসেছে বিশাল CLIP/DINO মডেল থেকে; **হালকা মডেলের কোনো স্বতন্ত্র benchmark নেই।**

---

## 5. Cross-cutting findings / সামগ্রিক পর্যবেক্ষণ

1. **Background removal helps a lot.** PlantDoc's own baseline jumps from ~30% (full image) to ~70% (cropped leaf).
   *বাংলা:* ব্যাকগ্রাউন্ড সরালে accuracy প্রায় দ্বিগুণ হয়।
2. **Pretraining source matters.** ImageNet+PlantVillage pretraining beat ImageNet-only by ~20 points on cropped PlantDoc.
   *বাংলা:* শুধু ImageNet নয়, আগে PlantVillage-এ fine-tune করে তারপর PlantDoc-এ করলে (two-stage transfer learning) ভালো হয়।
3. **Bigger ≠ better on small wild data.** A 69.6M-parameter ensemble got 60% while attention-MobileNetV2 got 82%.
   *বাংলা:* ছোট ডেটায় বড় মডেল overfit করে।
4. **Protocol inconsistency is the biggest problem in this literature** — different PlantDoc versions, splits, cropping, and augmentation timing make most numbers non-comparable; even MVPDR drops ~4 points under a different protocol.
   *বাংলা:* এই গবেষণাক্ষেত্রের সবচেয়ে বড় সমস্যা হলো একেকজন একেকভাবে পরীক্ষা করে — ফলে সংখ্যাগুলো তুলনা করা যায় না।
5. **Foundation models dominate PlantWild**, but none are deployable on a phone.
   *বাংলা:* PlantWild-এ বড় foundation model সেরা, কিন্তু মোবাইলে চলে না।

---

## 6. Research gaps (verified, specific) / গবেষণার ফাঁক (নির্দিষ্ট ও যাচাইকৃত)

Your proposal's current gap statement ("simple lightweight method + Grad-CAM") is too generic — reviewers will say MobileNet+Grad-CAM on PlantDoc already exists (Duhan 2026, RTR_Lite 2025). Below are sharper, defensible gaps.

**G1 — No standardized lightweight benchmark on PlantWild.** In my search, no peer-reviewed study reports MobileNetV3 / EfficientNet-B0/B1 / ShuffleNet results on PlantWild **alone** under its official split; results exist only for CLIP/DINO models or in a merged dataset (Kumar 2025 preprint).
*বাংলা:* PlantWild-এ আলাদাভাবে, অফিসিয়াল split-এ, হালকা মডেলের কোনো peer-reviewed benchmark পাইনি।

**G2 — Leakage-free, reproducible protocol for PlantDoc.** Reported PlantDoc numbers (60–90%) mix versions and augmentation practices. A clean protocol (official split, augmentation after split, multiple seeds, mean ± std, macro-F1) is itself a contribution.
*বাংলা:* PlantDoc-এ সঠিক, leakage-মুক্ত, একাধিক seed দিয়ে পুনরুৎপাদনযোগ্য ফলাফল — এটাই একটা অবদান।

**G3 — Lightweight cross-dataset generalization PlantDoc ↔ PlantWild.** Nobody trains a small model on one wild dataset and tests on the other using the overlapping classes (e.g., apple scab, corn rust, tomato early blight exist in both).
*বাংলা:* এক wild ডেটাসেটে ট্রেন করে অন্যটিতে (মিল থাকা ক্লাসগুলোতে) টেস্ট — হালকা মডেল দিয়ে কেউ করেনি।

**G4 — Quantitative explainability.** All papers show Grad-CAM *qualitatively*. PlantDoc has **leaf bounding boxes**, so you can measure *how much of the Grad-CAM energy falls inside the leaf box* (pointing-game / energy-in-box). This turns "the model looks at the leaf" into a number, and lets you prove whether background reduction actually changes where the model looks.
*বাংলা:* সবাই Grad-CAM ছবি দেখায়, কিন্তু কেউ মাপে না। PlantDoc-এ পাতার bounding box আছে — তাই Grad-CAM-এর কত শতাংশ পাতার ভিতরে পড়ছে তা সংখ্যায় মাপা যায়। এটা তোমাদের থিসিসের একটা শক্তিশালী ও নতুন দিক হতে পারে।

**G5 — Knowledge distillation from foundation models to mobile CNNs** on in-the-wild plant disease data (teacher: CLIP/DINOv2; student: MobileNetV3/EfficientNet-B0) has not been reported on PlantWild.
*বাংলা:* বড় মডেল থেকে ছোট মডেলে জ্ঞান স্থানান্তর (distillation) PlantWild-এ এখনও রিপোর্ট হয়নি।

**G6 — Honest efficiency reporting.** Few papers report params, FLOPs, model size, *and* measured CPU/phone latency together with macro-F1.
*বাংলা:* accuracy-র সাথে প্যারামিটার, FLOPs, সাইজ ও বাস্তব latency একসাথে খুব কম পেপারে আছে।

**G7 — Label noise handling.** PlantDoc authors and Salman et al. both report mislabeled/composite images, yet no lightweight study measures the effect of cleaning or noise-robust loss (e.g., label smoothing).
*বাংলা:* PlantDoc-এ ভুল লেবেল ও কোলাজ ছবি আছে — এগুলো পরিষ্কার করলে বা label smoothing দিলে ফল কতটা বদলায়, তা কেউ মাপেনি।

---

## 7. Recommendations for your thesis / থিসিসের জন্য পরামর্শ

1. **Sharpen the contribution** to: *"A leakage-free, multi-seed benchmark of lightweight CNNs on PlantDoc and PlantWild (official splits), with cross-dataset evaluation and quantitative Grad-CAM localization using PlantDoc bounding boxes."* Optional advanced step: CLIP/DINO → MobileNet distillation.
2. **Use PyTorch + timm** (one framework only), 224×224, ≥3 seeds, report mean ± std.
3. **Always report macro-F1** (PlantWild imbalance 44–589).
4. **Two-stage transfer**: ImageNet → PlantVillage → target (proven useful by PlantDoc paper).
5. **Compare fairly**: when you compare with Duhan (89.86%) or Salman (74%), state that protocols differ.
6. **Fix references**: correct years/DOIs listed in §8; drop the unverified 64.9% mAP unless confirmed.

**বাংলা।**
১. অবদানটা নির্দিষ্ট করো: "PlantDoc ও PlantWild-এ অফিসিয়াল split-এ, leakage-মুক্ত, একাধিক seed দিয়ে হালকা CNN-এর benchmark, cross-dataset মূল্যায়ন এবং bounding box দিয়ে Grad-CAM-এর সংখ্যাগত মূল্যায়ন।" চাইলে অতিরিক্ত ধাপ: বড় মডেল থেকে MobileNet-এ distillation।
২. শুধু একটা ফ্রেমওয়ার্ক (PyTorch + timm) ব্যবহার করো; অন্তত ৩টি seed চালিয়ে গড় ± বিচ্যুতি দাও।
৩. সবসময় macro-F1 দাও।
৪. ImageNet → PlantVillage → PlantDoc/PlantWild — দুই ধাপে transfer learning করো।
৫. অন্য পেপারের সাথে তুলনা করার সময় প্রোটোকলের পার্থক্য উল্লেখ করো।
৬. রেফারেন্স ঠিক করো (নিচে দেওয়া আছে)।

---

## 8. Verified reference list (corrected) / যাচাইকৃত রেফারেন্স তালিকা

1. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419. https://doi.org/10.3389/fpls.2016.01419
2. Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. In *Proc. 7th ACM IKDD CoDS & 25th COMAD* (pp. 249–253). https://doi.org/10.1145/3371158.3371196 (arXiv:1911.10317)
3. Wei, T., Chen, Z., Huang, Z., & Yu, X. (2024). Benchmarking in-the-wild multimodal plant disease recognition and a versatile baseline. In *Proc. 32nd ACM International Conference on Multimedia (MM '24)*. https://doi.org/10.1145/3664647.3680599 (arXiv:2408.03120)
4. Duhan, S., Gulia, P., Gill, N. S., Oliveira, T. A., & Singh, P. K. (2026). Evaluation of lightweight and efficient deep learning models for plant disease classification. *Discover Internet of Things*, 6, 43. https://doi.org/10.1007/s43926-026-00310-0
5. Krishna, M. S., Machado, P., Otuka, R. I., Yahaya, S. W., Neves dos Santos, F., & Ihianle, I. K. (2025). Plant leaf disease detection using deep learning: A multi-dataset approach. *J*, 8(1), 4. https://doi.org/10.3390/j8010004
6. Salman, Z., Muhammad, A., & Han, D. (2025). Plant disease classification in the wild using vision transformers and mixture of experts. *Frontiers in Plant Science*, 16, 1522985. https://doi.org/10.3389/fpls.2025.1522985
7. Zubair, F., Saleh, M., Akbari, Y., & Al Maadeed, S. (2025). A robust ensemble model for plant disease detection using deep learning architectures. *AgriEngineering*, 7(5), 159. https://doi.org/10.3390/agriengineering7050159
8. Miao, Y., Meng, W., & Zhou, X. (2025). SerpensGate-YOLOv8: An enhanced YOLOv8 model for accurate plant disease detection. *Frontiers in Plant Science*, 15, 1514832. https://doi.org/10.3389/fpls.2024.1514832
9. Bera, A., Bhattacharjee, D., & Krejcar, O. (2024). PND-Net: Plant nutrition deficiency and disease classification using graph convolutional network. *Scientific Reports*, 14, 15537. https://doi.org/10.1038/s41598-024-66543-7
10. Duhan, S., Gulia, P., Gill, N. S., & Narwal, E. (2025). RTR_Lite_MobileNetV2: A lightweight and efficient model for plant disease detection and classification. *Current Plant Biology*, 42, 100459. https://doi.org/10.1016/j.cpb.2025.100459
11. Kumar, A., Monga, H. P., Brahma, T., Kalra, S., & Sherif, N. (2025). Mobile-friendly deep learning for plant disease detection: A lightweight CNN benchmark across 101 classes of 33 crops. *arXiv:2508.10817* (preprint).
12. El Karch, H., Natij, Y., Benaly, M., El Gouri, R., & Mezouari, A. (2026). Backbone diversity beats text supervision: A systematic study of frozen multi-foundation model fusion for in-the-wild plant disease recognition. *Frontiers in Plant Science*. https://doi.org/10.3389/fpls.2026.1860665
13. Moupojou, E., et al. (2023). FieldPlant: A dataset of field plant images for plant disease detection and classification with deep learning. *IEEE Access*, 11, 35398–35410. https://doi.org/10.1109/ACCESS.2023.3263042

**Corrections vs. the proposal:** (a) PlantDoc is formally 2020 (CoDS-COMAD), arXiv 2019; (b) SerpensGate volume is 15 with a 2024 DOI though published Jan 2025; (c) "64.9% mAP" for SerpensGate is unverified; (d) Salman's 74% is on a cleaned PlantDoc subset; (e) Duhan's 89.86% uses a 28-class Kaggle version with class-balancing augmentation whose timing vs. the split is unclear; (f) add refs 1, 10, 11, 12, 13 — especially 11, the closest competing work.

**প্রপোজালের তুলনায় সংশোধন:** PlantDoc আনুষ্ঠানিকভাবে ২০২০ সালের; SerpensGate-এর ভলিউম ১৫; "৬৪.৯% mAP" যাচাই হয়নি; Salman-এর ৭৪% পরিষ্কার-করা subset-এ; Duhan-এর ৮৯.৮৬% ২৮-ক্লাসের Kaggle ভার্সনে ও augmentation-এর সময় অস্পষ্ট; নতুন রেফারেন্স ১, ১০, ১১, ১২, ১৩ যোগ করো — বিশেষ করে ১১ নম্বর, যেটা তোমাদের সবচেয়ে কাছের প্রতিযোগী কাজ।

---

### Suggested next step / পরবর্তী ধাপ
Step 2: download both datasets, run a data audit (class counts, duplicates across train/test, composite images, the class overlap table between PlantDoc and PlantWild), and lock the experimental protocol before any training.
ধাপ ২: দুটো ডেটাসেট ডাউনলোড করে ডেটা অডিট (ক্লাস অনুযায়ী সংখ্যা, train/test-এ ডুপ্লিকেট, কোলাজ ছবি, PlantDoc–PlantWild মিল থাকা ক্লাসের তালিকা) করা, এবং ট্রেনিং শুরুর আগে পরীক্ষার প্রোটোকল চূড়ান্ত করা।
