---
id: INDEX
title: Pestivid potato RAG corpus - index, coverage and dose policy
source_name: Pestivid corpus audit
source_url: https://ppqs.gov.in/divisions/cib-rc/registered-products
retrieved: 2026-08-22
jurisdiction: India
conditions: [Bacteria, Fungi, Healthy, Nematode, Pest, Phytophthora, Virus]
contains_doses: false
confidence: high
---

# Pestivid potato retrieval corpus

24 documents, 14,633 words of body text. Audited 22 August 2026.
Every source_url in the corpus was fetched on that date and returned HTTP 200.

## THE DOSE POLICY - read this before using any document

This corpus does not tell a farmer how much of anything to spray.

- No pesticide dose, spray rate, dilution, seed-treatment concentration, dip
  time or pre-harvest interval in days is recorded anywhere.
- A dose may be recorded only as a direct quotation whose exact source is named,
  and only when that source names both the formulation and the crop. It is then
  marked contains_doses: true and labelled in the body as quoted, not as this
  platform's recommendation.
- No dose may be inferred, averaged, converted, rounded, or given as a "typical
  range".
- Exactly one document carries a quoted figure. potato-virus-no-cure-prevention-only
  quotes ICAR-CPRI's equipment-cleaning strengths (3% trisodium phosphate, 1%
  calcium hypochlorite) for disinfecting tools against contact-spread viruses.
  Nothing is applied to the crop, so there is no residue or waiting-period
  question. Verified verbatim against the source PDF.
- Doses were available in nearly every source and were deliberately dropped:
  bleaching powder per hectare and mancozeb dips (NHB), mercury and boric acid
  dips (TNAU and NHB), gramoxone per hectare, imidacloprid and thiamethoxam
  percentages, carbofuran and furadan per hectare (ICAR-CPRI), diquat desiccant
  (AHDB), and the entire dose column of the CIB and RC Major Uses of Pesticides
  fungicide list.
- When a user asks "how much do I spray", the answer this corpus supports is the
  CIB and RC register, the product's own approved label, or a Krishi Vigyan
  Kendra. Three documents exist to teach exactly that:
  cibrc-register-how-to-check-a-pesticide, read-a-pesticide-label-india, and
  pre-harvest-interval-and-residues. Retrieval should pair them with the
  platform's withheld-dose message, never treat them as a dosage answer.

## The documents

| id | title | conditions | source | quoted doses |
|---|---|---|---|---|
| aphids-whitefly-virus-vectors | Aphids and whitefly on potato - direct damage and their role as virus vectors | Pest, Virus | ICAR-CPRI Shimla, Model Training Course on Disease and Pest Management in Potato, 2016 | no |
| bacterial-wilt-brown-rot-potato | Bacterial wilt and brown rot of potato, the glass-of-water test, and why rotation is the real control | Bacteria, Nematode | CIP, Integrated Control of Bacterial Wilt of Potato; NHB (GoI) | no |
| cibrc-register-how-to-check-a-pesticide | The CIB and RC Register and How to Check a Pesticide Yourself | all 6 disease classes | CIB and RC / Directorate of Plant Protection, Quarantine and Storage (GoI) | no |
| early-blight-potato-identification | Early blight of potato, and how to tell it from late blight in the field | Fungi, Phytophthora | NHB (GoI); TNAU Agritech Portal | no |
| india-seed-potato-system-and-seed-plot-technique | India's seed potato system - certified seed, the seed plot technique, sourcing clean seed | Virus, Healthy | NAAS Strategy Paper 14 (2021); ICAR-CPRI | no |
| late-blight-haulm-destruction-and-harvest | Cutting affected foliage, killing the haulm, and harvesting a blighted crop | Phytophthora | AHDB / Potato Council (GB), Managing the Risk of Late Blight | no |
| late-blight-identification-potato | Recognising late blight on potato, and telling it apart from early blight | Phytophthora, Fungi | TNAU Agritech Portal | no |
| late-blight-non-chemical-field-practices | Stopping late blight without chemicals - seed, planting, spacing and water | Phytophthora | Lal, Sharma, Yadav and Kumar (ICAR-CPRI), IntechOpen 2018 | no |
| late-blight-resistant-varieties-india | Indian potato varieties rated resistant to late blight, and who released them | Phytophthora, Nematode | ICAR-CPRI Technical Bulletin No. 78 (Revised) | no |
| late-blight-weather-and-forecasting-india | The weather that drives late blight, and how it is forecast in India | Phytophthora | ICAR-CPRI (Sanjeev Sharma), Status of Late Blight Management in India | no |
| nutrient-deficiency-and-abiotic-mimics-potato | Things that look like potato disease but are not - nutrients, frost, heat, water, drift | Healthy, Fungi, Phytophthora, Virus, Bacteria | UNECE Guide to Seed Potato Diseases, Pests and Defects (hosted by TNAU) | no |
| pesticide-ppe-and-safe-handling | Protective Gear, Safe Mixing and First Aid for Pesticide Users | all 6 disease classes | DPPQS (GoI) and TNAU, with Insecticides Rules 1971 and AIIMS NPIC | no |
| potato-irrigation-and-drainage | Irrigation and drainage in potato - how water decides disease and tuber quality | Phytophthora, Fungi, Bacteria, Healthy | AHDB, Seasonal Water Management for Potatoes (2nd ed.), with UC IPM and TNAU | no |
| potato-nematodes-cyst-and-root-knot | Potato cyst nematode and root-knot nematode - why you need a soil test, not a photo | Nematode | ICAR-CPRI Shimla | no |
| potato-scab-black-scurf-dry-rot | Common scab, black scurf and dry rot, and the cultural controls that work | Fungi, Bacteria | NHB (GoI); TNAU Agritech Portal | no |
| potato-seasons-and-regions-india | Potato seasons and growing regions of India, and why the same symptom means different things | Phytophthora, Fungi, Virus, Pest, Bacteria | ICAR-CPRI (Bhardwaj et al., Life, 2022) | no |
| potato-seed-and-storage-sanitation | How potato disease travels in seed tubers, and how to handle and store them | Bacteria, Fungi, Phytophthora, Virus, Pest | NHB (GoI); NAAS; CIP | no |
| potato-tuber-moth | Potato tuber moth (Phthorimaea operculella) in field and store | Pest | ICAR-CPRI, A Manual on Potato Tuber Moth | no |
| potato-virus-no-cure-prevention-only | Why a potato virus cannot be cured - and why the money goes on prevention | Virus, Pest | ICAR-CPRI Shimla | YES - one equipment-cleaning quotation |
| potato-virus-symptoms-and-lookalikes | Potato viruses in India - PVY, PVX, PVA, leaf roll, apical leaf curl, and their lookalikes | Virus, Pest | ICAR-CPRI Shimla | no |
| pre-harvest-interval-and-residues | Pre-Harvest Interval, Residues and Maximum Residue Limits | all 6 disease classes | CIB and RC, and FSSAI | no |
| read-a-pesticide-label-india | How to Read a Pesticide Label in India | all 6 disease classes | Insecticides Rules 1971 and Insecticides Act 1968, with DPPQS | no |
| scouting-and-thresholds-potato | Scouting a potato field - deciding whether an infestation is worth acting on | Pest, Nematode, Virus | NIPHM and DPPQS (GoI), AESA based IPM Package - Potato | no |
| where-a-farmer-gets-real-help-india | Where a potato farmer in India gets real help - phone numbers, KVKs, soil labs, portals | all 7 | mKisan Portal, DA and FW (GoI) | no |

"all 6 disease classes" means Bacteria, Fungi, Nematode, Pest, Phytophthora and
Virus. Those five cross-cutting documents are tagged broadly because they apply
to any chemical decision. They contain no diagnostic content and should not be
counted as coverage of a condition.

## What the corpus covers

- Phytophthora (late blight) is the strongest topic, with six documents:
  identification, the weather and India's own forecasting models, non-chemical
  field practice, haulm destruction and harvest, resistant varieties with
  ICAR-CPRI's per-variety ratings, plus the early-blight differential.
- Virus has three documents plus the vector document: symptoms and lookalikes,
  why there is no cure, and India's certified seed chain and seed plot technique.
- Pest has potato tuber moth, aphids and whitefly as vectors, and how to scout a
  field and decide whether acting is worth it.
- Nematode has one document covering cyst and root-knot nematode, soil sampling,
  the Nilgiris quarantine and rotation.
- Bacteria has bacterial wilt and brown rot, including the glass-of-water
  vascular flow test a farmer can run themselves.
- Fungi has early blight, common scab, black scurf and dry rot.
- Healthy is served indirectly, by the abiotic-mimics document that rules
  disease out.
- Cross-cutting: label reading, the CIB and RC register, pre-harvest interval and
  MRLs, PPE and first aid, seasons and regions, irrigation and drainage, seed and
  storage sanitation, and where to get real help.

## What the corpus does NOT cover

Retrieval will come back empty or thin on all of these.

- Any dose, for anything. By policy. This corpus cannot answer "how much".
- Which products are legally registered on potato in India. No CIB and RC
  potato-specific registered list was transcribed. The corpus teaches the lookup
  but does not hold the answer.
- Soft rot and blackleg (Pectobacterium and Erwinia). Named as a consequence in
  five documents - after late blight, after waterlogging, after tuber moth,
  after harvest damage - and described in none. The largest single hole in the
  Bacteria class.
- Wart (Synchytrium endobioticum), powdery scab, charcoal rot, Sclerotium
  rolfsii tuber rot, and Fusarium and Verticillium wilt. The bacterial-wilt
  document tells a farmer that a non-streaming wilt is "probably Fusarium,
  Verticillium, or root damage", and then the corpus says nothing about any of
  them.
- White grub, cutworm (Agrotis), Epilachna beetle, mites, thrips, leaf miner and
  flea beetle. All appear in the scouting counting list with no identification
  document behind them. Several are locally serious in India.
- Per-pest economic thresholds beyond aphids and the two nematodes. ICAR-CPRI
  publishes a table, but its columns cannot be read reliably from the PDF, so
  those pairings were deliberately not reproduced.
- What a healthy crop looks like, stage by stage. The Healthy classifier class
  has no positive-case document.
- Potato stem necrosis virus (thrips-borne) and PSTVd symptoms. Apical leaf curl
  gets one paragraph despite 40 to 70 per cent reported incidence in the
  Indo-Gangetic plains.
- Tuta absoluta on potato, an expanding problem in India.
- State-level extension advice. No PAU, ANGRAU or UAS Dharwad bulletin was
  retrievable, so TNAU is the only state university source. No state agriculture
  department phone directory. No potato variety released by any body other than
  ICAR-CPRI surfaced.
- The live text of the Nilgiris potato cyst nematode quarantine. The corpus
  describes the practice from ICAR-CPRI's historical account, not the current
  notification.
- Photographs. No image references, which limits how well these documents can
  support the image classifier at the moment of diagnosis.
- A Telugu or Hindi glossary. Roguing, dehaulming, haulm, earthing up,
  lenticels, rugosity, SMD and vinekill have no everyday equivalent and will
  machine-translate badly. Roguing in particular is the corpus's single most
  important preventive action.

## Known defects to fix (audit, 22 August 2026)

1. potato-scab-black-scurf-dry-rot says methoxy ethyl mercuric chloride is on the
   banned list. In the cited PDF it sits under restricted, not banned - and that
   PDF is a 2011 snapshot. Re-cite the live ppqs.gov.in banned and restricted
   list (as on 31.07.2026) and correct the status.
2. late-blight-weather-and-forecasting-india states precise JHULSACAST and
   INDO-BLIGHTCAST thresholds (50 hours at 85% RH, 100 hours in 7.2 to 26.6 C,
   52.5 P-days, 525 RH units) that are not in any of the eight URLs it lists.
   Source them or remove them. The file is marked confidence: high.
3. india-seed-potato-system-and-seed-plot-technique gives seed cost as "40 to 50
   per cent" (ICAR-CPRI says about 50 per cent, NAAS says over one-third) and
   breeder seed as "2,600 to 3,500 tonnes", a range built from three different
   sources. Quote one source instead of synthesising a range.
4. bacterial-wilt-brown-rot-potato adds finger millet to CIP's rotation list.
   CIP names wheat and maize. Harmless agronomically, but unsourced.
5. potato-nematodes-cyst-and-root-knot dates Kufri Neelima to 2012. ICAR-CPRI
   Technical Bulletin 78 and ICAR Indian Horticulture both say 2010, as does
   late-blight-resistant-varieties-india. Internal contradiction.
6. Kufri Abhedya is stated as notified "in 2026"; the cited Tribune article says
   only that it has been notified. Re-source from ICAR or the Gazette.
7. late-blight-weather-and-forecasting-india cites a 2019 conference
   presentation as though it were a bulletin, and the URL serves a text sidecar
   rather than the slide deck itself. Label it as a presentation, 2019.
8. Mechanical: the potato tuber moth source URL contains literal parentheses,
   which breaks bare-URL parsing, and the CIB and RC document has markdown bold
   fused directly onto a bare URL.
9. Fifteen of 24 bodies exceed the 600-word target, the longest at 746. Nothing
   reads as padded, but the chunks are long.
10. Semantic duplication. aphids-whitefly-virus-vectors and
    potato-virus-no-cure-prevention-only repeat the same five management actions
    from the same source. The aphid threshold of 20 per 100 compound leaves
    appears in three documents. The early-versus-late-blight discriminator
    appears in two. Three UNECE quotations appear in both
    nutrient-deficiency-and-abiotic-mimics-potato and
    potato-irrigation-and-drainage.

## Contact details, verified 22 August 2026

- Kisan Call Centre 1800-180-1551. Verified on mkisan.gov.in and
  dackkms.gov.in. 6.00 am to 10.00 pm, all seven days, 22 local languages.
- National Poisons Information Centre, AIIMS New Delhi: 1800 116 117. Verified
  on aiims.edu, 24 hours a day.
- PMFBY crop loss and grievance: 14447. Verified on pmfby.gov.in, with the
  72-hour intimation deadline verified in the Revamped Operational Guidelines.
- soilhealth.dac.gov.in, kisansuvidha.gov.in and ppqs.gov.in all resolve.
- kvk.icar.gov.in does not resolve. Use Kisan Suvidha's Find KVK, or call
  1800-180-1551. The corpus already routes around this.

After the dose policy, these are the most safety-critical facts in the corpus.
Re-verify them, and the CIB and RC banned list, on a schedule. A stale banned
list or a wrong helpline number is itself a hazard.
