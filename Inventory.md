# Inventaire des datasets

## Vue d'ensemble

| Dataset | Type de données | Tâche principale | Images locales | Classes | Splits |
|---|---:|---|---:|---:|---|
| `APTOS2019` | Fond d'oeil | Rétinopathie diabétique, 5 grades | 3 662 | 5 | train / val / test |
| `IDRiD_data` | Fond d'oeil | Rétinopathie diabétique, 5 grades | 516 | 5 | train / val / test |
| `MESSIDOR2` | Fond d'oeil | Rétinopathie diabétique, 5 grades | 1 744 | 5 | train / val / test |
| `Glaucoma_fundus` | Fond d'oeil | Glaucome, 3 niveaux | 1 544 | 3 | train / val / test |
| `PAPILA` | Fond d'oeil | Glaucome, normal / suspect / glaucome | 488 | 3 | train / val / test |
| `Retina` | Fond d'oeil | Classification multi-pathologies | 601 | 4 | train / val / test |
| `OCTID` | OCT rétinien | Classification de pathologies OCT | 572 | 5 | train / val / test |
| `JSIEC` | Fond d'oeil | Classification multi-pathologies, 39 classes | 1 000 | 39 | train / val / test |

Total local: **10 127 images** réparties dans **8 datasets**.

## Convention des classes DR

Les datasets de rétinopathie diabétique utilisent une classification en 5 niveaux. Les noms de dossiers changent légèrement selon le dataset, mais la logique est la même.

| Grade | Sens clinique | Noms de classes locaux |
|---:|---|---|
| 0 | Pas de rétinopathie diabétique | `anoDR`, `anodr` |
| 1 | Rétinopathie légère | `bmildDR`, `bmilddr` |
| 2 | Rétinopathie modérée | `cmoderateDR`, `cmoderatedr` |
| 3 | Rétinopathie sévère | `dsevereDR`, `dseveredr` |
| 4 | Rétinopathie proliférante | `eproDR`, `eproliferativedr` |

Les préfixes `a`, `b`, `c`, `d`, `e` semblent conserver l'ordre des classes dans les dossiers.

## Rétinopathie Diabétique

### `APTOS2019`

- **Type**: images de fond d'oeil.
- **Usage local**: classification de la sévérité de la rétinopathie diabétique.
- **Images locales**: 3 662.
- **Répartition locale**: train 2 048, val 514, test 1 100.
- **Classes locales**: `anodr`, `bmilddr`, `cmoderatedr`, `dseveredr`, `eproliferativedr`.
- **Provenance**: images issues d'APTOS 2019 Blindness Detection, compétition Kaggle organisée autour de la détection de la cécité liée à la rétinopathie diabétique. Le `readme.txt` local indique que les images sont prétraitées avec AutoMorph.
- **Source**: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
- **Prétraitement indiqué localement**: https://github.com/rmaphoh/AutoMorph

### `IDRiD_data`

- **Type**: images de fond d'oeil.
- **Usage local**: classification de la sévérité de la rétinopathie diabétique.
- **Images locales**: 516.
- **Répartition locale**: train 329, val 84, test 103.
- **Classes locales**: `anoDR`, `bmildDR`, `cmoderateDR`, `dsevereDR`, `eproDR`.
- **Provenance**: Indian Diabetic Retinopathy Image Dataset. Le dataset original contient des images de fond d'oeil, des grades de rétinopathie diabétique et d'oedème maculaire diabétique, ainsi que des annotations de lésions pour une partie des images.
- **Source officielle indiquée localement**: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
- **Article descriptif**: https://www.mdpi.com/2306-5729/3/3/25
- **Prétraitement indiqué localement**: https://github.com/rmaphoh/AutoMorph

### `MESSIDOR2`

- **Type**: images de fond d'oeil.
- **Usage local**: classification de la sévérité de la rétinopathie diabétique.
- **Images locales**: 1 744.
- **Répartition locale**: train 972, val 246, test 526.
- **Classes locales**: `anodr`, `bmilddr`, `cmoderatedr`, `dseveredr`, `eproliferativedr`.
- **Provenance**: Messidor-2 est une base de rétinographies centrées macula pour l'étude de la rétinopathie diabétique. La page ADCIS décrit 874 examens et 1 748 images dans le jeu original; le dossier local en contient 1 744 après préparation.
- **Source officielle indiquée localement**: https://www.adcis.net/en/third-party/messidor2/

## Glaucome

### `Glaucoma_fundus`

- **Type**: images de fond d'oeil.
- **Usage local**: classification du glaucome en contrôle normal, glaucome précoce et glaucome avancé.
- **Images locales**: 1 544.
- **Répartition locale**: train 861, val 218, test 465.
- **Classes locales**: `anormal_control`, `bearly_glaucoma`, `cadvanced_glaucoma`.
- **Provenance**: dataset Glaucoma Fundus publié sur Harvard Dataverse. Le `readme.txt` local indique que les images sont prétraitées.
- **Source officielle indiquée localement**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1YRRAC
- **Article associé**: https://pmc.ncbi.nlm.nih.gov/articles/PMC6258525/

### `PAPILA`

- **Type**: images de fond d'oeil centrées sur la papille, avec données cliniques dans le dataset original.
- **Usage local**: classification normal / suspect glaucome / glaucome.
- **Images locales**: 488.
- **Répartition locale**: train 311, val 79, test 98.
- **Classes locales**: `anormal`, `bsuspectglaucoma`, `cglaucoma`.
- **Provenance**: PAPILA est un dataset public avec images de fond d'oeil et données cliniques des deux yeux de patients, destiné à l'évaluation du glaucome. L'article Scientific Data indique 244 patients et des labels clinique non-glaucomateux, suspect et glaucomateux.
- **Source officielle indiquée localement**: https://figshare.com/articles/dataset/PAPILA/14798004/1
- **Article descriptif**: https://www.nature.com/articles/s41597-022-01388-1

## Multi-pathologies Fond D'oeil

### `Retina`

- **Type**: images de fond d'oeil.
- **Usage local**: classification entre normal, cataracte, glaucome et maladie rétinienne.
- **Images locales**: 601.
- **Répartition locale**: train 336, val 84, test 181.
- **Classes locales**: `anormal`, `bcataract`, `cglaucoma`, `ddretina_disease`.
- **Provenance**: le `readme.txt` local pointe vers le dataset Kaggle `cataractdataset` et indique un prétraitement AutoMorph. Cette provenance doit être considérée comme une source de distribution Kaggle plutôt qu'une source clinique primaire parfaitement documentée.
- **Source indiquée localement**: https://www.kaggle.com/datasets/jr2ngb/cataractdataset
- **Prétraitement indiqué localement**: https://github.com/rmaphoh/AutoMorph

### `JSIEC`

- **Type**: images de fond d'oeil.
- **Usage local**: classification fine de 39 maladies ou conditions du fond d'oeil.
- **Images locales**: 1 000.
- **Répartition locale**: train 534, val 150, test 316.
- **Classes locales**:
  - `0.0.Normal`
  - `0.1.Tessellated fundus`
  - `0.2.Large optic cup`
  - `0.3.DR1`
  - `1.0.DR2`
  - `1.1.DR3`
  - `2.0.BRVO`
  - `2.1.CRVO`
  - `3.RAO`
  - `4.Rhegmatogenous RD`
  - `5.0.CSCR`
  - `5.1.VKH disease`
  - `6.Maculopathy`
  - `7.ERM`
  - `8.MH`
  - `9.Pathological myopia`
  - `10.0.Possible glaucoma`
  - `10.1.Optic atrophy`
  - `11.Severe hypertensive retinopathy`
  - `12.Disc swelling and elevation`
  - `13.Dragged Disc`
  - `14.Congenital disc abnormality`
  - `15.0.Retinitis pigmentosa`
  - `15.1.Bietti crystalline dystrophy`
  - `16.Peripheral retinal degeneration and break`
  - `17.Myelinated nerve fiber`
  - `18.Vitreous particles`
  - `19.Fundus neoplasm`
  - `20.Massive hard exudates`
  - `21.Yellow-white spots-flecks`
  - `22.Cotton-wool spots`
  - `23.Vessel tortuosity`
  - `24.Chorioretinal atrophy-coloboma`
  - `25.Preretinal hemorrhage`
  - `26.Fibrosis`
  - `27.Laser Spots`
  - `28.Silicon oil in eye`
  - `29.0.Blur fundus without PDR`
  - `29.1.Blur fundus with suspected PDR`
- **Provenance**: jeu de 1 000 images de fond d'oeil avec 39 catégories, publié par le Joint Shantou International Eye Centre (JSIEC) sur Zenodo. Le dataset est associé à des travaux de classification automatique de maladies du fond d'oeil.
- **Source officielle indiquée localement**: https://zenodo.org/records/3477553
- **Article associé**: https://www.nature.com/articles/s41467-021-25138-w

## OCT Rétinien

### `OCTID`

- **Type**: images OCT rétiniennes.
- **Usage local**: classification de pathologies sur OCT.
- **Images locales**: 572.
- **Répartition locale**: train 316, val 82, test 174.
- **Classes locales**: `ANormal`, `ARMD`, `CSR`, `Diabetic_retinopathy`, `Macular_Hole`.
- **Provenance**: Optical Coherence Tomography Image Retinal Database, hébergé par Borealis / University of Waterloo Dataverse. La base originale contient des images OCT haute résolution classées en conditions pathologiques comme normal, macular hole, AMD/ARMD, CSR et rétinopathie diabétique.
- **Source officielle indiquée localement**: https://borealisdata.ca/dataverse/OCTID
- **Article descriptif**: https://doi.org/10.1016/j.compeleceng.2019.106532
