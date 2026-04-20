# Rapport d'evaluation RETFound DR

Chaque ligne correspond a un checkpoint fine-tune sur le dataset source, puis evalue sur le dataset cible **complet, fusion train, val et test**.

| Source checkpoint | Dataset cible | N images | Accuracy | AUROC | F1 macro | Kappa | Composite |
|---|---|---:|---:|---:|---:|---:|---:|
| APTOS2019 | APTOS2019 | 3662 | 0.8441 | 0.9593 | 0.6596 | 0.7574 | 0.7921 |
| APTOS2019 | IDRiD_data | 516 | 0.5078 | 0.8252 | 0.3622 | 0.3183 | 0.5019 |
| APTOS2019 | MESSIDOR2 | 1744 | 0.5940 | 0.7509 | 0.3571 | 0.2452 | 0.4510 |
| IDRiD_data | APTOS2019 | 3662 | 0.4511 | 0.7572 | 0.3249 | 0.2299 | 0.4373 |
| IDRiD_data | IDRiD_data | 516 | 0.7132 | 0.9208 | 0.5760 | 0.5996 | 0.6988 |
| IDRiD_data | MESSIDOR2 | 1744 | 0.5109 | 0.8178 | 0.3218 | 0.2373 | 0.4590 |
| MESSIDOR2 | APTOS2019 | 3662 | 0.7070 | 0.8227 | 0.4829 | 0.5513 | 0.6190 |
| MESSIDOR2 | IDRiD_data | 516 | 0.6105 | 0.8629 | 0.4507 | 0.4468 | 0.5868 |
| MESSIDOR2 | MESSIDOR2 | 1744 | 0.7328 | 0.9051 | 0.6205 | 0.5073 | 0.6776 |



Chaque ligne correspond a un checkpoint fine-tune sur le dataset source, puis evalue sur le dataset cible avec **seulement test**.

| Source checkpoint | Dataset cible | N images | Accuracy | AUROC | F1 macro | Kappa | Composite |
|---|---|---:|---:|---:|---:|---:|---:|
| APTOS2019 | APTOS2019 | 1100 | 0.8218 | 0.9447 | 0.6223 | 0.7237 | 0.7636 |
| APTOS2019 | IDRiD_data | 103 | 0.4369 | 0.8057 | 0.3386 | 0.2505 | 0.4650 |
| APTOS2019 | MESSIDOR2 | 526 | 0.5932 | 0.7255 | 0.3911 | 0.2381 | 0.4516 |
| IDRiD_data | APTOS2019 | 1100 | 0.4427 | 0.7476 | 0.3200 | 0.2207 | 0.4295 |
| IDRiD_data | IDRiD_data | 103 | 0.5146 | 0.8248 | 0.4075 | 0.3225 | 0.5182 |
| IDRiD_data | MESSIDOR2 | 526 | 0.5247 | 0.8297 | 0.3218 | 0.2522 | 0.4679 |
| MESSIDOR2 | APTOS2019 | 1100 | 0.6973 | 0.8184 | 0.4732 | 0.5394 | 0.6103 |
| MESSIDOR2 | IDRiD_data | 103 | 0.5146 | 0.8081 | 0.3829 | 0.3146 | 0.5019 |
| MESSIDOR2 | MESSIDOR2 | 526 | 0.6977 | 0.8827 | 0.5936 | 0.4447 | 0.6403 |

## Comparaison avec résultat partagé
|Dataset| AUC partagé | AUC obtenu |
|---|---|---:|
| APTOS2019| 0.943|0.945|
| IDRiD|0.822|0.825|
|MESSIDOR-2|0.884|0.883|

On retrouve bien les résultats attendus.



# Analyse des Performances de Généralisation des Checkpoints Fine-Tunés

Le checkpoint fine-tuné sur **MESSIDOR2** est le plus robuste en validation externe, même s’il n’est pas le meilleur sur sa diagonale interne. Il généralise mieux aux autres datasets que les checkpoints **APTOS2019** et **IDRiD**, avec un composite externe moyen d’environ **0.603**, contre **0.477** pour APTOS2019 et **0.448** pour IDRiD.


## Lecture Synthétique des Résats

### Meilleure performance externe

| Source     | Cible       | Accuracy | AUROC  | F1 macro | Kappa  | Composite |
|------------|------------|----------|--------|----------|--------|-----------|
| MESSIDOR2  | APTOS2019  | 0.7070   | 0.8227 | 0.4829   | 0.5513 | 0.6190    |

### Deuxième meilleure performance externe

| Source     | Cible        | Accuracy | AUROC  | F1 macro | Kappa  | Composite |
|------------|-------------|----------|--------|----------|--------|-----------|
| MESSIDOR2  | IDRiD_data  | 0.6105   | 0.8629 | 0.4507   | 0.4468 | 0.5868    |

Cela indique que **MESSIDOR2 fournit le fine-tuning le plus transférable** parmi les trois sources testées.

- **APTOS2019** : excellente performance interne (composite = 0.7921), mais chute hors domaine  
- **IDRiD** : encore plus fragile, probablement à cause de sa petite taille (516 images)

---


## Comparaison des Performances Externes

| Checkpoint source | Composite externe moyen | AUROC externe moyen | F1 macro externe moyen | Kappa externe moyen |
|-------------------|------------------------|---------------------|------------------------|---------------------|
| MESSIDOR2         | 0.6029                 | 0.8428              | 0.4668                 | 0.4990              |
| APTOS2019         | 0.4765                 | 0.7880              | 0.3596                 | 0.2817              |
| IDRiD_data        | 0.4482                 | 0.7875              | 0.3234                 | 0.2336              |

### Observation clé

- **MESSIDOR2** : très stable  
  - Kappa : **0.5073 → 0.4990**
- **APTOS2019** : forte chute  
  - Kappa : **0.7574 → 0.2817**

Indique une **sur-adaptation au domaine source**

---

## Interprétation Clinique des Erreurs

Le modèle distingue correctement :
- **Absence de DR (Grade 0)**
- **DR modérée (Grade 2)**

Mais échoue sur :
- classes intermédiaires
- classes rares

---

## Déséquilibre des Classes

| Dataset      | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 |
|-------------|--------|--------|--------|--------|--------|
| APTOS2019   | 1805   | 370    | 999    | 193    | 295    |
| IDRiD_data  | 168    | 25     | 168    | 93     | 62     |
| MESSIDOR2   | 1017   | 270    | 347    | 75     | 35     |

---

## Faiblesse Majeure : Grade 1 (DR légère)

| Paire                        | F1 Grade 1 |
|-----------------------------|------------|
| APTOS2019 → MESSIDOR2       | 0.000      |
| IDRiD_data → APTOS2019      | 0.000      |
| IDRiD_data → MESSIDOR2      | 0.000      |
| MESSIDOR2 → APTOS2019       | 0.004      |
| MESSIDOR2 → IDRiD_data      | 0.000      |

Le modèle :
- ne détecte quasiment jamais la DR légère
- confond souvent :
  - Grade 1 → Grade 0
  - Grade 1 → Grade 2

---


## Synthèse Finale

**MESSIDOR2 est le meilleur dataset source pour la généralisation dans cette expérience.**

Cependant :

- le modèle reste sensible :
  - au décalage de domaine
  - au déséquilibre des classes
- la détection de la **DR légère est un point critique non résolu**
