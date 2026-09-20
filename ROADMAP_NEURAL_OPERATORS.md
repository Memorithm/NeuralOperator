# Bilan et feuille de route — Neural Operators

Date de l'audit : 20 septembre 2026.

## 1. État constaté

La pièce `discussion-gemini.txt` est une note théorique de 347 lignes portant
sur :

- le lien possible entre théorie quantique des champs et apprentissage profond ;
- DeepONet ;
- Fourier Neural Operator (FNO) ;
- Physics-Informed Neural Operator (PINO) ;
- différentiation spectrale des résidus d'EDP.

Elle ne contient ni dépôt logiciel, ni jeu de données, ni modèle entraîné, ni
mesure reproductible. Le workspace contient Python 3.12, NumPy et SciPy ; Rust,
Cargo et PyTorch ne sont pas disponibles dans cet environnement.

Le socle initial implémenté dans `neural_operator_lab/` valide les opérations
numériques qui seront réutilisées par le FNO/PINO. Le complément M1 compare
désormais la convergence spectrale à une différence finie périodique sur une
fonction lisse.
Le lot M2 contient maintenant un FNO 1D minimal entraînable par SciPy et une
expérience de transfert de résolution sur un opérateur synthétique à modes bas.

## 2. Évaluation scientifique de la note

### Ce qui est solide

- L'idée centrale est correcte : un neural operator vise une application entre
  espaces de fonctions et non seulement entre vecteurs de dimension fixe.
- FNO paramètre une convolution intégrale dans l'espace de Fourier et a été
  évalué sur plusieurs familles d'EDP dans l'article fondateur.
- DeepONet sépare bien l'encodage de la fonction d'entrée (branch) et celui du
  point de requête (trunk).
- PINO combine apprentissage d'opérateur et contraintes provenant de l'EDP.
- La relation `F(∂ⁿu) = (ik)ⁿ F(u)` est le bon point de départ pour une grille
  périodique régulière.

### Ce qui doit être corrigé ou mesuré

| Affirmation de la note | Statut | Formulation de travail retenue |
| --- | --- | --- |
| « Invariance à la résolution » | Conditionnelle | Elle doit être testée sur des résolutions et domaines explicitement choisis. Un FNO ne garantit pas une généralisation hors distribution, ni des bords non périodiques sans traitement adapté. |
| Gain `10³–10⁵` | Non généralisable | Le papier FNO rapporte jusqu'à trois ordres de grandeur dans ses expériences ; tout chiffre propre au projet devra inclure le solveur, la précision, le matériel, la tolérance, le coût d'entraînement amorti et la stabilité du rollout. |
| Différentiation spectrale « exacte » | Trop forte | Elle est algébriquement exacte pour la représentation de Fourier discrète, mais la dérivée de la fonction continue est approchée par la grille, et les discontinuités, bords et aliasing peuvent dominer l'erreur. |
| Faible coût par rapport à Autograd | Dépend du cas | Le coût doit être benchmarké sur la même architecture, taille de champ, ordre de dérivée, mémoire et matériel. La FFT coûte `O(N log N)` ; ce n'est pas automatiquement plus rapide dans tous les régimes. |
| PINO sans aucune donnée | Possible dans certains cas | Le résidu PDE seul ne suffit pas à garantir une solution unique : conditions initiales, limites, paramètres et identifiabilité doivent être imposés et testés. |
| Potentiel vecteur ⇒ incompressibilité | Valable sous hypothèses | La construction doit respecter la dimension, la régularité, la convention de signe et les conditions aux limites. Elle est ajoutée au socle comme invariant testable, pas comme slogan général. |
| FourCastNet = FNO standard | À préciser | FourCastNet utilise une architecture Adaptive Fourier Neural Operator (AFNO), ce qui doit être distingué d'un FNO minimal. |
| TQC/RG comme avantage immédiat | Hypothèse de recherche | Les analogies RG, limites `1/N` et diagrammes peuvent guider une étude théorique, mais elles ne sont pas un composant logiciel prioritaire tant qu'une hypothèse falsifiable et une mesure n'ont pas été définies. |

## 3. Objectif du programme

Construire un moteur d'opérateurs neuronaux vérifiable, capable d'apprendre des
familles d'EDP et de démontrer, par mesures reproductibles, dans quelles
conditions il apporte un avantage sur un solveur numérique et sur des baselines
neurales simples.

La priorité n'est pas de maximiser une promesse de vitesse isolée. La cible est
un compromis documenté entre :

1. erreur de solution et erreur hors distribution ;
2. résidu physique et respect des conditions aux limites ;
3. stabilité des prédictions à plusieurs pas ;
4. coût mémoire, latence et débit ;
5. portabilité grille régulière, grille non uniforme et domaine non périodique.

## 4. Travaux restants, par lots

### Lot M0 — Spécification et reproductibilité — en cours

- fixer une licence et une politique de provenance lorsque le dépôt cible sera
  choisi ;
- définir les métriques et les tolérances avant les expériences ;
- conserver les versions des solveurs, bibliothèques, pilotes et jeux de
  données ;
- séparer clairement faits publiés, hypothèses et résultats propres au projet.

### Lot M1 — Socle numérique — premier lot implémenté

- dérivées spectrales 1D/2D ;
- Laplacien, divergence et construction incompressible ;
- résidu de Burgers ;
- masque anti-aliasing `2/3` ;
- tests analytiques et cas invalides.

Reste à faire dans M1 :

- validation des bords non périodiques par une méthode distincte ;
- comparaison avec différences finies et dérivation automatique lorsque le
  backend ML sera disponible.

La convergence sur plusieurs tailles de grille et la comparaison avec la
différence finie périodique sont implémentées dans
`scripts/convergence_experiment.py`. La comparaison avec Autograd reste
explicitement reportée jusqu'à l'ajout d'un backend ML.

### Lot M2 — Baselines d'apprentissage

- FNO 1D minimal avec couche spectrale, transformation locale, activation et
  projection ;
- entraînement de référence par différences finies via SciPy ;
- séparation entraînement / retenu / résolution double sur un opérateur
  synthétique contrôlé ;
- DeepONet sur un opérateur 1D simple ;
- baseline convolutionnelle et baseline interpolation ;
- protocole de séparation des familles de paramètres, pas seulement des points
  de grille ;
- entraînement déterministe avec plusieurs graines.

Le FNO minimal et son script sont dans `src/neural_operator_reference/fno1d.py`
et `scripts/fno_experiment.py`. Sur l'expérience actuelle, la MSE passe de
`0,3998` à `3,76×10^-6` sur l'entraînement ; la résolution doublée atteint une
MSE de `1,29×10^-4` et une erreur relative de `1,45 %`. Ces chiffres ne sont
pas extrapolés à une EDP ou à un autre régime.

Une baseline non neuronale de rééchantillonnage linéaire périodique est ajoutée
dans `baselines.py`. Le script
`scripts/burgers_resolution_baseline_experiment.py` compare l'interpolation de
la sortie FNO grossière à une évaluation directe du même FNO sur une entrée
rééchantillonnée fine. La baseline ne prétend pas apprendre l'opérateur : elle
mesure le coût de ne faire que transporter une prédiction déjà calculée.

Reste à faire dans M2 : remplacer l'optimisation par différences finies par un
backend autodiff, ajouter DeepONet, puis comparer avec une baseline
convolutionnelle et une baseline interpolation.

Le code de référence `neuraloperator` est utile pour comparer les résultats,
mais il ne doit pas être confondu avec une implémentation Memorithm. Sa
compatibilité actuelle et son coût devront être gelés dans l'environnement de
benchmark.

### Lot M3 — Jeux d'EDP et solveurs de vérité terrain

Le solveur Burgers 1D périodique de référence est maintenant implémenté dans
`src/neural_operator_reference/burgers.py`. Il utilise RK4, le produit
anti-aliasé et des contrôles de moyenne et d'énergie. La génération de datasets
déterministe est maintenant disponible dans `datasets.py`, et un premier FNO
est entraîné sur Burgers dans `scripts/burgers_fno_experiment.py`.

Ordre recommandé :

1. Burgers 1D périodique : test de base et résidu non linéaire ;
2. Darcy 2D : elliptique et conditions aux limites ;
3. advection-diffusion : robustesse et changement de régime ;
4. Navier–Stokes 2D : rollout long, turbulence et conservation.

Chaque dataset doit enregistrer l'équation, les paramètres, la résolution, les
conditions initiales/limites, la méthode numérique et la tolérance du solveur.

Le premier cas Burgers est contrôlé mais volontairement petit : 4 trajectoires
à 16 points pour l'apprentissage, 2 trajectoires retenues à 16 points et une
évaluation exploratoire à 32 points. L'erreur relative observée à 32 points est
environ `6,9 %`; ce résultat montre que le transfert de résolution n'est pas
automatiquement garanti.

### Lot M4 — PINO et invariants physiques

Le diagnostic de transition Burgers et un entraînement pondéré données/PDE de
référence sont maintenant disponibles dans
`physics.py`. Il sépare l'erreur de données du résidu PDE, mais il ne constitue
pas encore une perte PINO complète : la dérivée temporelle est ici une
approximation entre deux états. `training.py` expose ce protocole pour mesurer
le compromis, sans le présenter comme une solution continue.

- perte de données `L_data` ;
- perte PDE `L_pde` ;
- perte conditions initiales/limites `L_bc` ;
- pondération et suivi séparé de chaque terme ;
- contraintes fortes quand elles sont mathématiquement justifiées ;
- détection des solutions triviales ou non identifiables ;
- vérification des quantités conservées à chaque rollout.

La différentiation spectrale ne sera activée que lorsque la grille, les bords et
les hypothèses de régularité l'autorisent. Sinon, il faudra un opérateur adapté
(Chebyshev, bases polynomiales, méthode intégrale, grille arbitraire ou autre)
et un benchmark séparé.

### Lot M5 — Résolution, géométrie et généralisation

- entraînement à résolution grossière puis évaluation fine ;
- maillages non uniformes ;
- domaines non rectangulaires ;
- conditions de Dirichlet, Neumann et mixtes ;
- tests OOD sur amplitude, viscosité, forcing et géométrie ;
- bornes d'erreur empiriques et détection de dérive.

Ce lot est indispensable avant d'employer le terme « résolution-invariant ».

### Lot M6 — Performance et portage système

- benchmark CPU puis GPU ;
- latence p50/p95, débit, mémoire et énergie si mesurable ;
- coût total : génération des données + entraînement + inférences ;
- comparaison avec solveurs classiques à précision comparable ;
- export d'un format de poids documenté ;
- noyaux FFT, tenseurs et parallélisme dans SciRust/NNIS après validation de
  l'oracle Python ;
- vérification bitwise ou tolérée entre référence et backend Rust.

### Lot M7 — Industrialisation et intégration

- API de prédiction avec version de modèle et métadonnées d'équation ;
- validation avant déploiement : erreur, résidu, domaine de validité ;
- arrêt sûr lorsqu'une entrée sort du domaine entraîné ;
- journaux d'expériences et artefacts reproductibles ;
- connexion éventuelle à Forge pour rechercher architectures/hyperparamètres,
  sans laisser la recherche remplacer les preuves numériques ;
- documentation et tests CI.

## 5. Critères de sortie

Un jalon ne sera considéré comme terminé que si :

- les tests passent sur une machine propre ;
- les métriques sont produites par un script versionné ;
- les résultats incluent au moins une baseline non neuronale et une baseline
  neuronale ;
- les gains annoncés sont accompagnés de leur protocole ;
- l'erreur, le résidu PDE, la stabilité et le coût sont tous rapportés ;
- les limitations et les cas d'échec sont documentés.

## 6. Sources primaires consultées

- Zongyi Li et al. (2020), [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895).
- Zongyi Li et al. (2021), [Physics-Informed Neural Operator for Learning Partial Differential Equations](https://arxiv.org/abs/2111.03794).
- Lu, Jin et Karniadakis (2019), [DeepONet: Learning nonlinear operators...](https://arxiv.org/abs/1910.03193).
- Pathak et al. (2022), [FourCastNet: ... Adaptive Fourier Neural Operators](https://arxiv.org/abs/2202.11214).
- Lingsch et al. (2023), [Beyond Regular Grids: Fourier-Based Neural Operators on Arbitrary Domains](https://arxiv.org/abs/2305.19663).
- Liu et al. (2023), [SPFNO: Spectral operator learning for PDEs with Dirichlet and Neumann boundary conditions](https://arxiv.org/abs/2312.06980).
- [NeuralOperator — implementation PyTorch de référence](https://github.com/neuraloperator/neuraloperator), dépôt consulté le 20 septembre 2026.
