## Current checkpoint — 23 September 2026 (tensor learned baseline)

The full-SPD truth oracle is now connected to the three existing 2D learned
operator references without architecture-specific tensor preprocessing. FNO 2D,
DeepONet 2D and the local convolution baseline consume the three physical input
channels `K_xx`, `K_xy`, and `K_yy`. A dedicated evaluator reports data
error, relative L2 error, exact boundary violation and the Q1 algebraic residual
MSE.

Parameter budgets are matched within 3%: 8,379 parameters for DeepONet, 8,465
for FNO and 8,625 for the local convolution baseline. The first deterministic
protocol trains on ratio-4 SPD tensors and evaluates independent IID samples,
an orientation-family shift and a principal-ratio shift to 16.

The corrected implementation at
`ef219ad3bf20309b03e3010ccbc0035d58d800e2` passed the standard suite
(85 tests, with optional PyTorch tests skipped there) and was independently
qualified through Memorithm/RemoteOps run 35819149284 on NVIDIA Thor. The
targeted PyTorch/tensor suite passed 6/6 tests under NumPy 2.5.3, SciPy 1.18.1
and PyTorch 2.14.0+cu130 with CUDA enabled. The tensor learned benchmark artifact
has SHA-256
`2de99a4bd64e922227616d1ad6b92336031b1bb9db75f68130306ac0e2f1371e`.

For this single-seed protocol, FNO relative L2 is 0.1640 IID, 0.6498 under the
orientation shift and 0.7307 at principal ratio 16. DeepONet is 0.2353, 0.2265
and 0.9465 respectively. The local convolution baseline is 0.2384, 1.4753 and
1.1339. These are descriptive measurements of one controlled run, not an
architecture ranking or a universal performance claim.

The next evidence gate is multi-seed replication with fixed evaluation cohorts
and mean/sample-standard-deviation/min/max summaries. Tensor-specific
normalization or equivariance changes should be evaluated only after that
replication, followed by non-rectangular geometry, mixed/Neumann boundaries and
discontinuous facies.

---

## Previous checkpoint — 23 September 2026 (full-SPD truth oracle)

The full-SPD Darcy truth path is now implemented and qualified on Thor through
Memorithm/RemoteOps. The reference solves `-div(K grad(u))=f` for nodal
symmetric positive-definite 2x2 tensor fields with Q1 finite elements and 2x2
Gauss quadrature. This removes the previous grid-aligned restriction:
`K_xy` may be non-zero and the principal-axis orientation may vary spatially.

The deterministic dataset stores three channels (`K_xx`, `K_xy`, `K_yy`).
It preserves a controlled principal-value ratio while the existing scalar
log-permeability field controls determinant scale and a separate seeded smooth
field controls orientation.

RemoteOps run 35818381772 qualified commit
`7c918dcde8e43f90130f25221c3ecf2f476cf135` on
`thor-remoteops-arm64-01`: all five tensor tests passed. For the constant
rotated tensor [[2, 0.6], [0.6, 1]], relative L2 error decreased from
`1.3094e-2` at 9x9 to `2.0670e-4` at 65x65 with observed orders
`1.9888`, `1.9972`, and `1.9993`. The largest reported algebraic residual
was `3.083e-15`.

Spatially oriented families with principal ratios 4 and 16 produced non-zero
cross terms while preserving the eigenvalue ratio to approximately
`3.11e-15` and `4.62e-14` maximum absolute error respectively. The largest
algebraic residual across those generated families was below `1.5e-16`.

This tranche establishes truth-solver correctness, not learned-operator
superiority. The next gate is now a matched-budget three-channel learned
comparison on rotated/spatial tensor OOD families, followed by
non-rectangular geometry, mixed/Neumann boundary conditions and discontinuous
facies.

---

## Previous checkpoint — 23 September 2026 (diagonal anisotropy)

This checkpoint supersedes the 21 September snapshot below.

The diagonal-anisotropic Darcy truth path is now implemented and qualified.
The conservative stencil solves

`-d_x(k_x d_x u) - d_y(k_y d_y u) = f`

with strictly positive diagonal coefficients, harmonic face means and
homogeneous Dirichlet boundaries. The deterministic dataset represents the
tensor field with two input channels and constructs a controlled family with
`k_x/k_y = r` while preserving the scalar geometric scale
`sqrt(k_x k_y)`.

The qualification includes exact reduction to the scalar Darcy solver when
`k_x = k_y`, deterministic dataset reproduction, invalid-input guards,
second-order analytic convergence and a versioned JSON benchmark. On the
self-hosted ARM64 qualification run, all five anisotropic tests passed. The
isotropic-reduction maximum solution difference was exactly `0.0`. The
relative L2 errors at 9, 17, 33 and 65 points were respectively
`1.2950746721879309e-2`, `3.218964440080193e-3`,
`8.035776793720459e-4` and `2.0082180969406666e-4`, with observed orders
`2.0084`, `2.0021` and `2.0005`.

For controlled anisotropy ratios `0.25`, `1`, `4` and `16`, the
pointwise ratio error was exactly zero and the largest measured absolute
discrete residual was `4.218847493575595e-14`. These are truth-solver
properties only: no learned-operator advantage is claimed by this tranche.

The global permeability-scale equivariance tranche was separately qualified on
the same ARM64 control plane with the optional PyTorch backend: all 26
`test_torch_*.py` tests passed under Python 3.12.3, NumPy 2.5.3,
SciPy 1.18.1 and PyTorch 2.14.0+cu130 with CUDA available. For the FNO2D
benchmark, the mean relative L2 error on the `mean_high_extreme` family was
`2.03551979245677` in raw mode and `0.10035770485973827` with exact scale
equivariance; IID behavior was unchanged to floating-point precision.

The next gate is no longer basic diagonal anisotropy. It is rotated/spatially
varying anisotropy, non-rectangular geometry and boundary-condition shifts,
then discontinuous facies. Learned FNO/DeepONet/local-convolution comparisons
must follow the truth-solver extension instead of preceding it.

---

## Previous checkpoint — 21 September 2026

This checkpoint supersedes the 20 September snapshot below.

Since that checkpoint, the repository has advanced through the optional PyTorch
autodiff FNO, NumPy/PyTorch parity checks, a nonlinear PyTorch DeepONet
reference, a controlled FNO/DeepONet Burgers comparison, and PR #20's
differentiable spectral transition-PINO training path. The transition PINO
still uses a finite temporal transition quotient; it is not a continuous-time
collocation residual.

The current Darcy tranche now contains both the conservative 2D truth solver
and a first trainable PyTorch FNO 2D. The model appends coordinates, mixes
complex low Fourier modes, accepts new grid resolutions without changing
weights, and imposes homogeneous Dirichlet boundaries with a hard analytic
output envelope.

The benchmark now also contains matched-budget 2D comparators: a nonlinear
DeepONet using the 9x9 permeability grid as branch sensors and a local
coordinate-aware convolution baseline. The reference parameter counts are
8,449 (FNO), 8,649 (DeepONet) and 8,672 (local convolution), with identical
training samples, validation/OOD cohorts and Adam step counts.

Training wall time is recorded separately because equal optimizer steps do not
imply equal compute. A differentiable conservative Darcy residual now mirrors
the SciPy truth stencil inside PyTorch autograd, and the FNO training path
reports data, physics and weighted total losses separately.

The first PINO experiment is a controlled physics-weight sensitivity sweep with
an explicit zero-weight data-only control. The matched-budget architecture
comparison is now also repeated over three training-set sizes and three paired
replicates, with raw runs plus mean/sample-standard-deviation summaries.

The Darcy generator now exposes a provenance-preserving spectral-decay
parameter. The OOD benchmark evaluates fixed families that separately shift
coefficient contrast, spectral roughness/smoothness and mean permeability,
with three model/training replicates and shared evaluation cohorts.

The same Darcy PINO weight sweep is now repeated at 9x9 and 17x17 with paired
initialization seeds, a fixed architecture and identical weight candidates.
This measures whether the data/physics trade-off is resolution-sensitive
instead of assuming one global physics coefficient.

The coefficient-family benchmark exposed a distinct weakness under global
permeability-scale shifts. For scalar Darcy with fixed forcing and homogeneous
Dirichlet pressure, this symmetry is exact: k=c*k_hat implies u=u_hat/c.
The pipeline now factors the per-sample geometric-mean coefficient scale,
trains on scale-normalized permeability/pressure, and restores physical
pressure analytically without adding trainable parameters.

The next gate is anisotropic coefficients and non-rectangular/boundary-condition
shifts, followed by discontinuous facies. If those remain controlled, the
program can move to advection-diffusion with the same evidence protocol.

---
## Previous checkpoint — 20 September 2026

Since the initial roadmap was written, the following repository changes are merged:

- PR #11: deterministic recursive Burgers rollout evaluation with per-step error, finite-horizon physics residual, and mean/energy mismatch metrics.
- PR #12: explicit train/validation/OOD partition specifications with shared-grid and shared-horizon checks.
- PR #13: PolyForm Noncommercial 1.0.0 license, copyright Required Notice, and commercial licensing guide.

These changes do not establish a production performance advantage, continuous-time PINO validity, or resolution invariance. The next technical gate remains an optional autodiff backend retained against the NumPy/SciPy oracle, followed by a nonlinear DeepONet and larger PDE families.

---
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

La baseline `PeriodicConv1D` dans `baselines.py` ajuste aussi un stencil local
périodique par moindres carrés. Elle fournit un point de comparaison
convolutionnel transparent, mais ses poids sont liés à la grille discrète et
ne constituent donc pas une garantie de transfert de résolution. Sur le petit
cas Burgers actuel, elle atteint environ `0,060 %` d'erreur relative à 32
points, contre `1,87 %` pour le FNO évalué sur l'entrée fine exacte : ce
contre-résultat indique que l'horizon et la famille de données sont trop
simples pour attribuer un avantage au mélange spectral.

Reste à faire dans M2 : remplacer l'optimisation par différences finies par un
backend autodiff et ajouter DeepONet, avec une séparation stricte des familles
de paramètres et plusieurs graines.

Une première référence `LinearDeepONet1D` est maintenant disponible dans
`deeponet.py`. Elle sépare explicitement le branch (capteurs de la fonction)
et le trunk (coordonnées de requête), et vérifie la sortie à 16 puis 32 points.
Le trunk est fixe et Fourier, et le branch est linéaire : cette étape valide le
contrat de coordonnées mais ne remplace pas encore un DeepONet non linéaire
entraîné par autodiff.

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

Le deuxième problème de vérité terrain est maintenant initialisé avec Darcy 2D.
`darcy.py` implémente un stencil de flux conservatif à coefficient variable avec
moyennes harmoniques aux faces et conditions de Dirichlet homogènes.
`darcy_datasets.py` produit des champs de perméabilité positifs et déterministes,
et `scripts/darcy_benchmark.py` mesure convergence analytique et résidu discret.
Un premier opérateur 2D entraînable est désormais inclus dans torch_fno2d.py, avec benchmark IID/OOD/multi-résolution versionné.

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
