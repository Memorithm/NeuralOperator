# Neural Operator Lab

Socle de référence reproductible pour le programme Neural Operators.

Le workspace ne disposait pas encore d'un dépôt ou d'un modèle entraînable. Ce
premier lot fournit donc un oracle numérique indépendant d'un framework de
machine learning :

- différentiation spectrale périodique en 1D et 2D ;
- résidu de Burgers à partir d'une dérivée temporelle fournie ;
- construction d'un champ de vitesse incompressible à partir d'une fonction de
  courant ;
- masque et produit anti-aliasés selon la règle des `2/3` ;
- baseline d'opérateur de Fourier linéaire ajustée par moindres carrés ;
- FNO 1D minimal entraînable par SciPy pour validation de protocole ;
- solveur de référence Burgers 1D périodique par RK4 avec anti-aliasing ;
- solveur Darcy 2D à coefficient variable, stencil conservatif et bords de Dirichlet homogènes ;
- génération déterministe de champs de perméabilité log-normaux lisses pour Darcy ;
- FNO 2D PyTorch coordonné, avec mélange spectral complexe et contrainte Dirichlet forte ;
- DeepONet 2D non linéaire et baseline convolutionnelle locale à budget de paramètres rapproché ;
- résidu Darcy conservatif différentiable et entraînement FNO data+physique avec sweep du poids PINO ;
- étude multi-seed et sample-scaling pour FNO 2D, DeepONet 2D et convolution locale ;
- familles OOD Darcy contrôlées par contraste, rugosité spectrale et échelle moyenne de perméabilité ;
- génération déterministe de datasets Burgers avec métadonnées ;
- diagnostic séparé de perte de données et de résidu PDE Burgers ;
- baseline d'interpolation linéaire périodique pour le transfert de résolution ;
- baseline convolutionnelle locale périodique ajustée par moindres carrés ;
- référence DeepONet séparable à trunk Fourier fixe ;
- évaluation récursive multi-pas avec erreur, résidu et écarts de conservation ;
- partitions déterministes train/validation/OOD avec provenance des paramètres ;
- tests déterministes sur fonctions analytiques ;
- benchmark contrôlé PyTorch FNO/DeepONet avec budget de paramètres quasi identique,
  métriques validation/OOD/rollout et coût d'inférence ;
- résidu de Burgers spectral différentiable sous PyTorch et entraînement FNO
  data+physique dans le graphe d'autograd.

Le FNO NumPy/SciPy est un oracle de recherche volontairement petit et lent,
entraîné par différences finies via SciPy. Il ne remplace pas encore un backend
PyTorch ou Rust. La baseline linéaire n'est pas présentée comme un réseau
neuronal. Cette séparation est
volontaire : le socle numérique doit être validé avant de servir de cible à une
implémentation PyTorch puis Rust.

## Exécution

Depuis la racine du workspace :

```bash
cd NeuralOperator
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/reference_experiment.py
PYTHONPATH=src python3 scripts/fno_experiment.py
PYTHONPATH=src python3 scripts/burgers_benchmark.py
PYTHONPATH=src python3 scripts/darcy_benchmark.py
PYTHONPATH=src python3 scripts/torch_darcy_fno2d_experiment.py
PYTHONPATH=src python3 scripts/torch_darcy_operator_comparison.py
PYTHONPATH=src python3 scripts/torch_darcy_pino_experiment.py
PYTHONPATH=src python3 scripts/torch_darcy_multiseed_scaling.py
PYTHONPATH=src python3 scripts/torch_darcy_ood_families.py
PYTHONPATH=src python3 scripts/burgers_dataset_experiment.py
PYTHONPATH=src python3 scripts/burgers_fno_experiment.py
PYTHONPATH=src python3 scripts/burgers_pino_experiment.py
PYTHONPATH=src python3 scripts/burgers_resolution_baseline_experiment.py
PYTHONPATH=src python3 scripts/deeponet_experiment.py
PYTHONPATH=src python3 scripts/burgers_rollout_experiment.py
PYTHONPATH=src python3 scripts/burgers_split_experiment.py
PYTHONPATH=src python3 scripts/torch_operator_comparison.py
PYTHONPATH=src python3 scripts/torch_pino_experiment.py
```

Dépendances utilisées : NumPy et SciPy. Aucun téléchargement de données ni
aucun accès réseau n'est nécessaire pour ces vérifications.

## Prochain jalon

Le socle Burgers dispose désormais des baselines, du backend autodiff PyTorch,
de la comparaison FNO/DeepONet et d'un PINO de transition différentiable.
Le deuxième problème de vérité terrain est Darcy 2D : le dépôt contient un
solveur conservatif à coefficient variable, des jeux déterministes et un
benchmark de convergence analytique. Un premier FNO 2D PyTorch entraînable
apprend désormais le mapping perméabilité -> pression sur grille rectangulaire.

La suite technique est maintenant :

1. calibrer la pondération physique selon résolution et échelle du forcing ;
2. tester le scaling de résolution et la stabilité des conclusions multi-seed ;
3. étendre les familles OOD à l'anisotropie et aux géométries non rectangulaires ;
4. ajouter advection-diffusion avant Navier-Stokes 2D ;
5. ne porter vers Rust/SciRust que les noyaux acceptés par les oracles numériques.

Aucun avantage de performance ou d'invariance de résolution n'est revendiqué
pour Darcy tant que ces comparaisons n'ont pas été mesurées.

## Licence

Le code source est disponible sous la [licence PolyForm Noncommercial 1.0.0](LICENSE.md). L’usage commercial n’est pas accordé par cette licence. Consultez [LICENSING.md](LICENSING.md) pour la voie de licence commerciale séparée.

Copyright © 2026 Tarek Zekriti.
