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
- génération déterministe de datasets Burgers avec métadonnées ;
- diagnostic séparé de perte de données et de résidu PDE Burgers ;
- baseline d'interpolation linéaire périodique pour le transfert de résolution ;
- baseline convolutionnelle locale périodique ajustée par moindres carrés ;
- référence DeepONet séparable à trunk Fourier fixe ;
- évaluation récursive multi-pas avec erreur, résidu et écarts de conservation ;
- partitions déterministes train/validation/OOD avec provenance des paramètres ;
- backend FNO PyTorch optionnel entraîné par autodifférentiation ;
- tests déterministes sur fonctions analytiques.

Le FNO NumPy/SciPy reste un oracle de recherche volontairement petit et lent,
entraîné par différences finies via SciPy. Le backend PyTorch est maintenant
disponible en option pour l'autodifférentiation ; il n'est pas ajouté aux
dépendances de base et aucune équivalence de performance n'est encore
revendiquée. La baseline linéaire n'est pas présentée comme un réseau neuronal.
Cette séparation conserve l'oracle numérique comme cible d'acceptation avant
un futur portage Rust.

## Exécution

Depuis la racine du workspace :

```bash
cd NeuralOperator
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/reference_experiment.py
PYTHONPATH=src python3 scripts/fno_experiment.py
PYTHONPATH=src python3 scripts/burgers_benchmark.py
PYTHONPATH=src python3 scripts/burgers_dataset_experiment.py
PYTHONPATH=src python3 scripts/burgers_fno_experiment.py
PYTHONPATH=src python3 scripts/burgers_pino_experiment.py
PYTHONPATH=src python3 scripts/burgers_resolution_baseline_experiment.py
PYTHONPATH=src python3 scripts/deeponet_experiment.py
PYTHONPATH=src python3 scripts/burgers_rollout_experiment.py
PYTHONPATH=src python3 scripts/burgers_split_experiment.py
```

Dépendances de base : NumPy et SciPy. Aucun téléchargement de données ni aucun
accès réseau n'est nécessaire pour ces vérifications.

Pour activer le backend autodiff optionnel :

```bash
python3 -m pip install -r requirements-autodiff.txt
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/torch_fno_experiment.py
```

Sans PyTorch, les tests optionnels sont ignorés et la suite NumPy/SciPy reste
exécutable.

## État du backend autodiff

Le backend optionnel `TorchFNO1D` reprend le contrat `(batch, points, channels)`
du FNO de référence et entraîne ses paramètres par autodifférentiation. Le
prochain contrôle est une comparaison sur les mêmes partitions train,
validation et OOD, avec plusieurs graines et métriques de rollout.

## Prochain jalon

Étendre le benchmark d'opérateur 1D périodique basé sur Burgers avec :

1. données générées par le solveur de référence contrôlé ;
2. baseline interpolation/conv1d ;
3. FNO entraînable sur données d'EDP ;
4. pertes données + résidu PDE ;
5. évaluation multi-résolution et stabilité du rollout.

Les résultats devront inclure l'erreur relative, le résidu physique, la
conservation pertinente, le coût d'entraînement amorti et le coût d'inférence.

L'expérience FNO est volontairement synthétique : elle vérifie le contrat
spectral et le transfert de résolution sur un opérateur linéaire à modes bas.
Elle ne constitue pas encore une validation sur une EDP ni une comparaison de
performance avec PyTorch.


## Licence

Le code source est disponible sous la [licence PolyForm Noncommercial 1.0.0](LICENSE.md). L’usage commercial n’est pas accordé par cette licence. Consultez [LICENSING.md](LICENSING.md) pour la voie de licence commerciale séparée.

Copyright © 2026 Tarek Zekriti.
