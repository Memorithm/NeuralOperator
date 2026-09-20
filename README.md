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
- tests déterministes sur fonctions analytiques.

Ce n'est pas encore un FNO ni un PINO entraînable. La baseline linéaire n'est
pas présentée comme un réseau neuronal. Cette séparation est
volontaire : le socle numérique doit être validé avant de servir de cible à une
implémentation PyTorch puis Rust.

## Exécution

Depuis la racine du workspace :

```bash
cd NeuralOperator
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/reference_experiment.py
```

Dépendance utilisée : NumPy. Aucun téléchargement de données ni aucun accès
réseau n'est nécessaire pour ces vérifications.

## Prochain jalon

Ajouter un benchmark d'opérateur 1D périodique (Burgers) avec :

1. solveur de référence contrôlé ;
2. baseline interpolation/conv1d ;
3. FNO entraînable ;
4. pertes données + résidu PDE ;
5. évaluation multi-résolution et stabilité du rollout.

Les résultats devront inclure l'erreur relative, le résidu physique, la
conservation pertinente, le coût d'entraînement amorti et le coût d'inférence.
