# Premier noyau Rust qualifiable — B-NO-01

`neural-operator-rust` porte uniquement le résidu Darcy isotrope conservatif
`darcy_residual_2d` de l'oracle Python, observé au commit
`fcd34147c9623ffd8ea49dbb7894e0516fb88be0`. Les moyennes harmoniques aux faces,
le stencil à cinq points, l'ordre des axes, les longueurs et les valeurs aux
bords fournies sont conservés. Le code Python/NumPy/SciPy et tous les modèles
PyTorch restent les références de recherche indépendantes ; aucun n'est remplacé.

L'API Rust est fallible, safe, sans dépendance externe, MSRV 1.89. Elle accepte
des tranches empruntées de taille exacte, une force scalaire ou un champ complet.
Le budget de points, opérations et octets de sortie est vérifié avant allocation
et scan ; un plafond indépendant de 1 million de points reste obligatoire.
Un dépassement, une forme erronée, k≤0, une entrée non finie, un espacement
sous-flux/overflow ou un résultat non fini est rejeté sans résultat partiel ni
mutation des entrées. Les autres broadcasts Python ne sont pas implicitement
acceptés. L'allocation préalable des entrées appartient à l'appelant.

## Protocole figé avant exécution

32 cas : quatre grilles (3×3, 5×7, 9×9, 17×29), quatre familles déterministes,
deux états (fourni et résolu par `solve_darcy_2d` SciPy). Aucun dataset/holdout de
modèle existant n'est consommé. La tolérance est rtol=1e-12, atol=1e-10, annoncée
avant observation. Les flux NumPy de l'oracle original sont indépendants de la
portée Rust ; les champs d'entrée communs ne calculent aucun résultat de référence.

La qualification vérifie aussi les comptes structuraux exacts : une sortie
f64 par cellule intérieure et `4 + 30 * cellules_intérieures` opérations
arithmétiques scalaires du stencil/espacements. Validation, indexing, allocation
et instructions machine sont exclus de ce compte ; il ne signifie pas FLOPS
mesurés ou RSS. Chaque appel Rust inclut ces coûts lors d'une mesure de 5 warmups
et 31 répétitions, avec médiane et p95 en ns sur le runner CPU effectivement
utilisé. Ces timings ne sont pas une supériorité Python, un coût GPU ou une
référence stable de performance hardware.

Commandes : `cargo test --all-targets --locked`, `cargo clippy --all-targets
--locked -- -D warnings`, `cargo build --release --frozen --example darcy_probe`,
puis `PYTHONPATH=src python scripts/qualify_rust_darcy.py --binary
target/release/examples/darcy_probe --output <dossier-isolé>`.

Le workflow épinglé conserve le SHA exact, toolchain/target, locks, hash binaire,
hash oracle/corpus, les 32 entrées, sorties indépendantes et métriques de coût.
Il faut tous les checks habituels et ce gate verts sur HEAD, puis requalifier la
version fusionnée. Un résultat prévu n'est jamais annoncé comme exécuté.

## Limites de convergence

Ce n'est pas le solveur Darcy, le tenseur SPD anisotrope, FNO/DeepONet/PINO,
l'autograd, une intégration complète SciRust ou la portabilité GPU. La sémantique
du résidu/protocole reste chez NeuralOperator. Les futurs solveurs/primitives
génériques doivent réutiliser leurs propriétaires SciRust ; cette première
tranche ne crée ni ordonnanceur ni schéma de preuve concurrent. Toute migration
suivante doit conserver sa propre méthode et ses résultats de parité/cost.
