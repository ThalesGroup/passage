# Style commun des trajectoires PASSAGE

| Auteur | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-10-17 | 1.0 |

## 1. Lire une figure

Les notebooks candidats, le papier et les images de présentation utilisent le
même profil. Le terrain va du vert sauge au sable. Les obstacles sont anthracite
avec des hachures claires. Un cercle blanc marque le départ ; une étoile blanche
marque le but. Les trajectoires ont un halo blanc pour rester visibles.

| Comparaison | Vert continu | Bleu pointillé court |
|---|---|---|
| Candidat contre référence | Route de référence | Sortie du candidat |
| Effet des obstacles | Route sur terrain libre | Route tenant compte des obstacles |

*Tableau 1 — Signification des deux tracés ; lire la légende de chaque figure.*

Pour comparer plusieurs coûts dans un même panneau, les couleurs bleu, orange
et violet peuvent distinguer altitude, pente et énergie, comme dans le papier.
La légende indique alors explicitement que la couleur représente le coût.
Dans la planche à six panneaux, les colonnes identifient ces coûts ; les couleurs
des lignes gardent le sens libre/obstacles du tableau 1.

## 2. Préserver les résultats

Le style conserve tous les sommets et leur ordre. Il ne résout aucun nouveau
chemin, ne lisse pas les trajectoires et ne remplace pas les échecs par des
tracés inventés. Une prédiction absente reste absente.

Les grandes cartes peuvent être affichées sur un aperçu de 512 cellules par
côté. Le sous-échantillonnage de l'altitude et l'agrégation prudente des obstacles
sont alors signalés dans la figure. Les coordonnées des routes restent natives.
Un rendu lisible ne constitue pas une nouvelle preuve d'optimalité.

## 3. Utiliser le même rendu

Le profil se trouve dans [trajectory_style.py](../scripts/trajectory_style.py).
Le [point d'entrée du papier](../../paper/scripts/figure_route_style.py) réutilise
ce module. Les notebooks candidats gardent leurs adaptateurs et scores figés ;
une couche de présentation distincte dessine leurs dix exemples avec ce profil.

Pour préparer les illustrations des sept résolutions, depuis `passage/` :

```bash
PYTHONPATH=src uv run --frozen python scripts/render_publication_previews.py
```

Les images, leurs sources et leurs empreintes sont préparées dans
[hf-presentation](../outputs/trajectory-style-v1-20261017/hf-presentation/).
Le script prend la première scène de validation de chaque résolution, sans
filtrage par qualité de route. Il ne lance ni solveur ni réseau.

> **Périmètre — présentation et publication**
>
> Ces fichiers sont des illustrations séparées. Les PNG intégrés aux Parquet
> déjà qualifiés restent intacts. Les remplacer demanderait une nouvelle version
> des artefacts et les audits correspondants. Aucun téléversement n'est effectué.
