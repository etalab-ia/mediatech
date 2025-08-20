# MEDIATECH

[![License](https://img.shields.io/github/license/etalab-ia/mediatech?label=licence&color=red)](https://github.com/etalab-ia/mediatech/blob/main/LICENSE)
[![English version](https://img.shields.io/badge/🇬🇧-English%20version-blue)](../README.md)
[![Collection Hugging Face ](https://img.shields.io/badge/🤗-Collection%20Hugging%20Face-yellow)](https://huggingface.co/collections/AgentPublic/mediatech-68309e15729011f49ef505e8)

## 📝 Description

Ce projet traite les données publiques mises à disposition par divers administrations afin de faciliter l'accès à des données publiques vectorisées et prêtes à l'emploi pour des usages IA dans la fonction publique.
Il inclut des scripts pour télécharger, traiter, embedder, insérer ces données dans une base PostgreSQL et faciliter leur exportation via divers moyens.

## 💡 Tutoriel

### 𖣘 Méthode 1 : Airflow

#### Installation et configuration des dépendances

1.  Exécutez le script de déploiement initial :
    ```bash
    sudo chmod +x ./scripts/initial_deployment.sh
    ./scripts/initial_deployment.sh
    ```

2.  Configurez les variables d'environnement dans un fichier [`.env`](.env) en vous basant sur l'exemple contenu dans [`.env.example`](.env.example).
    > La variable `AIRFLOW_UID` doit être obtenue en exécutant :
    ```bash
    echo $(id -u)
    ```
    > La variable `JWT_TOKEN` sera obtenue plus tard en utilisant l'API d'Airflow. Laissez-la vide pour le moment.

#### Initialisation des conteneurs Airflow et PostgreSQL (PgVector)

1.  Exécutez le script [`containers_deployment.sh`](./scripts/containers_deployment.sh) :
    ```bash
    sudo chmod +x ./scripts/containers_deployment.sh
    ./scripts/containers_deployment.sh
    ```

2.  Exportez les variables du fichier [`.env`](.env) :
    ```bash
    export $(grep -v '^#' .env | xargs)
    ```

3. Assurez-vous de supprimer le volume PostgreSQL (PgVector) :
    ```bash
    docker compose down -v
    ```
    > ⚠️ Cette opération supprimera tous les volumes !

4.  Utilisez l'API d'Airflow pour obtenir la variable `JWT_TOKEN` :
    ```bash
    curl -X 'POST' \
    'http://localhost:8080/auth/token' \
    -H 'Content-Type: application/json' \
    -d "{\"username\": \"${_AIRFLOW_WWW_USER_USERNAME}\", \"password\": \"${_AIRFLOW_WWW_USER_PASSWORD}\"}"
    ```

5.  Définissez la variable `JWT_TOKEN` dans le fichier [`.env`](.env) avec le `access_token` obtenu.

#### Téléchargement, Traitement et Téléversement des Données

Vous êtes maintenant prêt à utiliser Airflow et à exécuter les DAGs disponibles.
Chaque jeu de données a son propre DAG et un [`DAG_Controller`](./airflow/dags/dag_controller.py) est défini pour gérer tous les DAGs de jeux de données et leur ordre d'exécution.

### </> Méthode 2 : Utiliser le CLI en local

#### Installation des dépendances

1.  Installez les dépendances apt requises :
    ```bash
    sudo apt-get update
    sudo apt-get install -y $(cat config/requirements-apt-container.txt)
    ```

2.  Créez et activez un environnement virtuel :
    ```bash
    python3 -m venv .venv  # Créer l'environnement virtuel
    source .venv/bin/activate  # Activer l'environnement virtuel
    ```

3.  Installez les dépendances python requises :
    ```bash
    pip install -e .
    ```

> L'installation en mode développement (`-e`) permet d'utiliser la commande `mediatech` et de modifier le code sans réinstallation.

> **Note :** Assurez-vous que l'environnement est correctement configuré avant de continuer.

#### Configuration de la base de données PostgreSQL (PgVector)

1.  Configurez les variables d'environnement dans un fichier [`.env`](.env) en vous basant sur l'exemple contenu dans [`.env.example`](.env.example).

2.  Démarrez le conteneur PostgreSQL avec Docker :
    ```bash
    docker compose up -d postgres
    ```

3.  Vérifiez que le conteneur `pgvector_container` est en cours d'exécution :
    ```bash
    docker ps
    ```

#### Téléchargement, Traitement et Téléversement des Données

##### Utilisation de la commande `mediatech`

Après installation, la commande `mediatech` est disponible globalement et remplace `python main.py` :

> Si vous rencontrez des soucis avec la commande `mediatech`, il reste tout de même possible d'utiliser la commande `python main.py` à la place.

Le fichier [`main.py`](main.py) est le point d'entrée principal du projet et propose une interface en ligne de commande (CLI) pour exécuter chaque étape du pipeline séparément.
Vous pouvez l'utiliser ainsi :

```bash
mediatech <commande> [options]
```
ou

```bash
python main.py <commande> [options]
```

Exemples de commandes :
- Voir l'aide :
  ```bash
  mediatech --help
  ```
- Créer les tables PostgreSQL:
  ```bash
  mediatech create_tables --model BAAI/bge-m3
  ```
- Télécharger tous les fichiers répertoriés dans [`data_config.json`](config/data_config.json):
  ```bash
  mediatech download_files --all
  ```
- Télécharger les fichiers de la source `service_public` :
  ```bash
  mediatech download_files --source service_public
  ```
- Télécharger et traiter tous les fichiers répertoriés dans [`data_config.json`](config/data_config.json):
  ```bash
  mediatech download_and_process_files --all --model BAAI/bge-m3
  ```
- Traiter toutes les données :
  ```bash
  mediatech process_files --all --model BAAI/bge-m3
  ```
- Diviser une table en sous-tables basées sur différents critères (cf: [`main.py`](main.py)) :
  ```bash
  mediatech split_table --source legi
  ```
- Exporter les tables PostgreSQL en fichier parquet:
  ```bash
  mediatech export_tables --output data/parquet
  ```
- Téléverser les datasets en format parquet sur le repository Hugging Face:
  ```bash
  mediatech upload_dataset --input data/parquet/service_public.parquet --dataset-name service-public
  ```

Exécutez `mediatech --help` dans votre terminal pour voir toutes les options disponibles, ou consultez directement le code contenu dans [`main.py`](main.py).

##### Utilisation alternative avec `python main.py`

Si vous préférez utiliser directement le script Python, vous pouvez toujours utiliser :

```bash
python main.py <commande> [options]
```

Exemples :
```bash
python main.py download_files
python main.py create_tables --model BAAI/bge-m3
python main.py process_files --all --model BAAI/bge-m3
```
##### Utilisation du script [`update.sh`](update.sh)

Le script [`update.sh`](update.sh) permet d'exécuter l'ensemble du pipeline de traitement des données : téléchargement, création des tables, vectorisation et export.
Pour l'exécuter, lancez la commande suivante depuis la racine du projet :

```bash
./scripts/update.sh
```

Ce script va :
- Attendre que la base PostgreSQL soit disponible,
- Créer ou mettre à jour les tables nécessaires dans la base PostgreSQL,
- Télécharger les fichiers publics répertoriés dans [`data_config.json`](config/data_config.json),
- Traiter et vectoriser les données,
- Exporter les tables au format Parquet,
- Téléverser les fichiers Parquet sur [Hugging Face](https://huggingface.co/AgentPublic).

## 🗂️ Structure du projet

- **[`main.py`](main.py)** : Point d'entrée principal pour exécuter le pipeline complet via un CLI.
- **[`pyproject.toml`](pyproject.toml)** : Configuration du projet Python et des dépendances.
- **[`Dockerfile`](Dockerfile)** : Définit les instructions pour construire l'image Docker personnalisée pour Airflow, en installant les dépendances système, les paquets Python et en configurant l'environnement du projet.
- **[`docker-compose.yml`](docker-compose.yml)** : Orchestre la configuration multi-conteneurs, définit les services Airflow et la base de données PostgreSQL (PgVector).
- **[`.github/`](.github/)** : Contient les workflows GitHub Actions pour l'Intégration Continue et le Déploiement Continu (CI/CD), automatisant les tests et les processus de déploiement.
- **[`download_and_processing/`](download_and_processing/)** : Contient les scripts pour télécharger et extraire les fichiers.
- **[`database/`](database/)** : Contient les scripts pour gérer la base de données (création de tables, insertion de données).
- **[`utils/`](utils/)** : Contient des fonctions utilitaires partagées entre les différents modules.
- **[`config/`](config/)** : Contient les scripts de configuration du projet.
- **[`logs/`](logs/)** : Contient les fichiers journaux pour suivre l'exécution des [scripts](scripts/).
- **[`scripts/`](scripts/)** : Contient l'ensemble des scripts shell, exécutés soit automatiquement, soit manuellement dans certains cas.
  - **[`scripts/update.sh`](scripts/update.sh)** : Script shell pour exécuter l'ensemble du pipeline de traitement des données.
  - **[`scripts/periodic_update.sh`](scripts/periodic_update.sh)** : Script shell pour automatiser le pipeline sur la machine virtuelle. Ce script est exécuté périodiquement par [`cron_config.txt`](cron_config.txt).
  - **[`scripts/backup.sh`](scripts/backup.sh)** : Script shell pour sauvegarder le volume Pgvector (PostgreSQL) et certains fichiers de configuration. Ce script est exécuté périodiquement par [`cron_config.txt`](cron_config.txt).
  - **[`scripts/restore.sh`](scripts/restore.sh)** : Script shell pour restaurer le volume Pgvector (PostgreSQL) et les fichiers de configuration si nécessaire.
  - **[`scripts/initial_deployment.sh`](scripts/initial_deployment.sh)**: Met en place un nouvel environnement serveur en installant Docker, Docker Compose et d'autres dépendances système.
  - **[`scripts/containers_deployment.sh`](scripts/containers_deployment.sh)**: Gère le cycle de vie de l'application en construisant, initialisant et déployant les conteneurs Docker tels que définis dans [docker-compose.yml](docker-compose.yml). Il doit être exécuté après chaque mise à jour du CLI Mediatech ou d'un autre script non partagé avec le conteneur Airflow.
  - **[`scripts/check_running_dags.sh`](scripts/check_running_dags.sh)**: Vérifie l'API d'Airflow pour voir si des pipelines de données (DAGs) sont en cours d'exécution, utilisé pour verrouiller en toute sécurité le processus de déploiement.
  - **[`scripts/delete_old_logs.sh`](scripts/delete_old_logs.sh)** : Script shell pour supprimer automatiquement les anciens fichiers de logs du dossier [`logs/`](logs/). Il conserve les logs des X derniers jours et supprime les plus anciens. Ce script peut être exécuté manuellement ou programmé via cron pour garder le dossier de logs propre.
- **[`airflow`](airflow/)**: Contient tous les fichiers relatifs à Apache Airflow, y compris les définitions de DAGs (`dags/`), la configuration (`config/`), les logs (`logs/`) et les plugins (`plugins/`). C'est ici que les pipelines d'orchestration de données sont définis et gérés.

## ⚖️ Licence

Ce projet est sous [Licence MIT](../LICENSE)