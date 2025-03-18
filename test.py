import sqlite3

# Connexion à une base de données SQLite (création si elle n'existe pas)
conn = sqlite3.connect(':memory:')  # Connexion à une base en mémoire pour le test

# Création d'un curseur pour exécuter des requêtes
cursor = conn.cursor()

# Test de la version de SQLite
cursor.execute('SELECT sqlite_version();')

# Récupérer et afficher la version
version = cursor.fetchone()
print(f"SQLite Version: {version[0]}")

# Fermer la connexion
conn.close()
