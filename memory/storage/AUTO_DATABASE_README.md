# 🗄️ PostgreSQL Auto-Database Creation

## Übersicht

Das Kira Memory System kann jetzt automatisch PostgreSQL-Datenbanken erstellen, falls sie nicht existieren. Das bedeutet, dass du keine manuelle Datenbank-Einrichtung mehr benötigst.

## Features

### ✅ Automatische Datenbank-Erstellung
- Erstellt automatisch die `kira_memory` Datenbank falls sie nicht existiert
- Funktioniert mit benutzerdefinierten Datenbank-Namen
- Robuste Fehlerbehandlung

### ✅ Automatische Benutzer-Erstellung
- Erstellt automatisch den `kira` Benutzer falls er nicht existiert
- Setzt die notwendigen Berechtigungen
- Funktioniert mit benutzerdefinierten Benutzern

### ✅ Schema-Initialisierung
- Erstellt automatisch alle notwendigen Tabellen
- Conversation Memory Integration
- Memory entries, conversations, und alle anderen Tabellen

## Konfiguration

### Standard-Konfiguration
```python
# Verwendet Standard-Werte
storage = PostgreSQLStorage()

# Standard-Verbindung:
# host=localhost port=5432 dbname=kira_memory user=kira password=kira_password
```

### Benutzerdefinierte Konfiguration
```python
# Mit Config-Dict
config = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'meine_kira_db',
    'user': 'mein_user',
    'password': 'mein_password'
}
storage = PostgreSQLStorage(database_config=config)

# Mit Connection String
storage = PostgreSQLStorage(
    connection_string='host=localhost port=5432 dbname=kira_prod user=kira_prod password=secure_password'
)
```

## Voraussetzungen

### PostgreSQL Server
Der PostgreSQL Server muss laufen und erreichbar sein. Die Auto-Creation benötigt:

1. **Admin-Zugang** zum PostgreSQL Server (für Datenbank-Erstellung)
2. **Einen der folgenden Admin-Benutzer:**
   - `postgres` mit Passwort `postgres`
   - `postgres` mit einem anderen Passwort
   - Einen anderen Superuser

### Installation
```bash
# PostgreSQL Python Client
pip install psycopg2-binary

# oder für Entwicklung
pip install psycopg2
```

## Verwendung

### Einfache Initialisierung
```python
from memory.storage.postgresql_storage import PostgreSQLStorage

# Erstelle Storage
storage = PostgreSQLStorage()

# Initialisiere (erstellt automatisch DB falls nötig)
success = storage.initialize()

if success:
    print("✅ PostgreSQL Storage bereit")
else:
    print("❌ Initialisierung fehlgeschlagen")
```

### Mit Konfiguration
```python
# Für Produktionsumgebung
prod_config = {
    'host': 'prod-db.example.com',
    'port': 5432,
    'dbname': 'kira_production',
    'user': 'kira_prod',
    'password': 'secure_production_password'
}

storage = PostgreSQLStorage(database_config=prod_config)
storage.initialize()
```

## Automatische Erstellungslogik

### 1. Datenbank-Erstellung
```python
def _create_database_if_not_exists(self) -> bool:
    # 1. Verbinde zur 'postgres' Standard-DB
    # 2. Prüfe ob Ziel-DB existiert
    # 3. Erstelle DB falls sie nicht existiert
    # 4. Logge Erfolg/Fehler
```

### 2. Benutzer-Erstellung
```python
def _create_user_if_not_exists(self) -> bool:
    # 1. Verbinde als postgres-Admin
    # 2. Prüfe ob Benutzer existiert
    # 3. Erstelle Benutzer falls nötig
    # 4. Setze Berechtigungen
```

### 3. Schema-Initialisierung
```python
def _create_schema(self):
    # 1. Erstelle memory_entries Tabelle
    # 2. Erstelle conversations Tabelle
    # 3. Erstelle alle anderen Tabellen
    # 4. Setze Indizes und Constraints
```

## Fehlerbehandlung

### Häufige Probleme und Lösungen

#### 1. PostgreSQL Server nicht erreichbar
```
❌ PostgreSQL Connection Test fehlgeschlagen: connection to server at "localhost", port 5432 failed
```

**Lösung:**
- Starte PostgreSQL Server: `brew services start postgresql`
- Prüfe ob Port 5432 verfügbar ist: `lsof -i :5432`

#### 2. Keine Admin-Berechtigung
```
❌ Fehler beim Erstellen der Datenbank: permission denied to create database
```

**Lösung:**
- Verwende einen Superuser-Account
- Oder erstelle die Datenbank manuell: `CREATE DATABASE kira_memory;`

#### 3. Benutzer kann nicht erstellt werden
```
⚠️ Fehler beim Erstellen des Benutzers: role "kira" already exists
```

**Lösung:**
- Das ist normal - der Benutzer existiert bereits
- Die Initialisierung wird fortgesetzt

## Testing

### Test-Skript ausführen
```bash
cd /Users/Leon/Desktop/Kira_Home
python3 debug_auto_database.py
```

### Manueller Test
```python
from memory.storage.postgresql_storage import PostgreSQLStorage

# Test verschiedene Szenarien
configs = [
    None,  # Standard
    {'dbname': 'test_db', 'user': 'test_user'},  # Custom
    'host=localhost port=5432 dbname=direct user=kira password=kira_password'  # Connection String
]

for config in configs:
    storage = PostgreSQLStorage(database_config=config if isinstance(config, dict) else None,
                               connection_string=config if isinstance(config, str) else None)
    success = storage.initialize()
    print(f"Config {config}: {'✅' if success else '❌'}")
```

## Monitoring

### Logs prüfen
```bash
# Kira System Log
tail -f logs/kira_system.log | grep -i postgresql

# Oder direkt beim Start
python3 main.py 2>&1 | grep -i postgresql
```

### Datenbank-Status prüfen
```sql
-- Verbinde zur postgres DB
psql -h localhost -U postgres -d postgres

-- Prüfe verfügbare Datenbanken
\l

-- Prüfe Benutzer
\du

-- Verbinde zur Kira DB
\c kira_memory

-- Prüfe Tabellen
\dt
```

## Sicherheit

### Produktionsumgebung
```python
# Sichere Konfiguration für Produktion
prod_config = {
    'host': 'secure-db.internal',
    'port': 5432,
    'dbname': 'kira_production',
    'user': 'kira_app',
    'password': os.getenv('KIRA_DB_PASSWORD'),  # Aus Umgebungsvariable
    'sslmode': 'require'  # SSL-Verbindung
}
```

### Backup-Strategie
```bash
# Regelmäßige Backups
pg_dump -h localhost -U kira -d kira_memory > backup_$(date +%Y%m%d).sql

# Restore
psql -h localhost -U kira -d kira_memory < backup_20250708.sql
```

## Erweiterte Konfiguration

### Connection Pool
```python
# Erweiterte Pool-Konfiguration
storage = PostgreSQLStorage(database_config={
    'host': 'localhost',
    'port': 5432,
    'dbname': 'kira_memory',
    'user': 'kira',
    'password': 'kira_password',
    'max_connections': 10,
    'min_connections': 2
})
```

### SSL-Verbindung
```python
# SSL-Konfiguration
ssl_config = {
    'host': 'secure-db.example.com',
    'port': 5432,
    'dbname': 'kira_memory',
    'user': 'kira',
    'password': 'secure_password',
    'sslmode': 'require',
    'sslcert': 'client-cert.pem',
    'sslkey': 'client-key.pem',
    'sslrootcert': 'ca-cert.pem'
}
```

---

## 🚀 Zusammenfassung

Das PostgreSQL Auto-Database Creation System macht die Kira Memory-Einrichtung extrem einfach:

1. **Keine manuelle DB-Erstellung nötig**
2. **Automatische Benutzer-Verwaltung**
3. **Robuste Fehlerbehandlung**
4. **Flexible Konfiguration**
5. **Produktionsreife Sicherheit**

Einfach `storage.initialize()` aufrufen und das System erledigt den Rest!
