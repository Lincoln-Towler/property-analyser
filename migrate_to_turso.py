#!/usr/bin/env python3
"""
Migrate Local SQLite Database to Turso
========================================
This script copies all your data from local property_data.db to Turso cloud database.

Usage:
    python migrate_to_turso.py

Make sure you have:
1. Installed libsql: pip install libsql-experimental
2. Your Turso database URL and auth token ready
"""

import sqlite3
from libsql_experimental import dbapi2 as libsql

def migrate_to_turso(turso_url, turso_token, local_db_path="property_data.db"):
    """Migrate data from local SQLite to Turso"""
    
    print("🔄 Starting migration to Turso...")
    
    # Connect to local database
    print(f"📂 Connecting to local database: {local_db_path}")
    local_conn = sqlite3.connect(local_db_path)
    local_cursor = local_conn.cursor()
    
    # Connect to Turso
    print(f"☁️  Connecting to Turso: {turso_url}")
    turso_conn = libsql.connect(database=turso_url, auth_token=turso_token)
    turso_cursor = turso_conn.cursor()
    
    # Tables to migrate
    tables = ['economic_indicators', 'property_data', 'market_commentary']
    
    for table in tables:
        print(f"\n📊 Migrating table: {table}")
        
        try:
            # Get data from local
            local_cursor.execute(f"SELECT * FROM {table}")
            rows = local_cursor.fetchall()
            
            if not rows:
                print(f"   ℹ️  No data in {table}")
                continue
            
            # Get column names
            local_cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in local_cursor.fetchall()]
            
            # Insert into Turso
            placeholders = ','.join(['?' for _ in columns])
            insert_query = f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})"
            
            for row in rows:
                turso_cursor.execute(insert_query, row)
            
            turso_conn.commit()
            print(f"   ✅ Migrated {len(rows)} rows")
            
        except Exception as e:
            print(f"   ❌ Error migrating {table}: {e}")
    
    # Close connections
    local_conn.close()
    turso_conn.close()
    
    print("\n🎉 Migration complete!")
    print("\n📝 Next steps:")
    print("1. Push updated code to GitHub")
    print("2. Add Turso secrets to Streamlit Cloud")
    print("3. Your app will now use Turso for persistent storage!")

if __name__ == "__main__":
    print("=" * 60)
    print("  Turso Migration Tool")
    print("=" * 60)
    
    turso_url = input("\n📍 Enter your Turso database URL: ").strip()
    turso_token = input("🔑 Enter your Turso auth token: ").strip()
    
    if not turso_url or not turso_token:
        print("\n❌ Error: URL and token are required!")
        exit(1)
    
    # Confirm
    print(f"\n⚠️  About to migrate local data to: {turso_url}")
    confirm = input("Continue? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        migrate_to_turso(turso_url, turso_token)
    else:
        print("\n❌ Migration cancelled")
