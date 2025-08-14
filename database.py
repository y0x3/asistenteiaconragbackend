import sqlite3

def crear_conexion():
    conn = sqlite3.connect('base_rag.db')
    return conn

def crear_tablas():
    conn = crear_conexion()
    cursor = conn.cursor()

    # Tabla de conversaciones
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo TEXT,
            fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Tabla de mensajes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mensajes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversacion_id INTEGER,
            remitente TEXT, -- "usuario" o "ia"
            texto TEXT,
            fecha_envio TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(conversacion_id) REFERENCES conversaciones(id)
        )
    ''')

    # Tabla de documentos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            texto TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

# ------------------------------
# CRUD Documentos
# ------------------------------
def insertar_documento(texto):
    conn = crear_conexion()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO documentos (texto) VALUES (?)', (texto,))
    conn.commit()
    conn.close()

def obtener_documentos():
    conn = crear_conexion()
    cursor = conn.cursor()
    cursor.execute('SELECT id, texto FROM documentos')
    docs = cursor.fetchall()
    conn.close()
    return docs

# ------------------------------
# CRUD Conversaciones
# ------------------------------
def crear_conversacion(titulo):
    conn = crear_conexion()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversaciones (titulo)
        VALUES (?)
    ''', (titulo,))
    conn.commit()
    conversacion_id = cursor.lastrowid
    conn.close()
    return conversacion_id

def obtener_conversaciones():
    conn = crear_conexion()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, titulo, fecha_creacion
        FROM conversaciones
        ORDER BY fecha_creacion DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

# ------------------------------
# CRUD Mensajes
# ------------------------------
def guardar_mensaje(conversacion_id, remitente, texto):
    conn = crear_conexion()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO mensajes (conversacion_id, remitente, texto)
        VALUES (?, ?, ?)
    ''', (conversacion_id, remitente, texto))
    conn.commit()
    conn.close()

def obtener_mensajes(conversacion_id):
    conn = crear_conexion()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, remitente, texto, fecha_envio
        FROM mensajes
        WHERE conversacion_id = ?
        ORDER BY fecha_envio ASC
    ''', (conversacion_id,))
    data = cursor.fetchall()
    conn.close()
    return data
