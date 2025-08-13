import sqlite3

# ------------------------------
# Conexión
# ------------------------------
def crear_conexion():
    conn = sqlite3.connect('base_rag.db')
    return conn

# ------------------------------
# Crear todas las tablas
# ------------------------------
def crear_tablas():
    conn = crear_conexion()
    cursor = conn.cursor()

    # Tabla de usuarios
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            google_id TEXT UNIQUE NOT NULL,
            nombre TEXT,
            email TEXT,
            avatar_url TEXT
        )
    ''')

    # Tabla de conversaciones
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            titulo TEXT,
            fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(usuario_id) REFERENCES usuarios(id)
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

    # Tabla de documentos (la que ya tenías)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            texto TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

# ------------------------------
# CRUD Documentos (RAG)
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
# CRUD Usuarios
# ------------------------------
def obtener_o_crear_usuario(google_id, nombre, email, avatar_url):
    conn = crear_conexion()
    cursor = conn.cursor()

    cursor.execute('SELECT id FROM usuarios WHERE google_id = ?', (google_id,))
    user = cursor.fetchone()

    if user:
        conn.close()
        return user[0]

    cursor.execute('''
        INSERT INTO usuarios (google_id, nombre, email, avatar_url)
        VALUES (?, ?, ?, ?)
    ''', (google_id, nombre, email, avatar_url))
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id

# ------------------------------
# CRUD Conversaciones
# ------------------------------
def crear_conversacion(usuario_id, titulo):
    conn = crear_conexion()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversaciones (usuario_id, titulo)
        VALUES (?, ?)
    ''', (usuario_id, titulo))
    conn.commit()
    conversacion_id = cursor.lastrowid
    conn.close()
    return conversacion_id

def obtener_conversaciones(usuario_id):
    conn = crear_conexion()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, titulo, fecha_creacion
        FROM conversaciones
        WHERE usuario_id = ?
        ORDER BY fecha_creacion DESC
    """, (usuario_id,))
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
        SELECT remitente, texto, fecha_envio
        FROM mensajes
        WHERE conversacion_id = ?
        ORDER BY fecha_envio ASC
    ''', (conversacion_id,))
    data = cursor.fetchall()
    conn.close()
    return data
