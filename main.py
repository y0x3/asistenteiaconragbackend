import os
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import database
from apillm import client, MODEL
import requests

app = Flask(__name__)
CORS(app)

# Crea tablas al inicio
database.crear_tablas()

# ======== Config ========
APP_URL = os.environ.get("APP_URL", "http://localhost:5000")
EMBEDDING_MODEL = os.environ.get("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Splitter para trocear texto
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "]
)


def generar_embedding(texto: str) -> list:
    """Genera un embedding usando OpenRouter (sin SDK de OpenAI).
    Incluye logs claros si la API responde algo no-JSON.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY no está definido en variables de entorno.")

    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Estos dos ayudan a OpenRouter a identificar tu app (algunos planes lo requieren)
        "HTTP-Referer": APP_URL,
        "X-Title": "prueba_tecnica_activamente_backend",
    }
    data = {"model": EMBEDDING_MODEL, "input": texto}

    resp = requests.post(url, headers=headers, json=data, timeout=30)

    # Si no es 200, lanza error con detalle para que aparezca en logs de Render
    if resp.status_code != 200:
        raise ValueError(f"Error en embeddings: {resp.status_code} - {resp.text[:500]}")

    try:
        payload = resp.json()
    except Exception:
        # Log de ayuda si la API devuelve HTML/texto
        snippet = resp.text[:500]
        raise ValueError(f"Respuesta embeddings no-JSON (HTTP 200): {snippet}")

    if isinstance(payload, dict) and "error" in payload:
        raise ValueError(f"OpenRouter embeddings error: {payload['error']}")

    try:
        return payload["data"][0]["embedding"]
    except Exception as e:
        raise ValueError(f"Formato inesperado en respuesta de embeddings: {payload}") from e


def _empty_index(dim: int = 1536):
    """Index vacío por defecto con 1536 dims (text-embedding-3-small)."""
    return faiss.IndexFlatL2(dim)


def crear_indice_faiss():
    """Crea el índice FAISS desde documentos en DB. Si algo falla, no tumba la app."""
    try:
        docs = database.obtener_documentos()
        if not docs:
            return _empty_index(), []

        textos = [doc[1] for doc in docs]
        chunks = text_splitter.split_text(" ".join(textos))
        if not chunks:
            return _empty_index(), []

        # Genera embeddings de forma segura (si alguno falla, lo saltamos)
        vectors = []
        for ch in chunks:
            try:
                vec = generar_embedding(ch)
                if isinstance(vec, list) and vec:
                    vectors.append(vec)
            except Exception as e:
                print("[WARN] Falla embedding de un chunk:", e)
                continue

        if not vectors:
            print("[WARN] No se generaron embeddings; índice vacío.")
            return _empty_index(), []

        embeddings = np.array(vectors, dtype="float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index, chunks[: len(vectors)]  # Alinea chunks con 'vectors'

    except Exception as e:
        print("[ERROR] crear_indice_faiss:", e)
        return _empty_index(), []


# Construye índice al arrancar, pero no tires la app si falla
index, doc_chunks = crear_indice_faiss()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "has_api_key": bool(OPENROUTER_API_KEY),
        "embedding_model": EMBEDDING_MODEL,
        "index_size": int(index.ntotal) if index else 0,
    })


@app.route("/add_doc", methods=["POST"])
def add_doc():
    data = request.get_json()
    texto = (data.get("texto", "") or "").strip()
    if not texto:
        return jsonify({"status": "No hay texto para agregar"}), 400

    database.insertar_documento(texto)
    global index, doc_chunks
    index, doc_chunks = crear_indice_faiss()
    return jsonify({"status": "Documento agregado", "index_size": int(index.ntotal)})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    mensaje = data.get("mensaje", "")
    usuario_id = data.get("usuario_id")
    conversacion_id = data.get("conversacion_id")

    if not conversacion_id:
        conversacion_id = database.crear_conversacion(usuario_id, titulo="Nueva conversación")

    database.guardar_mensaje(conversacion_id, "usuario", mensaje)

    # Embedding del mensaje (seguro ante errores)
    resultados_similares = []
    try:
        chunks = text_splitter.split_text(mensaje)
        if chunks and index is not None and index.ntotal > 0:
            emb0 = np.array([generar_embedding(chunks[0])], dtype="float32")
            k = min(3, index.ntotal)
            D, I = index.search(emb0, k)
            for idx_hit in I[0]:
                if 0 <= idx_hit < len(doc_chunks):
                    resultados_similares.append(doc_chunks[idx_hit])
    except Exception as e:
        print("[WARN] No se pudo calcular similitud para el mensaje:", e)

    try:
        contexto = "\n".join(resultados_similares)
        prompt = (
            f"Usa el siguiente contexto si es útil. Si no, responde solo con tu conocimiento.\n\n"
            f"Contexto:\n{contexto}\n\n"
            f"Pregunta: {mensaje}"
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            top_p=1.0,
        )
        respuesta = response.choices[0].message.content
    except Exception as e:
        respuesta = f"Error al generar respuesta: {str(e)}"

    database.guardar_mensaje(conversacion_id, "ia", respuesta)

    return jsonify({
        "conversacion_id": conversacion_id,
        "mensaje_original": mensaje,
        "resultados_similares": resultados_similares,
        "respuesta_generada": respuesta,
    })


@app.route("/conversaciones/<int:usuario_id>", methods=["GET"])
def get_conversaciones(usuario_id):
    conversaciones = database.obtener_conversaciones(usuario_id)
    return jsonify([
        {"id": c[0], "titulo": c[1], "fecha": c[2]} for c in conversaciones
    ])


@app.route("/mensajes/<int:conversacion_id>", methods=["GET"])
def get_mensajes(conversacion_id):
    mensajes = database.obtener_mensajes(conversacion_id)
    return jsonify([
        {"id": m[0], "remitente": m[1], "texto": m[2], "fecha": m[3]} for m in mensajes
    ])


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    text = file.read().decode("utf-8")
    database.insertar_documento(text)
    global index, doc_chunks
    index, doc_chunks = crear_indice_faiss()
    return jsonify({"status": "Documento agregado", "index_size": int(index.ntotal)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
