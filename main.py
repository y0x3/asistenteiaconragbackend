from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import database  # tu archivo para SQLite
import os
from openai import OpenAI  # Para OpenRouter

# ------------------ CONFIGURAR OPENROUTER ------------------
API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Falta OPENROUTER_API_KEY en variables de entorno. "
        "En PowerShell: $env:OPENROUTER_API_KEY = 'sk-or-...'"
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost:5000",  # cámbialo por tu dominio si aplica
        "X-Title": "prueba_tecnica_activamente_backend",
    },
)

model = "mistralai/mistral-7b-instruct"  
# -----------------------------------------------------------

app = Flask(__name__)
CORS(app)

database.crear_tablas()

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def _empty_index(dim: int = 384):
    return faiss.IndexFlatL2(dim)


def crear_indice_faiss():
    docs = database.obtener_documentos()
    if not docs:
        return _empty_index(), []

    textos = [doc[1] for doc in docs]
    chunks = text_splitter.split_text(" ".join(textos))
    if not chunks:
        return _empty_index(), []

    embeddings = embedding_model.encode(chunks).astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, chunks


index, doc_chunks = crear_indice_faiss()


@app.route("/add_doc", methods=["POST"])
def add_doc():
    data = request.get_json()
    texto = (data.get("texto", "") or "").strip()
    if texto:
        database.insertar_documento(texto)
        global index, doc_chunks
        index, doc_chunks = crear_indice_faiss()
        return jsonify({"status": "Documento agregado"})
    else:
        return jsonify({"status": "No hay texto para agregar"}), 400


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    mensaje = data.get("mensaje", "")
    usuario_id = data.get("usuario_id")  # Lo recibimos del front
    conversacion_id = data.get("conversacion_id")  # Puede venir vacío si es nueva


    # Si no hay conversacion_id, creamos una nueva
    if not conversacion_id:
        conversacion_id = database.crear_conversacion(usuario_id, titulo="Nueva conversación")

    # Guardar el mensaje del usuario
    database.guardar_mensaje(conversacion_id, "usuario", mensaje)

    # Dividir mensaje en chunks y generar embeddings
    chunks = text_splitter.split_text(mensaje)
    embeddings = embedding_model.encode(chunks).astype('float32') if chunks else np.array([])

    resultados_similares = []

    if embeddings.size > 0 and index is not None and index.ntotal > 0:
        k = min(3, index.ntotal)
        D, I = index.search(embeddings[0:1], k)
        for idx_hit in I[0]:
            if 0 <= idx_hit < len(doc_chunks):
                resultados_similares.append(doc_chunks[idx_hit])

    try:
        contexto = "\n".join(resultados_similares)
        prompt = (f"Usa el siguiente contexto si es útil. Si no, responde solo con tu conocimiento.\n\n"
                  f"Contexto:\n{contexto}\n\n"
                  f"Pregunta: {mensaje}")

        # Llamada a OpenRouter
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            top_p=1.0,
        )

        respuesta = response.choices[0].message.content

    except Exception as e:
        respuesta = f"Error al generar respuesta: {str(e)}"

    # Guardar respuesta de la IA
    database.guardar_mensaje(conversacion_id, "ia", respuesta)

    return jsonify({
        "conversacion_id": conversacion_id,
        "mensaje_original": mensaje,
        "chunks": chunks,
        "embeddings": embeddings.tolist() if embeddings.size > 0 else [],
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
    mensajes = database.obtener_mensajes(conversacion_id)  # lista de (id, remitente, texto, fecha)
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
    return jsonify({"status": "Documento agregado"})



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
