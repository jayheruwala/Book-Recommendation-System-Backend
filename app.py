from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
from scipy.sparse import csr_matrix

# -------------------------------------------------------------
# INITIALIZE FLASK APP
# -------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------------------
# LOAD DATA & MODEL
# -------------------------------------------------------------
DATA_FILE = "Data.xlsx"
MODEL_FILE = "knn_book_model.pkl"

print("📂 Loading data and trained model...")

df = pd.read_excel(DATA_FILE)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

for col in ['mem_code', 'accesion_no', 'title']:
    df[col] = df[col].astype(str).str.strip()

df = df.dropna(subset=['mem_code', 'accesion_no', 'title'])
df = df.drop_duplicates(subset=['mem_code', 'accesion_no'])

knn_model = joblib.load(MODEL_FILE)

print("✅ Model Loaded Successfully!")

# -------------------------------------------------------------
# BUILD USER-ITEM MATRIX (EVERY START)
# -------------------------------------------------------------
print("⚙ Building user-item matrix...")

user_item_matrix = df.pivot_table(
    index='mem_code',
    columns='accesion_no',
    aggfunc='size',
    fill_value=0
)

sparse_matrix = csr_matrix(user_item_matrix.values)

print("✅ User-Item Matrix Ready!")


# -------------------------------------------------------------
# RECOMMENDATION FUNCTION
# -------------------------------------------------------------
def recommend_books(user_id, model, user_item_matrix, df, top_n=10):

    if user_id not in user_item_matrix.index:
        return []

    user_idx = user_item_matrix.index.get_loc(user_id)

    distances, indices = model.kneighbors(
        user_item_matrix.iloc[user_idx, :].values.reshape(1, -1),
        n_neighbors=6
    )

    similar_users = user_item_matrix.index[indices.flatten()[1:]]

    books_count = {}

    for u in similar_users:
        books = df[df['mem_code'] == u]['accesion_no'].unique()
        for b in books:
            books_count[b] = books_count.get(b, 0) + 1

    if not books_count:
        return []

    rec = sorted(books_count.items(), key=lambda x: x[1], reverse=True)[:top_n]
    rec_df = pd.DataFrame(rec, columns=['accesion_no', 'score'])

    titles = df[['accesion_no', 'title']].drop_duplicates()
    rec_df = rec_df.merge(titles, on='accesion_no', how='left')

    return rec_df.to_dict(orient="records")


# -------------------------------------------------------------
# API ROUTE: HOME TEST
# -------------------------------------------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "📚 Book Recommendation API is Running!",
        "usage": "/recommend?user_id=123"
    })


# -------------------------------------------------------------
# API ROUTE: GET RECOMMENDATIONS
# -------------------------------------------------------------
@app.route("/recommend", methods=["GET"])
def recommend_api():

    user_id = request.args.get("user_id")

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    recommendations = recommend_books(
        user_id=user_id,
        model=knn_model,
        user_item_matrix=user_item_matrix,
        df=df,
        top_n=10
    )

    if not recommendations:
        return jsonify({
            "user_id": user_id,
            "message": "No recommendations found."
        })

    return jsonify({
        "user_id": user_id,
        "recommendations": recommendations
    })


# -------------------------------------------------------------
# RUN FLASK SERVER
# -------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
