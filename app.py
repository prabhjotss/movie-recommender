from flask import Flask, render_template_string, request, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# ── Load data ────────────────────────────────────────────────────────────────
movies_data = pd.read_csv("movie.csv")
movies_data.fillna('', inplace=True)

selected_features = ['genre', 'overview', 'director', 'cast']
for f in selected_features:
    movies_data[f] = movies_data[f].fillna('')

combined_features = (
    movies_data['genre'] + ' ' +
    movies_data['overview'] + ' ' +
    movies_data['director'] + ' ' +
    movies_data['cast']
)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# ── Recommendation logic ─────────────────────────────────────────────────────
def recommend_movies(prompt, genre_filter='', top_n=10):
    prompt_vector = vectorizer.transform([prompt])
    similarity_scores = cosine_similarity(prompt_vector, feature_vectors)[0]
    sorted_indices = np.argsort(similarity_scores)[::-1]

    results = []
    for i in sorted_indices:
        row = movies_data.iloc[i]
        if genre_filter and genre_filter.lower() not in row['genre'].lower():
            continue
        results.append({
            'title':    row['movie_name'],
            'year':     row.get('year', ''),
            'genre':    row.get('genre', ''),
            'director': row.get('director', ''),
            'score':    round(float(similarity_scores[i]) * 100, 1),
        })
        if len(results) == top_n:
            break
    return results

# ── Template ─────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Filmatch — Movie Recommender</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #f7f8fa;
    --surface:   #ffffff;
    --border:    #e4e7ec;
    --border2:   #cdd2db;
    --text:      #111318;
    --muted:     #6b7280;
    --hint:      #9ca3af;
    --accent:    #1a1a2e;
    --accent-fg: #ffffff;
    --info-bg:   #eff6ff;
    --info-fg:   #1d4ed8;
    --chip-act:  #1a1a2e;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow:    0 1px 4px rgba(0,0,0,0.06);
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --bg:       #0f1117;
      --surface:  #1a1c23;
      --border:   #2e3038;
      --border2:  #454850;
      --text:     #f0f1f5;
      --muted:    #9ca3af;
      --hint:     #6b7280;
      --accent:   #e8e8f0;
      --accent-fg:#0f1117;
      --info-bg:  #1e2a42;
      --info-fg:  #93c5fd;
      --chip-act: #e8e8f0;
    }
  }

  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── NAV ── */
  nav {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 1.5rem;
    height: 54px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .nav-logo { display: flex; align-items: center; gap: 8px; text-decoration: none; }
  .nav-icon {
    width: 30px; height: 30px;
    background: var(--accent); color: var(--accent-fg);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
  }
  .nav-icon svg { width: 16px; height: 16px; }
  .nav-title { font-size: 16px; font-weight: 600; color: var(--text); }
  .nav-tag { margin-left: auto; font-size: 12px; color: var(--hint); }

  /* ── MAIN ── */
  main { flex: 1; max-width: 780px; width: 100%; margin: 0 auto; padding: 2rem 1rem 4rem; }

  .hero { text-align: center; margin-bottom: 2rem; }
  .hero h1 { font-size: 28px; font-weight: 600; margin-bottom: 6px; }
  .hero p { font-size: 15px; color: var(--muted); }

  /* ── SEARCH CARD ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem;
    box-shadow: var(--shadow);
  }

  .search-row { display: flex; gap: 8px; margin-bottom: 1rem; }
  input[type=text], select {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 10px 14px;
    font-size: 15px;
    color: var(--text);
    font-family: inherit;
    outline: none;
    transition: border-color 0.15s;
  }
  input[type=text]:focus { border-color: var(--border2); }
  #searchInput { flex: 1; }
  select { min-width: 130px; cursor: pointer; }

  /* ── BUTTONS ── */
  .btn {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius-md);
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 500;
    color: var(--text);
    cursor: pointer;
    font-family: inherit;
    transition: background 0.12s, transform 0.1s;
    white-space: nowrap;
  }
  .btn:hover { background: var(--bg); }
  .btn:active { transform: scale(0.97); }
  .btn-primary {
    background: var(--accent);
    color: var(--accent-fg);
    border-color: var(--accent);
  }
  .btn-primary:hover { opacity: 0.88; background: var(--accent); }
  .btn-sm { padding: 6px 12px; font-size: 13px; }

  /* ── GENRE CHIPS ── */
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 1rem; }
  .chip {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 13px;
    color: var(--muted);
    cursor: pointer;
    transition: all 0.12s;
    user-select: none;
    text-decoration: none;
    display: inline-block;
  }
  .chip:hover { border-color: var(--border2); color: var(--text); }
  .chip.active { background: var(--chip-act); color: var(--accent-fg); border-color: var(--chip-act); }
  @media (prefers-color-scheme: dark) {
    .chip.active { color: var(--accent-fg); }
  }

  /* ── SUGGESTIONS ── */
  .suggestions { margin-bottom: 1.5rem; }
  .section-label { font-size: 12px; color: var(--hint); margin-bottom: 0.5rem; }
  .suggest-row { display: flex; flex-wrap: wrap; gap: 6px; }
  .suggest-btn {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 6px 12px;
    font-size: 13px;
    color: var(--muted);
    cursor: pointer;
    font-family: inherit;
    transition: all 0.12s;
  }
  .suggest-btn:hover { border-color: var(--border2); color: var(--text); }

  /* ── HISTORY ── */
  .history { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border); }
  .history-label { font-size: 12px; color: var(--hint); }
  .history-pill {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 12px;
    color: var(--muted);
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
  }
  .history-pill:hover { color: var(--text); }

  /* ── RESULTS ── */
  .results-header { display: flex; align-items: center; justify-content: space-between; margin: 1.5rem 0 0.75rem; }
  .results-label { font-size: 14px; color: var(--muted); }
  .results-count { font-size: 13px; color: var(--hint); }

  .movie-grid { display: flex; flex-direction: column; gap: 8px; }
  .movie-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: border-color 0.12s, background 0.12s;
  }
  .movie-card:hover { border-color: var(--border2); background: var(--bg); }

  .rank { width: 28px; text-align: center; font-size: 13px; color: var(--hint); font-weight: 500; flex-shrink: 0; }
  .rank.top3 { color: #b45309; }

  .thumb {
    width: 42px; height: 58px;
    border-radius: 6px;
    background: var(--bg);
    border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .thumb svg { width: 18px; height: 18px; color: var(--hint); }

  .info { flex: 1; min-width: 0; }
  .movie-title { font-size: 15px; font-weight: 500; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 5px; }
  .meta { display: flex; gap: 5px; flex-wrap: wrap; align-items: center; }
  .tag { font-size: 12px; color: var(--muted); background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 2px 7px; }
  .tag.genre { background: var(--info-bg); color: var(--info-fg); border-color: transparent; }
  .director { font-size: 12px; color: var(--hint); margin-left: 2px; }

  .score { flex-shrink: 0; text-align: right; width: 52px; }
  .score-val { font-size: 13px; font-weight: 500; color: var(--muted); }
  .score-track { height: 3px; background: var(--border); border-radius: 2px; margin-top: 5px; }
  .score-fill { height: 100%; background: var(--info-fg); border-radius: 2px; }

  /* ── EMPTY STATE ── */
  .empty {
    text-align: center;
    padding: 3.5rem 1rem;
    color: var(--hint);
    border: 1px dashed var(--border2);
    border-radius: var(--radius-lg);
    margin-top: 1.5rem;
  }
  .empty svg { width: 42px; height: 42px; margin-bottom: 1rem; opacity: 0.35; }
  .empty p { font-size: 14px; }

  /* ── FOOTER ── */
  footer { text-align: center; margin-top: 3rem; font-size: 13px; color: var(--hint); }
  footer strong { color: var(--muted); }

  /* ── RESPONSIVE ── */
  @media (max-width: 520px) {
    .search-row { flex-direction: column; }
    .btn-primary { width: 100%; }
    select { width: 100%; }
    .score { display: none; }
    .thumb { display: none; }
    .hero h1 { font-size: 22px; }
  }
</style>
</head>

<body>
<nav>
  <a class="nav-logo" href="/">
    <div class="nav-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="2" y="4" width="20" height="16" rx="2"/>
        <path d="M8 4v16M2 9h6M2 15h6M16 4v16"/>
      </svg>
    </div>
    <span class="nav-title">Filmatch</span>
  </a>
  <span class="nav-tag">Content-based movie recommender</span>
</nav>

<main>
  <div class="hero">
    <h1>Find your next favourite movie</h1>
    <p>Search by title, mood, genre, actor, director, or describe a plot.</p>
  </div>

  <!-- Search form -->
  <div class="card" style="margin-bottom: 1.25rem;">
    <form method="post" action="/">
      <div class="search-row">
        <input type="text" id="searchInput" name="prompt"
               value="{{ prompt or '' }}"
               placeholder='e.g. "Christopher Nolan thriller" or "feel-good 90s comedy"'
               autocomplete="off" required>
        <select name="genre_filter">
          <option value="">All genres</option>
          {% for g in genres %}
            <option value="{{ g }}" {% if genre_filter == g %}selected{% endif %}>{{ g }}</option>
          {% endfor %}
        </select>
        <button class="btn btn-primary" type="submit">Search</button>
        {% if prompt %}
          <a href="/" class="btn btn-sm">Clear</a>
        {% endif %}
      </div>

      <!-- Genre chips (quick-fill) -->
      <div class="chips">
        {% for g in genres %}
          <span class="chip {% if genre_filter == g %}active{% endif %}"
                onclick="fillGenre('{{ g }}')">{{ g }}</span>
        {% endfor %}
      </div>

      <!-- Top-N selector -->
      <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:13px; color:var(--muted);">Results:</span>
        {% for n in [5, 10, 15, 20] %}
          <label style="font-size:13px; color:var(--muted); cursor:pointer;">
            <input type="radio" name="top_n" value="{{ n }}"
                   {% if top_n == n %}checked{% endif %}
                   style="accent-color: var(--info-fg);"> {{ n }}
          </label>
        {% endfor %}
      </div>
    </form>

    <!-- Search history -->
    {% if history %}
    <div class="history">
      <span class="history-label">Recent searches:</span>
      {% for h in history %}
        <a class="history-pill" href="/?prompt={{ h | urlencode }}&top_n={{ top_n }}">{{ h }}</a>
      {% endfor %}
      <a href="/clear-history" class="history-pill" style="color:var(--hint);">✕ clear</a>
    </div>
    {% endif %}
  </div>

  <!-- Suggestions -->
  {% if not prompt %}
  <div class="suggestions">
    <p class="section-label">Try a search like</p>
    <div class="suggest-row">
      {% for s in suggestions %}
        <button class="suggest-btn" type="button" onclick="setAndGo('{{ s }}')">{{ s }}</button>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  <!-- Results -->
  {% if recommendations is not none %}
    {% if recommendations %}
      <div class="results-header">
        <span class="results-label">Results for "<strong>{{ prompt }}</strong>"</span>
        <span class="results-count">{{ recommendations | length }} movies</span>
      </div>
      <div class="movie-grid">
        {% for m in recommendations %}
        <div class="movie-card">
          <span class="rank {% if loop.index <= 3 %}top3{% endif %}">
            {{ loop.index }}{% if loop.index == 1 %}st{% elif loop.index == 2 %}nd{% elif loop.index == 3 %}rd{% else %}th{% endif %}
          </span>
          <div class="thumb">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <rect x="2" y="4" width="20" height="16" rx="2"/>
              <path d="M8 4v16M2 9h6M2 15h6"/>
            </svg>
          </div>
          <div class="info">
            <div class="movie-title">{{ m.title }}</div>
            <div class="meta">
              {% if m.year %}<span class="tag">{{ m.year }}</span>{% endif %}
              {% if m.genre %}<span class="tag genre">{{ m.genre | truncate(30, true, '…') }}</span>{% endif %}
              {% if m.director %}<span class="director">dir. {{ m.director | truncate(25, true, '…') }}</span>{% endif %}
            </div>
          </div>
          <div class="score">
            <div class="score-val">{{ m.score }}%</div>
            <div class="score-track">
              <div class="score-fill" style="width: {{ [m.score, 100] | min }}%"></div>
            </div>
          </div>
        </div>
        {% endfor %}
      </div>
    {% else %}
      <div class="empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
        </svg>
        <p>No results found for "<strong>{{ prompt }}</strong>".<br>Try a different search or remove the genre filter.</p>
      </div>
    {% endif %}
  {% endif %}

</main>

<footer>
  Made by <strong>Prabhjot Singh</strong> &nbsp;·&nbsp; Filmatch Movie Recommender
</footer>

<script>
function fillGenre(g) {
  const select = document.querySelector('select[name=genre_filter]');
  const chips  = document.querySelectorAll('.chip');
  chips.forEach(c => c.classList.toggle('active', c.textContent === g));
  select.value = g;
}

function setAndGo(q) {
  document.getElementById('searchInput').value = q;
  document.querySelector('form').submit();
}

// Persist chip highlight on load
(function() {
  const sel = document.querySelector('select[name=genre_filter]');
  if (sel && sel.value) {
    document.querySelectorAll('.chip').forEach(c => {
      if (c.textContent === sel.value) c.classList.add('active');
    });
  }
})();
</script>
</body>
</html>
"""

# ── Config ───────────────────────────────────────────────────────────────────
GENRES = [
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western',
]

SUGGESTIONS = [
    'mind-bending sci-fi with twists',
    'romantic comedy set in New York',
    'psychological thriller heist',
    'animated family adventure',
    'Christopher Nolan epic',
    'feel-good 90s comedy',
    'Oscar-winning drama',
    'superhero action blockbuster',
]

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    prompt        = None
    recommendations = None
    genre_filter  = ''
    top_n         = 10

    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        prompt       = request.form.get('prompt', '').strip()
        genre_filter = request.form.get('genre_filter', '')
        top_n        = int(request.form.get('top_n', 10))

        if prompt:
            # Update history (max 6 entries)
            hist = session['history']
            if prompt not in hist:
                hist.insert(0, prompt)
                if len(hist) > 6:
                    hist.pop()
            session['history'] = hist
            session.modified = True

            recommendations = recommend_movies(prompt, genre_filter, top_n)
    else:
        # Support GET params so history links work
        prompt       = request.args.get('prompt', '').strip() or None
        genre_filter = request.args.get('genre_filter', '')
        top_n        = int(request.args.get('top_n', 10))
        if prompt:
            recommendations = recommend_movies(prompt, genre_filter, top_n)

    return render_template_string(
        HTML,
        prompt=prompt,
        recommendations=recommendations,
        genre_filter=genre_filter,
        top_n=top_n,
        genres=GENRES,
        history=session.get('history', []),
        suggestions=SUGGESTIONS,
    )


@app.route('/clear-history')
def clear_history():
    session['history'] = []
    return app.make_response(('<script>history.back()</script>', 200))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
