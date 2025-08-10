async function postJson(url, body){
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const data = await res.json().catch(()=>({}));
  if(!res.ok){
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

function formatScorecard(card){
  return JSON.stringify(card, null, 2);
}

function makeResultItem(r){
  const div = document.createElement('div');
  div.className = 'result-item';
  div.innerHTML = `
    <h3><a href="${r.article.url}" target="_blank" rel="noopener noreferrer">${r.article.title || r.article.url}</a></h3>
    <div class="meta">
      <span class="score">Score: ${r.overall_score}</span>
      • Readability ${r.readability_score}
      • NER ${r.ner_density_score}
      • Sentiment ${r.sentiment_score}
      • TF-IDF ${r.tfidf_relevance_score}
      • Recency ${r.recency_score}
    </div>
    <p>${r.article.summary || ''}</p>
  `;
  return div;
}

window.addEventListener('DOMContentLoaded', () => {
  const analyzeForm = document.getElementById('analyze-form');
  const analyzeResult = document.getElementById('analyze-result');
  analyzeForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    analyzeResult.textContent = 'Analyzing…';
    const url = document.getElementById('analyze-url').value.trim();
    const query = document.getElementById('analyze-query').value.trim();
    try{
      const data = await postJson('/analyze', { url, query });
      analyzeResult.textContent = formatScorecard(data);
    }catch(err){
      analyzeResult.textContent = `Error: ${err.message}`;
    }
  });

  const curateForm = document.getElementById('curate-form');
  const curateResult = document.getElementById('curate-result');
  curateForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    curateResult.innerHTML = '<pre class="result">Curating…</pre>';
    const topic = document.getElementById('curate-topic').value.trim();
    const maxArticlesStr = document.getElementById('curate-max').value;
    const max_articles = maxArticlesStr ? parseInt(maxArticlesStr, 10) : undefined;
    try{
      const data = await postJson('/curate-topic', { topic, max_articles });
      curateResult.innerHTML = '';
      (data.results || []).forEach(r => {
        curateResult.appendChild(makeResultItem(r));
      });
      if((data.results || []).length === 0){
        curateResult.innerHTML = '<pre class="result">No results</pre>';
      }
    }catch(err){
      curateResult.innerHTML = `<pre class="result">Error: ${err.message}</pre>`;
    }
  });
});
