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

function setText(id, text){
  const el = document.getElementById(id);
  if(el){ el.textContent = text; }
}

function setBar(id, value){
  const el = document.getElementById(id);
  if(el){ el.style.width = `${Math.max(0, Math.min(100, value))}%`; }
}

function show(id){ const el = document.getElementById(id); if(el){ el.classList.remove('hidden'); } }
function hide(id){ const el = document.getElementById(id); if(el){ el.classList.add('hidden'); } }

function createCurationCard(r){
  const safeTitle = r.article.title || r.article.url;
  const wrap = document.createElement('div');
  wrap.className = 'list-card';
  wrap.innerHTML = `
    <div class="flex items-start justify-between gap-3">
      <div>
        <a href="${r.article.url}" target="_blank" rel="noopener" class="text-base font-semibold hover:underline">${safeTitle}</a>
        <div class="mt-1 meta-row">
          <span class="font-bold" style="color:#16a34a">${Math.round(r.overall_score)}</span>
          • Read ${Math.round(r.readability_score)}
          • NER ${Math.round(r.ner_density_score)}
          • Sent ${Math.round(r.sentiment_score)}
          • Relevance ${Math.round(r.tfidf_relevance_score)}
          • Recency ${Math.round(r.recency_score)}
        </div>
      </div>
      <div class="badge-score-lg">${Math.round(r.overall_score)}</div>
    </div>
    ${r.article.summary ? `<p class="mt-2 text-sm" style="color:#334155">${r.article.summary}</p>` : ''}
  `;
  return wrap;
}

window.addEventListener('DOMContentLoaded', () => {
  // Analyze form behavior
  const analyzeForm = document.getElementById('analyze-form');
  if(analyzeForm){
    analyzeForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      hide('analyze-error');
      show('analyze-loading');
      const url = document.getElementById('analyze-url').value.trim();
      let query = document.getElementById('analyze-query').value.trim();
      if(!query){
        const t = document.getElementById('analyze-url').value.trim();
        try{ const u = new URL(t); query = u.hostname.replace(/^www\./,''); }catch(_){ /* ignore */ }
      }
      try{
        const data = await postJson('/analyze', { url, query });
        // Fill UI
        setText('an-title', data.article.title || data.article.url || 'Article');
        setText('an-meta', data.article.author ? `By ${data.article.author}` : '');
        setText('an-overall', Math.round(data.overall_score));

        setText('an-readability-v', `${data.readability_score}`);
        setBar('an-readability', data.readability_score);
        setText('an-ner-v', `${data.ner_density_score}`);
        setBar('an-ner', data.ner_density_score);
        setText('an-sentiment-v', `${data.sentiment_score}`);
        setBar('an-sentiment', data.sentiment_score);
         setText('an-tfidf-v', `${data.tfidf_relevance_score}`);
        setBar('an-tfidf', data.tfidf_relevance_score);
        setText('an-recency-v', `${data.recency_score}`);
        setBar('an-recency', data.recency_score);

        setText('an-summary', data.article.summary || '');
        // Entities placeholder (not provided by API yet). Hide section.
        hide('an-entities');

        show('analyze-result-card');
      }catch(err){
        const el = document.getElementById('analyze-error');
        if(el){ el.textContent = `Error: ${err.message}`; show('analyze-error'); }
      }finally{
        hide('analyze-loading');
      }
    });
  }

  // Curation behavior
  const curateForm = document.getElementById('curate-form');
  const curateResult = document.getElementById('curate-result');
  const curateSort = document.getElementById('curate-sort');
  const curateMinScore = document.getElementById('curate-min-score');
  const curateSearch = document.getElementById('curate-search');
  // Optional controls (may not exist)
  const curateDiversify = document.getElementById('curate-diversify');
  const curatePrev = document.getElementById('curate-prev');
  const curateNext = document.getElementById('curate-next');
  const curatePageInfo = document.getElementById('curate-page-info');
  const curatePageSize = document.getElementById('curate-page-size');
  let curateData = [];
  let currentPage = 1;

  function renderCuration(){
    if(!curateResult) return;
    const minScore = Math.max(0, Math.min(100, parseInt(curateMinScore?.value || '0', 10) || 0));
    const search = (curateSearch?.value || '').toLowerCase();
    let results = [...curateData];
    results = results.filter(r => r.overall_score >= minScore);
    if(search){
      results = results.filter(r => (r.article.title || r.article.url || '').toLowerCase().includes(search));
    }
    const sortVal = curateSort?.value || 'score_desc';
    results.sort((a,b) => sortVal === 'score_asc' ? a.overall_score - b.overall_score : b.overall_score - a.overall_score);

    // Pagination
    const pageSize = parseInt(curatePageSize?.value || '10', 10) || 10;
    const totalPages = Math.max(1, Math.ceil(results.length / pageSize));
    if(currentPage > totalPages) currentPage = totalPages;
    if(currentPage < 1) currentPage = 1;
    const start = (currentPage - 1) * pageSize;
    const pageItems = results.slice(start, start + pageSize);

    setText('curate-count', String(results.length));
    if(curatePageInfo){ curatePageInfo.textContent = `Page ${currentPage} / ${totalPages}`; }
    if(curatePrev){ curatePrev.disabled = currentPage <= 1; }
    if(curateNext){ curateNext.disabled = currentPage >= totalPages; }

    curateResult.innerHTML = '';
    if(pageItems.length === 0){
      const empty = document.createElement('div');
      empty.className = 'list-card';
      empty.textContent = 'No results';
      curateResult.appendChild(empty);
      return;
    }
    pageItems.forEach(r => curateResult.appendChild(createCurationCard(r)));
  }

  function attachCurationFilters(){
    if(curateSort){ curateSort.addEventListener('change', renderCuration); }
    if(curateMinScore){ curateMinScore.addEventListener('input', renderCuration); }
    if(curateSearch){ curateSearch.addEventListener('input', renderCuration); }
    if(curatePageSize){ curatePageSize.addEventListener('change', () => { currentPage = 1; renderCuration(); }); }
    if(curatePrev){ curatePrev.addEventListener('click', () => { currentPage = Math.max(1, currentPage - 1); renderCuration(); }); }
    if(curateNext){ curateNext.addEventListener('click', () => { currentPage = currentPage + 1; renderCuration(); }); }
  }

  if(curateForm){
    attachCurationFilters();
    curateForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      hide('curate-error');
      show('curate-loading');
      const topic = document.getElementById('curate-topic').value.trim();
      const maxStr = document.getElementById('curate-max').value;
      const max_articles = maxStr ? parseInt(maxStr, 10) : undefined;
      const apply_diversity = curateDiversify ? !!curateDiversify.checked : true;
      try{
        const data = await postJson('/curate-topic', { topic, max_articles, apply_diversity });
        curateData = data.results || [];
        currentPage = 1;
        renderCuration();
      }catch(err){
        const el = document.getElementById('curate-error');
        if(el){ el.textContent = `Error: ${err.message}`; show('curate-error'); }
      }finally{
        hide('curate-loading');
      }
    });
  }
});
