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
  wrap.className = 'article-card';
  
  // Format publish date
  const publishDate = r.article.publish_date ? new Date(r.article.publish_date).toLocaleDateString() : '';
  const author = r.article.author ? `By ${r.article.author}` : '';
  const domain = r.domain || '';
  const metaInfo = [author, publishDate, domain].filter(Boolean).join(' • ');
  
  wrap.innerHTML = `
    <div class="flex items-start justify-between gap-4 mb-4">
      <div class="flex-1">
        <h3 class="text-lg font-semibold text-gray-900 mb-2">
          <a href="${r.article.url}" target="_blank" rel="noopener" class="hover:text-blue-600 transition-colors">${safeTitle}</a>
        </h3>
        ${metaInfo ? `<p class="text-sm text-gray-500 mb-2">${metaInfo}</p>` : ''}
        <div class="flex flex-wrap gap-2 text-xs text-gray-600">
          <span class="bg-green-100 text-green-800 px-2 py-1 rounded-full font-medium">Score: ${Math.round(r.overall_score)}</span>
          <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded-full">Read: ${Math.round(r.readability_score)}</span>
          <span class="bg-purple-100 text-purple-800 px-2 py-1 rounded-full">NER: ${Math.round(r.ner_density_score)}</span>
          <span class="bg-orange-100 text-orange-800 px-2 py-1 rounded-full">Sent: ${Math.round(r.sentiment_score)}</span>
          <span class="bg-indigo-100 text-indigo-800 px-2 py-1 rounded-full">Rel: ${Math.round(r.tfidf_relevance_score)}</span>
          <span class="bg-pink-100 text-pink-800 px-2 py-1 rounded-full">Rec: ${Math.round(r.recency_score)}</span>
        </div>
      </div>
      <div class="score-badge">${Math.round(r.overall_score)}</div>
    </div>
    ${r.article.summary ? `<p class="text-sm text-gray-600 leading-relaxed mb-4">${r.article.summary}</p>` : ''}
    ${r.article.entities && r.article.entities.length ? `
      <div class="flex flex-wrap gap-1">
        ${r.article.entities.slice(0, 8).map(e => `<span class="entity-chip">${e.label}: ${e.text}</span>`).join('')}
      </div>
    ` : ''}
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
      const use_embeddings = document.getElementById('use-embeddings')?.checked;
      try{
        const data = await postJson('/analyze', { url, query, use_embeddings });
        // Fill UI
        setText('an-title', data.article.title || data.article.url || 'Article');
        const pd = data.article.publish_date ? new Date(data.article.publish_date).toLocaleDateString() : '';
        const author = data.article.author ? `By ${data.article.author}` : '';
        setText('an-meta', [author, pd, data.stats?.domain ? `• ${data.stats.domain}` : ''].filter(Boolean).join(' '));
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
        // Entities list
        const entsWrap = document.getElementById('an-entities');
        const entsList = document.getElementById('an-entities-list');
        if(Array.isArray(data.article.entities) && data.article.entities.length){
          entsList.innerHTML = '';
          data.article.entities.slice(0, 12).forEach(e => {
            const span = document.createElement('span');
            span.className = 'entity-chip';
            span.textContent = `${e.label}: ${e.text}`;
            entsList.appendChild(span);
          });
          show('an-entities');
        }else{
          hide('an-entities');
        }

        // Stats
        setText('an-word-count', data.stats?.word_count ?? '');
        setText('an-entity-count', data.stats?.entity_count ?? '');
        setText('an-rel-method', data.stats?.relevance_method ?? 'tfidf');

        show('analyze-result-card');
      }catch(err){
        const el = document.getElementById('analyze-error');
        if(el){ el.textContent = `Error: ${err.message}`; show('analyze-error'); }
      }finally{
        hide('analyze-loading');
      }
    });

    // Export handlers for Analyze
    const exportAnalyzeJson = document.getElementById('analyze-export-json');
    const exportAnalyzeCsv = document.getElementById('analyze-export-csv');
    function download(filename, text){
      const blob = new Blob([text], {type: 'text/plain;charset=utf-8'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = filename; a.click();
      URL.revokeObjectURL(url);
    }
    function getAnalyzeData(){
      const title = document.getElementById('an-title').textContent;
      const authorMeta = document.getElementById('an-meta').textContent;
      return {
        title,
        meta: authorMeta,
        overall: document.getElementById('an-overall').textContent,
        readability: document.getElementById('an-readability-v').textContent,
        ner: document.getElementById('an-ner-v').textContent,
        sentiment: document.getElementById('an-sentiment-v').textContent,
        relevance: document.getElementById('an-tfidf-v').textContent,
        recency: document.getElementById('an-recency-v').textContent,
        words: document.getElementById('an-word-count').textContent,
        entities: document.getElementById('an-entity-count').textContent,
        method: document.getElementById('an-rel-method').textContent,
        summary: document.getElementById('an-summary').textContent,
      };
    }
    exportAnalyzeJson?.addEventListener('click', () => {
      const data = getAnalyzeData();
      download('analysis.json', JSON.stringify(data, null, 2));
    });
    exportAnalyzeCsv?.addEventListener('click', () => {
      const d = getAnalyzeData();
      const csv = ['title,meta,overall,readability,ner,sentiment,relevance,recency,words,entities,method,summary',
        `${JSON.stringify(d.title)},${JSON.stringify(d.meta)},${d.overall},${d.readability},${d.ner},${d.sentiment},${d.relevance},${d.recency},${d.words},${d.entities},${JSON.stringify(d.method)},${JSON.stringify(d.summary)}`
      ].join('\n');
      download('analysis.csv', csv);
    });
  }

  // Curation behavior
  const curateForm = document.getElementById('curate-form');
  const curateResult = document.getElementById('curate-result');
  const curateSort = document.getElementById('curate-sort');
  const curateMinScore = document.getElementById('curate-min-score');
  const curateSearch = document.getElementById('curate-search');
  const curateDomain = document.getElementById('curate-domain');
  const curateDateFrom = document.getElementById('curate-date-from');
  const curateDateTo = document.getElementById('curate-date-to');
  const curateWordMin = document.getElementById('curate-word-min');
  const curateWordMax = document.getElementById('curate-word-max');
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
    // Facets
    const domain = (curateDomain?.value || '').toLowerCase();
    const fromTs = curateDateFrom?.value ? Date.parse(curateDateFrom.value) : null;
    const toTs = curateDateTo?.value ? Date.parse(curateDateTo.value) : null;
    const wordMin = parseInt(curateWordMin?.value || '0', 10) || 0;
    const wordMax = parseInt(curateWordMax?.value || '0', 10) || 0;

    if(domain){
      results = results.filter(r => (r.domain || '').toLowerCase().includes(domain));
    }
    if(fromTs){
      results = results.filter(r => r.article.publish_date ? Date.parse(r.article.publish_date) >= fromTs : true);
    }
    if(toTs){
      results = results.filter(r => r.article.publish_date ? Date.parse(r.article.publish_date) <= toTs : true);
    }
    if(wordMin){
      results = results.filter(r => (r.word_count || 0) >= wordMin);
    }
    if(wordMax){
      results = results.filter(r => (r.word_count || 0) <= wordMax);
    }
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
      empty.className = 'article-card text-center py-8';
      empty.innerHTML = `
        <div class="text-gray-400 mb-2">
          <i class="fa-solid fa-search text-2xl"></i>
        </div>
        <p class="text-gray-600">No articles found matching your criteria</p>
      `;
      curateResult.appendChild(empty);
      return;
    }
    pageItems.forEach(r => curateResult.appendChild(createCurationCard(r)));
  }

  function attachCurationFilters(){
    if(curateSort){ curateSort.addEventListener('change', renderCuration); }
    if(curateMinScore){ curateMinScore.addEventListener('input', renderCuration); }
    if(curateSearch){ curateSearch.addEventListener('input', renderCuration); }
    if(curateDomain){ curateDomain.addEventListener('input', renderCuration); }
    if(curateDateFrom){ curateDateFrom.addEventListener('change', renderCuration); }
    if(curateDateTo){ curateDateTo.addEventListener('change', renderCuration); }
    if(curateWordMin){ curateWordMin.addEventListener('input', renderCuration); }
    if(curateWordMax){ curateWordMax.addEventListener('input', renderCuration); }
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

    // Export handlers for Curation
    const exportCurateJson = document.getElementById('curate-export-json');
    const exportCurateCsv = document.getElementById('curate-export-csv');
    exportCurateJson?.addEventListener('click', () => {
      const payload = { count: curateData.length, results: curateData };
      download('curation.json', JSON.stringify(payload, null, 2));
    });
    exportCurateCsv?.addEventListener('click', () => {
      const header = 'title,url,domain,overall,readability,ner,sentiment,relevance,recency,words,date';
      const rows = curateData.map(r => [
        JSON.stringify(r.article.title || ''),
        JSON.stringify(r.article.url || ''),
        JSON.stringify(r.domain || ''),
        Math.round(r.overall_score),
        Math.round(r.readability_score),
        Math.round(r.ner_density_score),
        Math.round(r.sentiment_score),
        Math.round(r.tfidf_relevance_score),
        Math.round(r.recency_score),
        r.word_count || 0,
        JSON.stringify(r.article.publish_date || ''),
      ].join(','));
      download('curation.csv', [header, ...rows].join('\n'));
    });
  }
});
