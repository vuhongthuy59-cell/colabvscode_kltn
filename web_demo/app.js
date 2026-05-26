const state = {
  data: null,
  articleId: null,
  model: null,
  ticker: null,
  customAnalysis: null,
  activeTab: "real",
  search: "",
  dateFilter: "",
  category: "",
};

const relationColors = {
  parent_to_subsidiary: "#1f73d6",
  subsidiary_to_parent: "#087f8c",
  same_group: "#16875d",
  same_industry: "#b7791f",
  news_co_mention: "#6b4fc4",
};

const el = {
  tabButtons: [...document.querySelectorAll(".tab-button")],
  tabSections: [...document.querySelectorAll(".tab-section")],
  searchInput: document.getElementById("searchInput"),
  dateFilter: document.getElementById("dateFilter"),
  customTitle: document.getElementById("customTitle"),
  customDate: document.getElementById("customDate"),
  analyzeButton: document.getElementById("analyzeButton"),
  manualWarning: document.getElementById("manualWarning"),
  clearFiltersButton: document.getElementById("clearFiltersButton"),
  dataScopeNote: document.getElementById("dataScopeNote"),
  categoryFilter: document.getElementById("categoryFilter"),
  modelSelect: document.getElementById("modelSelect"),
  articleList: document.getElementById("articleList"),
  eventMeta: document.getElementById("eventMeta"),
  eventTitle: document.getElementById("eventTitle"),
  eventTags: document.getElementById("eventTags"),
  kpiTickersLabel: document.getElementById("kpiTickersLabel"),
  kpiTrueLabel: document.getElementById("kpiTrueLabel"),
  kpiPredLabel: document.getElementById("kpiPredLabel"),
  kpiErrorLabel: document.getElementById("kpiErrorLabel"),
  kpiTickers: document.getElementById("kpiTickers"),
  kpiTrue: document.getElementById("kpiTrue"),
  kpiPred: document.getElementById("kpiPred"),
  kpiError: document.getElementById("kpiError"),
  selectedModelLabel: document.getElementById("selectedModelLabel"),
  impactTitle: document.getElementById("impactTitle"),
  impactTable: document.getElementById("impactTable"),
  tickerSelect: document.getElementById("tickerSelect"),
  priceChartLabel: document.getElementById("priceChartLabel"),
  priceCanvas: document.getElementById("priceCanvas"),
  forecastChartLabel: document.getElementById("forecastChartLabel"),
  forecastActualValue: document.getElementById("forecastActualValue"),
  forecastPredValue: document.getElementById("forecastPredValue"),
  forecastGapValue: document.getElementById("forecastGapValue"),
  forecastCanvas: document.getElementById("forecastCanvas"),
  relationGraph: document.getElementById("relationGraph"),
  graphLegend: document.getElementById("graphLegend"),
  modelBars: document.getElementById("modelBars"),
  bottomPanelTitle: document.getElementById("bottomPanelTitle"),
  bottomPanelSubtitle: document.getElementById("bottomPanelSubtitle"),
  movementTableWrap: document.getElementById("movementTableWrap"),
  movementTable: document.getElementById("movementTable"),
};

const categoryRules = [
  ["legal_regulatory", ["xử phạt", "bị phạt", "thanh tra", "điều tra", "vi phạm", "đình chỉ", "hủy niêm yết", "cảnh báo"]],
  ["debt_bond", ["trái phiếu", "nợ", "đáo hạn", "chậm trả", "tái cấu trúc nợ"]],
  ["earnings", ["lợi nhuận", "lãi", "lỗ", "doanh thu", "kết quả kinh doanh", "bctc"]],
  ["ma_ownership", ["mua lại", "sáp nhập", "thâu tóm", "chuyển nhượng", "thoái vốn", "cổ đông lớn", "sở hữu"]],
  ["capital_issuance", ["phát hành", "tăng vốn", "cổ phiếu thưởng", "esop", "chào bán"]],
  ["dividend", ["cổ tức", "trả cổ tức", "giao dịch không hưởng quyền"]],
  ["leadership", ["bổ nhiệm", "từ nhiệm", "chủ tịch", "tổng giám đốc", "hđqt"]],
  ["project_contract", ["dự án", "hợp đồng", "hợp tác", "trúng thầu", "ký kết", "khởi công", "phát triển"]],
  ["market_industry", ["giá thép", "giá dầu", "ngành", "vn-index", "thị trường chứng khoán", "xuất khẩu"]],
];

const positiveWords = ["hợp tác", "phát triển", "tăng", "lãi", "trúng thầu", "ký kết", "mở rộng", "hoàn thành", "phục hồi"];
const negativeWords = ["lỗ", "giảm", "nợ", "chậm trả", "xử phạt", "vi phạm", "điều tra", "cảnh báo", "bán tháo"];

const asciiCategoryRules = [
  ["legal_regulatory", ["xu phat", "bi phat", "thanh tra", "dieu tra", "vi pham", "dinh chi", "huy niem yet", "canh bao"]],
  ["debt_bond", ["trai phieu", "no", "dao han", "cham tra", "tai cau truc no"]],
  ["earnings", ["loi nhuan", "lai", "lo", "doanh thu", "ket qua kinh doanh", "bctc"]],
  ["ma_ownership", ["mua lai", "sap nhap", "thau tom", "chuyen nhuong", "thoai von", "co dong lon", "so huu"]],
  ["capital_issuance", ["phat hanh", "tang von", "co phieu thuong", "esop", "chao ban"]],
  ["dividend", ["co tuc", "tra co tuc", "giao dich khong huong quyen"]],
  ["leadership", ["bo nhiem", "tu nhiem", "chu tich", "tong giam doc", "hdqt"]],
  ["project_contract", ["du an", "hop dong", "hop tac", "trung thau", "ky ket", "khoi cong", "phat trien"]],
  ["market_industry", ["gia thep", "gia dau", "nganh", "vn-index", "thi truong chung khoan", "xuat khau"]],
];

const asciiPositiveWords = [
  "hop tac",
  "phat trien",
  "dau tu",
  "du an moi",
  "nang luong moi",
  "mo rong",
  "tang",
  "lai",
  "trung thau",
  "ky ket",
  "tro lai",
  "co dong lon",
  "lam co dong lon",
  "mua vao",
  "chi hon",
  "hoan thanh",
  "phuc hoi",
];
const asciiNegativeWords = ["lo", "giam", "no", "cham tra", "xu phat", "vi pham", "dieu tra", "canh bao", "ban thao"];

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function byArticle(items) {
  const map = new Map();
  items.forEach((item) => {
    if (!map.has(item.article_id)) map.set(item.article_id, []);
    map.get(item.article_id).push(item);
  });
  return map;
}

function currentArticle() {
  if (state.customAnalysis) return state.customAnalysis.article;
  return state.data.articles.find((article) => article.article_id === state.articleId);
}

function articleMentions(articleId) {
  if (state.customAnalysis && articleId === "CUSTOM") return state.customAnalysis.mentions;
  return state.mentionsByArticle.get(articleId) || [];
}

function articlePredictions(articleId, model = state.model) {
  if (state.customAnalysis && articleId === "CUSTOM") return state.customAnalysis.predictions;
  return (state.predictionsByArticle.get(articleId) || [])
    .filter((row) => row.model === model)
    .sort((a, b) => Number(b.y_true || 0) - Number(a.y_true || 0));
}

function normalizeText(text) {
  return String(text || "")
    .replace(/\u0111/g, "d")
    .replace(/\u0110/g, "D")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\u0111/g, "d")
    .replace(/\u0110/g, "D")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function classifyCategory(title) {
  const clean = normalizeText(title);
  const found = asciiCategoryRules.find(([, words]) => words.some((word) => clean.includes(word)));
  return found ? found[0] : "other";
}

function hasPhrase(clean, phrase) {
  const escaped = phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
  return new RegExp(`(^|[^0-9a-zA-Z])${escaped}($|[^0-9a-zA-Z])`).test(clean);
}

function scoreSentiment(title) {
  const clean = normalizeText(title);
  const pos = asciiPositiveWords.filter((word) => hasPhrase(clean, word)).length;
  const neg = asciiNegativeWords.filter((word) => hasPhrase(clean, word)).length;
  return Math.max(-1, Math.min(1, (pos - neg) / Math.max(pos + neg, 1)));
}

function sentimentLabel(score) {
  if (score > 0.15) return "positive";
  if (score < -0.15) return "negative";
  return "neutral";
}

function titleTickerPrefixes(title) {
  const match = String(title || "").match(/^\s*([A-Z]{2,5}(?:\s*[,/&-]\s*[A-Z]{2,5})*)\s*:/);
  if (!match) return [];
  return match[1]
    .split(/[,/&-]/)
    .map((value) => value.trim().toUpperCase())
    .filter(Boolean);
}

function outOfUniverseTitleTickers(article) {
  const known = new Set(state.data.tickerMetadata.map((row) => row.ticker));
  return titleTickerPrefixes(article.title).filter((ticker) => !known.has(ticker));
}

function setActiveTab(tab) {
  state.activeTab = tab;
  if (tab === "model" && state.data && !state.predictionArticleIds.has(state.articleId)) {
    const firstModelPrediction = state.data.predictions.find((row) => row.model === state.model) || state.data.predictions[0];
    if (firstModelPrediction) {
      state.articleId = firstModelPrediction.article_id;
      state.ticker = firstModelPrediction.ticker;
    }
  }
  el.tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  el.tabSections.forEach((section) => {
    const visible = section.classList.contains(`${tab}-controls`);
    section.classList.toggle("hidden", !visible);
  });
  el.articleList.classList.toggle("hidden", tab === "manual");
  document.body.dataset.view = tab;
}

function detectTickers(title) {
  const clean = normalizeText(title);
  const direct = new Set();
  state.data.tickerMetadata.forEach((row) => {
    const ticker = row.ticker;
    const pattern = new RegExp(`(^|[^0-9a-zA-Z])${ticker}([^0-9a-zA-Z]|$)`, "i");
    if (pattern.test(title)) direct.add(ticker);
  });
  state.data.tickerAliases.forEach((row) => {
    const alias = normalizeText(row.alias);
    if (alias && alias.length >= 3 && clean.includes(alias)) direct.add(row.ticker);
  });
  return [...direct];
}

function metadata(ticker) {
  return state.metadataByTicker.get(ticker) || { ticker, industry: "unknown", sector: "unknown" };
}

function tradingIndexForDate(series, date) {
  const target = new Date(date);
  return series.findIndex((row) => new Date(row.date) >= target);
}

function movementForTicker(ticker, eventDate) {
  const series = state.pricesByTicker.get(ticker) || [];
  const start = tradingIndexForDate(series, eventDate);
  const horizons = [1, 2, 3, 5, 10];
  if (start < 0) return { ticker, baseDate: null, baseClose: null, returns: {} };
  const base = Number(series[start].close);
  const returns = {};
  horizons.forEach((days) => {
    const row = series[start + days];
    returns[days] = row ? (Number(row.close) / base - 1) : null;
  });
  return { ticker, baseDate: series[start].date, baseClose: base, returns };
}

function relatedTickersForDirect(directTickers, limit = 12) {
  const directSet = new Set(directTickers);
  const relatedRows = [];
  state.data.relationships.forEach((edge) => {
    const touchesDirect = directSet.has(edge.source_ticker) || directSet.has(edge.target_ticker);
    if (!touchesDirect) return;
    const ticker = directSet.has(edge.source_ticker) ? edge.target_ticker : edge.source_ticker;
    if (directSet.has(ticker)) return;
    relatedRows.push({
      ticker,
      relation_type: edge.relation_type,
      source_ticker: edge.source_ticker,
      target_ticker: edge.target_ticker,
      weight: Number(edge.weight || 0),
      score: relationshipScore(edge, directSet),
    });
  });
  return relatedRows
    .sort((a, b) => b.score - a.score)
    .filter((row, idx, arr) => arr.findIndex((item) => item.ticker === row.ticker) === idx)
    .slice(0, limit);
}

function movementsForEvent(article, predictions) {
  if (state.customAnalysis) return state.customAnalysis.movements;
  const directTickers = [
    ...new Set([
      ...articleMentions(article.article_id).map((row) => row.ticker),
      ...predictions.map((row) => row.ticker),
    ]),
  ];
  const related = relatedTickersForDirect(directTickers, 12);
  const directSet = new Set(directTickers);
  return [...directTickers, ...related.map((row) => row.ticker)].map((ticker) => {
    const relatedInfo = related.find((row) => row.ticker === ticker);
    return {
      ...movementForTicker(ticker, article.event_trading_date),
      meta: metadata(ticker),
      role: directSet.has(ticker) ? "Direct" : "Related",
      relation: relatedInfo?.relation_type || "mentioned",
      relationScore: relatedInfo?.score || 1,
    };
  });
}

function relationshipScore(edge, directTickers) {
  const relationBoost = {
    parent_to_subsidiary: 1.1,
    subsidiary_to_parent: 1.0,
    same_group: 0.85,
    same_industry: 0.45,
    news_co_mention: 0.65,
  }[edge.relation_type] || 0.5;
  const base = Number(edge.weight || 0.5) * relationBoost;
  return directTickers.has(edge.source_ticker) || directTickers.has(edge.target_ticker) ? base : base * 0.5;
}

function buildCustomAnalysis() {
  const title = el.customTitle.value.trim();
  const eventDate = el.customDate.value;
  if (!title || !eventDate) return;

  const tickers = detectTickers(title);
  el.manualWarning.classList.add("hidden");
  el.manualWarning.textContent = "";
  if (!tickers.length) {
    state.customAnalysis = null;
    el.manualWarning.textContent = "Không nhận diện được mã cổ phiếu thuộc universe 118 mã, nên không thực hiện phân tích.";
    el.manualWarning.classList.remove("hidden");
    return;
  }
  const category = classifyCategory(title);
  const sentiment = scoreSentiment(title);
  const directSet = new Set(tickers);
  const relatedRows = [];
  const industrySet = new Set(tickers.map((ticker) => metadata(ticker).industry).filter(Boolean));

  state.data.relationships.forEach((edge) => {
    const touchesDirect = directSet.has(edge.source_ticker) || directSet.has(edge.target_ticker);
    const sourceMeta = metadata(edge.source_ticker);
    const targetMeta = metadata(edge.target_ticker);
    const sameIndustryTouched =
      industrySet.has(sourceMeta.industry) || industrySet.has(targetMeta.industry);
    if (!touchesDirect && edge.relation_type !== "same_industry") return;
    if (!touchesDirect && !sameIndustryTouched) return;
    const candidate = directSet.has(edge.source_ticker) ? edge.target_ticker : edge.source_ticker;
    if (!directSet.has(candidate)) {
      relatedRows.push({
        ticker: candidate,
        relation_type: edge.relation_type,
        source_ticker: edge.source_ticker,
        target_ticker: edge.target_ticker,
        weight: Number(edge.weight || 0),
        score: relationshipScore(edge, directSet),
      });
    }
  });

  const related = relatedRows
    .sort((a, b) => b.score - a.score)
    .filter((row, idx, arr) => arr.findIndex((item) => item.ticker === row.ticker) === idx)
    .slice(0, 12);
  const allTickers = [...new Set([...tickers, ...related.map((row) => row.ticker)])];
  const movements = allTickers.map((ticker) => {
    const relatedInfo = related.find((row) => row.ticker === ticker);
    return {
      ...movementForTicker(ticker, eventDate),
      meta: metadata(ticker),
      role: directSet.has(ticker) ? "Direct" : "Related",
      relation: relatedInfo?.relation_type || "mentioned",
      relationScore: relatedInfo?.score || 1,
    };
  });

  state.customAnalysis = {
    article: {
      article_id: "CUSTOM",
      source: "User input",
      published_date: eventDate,
      event_trading_date: eventDate,
      title,
      category,
      general_sentiment: sentiment,
      sentiment_label: sentimentLabel(sentiment),
      n_mapped_tickers: tickers.length,
    },
    mentions: tickers.map((ticker) => ({
      article_id: "CUSTOM",
      ticker,
      company_name: metadata(ticker).company_name,
      is_primary: 1,
      mention_count: 1,
      relevance_score: 1,
      company_sentiment: sentiment,
    })),
    predictions: tickers.map((ticker) => {
      const move = movements.find((item) => item.ticker === ticker);
      return {
        model: "Post-news movement",
        article_id: "CUSTOM",
        event_trading_date: eventDate,
        ticker,
        y_true: move?.returns[5] ?? null,
        y_pred: null,
        absolute_error: null,
        category,
        sentiment_label: sentimentLabel(sentiment),
        general_sentiment: sentiment,
      };
    }),
    related,
    movements,
  };
  state.articleId = "CUSTOM";
  state.ticker = tickers[0] || related[0]?.ticker || null;
  setActiveTab("manual");
  render();
}

function filteredArticles() {
  const query = state.search.trim().toLowerCase();
  return state.data.articles.filter((article) => {
    if (state.activeTab === "model" && !state.predictionArticleIds.has(article.article_id)) return false;
    if (state.category && article.category !== state.category) return false;
    if (
      state.dateFilter &&
      article.event_trading_date !== state.dateFilter &&
      article.published_date !== state.dateFilter
    ) {
      return false;
    }
    if (!query) return true;
    const mentions = articleMentions(article.article_id).map((m) => m.ticker).join(" ");
    return `${article.title} ${article.category} ${mentions}`.toLowerCase().includes(query);
  });
}

function renderFilters() {
  const categories = [...new Set(state.data.articles.map((a) => a.category).filter(Boolean))].sort();
  categories.forEach((category) => {
    const option = document.createElement("option");
    option.value = category;
    option.textContent = category;
    el.categoryFilter.appendChild(option);
  });

  state.data.meta.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    el.modelSelect.appendChild(option);
  });
  el.modelSelect.value = state.model;
}

function renderArticleList() {
  const articles = filteredArticles();
  el.articleList.innerHTML = "";
  if (!articles.length) {
    el.articleList.innerHTML = `<div class="empty-state">Không có tin thật khớp bộ lọc hiện tại.</div>`;
    return;
  }
  const renderLimit = state.search.trim() ? 1000 : Number(state.data.meta.default_render_limit || 500);
  articles.slice(0, renderLimit).forEach((article) => {
    const mentions = articleMentions(article.article_id).map((m) => m.ticker).slice(0, 4).join(", ");
    const button = document.createElement("button");
    button.className = `article-card${article.article_id === state.articleId ? " active" : ""}`;
    button.innerHTML = `<strong>${article.title}</strong><span>${article.event_trading_date} · ${article.category} · ${mentions || "no ticker"}</span>`;
    button.addEventListener("click", () => {
      state.customAnalysis = null;
      setActiveTab(state.activeTab === "model" ? "model" : "real");
      state.articleId = article.article_id;
      const firstPrediction = articlePredictions(state.articleId)[0];
      state.ticker = firstPrediction?.ticker || articleMentions(state.articleId)[0]?.ticker || null;
      render();
    });
    el.articleList.appendChild(button);
  });
  if (articles.length > renderLimit) {
    const note = document.createElement("div");
    note.className = "empty-state";
    note.textContent = `Đang hiển thị ${renderLimit}/${articles.length} tin. Nhập mã cổ phiếu hoặc từ khóa để lọc hẹp hơn.`;
    el.articleList.appendChild(note);
  }
}

function renderHeader(article, predictions) {
  const mentions = articleMentions(article.article_id);
  const hasPredictions = predictions.length > 0;
  const avgTrue = predictions.reduce((sum, row) => sum + Number(row.y_true || 0), 0) / Math.max(predictions.length, 1);
  const avgPred = predictions.reduce((sum, row) => sum + Number(row.y_pred || 0), 0) / Math.max(predictions.length, 1);
  const avgError = predictions.reduce((sum, row) => sum + Number(row.absolute_error || 0), 0) / Math.max(predictions.length, 1);

  el.eventMeta.textContent = `${article.source || "Nguồn tin"} · Published ${article.published_date} · Event date ${article.event_trading_date}`;
  el.eventTitle.textContent = article.title;
  el.kpiTickersLabel.textContent = state.customAnalysis ? "Direct" : "Tickers";
  el.kpiTrueLabel.textContent = state.customAnalysis ? "Related" : hasPredictions ? "True vol" : "Model";
  el.kpiPredLabel.textContent = state.customAnalysis ? "Industries" : hasPredictions ? "Predicted" : "Category";
  el.kpiErrorLabel.textContent = state.customAnalysis ? "Max |+5d|" : hasPredictions ? "MAE" : "Sentiment";
  el.kpiTickers.textContent = String(mentions.length || predictions.length);
  el.kpiTrue.textContent = state.customAnalysis ? `${state.customAnalysis.related.length}` : hasPredictions ? pct(avgTrue) : "Chưa có";
  el.kpiPred.textContent = state.customAnalysis ? `${new Set(state.customAnalysis.movements.map((m) => m.meta.industry)).size}` : hasPredictions ? pct(avgPred) : article.category;
  el.kpiError.textContent = state.customAnalysis ? pct(Math.max(...state.customAnalysis.movements.map((m) => Math.abs(m.returns[5] || 0)), 0)) : hasPredictions ? pct(avgError) : article.sentiment_label;

  const sentimentClass = article.sentiment_label === "positive" ? "positive" : article.sentiment_label === "negative" ? "negative" : "";
  const outsideUniverse = state.customAnalysis ? [] : outOfUniverseTitleTickers(article);
  el.eventTags.innerHTML = [
    `<span class="tag">${article.category}</span>`,
    `<span class="tag ${sentimentClass}">${article.sentiment_label}</span>`,
    `<span class="tag">sentiment ${fmt(article.general_sentiment, 2)}</span>`,
    ...mentions.slice(0, 8).map((m) => `<span class="tag">${m.ticker}</span>`),
    ...outsideUniverse.map((ticker) => `<span class="tag warning">${ticker} ngoài universe</span>`),
  ].join("");
}

function renderImpactTable(predictions) {
  el.impactTitle.textContent = state.customAnalysis ? "Công ty bị ảnh hưởng trực tiếp" : predictions.length ? "Ảnh hưởng theo cổ phiếu" : "Ticker nhận diện trong tin thật";
  el.selectedModelLabel.textContent = state.customAnalysis
    ? "Các mã được nhận diện trực tiếp từ tiêu đề"
    : predictions.length
      ? `${state.model} · realized volatility horizon 5 ngày`
      : "Tin thật có thể chưa thuộc tập đánh giá model, nên chỉ hiển thị ticker/sentiment đã nhận diện";
  const headers = el.impactTable.closest("table").querySelectorAll("th");
  const labels = state.customAnalysis
    ? ["Mã", "Ngày gốc", "Giá đóng cửa", "+5 phiên", "Sentiment"]
    : predictions.length
      ? ["Mã", "Thực tế", "Dự báo", "Sai số", "Sentiment"]
      : ["Mã", "Vai trò", "Ngành", "Mapping", "Sentiment"];
  headers.forEach((header, idx) => {
    header.textContent = labels[idx] || header.textContent;
  });
  if (!state.customAnalysis && !predictions.length) {
    const mentions = articleMentions(currentArticle().article_id);
    el.impactTable.innerHTML = mentions.length
      ? mentions.map((mention) => {
        const meta = metadata(mention.ticker);
        return `
          <tr>
            <td class="ticker-cell">${mention.ticker}</td>
            <td>${Number(mention.is_primary) ? "Direct" : "Mention"}</td>
            <td>${meta.industry || "unknown"}</td>
            <td>${mention.mapping_method || "-"}</td>
            <td>${fmt(mention.company_sentiment, 2)}</td>
          </tr>
        `;
      }).join("")
      : `<tr><td colspan="5">Tin này chưa map được mã cổ phiếu trong universe.</td></tr>`;
    return;
  }
  el.impactTable.innerHTML = predictions
    .map((row) => {
      const mention = articleMentions(row.article_id).find((item) => item.ticker === row.ticker);
      const sentiment = mention ? fmt(mention.company_sentiment, 2) : "-";
      const meta = metadata(row.ticker);
      if (state.customAnalysis) {
        const move = state.customAnalysis.movements.find((item) => item.ticker === row.ticker);
        return `
          <tr>
            <td class="ticker-cell">${row.ticker}<span class="impact-note">${meta.industry || "unknown"}</span></td>
            <td>${move?.baseDate || "-"}</td>
            <td>${fmt(move?.baseClose, 2)}</td>
            <td>${pct(move?.returns[5])}</td>
            <td>${sentiment}</td>
          </tr>
        `;
      }
      return `
        <tr>
          <td class="ticker-cell">${row.ticker}</td>
          <td>${pct(row.y_true)}</td>
          <td>${pct(row.y_pred)}</td>
          <td>${pct(row.absolute_error)}</td>
          <td>${sentiment}</td>
        </tr>
      `;
    })
    .join("");
}

function renderTickerSelect(predictions) {
  const tickers = state.customAnalysis
    ? state.customAnalysis.movements.map((row) => row.ticker)
    : predictions.length
      ? predictions.map((row) => row.ticker)
      : articleMentions(currentArticle().article_id).map((row) => row.ticker);
  if (!tickers.includes(state.ticker)) state.ticker = tickers[0] || null;
  el.tickerSelect.innerHTML = tickers.map((ticker) => `<option value="${ticker}">${ticker}</option>`).join("");
  el.tickerSelect.value = state.ticker || "";
}

function renderPriceChart(article) {
  const canvas = el.priceCanvas;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(640, Math.floor(rect.width * dpr));
  canvas.height = Math.floor(300 * dpr);
  ctx.scale(dpr, dpr);

  const width = canvas.width / dpr;
  const height = canvas.height / dpr;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  if (!state.ticker) {
    el.priceChartLabel.textContent = "Không có ticker để vẽ";
    return;
  }

  const series = (state.pricesByTicker.get(state.ticker) || []).filter((row) => {
    const delta = (new Date(row.date) - new Date(article.event_trading_date)) / 86400000;
    return delta >= -30 && delta <= 15;
  });

  el.priceChartLabel.textContent = `${state.ticker} · close price window`;
  if (series.length < 2) {
    ctx.fillStyle = "#667085";
    ctx.fillText("Không đủ dữ liệu giá cho mã này.", 24, 40);
    return;
  }

  const values = series.map((row) => Number(row.close));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = 36;
  const x = (idx) => pad + (idx / (series.length - 1)) * (width - pad * 2);
  const y = (value) => height - pad - ((value - min) / Math.max(max - min, 0.0001)) * (height - pad * 2);

  ctx.strokeStyle = "#d9e1ec";
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i += 1) {
    const gy = pad + i * ((height - pad * 2) / 3);
    ctx.beginPath();
    ctx.moveTo(pad, gy);
    ctx.lineTo(width - pad, gy);
    ctx.stroke();
  }

  const eventIndex = series.findIndex((row) => row.date >= article.event_trading_date);
  if (eventIndex >= 0) {
    ctx.strokeStyle = "#c2413a";
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(x(eventIndex), pad);
    ctx.lineTo(x(eventIndex), height - pad);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.strokeStyle = "#1f73d6";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  series.forEach((row, idx) => {
    if (idx === 0) ctx.moveTo(x(idx), y(Number(row.close)));
    else ctx.lineTo(x(idx), y(Number(row.close)));
  });
  ctx.stroke();

  ctx.fillStyle = "#182230";
  ctx.font = "12px system-ui";
  ctx.fillText(`${fmt(max, 2)}`, 8, pad + 4);
  ctx.fillText(`${fmt(min, 2)}`, 8, height - pad);
  ctx.fillText(series[0].date, pad, height - 10);
  ctx.fillText(series[series.length - 1].date, width - pad - 78, height - 10);
  ctx.fillStyle = "#c2413a";
  ctx.fillText("event", Math.min(width - 72, Math.max(pad, x(Math.max(eventIndex, 0)) + 6)), pad + 14);
}

function setupCanvas(canvas, cssHeight, minWidth = 640) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(minWidth, Math.floor(rect.width * dpr));
  canvas.height = Math.floor(cssHeight * dpr);
  ctx.scale(dpr, dpr);
  const width = canvas.width / dpr;
  const height = canvas.height / dpr;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  return { ctx, width, height };
}

function drawLine(ctx, rows, x, y, key, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  rows.forEach((row, idx) => {
    const value = Number(row[key]);
    if (idx === 0) ctx.moveTo(x(idx), y(value));
    else ctx.lineTo(x(idx), y(value));
  });
  ctx.stroke();
}

function drawPoints(ctx, rows, x, y, key, color) {
  rows.forEach((row, idx) => {
    const value = Number(row[key]);
    ctx.beginPath();
    ctx.arc(x(idx), y(value), 4.5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = color;
    ctx.stroke();
  });
}

function renderForecastChart(article) {
  const { ctx, width, height } = setupCanvas(el.forecastCanvas, 260);
  const padLeft = 58;
  const padRight = 24;
  const padTop = 30;
  const padBottom = 46;

  if (state.customAnalysis) {
    el.forecastChartLabel.textContent = "Xanh là dữ liệu thật, đỏ là dữ liệu dự đoán. Tin nhập thủ công chưa có dữ liệu dự đoán từ model.";
    el.forecastActualValue.textContent = "-";
    el.forecastPredValue.textContent = "-";
    el.forecastGapValue.textContent = "-";
    ctx.fillStyle = "#667085";
    ctx.font = "13px system-ui";
    ctx.fillText("Không có y_pred cho tin nhập thủ công.", 24, 44);
    return;
  }

  const rows = state.data.predictions
    .filter((row) => row.model === state.model && row.ticker === state.ticker)
    .filter((row) => row.y_true !== null && row.y_pred !== null)
    .sort((a, b) => new Date(a.event_trading_date) - new Date(b.event_trading_date));

  el.forecastChartLabel.textContent = `${state.ticker || "-"} · ${state.model} · volatility 5 phiên`;
  if (rows.length < 2) {
    el.forecastActualValue.textContent = "-";
    el.forecastPredValue.textContent = "-";
    el.forecastGapValue.textContent = "-";
    ctx.fillStyle = "#667085";
    ctx.font = "13px system-ui";
    ctx.fillText("Không đủ điểm dự báo để vẽ đường so sánh.", 24, 44);
    return;
  }

  const selectedIndex = rows.findIndex((row) => row.article_id === article.article_id);
  const selected = selectedIndex >= 0 ? rows[selectedIndex] : rows[rows.length - 1];
  el.forecastActualValue.textContent = pct(selected.y_true);
  el.forecastPredValue.textContent = pct(selected.y_pred);
  el.forecastGapValue.textContent = pct(Math.abs(Number(selected.y_true) - Number(selected.y_pred)));
  const values = rows.flatMap((row) => [Number(row.y_true), Number(row.y_pred)]).filter(Number.isFinite);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const yMin = Math.max(0, min - (max - min) * 0.15);
  const yMax = max + (max - min) * 0.15;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;
  const x = (idx) => padLeft + (idx / (rows.length - 1)) * plotWidth;
  const y = (value) => padTop + (1 - (value - yMin) / Math.max(yMax - yMin, 0.0001)) * plotHeight;

  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(padLeft, padTop, plotWidth, plotHeight);
  ctx.strokeStyle = "#d9e1ec";
  ctx.lineWidth = 1;
  ctx.strokeRect(padLeft, padTop, plotWidth, plotHeight);

  ctx.fillStyle = "#475467";
  ctx.font = "12px system-ui";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const ticks = 5;
  for (let i = 0; i < ticks; i += 1) {
    const value = yMax - (i / (ticks - 1)) * (yMax - yMin);
    const gy = y(value);
    ctx.beginPath();
    ctx.moveTo(padLeft, gy);
    ctx.lineTo(width - padRight, gy);
    ctx.stroke();
    ctx.fillText(pct(value), padLeft - 10, gy);
  }

  drawLine(ctx, rows, x, y, "y_true", "#1f73d6");
  drawLine(ctx, rows, x, y, "y_pred", "#c2413a");
  if (rows.length <= 80) {
    drawPoints(ctx, rows, x, y, "y_true", "#1f73d6");
    drawPoints(ctx, rows, x, y, "y_pred", "#c2413a");
  }

  if (selectedIndex >= 0) {
    const sx = x(selectedIndex);
    ctx.fillStyle = "rgba(194, 65, 58, 0.08)";
    ctx.fillRect(Math.max(padLeft, sx - 8), padTop, 16, plotHeight);
    ctx.strokeStyle = "#c2413a";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(sx, padTop);
    ctx.lineTo(sx, height - padBottom);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#c2413a";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.font = "12px system-ui";
    ctx.fillText("tin đang chọn", sx, padTop + 8);

    ctx.fillStyle = "#182230";
    ctx.textBaseline = "bottom";
    ctx.fillText(`Thật ${pct(selected.y_true)} | Dự đoán ${pct(selected.y_pred)}`, sx, padTop - 6);

    [["y_true", "#1f73d6"], ["y_pred", "#c2413a"]].forEach(([key, color]) => {
      ctx.beginPath();
      ctx.arc(sx, y(Number(selected[key])), 6, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      ctx.stroke();
    });
  }

  ctx.fillStyle = "#182230";
  ctx.font = "12px system-ui";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(rows[0].event_trading_date, padLeft, height - padBottom + 14);
  ctx.textAlign = "right";
  ctx.fillText(rows[rows.length - 1].event_trading_date, width - padRight, height - padBottom + 14);
  ctx.save();
  ctx.translate(14, padTop + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillStyle = "#667085";
  ctx.fillText("Volatility (%)", 0, 0);
  ctx.restore();
}

function renderGraph(article, predictions) {
  const directTickers = new Set([
    ...articleMentions(article.article_id).map((m) => m.ticker),
    ...predictions.map((p) => p.ticker),
  ]);
  const relatedEdges = state.data.relationships.filter(
    (edge) => directTickers.has(edge.source_ticker) || directTickers.has(edge.target_ticker),
  );
  const edges = relatedEdges.slice(0, 40);
  const tickers = new Set(directTickers);
  edges.forEach((edge) => {
    tickers.add(edge.source_ticker);
    tickers.add(edge.target_ticker);
  });

  const nodes = [...tickers].slice(0, 18).map((ticker, idx, arr) => {
    const angle = (idx / Math.max(arr.length, 1)) * Math.PI * 2 - Math.PI / 2;
    const radius = directTickers.has(ticker) ? 112 : 168;
    return {
      ticker,
      primary: directTickers.has(ticker),
      x: 380 + Math.cos(angle) * radius,
      y: 210 + Math.sin(angle) * radius,
    };
  });
  const nodeSet = new Set(nodes.map((node) => node.ticker));
  const visibleEdges = edges.filter((edge) => nodeSet.has(edge.source_ticker) && nodeSet.has(edge.target_ticker));

  el.relationGraph.innerHTML = "";
  visibleEdges.forEach((edge) => {
    const source = nodes.find((node) => node.ticker === edge.source_ticker);
    const target = nodes.find((node) => node.ticker === edge.target_ticker);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", source.x);
    line.setAttribute("y1", source.y);
    line.setAttribute("x2", target.x);
    line.setAttribute("y2", target.y);
    line.setAttribute("stroke", relationColors[edge.relation_type] || "#98a2b3");
    line.setAttribute("stroke-width", String(1 + Number(edge.weight || 0)));
    line.setAttribute("opacity", "0.72");
    el.relationGraph.appendChild(line);
  });

  nodes.forEach((node) => {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", node.x);
    circle.setAttribute("cy", node.y);
    circle.setAttribute("r", node.primary ? "22" : "17");
    circle.setAttribute("fill", node.primary ? "#1f73d6" : "#ffffff");
    circle.setAttribute("stroke", node.primary ? "#1f73d6" : "#98a2b3");
    circle.setAttribute("stroke-width", "2");
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", node.x);
    text.setAttribute("y", node.y + 4);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "12");
    text.setAttribute("font-weight", "700");
    text.setAttribute("fill", node.primary ? "#ffffff" : "#182230");
    text.textContent = node.ticker;
    group.appendChild(circle);
    group.appendChild(text);
    el.relationGraph.appendChild(group);
  });

  const usedTypes = [...new Set(visibleEdges.map((edge) => edge.relation_type))];
  el.graphLegend.innerHTML = usedTypes.length
    ? usedTypes.map((type) => `<span><i style="background:${relationColors[type] || "#98a2b3"}"></i>${type}</span>`).join("")
    : "<span>Không có quan hệ doanh nghiệp trong tập demo cho tin này.</span>";
}

function renderModelBars() {
  if (state.customAnalysis) {
    el.bottomPanelTitle.textContent = "Biến động sau tin";
    el.bottomPanelSubtitle.textContent = "Return theo giá đóng cửa sau 1/2/3/5/10 phiên giao dịch";
    el.modelBars.classList.add("hidden");
    el.movementTableWrap.classList.remove("hidden");
    el.movementTable.innerHTML = state.customAnalysis.movements
      .map((row) => `
        <tr>
          <td class="ticker-cell">${row.ticker}</td>
          <td>${row.role}<span class="impact-note">${row.relation}</span></td>
          <td>${row.meta.industry || "unknown"}</td>
          <td>${pct(row.returns[1])}</td>
          <td>${pct(row.returns[2])}</td>
          <td>${pct(row.returns[3])}</td>
          <td>${pct(row.returns[5])}</td>
          <td>${pct(row.returns[10])}</td>
        </tr>
      `)
      .join("");
    return;
  }
  el.bottomPanelTitle.textContent = "So sánh mô hình";
  el.bottomPanelSubtitle.textContent = "MAE/RMSE trên tập test";
  el.modelBars.classList.remove("hidden");
  el.movementTableWrap.classList.add("hidden");
  const maxMae = Math.max(...state.data.modelMetrics.map((row) => Number(row.mae)));
  el.modelBars.innerHTML = state.data.modelMetrics
    .map((row) => {
      const width = Math.max(4, (Number(row.mae) / maxMae) * 100);
      const flags = [
        row.news_features === "Yes" ? "news" : null,
        row.relationship_edges === "Yes" ? "relationship" : null,
        row.co_mention_edges === "Yes" ? "co-mention" : null,
      ].filter(Boolean);
      return `
        <div class="bar-row">
          <div class="bar-top"><strong>${row.model}</strong><span>MAE ${pct(row.mae)} · RMSE ${pct(row.rmse)}</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <div class="bar-note">${flags.length ? flags.join(" · ") : "tabular/price baseline"}</div>
        </div>
      `;
    })
    .join("");
}

function renderModelBarsForEvent() {
  const article = currentArticle();
  const predictions = articlePredictions(article.article_id);
  const movements = movementsForEvent(article, predictions);
  el.bottomPanelTitle.textContent = state.customAnalysis ? "Biến động giá thật sau tin nhập" : "Biến động giá thật sau snapshot";
  el.bottomPanelSubtitle.textContent = "Dữ liệu thật từ giá đóng cửa sau 1/2/3/5/10 phiên; Direct là mã trong tin, Related là mã có quan hệ";
  el.modelBars.classList.add("hidden");
  el.movementTableWrap.classList.remove("hidden");
  el.movementTable.innerHTML = movements.length
    ? movements
      .map((row) => `
        <tr>
          <td class="ticker-cell">${row.ticker}</td>
          <td>${row.role}<span class="impact-note">${row.relation}</span></td>
          <td>${row.meta.industry || "unknown"}</td>
          <td>${pct(row.returns[1])}</td>
          <td>${pct(row.returns[2])}</td>
          <td>${pct(row.returns[3])}</td>
          <td>${pct(row.returns[5])}</td>
          <td>${pct(row.returns[10])}</td>
        </tr>
      `)
      .join("")
    : `<tr><td colspan="8">Không có dữ liệu giá sau snapshot cho tin này.</td></tr>`;
}

function render() {
  const article = currentArticle();
  if (!article) return;
  setActiveTab(state.activeTab);
  const predictions = articlePredictions(article.article_id);
  renderArticleList();
  renderHeader(article, predictions);
  renderImpactTable(predictions);
  renderTickerSelect(predictions);
  renderPriceChart(article);
  renderForecastChart(article);
  renderGraph(article, predictions);
  renderModelBarsForEvent();
}

async function init() {
  const response = await fetch("data/demo-data.json");
  const data = await response.json();
  state.data = data;
  state.model = data.meta.primary_model;
  state.articleId = data.articles[0]?.article_id;
  state.mentionsByArticle = byArticle(data.mentions);
  state.predictionsByArticle = byArticle(data.predictions);
  state.predictionArticleIds = new Set(data.predictions.map((row) => row.article_id));
  state.pricesByTicker = new Map();
  state.metadataByTicker = new Map();
  data.prices.forEach((row) => {
    if (!state.pricesByTicker.has(row.ticker)) state.pricesByTicker.set(row.ticker, []);
    state.pricesByTicker.get(row.ticker).push(row);
  });
  data.tickerMetadata.forEach((row) => state.metadataByTicker.set(row.ticker, row));
  state.ticker = articlePredictions(state.articleId)[0]?.ticker || null;
  const articleDates = data.articles.map((row) => row.event_trading_date).filter(Boolean).sort();
  if (articleDates.length) {
    el.dateFilter.min = articleDates[0];
    el.dateFilter.max = articleDates[articleDates.length - 1];
    el.dateFilter.title = `Dữ liệu thật: ${articleDates[0]} đến ${articleDates[articleDates.length - 1]}`;
    el.dataScopeNote.textContent = `Tin thật: ${data.meta.article_count}/${data.meta.available_processed_articles || data.meta.article_count} tin đã xử lý. Tin có dự báo model: ${data.meta.model_ready_article_count || state.predictionArticleIds.size}.`;
  }

  renderFilters();
  setActiveTab("real");
  render();
}

el.searchInput.addEventListener("input", (event) => {
  state.search = event.target.value;
  const articles = filteredArticles();
  if (articles.length && !articles.some((article) => article.article_id === state.articleId)) {
    state.articleId = articles[0].article_id;
  }
  render();
});

el.tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const tab = button.dataset.tab;
    if (tab === "real") {
      state.customAnalysis = null;
    }
    if (tab === "manual" && !state.customAnalysis) {
      el.manualWarning.textContent = "Nhập tiêu đề có mã cổ phiếu trong universe rồi bấm Phân tích tin nhập.";
      el.manualWarning.classList.remove("hidden");
    }
    setActiveTab(tab);
    render();
  });
});

el.clearFiltersButton.addEventListener("click", () => {
  state.search = "";
  state.dateFilter = "";
  state.category = "";
  el.searchInput.value = "";
  el.dateFilter.value = "";
  el.categoryFilter.value = "";
  const articles = filteredArticles();
  if (articles.length) {
    state.articleId = articles[0].article_id;
    const firstPrediction = articlePredictions(state.articleId)[0];
    state.ticker = firstPrediction?.ticker || articleMentions(state.articleId)[0]?.ticker || null;
  }
  render();
});

el.dateFilter.addEventListener("change", (event) => {
  state.dateFilter = event.target.value;
  state.customAnalysis = null;
  const articles = filteredArticles();
  if (articles.length) {
    state.articleId = articles[0].article_id;
    const firstPrediction = articlePredictions(state.articleId)[0];
    state.ticker = firstPrediction?.ticker || articleMentions(state.articleId)[0]?.ticker || null;
  }
  render();
});

el.categoryFilter.addEventListener("change", (event) => {
  state.category = event.target.value;
  const articles = filteredArticles();
  if (articles.length) state.articleId = articles[0].article_id;
  render();
});

el.modelSelect.addEventListener("change", (event) => {
  state.model = event.target.value;
  render();
});

el.tickerSelect.addEventListener("change", (event) => {
  state.ticker = event.target.value;
  renderPriceChart(currentArticle());
  renderForecastChart(currentArticle());
});

el.analyzeButton.addEventListener("click", buildCustomAnalysis);

window.addEventListener("resize", () => {
  if (state.data) {
    renderPriceChart(currentArticle());
    renderForecastChart(currentArticle());
  }
});

init().catch((error) => {
  console.error(error);
  el.eventMeta.textContent = "Không tải được dữ liệu demo";
  el.eventTitle.textContent = "Hãy chạy build_demo_data.py trước khi mở web.";
});
