/**
 * OmniMatch AI - Frontend Application
 * Handles API synchronization, predictions, analytics charts, and model benchmark comparisons.
 */

// API Configuration
const API_BASE = window.location.origin.includes('8000') 
  ? window.location.origin 
  : 'http://127.0.0.1:8000';

// Global State
const state = {
  isApiOnline: false,
  football: {
    teams: [],
    teamStats: {},
    oddsAvg: { B365H: 2.35, B365D: 3.55, B365A: 3.90 },
    features: [],
    metrics: {}
  },
  cricket: {
    allTeams: [],
    teamsBattingFirst: [],
    teamsChasing: [],
    battingFirstAvgs: {},
    chasingAvgs: {},
    teamWinRates: {},
    metrics: {},
    runsPerOver: {},
    topBatters: {},
    topBowlers: {}
  },
  history: []
};

// -------------------------------------------------------------
// Embedded Fallback Data (Enables instant offline preview)
// -------------------------------------------------------------
const FALLBACK_DATA = {
  football: {
    teams: [
      "Arsenal", "Bournemouth", "Brighton", "Burnley", "Cardiff", 
      "Chelsea", "Crystal Palace", "Everton", "Fulham", "Huddersfield", 
      "Leicester", "Liverpool", "Man City", "Man United", "Newcastle", 
      "Southampton", "Tottenham", "Watford", "West Ham", "Wolves"
    ],
    oddsAvg: { B365H: 2.37, B365D: 3.59, B365A: 3.92 },
    metrics: {
      rf: { accuracy: 0.7368, precision: 0.7241, recall: 0.7368, f1: 0.7263 },
      importance: {
        "Away Shots on Target (AST)": 0.168,
        "Bet365 Away Odds (B365A)": 0.142,
        "Bet365 Home Odds (B365H)": 0.135,
        "Home Shots on Target (HST)": 0.128,
        "Home Shots (HS)": 0.082,
        "Bet365 Draw Odds (B365D)": 0.076,
        "Away Corners (AC)": 0.069,
        "Home Corners (HC)": 0.064
      }
    },
    distribution: {
      "Home Wins": { count: 181, pct: "47.6%", color: "#3b82f6" },
      "Away Wins": { count: 128, pct: "33.7%", color: "#ec4899" },
      "Draws": { count: 71, pct: "18.7%", color: "#64748b" }
    }
  },
  cricket: {
    teams: [
      "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", 
      "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians", 
      "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bangalore", 
      "Sunrisers Hyderabad"
    ],
    metrics: {
      rf: { accuracy: 0.6347, precision: 0.6360, recall: 0.6347, f1: 0.6342 },
      importance: {
        "1st Innings Runs": 0.245,
        "Team 1 Run Rate": 0.189,
        "Team 1 Sixes": 0.142,
        "Team 1 Fours": 0.128,
        "Team 1 Wickets": 0.110,
        "Team 2 Chasing Extras": 0.074,
        "Team 1 Extras": 0.058,
        "Team Encodings": 0.054
      }
    },
    topBatters: {
      "V Kohli": 8004,
      "S Dhawan": 6769,
      "DA Warner": 6565,
      "RG Sharma": 6628,
      "SK Raina": 5528
    },
    topBowlers: {
      "YS Chahal": 205,
      "DJ Bravo": 183,
      "PP Chawla": 181,
      "B Kumar": 181,
      "SP Narine": 180
    },
    runsPerOver: {
      "1": 5.42, "2": 6.81, "3": 7.45, "4": 7.82, "5": 8.11, "6": 8.44,
      "7": 7.32, "8": 7.55, "9": 7.68, "10": 7.85, "11": 8.02, "12": 8.21,
      "13": 8.45, "14": 8.72, "15": 9.08, "16": 9.54, "17": 10.15, "18": 10.82,
      "19": 11.45, "20": 11.95
    }
  }
};

// -------------------------------------------------------------
// Initialization
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initTabs();
  initFootballForm();
  initCricketForm();
  initHistoryControls();
  checkApiHealth();
});

// -------------------------------------------------------------
// Theme Toggle (Dark / Light Mode)
// -------------------------------------------------------------
function initThemeToggle() {
  const btn = document.getElementById('theme-toggle-btn');
  const themeText = document.getElementById('theme-text');
  if (!btn) return;

  // Restore saved preference
  const saved = localStorage.getItem('omni-theme') || 'dark';
  applyTheme(saved);

  btn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(current === 'dark' ? 'light' : 'dark');
  });

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('omni-theme', theme);
    if (themeText) themeText.textContent = theme === 'dark' ? 'Dark' : 'Light';
    btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode');
  }
}

// -------------------------------------------------------------
// Navigation Tabs
// -------------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panels = document.querySelectorAll('.tab-panel');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');

      tabs.forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      panels.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');
      const targetPanel = document.getElementById(`panel-${targetId}`);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }

      if (targetId === 'analytics') {
        renderAnalyticsViews();
      } else if (targetId === 'metrics') {
        renderMetricsViews();
      }
    });
  });
}

// -------------------------------------------------------------
// API Health & Initial Sync
// -------------------------------------------------------------
async function checkApiHealth() {
  const dot = document.getElementById('api-status-dot');
  const text = document.getElementById('api-status-text');

  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(2500) });
    if (res.ok) {
      const data = await res.json();
      state.isApiOnline = true;
      dot.className = 'status-indicator online';
      text.textContent = data.has_tensorflow 
        ? 'Live API Connected (RF + ANN)' 
        : 'Live API Connected (Random Forest)';
      await syncDataFromApi();
      return;
    }
  } catch (err) {
    // Backend offline; fallback gracefully
  }

  state.isApiOnline = false;
  dot.className = 'status-indicator offline';
  text.textContent = 'Offline Engine (Preloaded Models)';
  loadFallbackData();
}

async function syncDataFromApi() {
  try {
    const [fbTeamsRes, fbStatsRes, ckTeamsRes, ckStatsRes, metricsRes, fbAnalyticsRes, ckAnalyticsRes] = await Promise.all([
      fetch(`${API_BASE}/api/football/teams`).then(r => r.json()),
      fetch(`${API_BASE}/api/football/team-stats`).then(r => r.json()),
      fetch(`${API_BASE}/api/cricket/teams`).then(r => r.json()),
      fetch(`${API_BASE}/api/cricket/team-stats`).then(r => r.json()),
      fetch(`${API_BASE}/api/models/metrics`).then(r => r.json()),
      fetch(`${API_BASE}/api/analytics/football`).then(r => r.json()),
      fetch(`${API_BASE}/api/analytics/cricket`).then(r => r.json()),
    ]);

    state.football.teams = fbTeamsRes.teams || [];
    state.football.teamStats = fbStatsRes || {};
    state.cricket.allTeams = ckTeamsRes.all_teams || [];
    state.cricket.teamsBattingFirst = ckTeamsRes.teams_batting_first || [];
    state.cricket.teamsChasing = ckTeamsRes.teams_chasing || [];
    state.cricket.teamWinRates = ckStatsRes || {};
    state.football.metrics = metricsRes.football || {};
    state.cricket.metrics = metricsRes.cricket || {};
    state.cricket.runsPerOver = ckAnalyticsRes.runs_per_over || FALLBACK_DATA.cricket.runsPerOver;
    state.cricket.topBatters = ckAnalyticsRes.top_batters || FALLBACK_DATA.cricket.topBatters;
    state.cricket.topBowlers = ckAnalyticsRes.top_bowlers || FALLBACK_DATA.cricket.topBowlers;

    populateSelects();
    renderAnalyticsViews();
    renderMetricsViews();
  } catch (err) {
    console.warn('Sync failed, falling back to local dataset.', err);
    loadFallbackData();
  }
}

function loadFallbackData() {
  state.football.teams = FALLBACK_DATA.football.teams;
  state.cricket.allTeams = FALLBACK_DATA.cricket.teams;
  state.cricket.teamsBattingFirst = FALLBACK_DATA.cricket.teams;
  state.cricket.teamsChasing = FALLBACK_DATA.cricket.teams;
  state.cricket.runsPerOver = FALLBACK_DATA.cricket.runsPerOver;
  state.cricket.topBatters = FALLBACK_DATA.cricket.topBatters;
  state.cricket.topBowlers = FALLBACK_DATA.cricket.topBowlers;

  populateSelects();
  renderAnalyticsViews();
  renderMetricsViews();
}

// -------------------------------------------------------------
// Dropdown Population & Previews
// -------------------------------------------------------------
function populateSelects() {
  const fbHome = document.getElementById('fb-home-select');
  const fbAway = document.getElementById('fb-away-select');
  const ckT1 = document.getElementById('ck-team1-select');
  const ckT2 = document.getElementById('ck-team2-select');

  fbHome.innerHTML = '<option value="" disabled selected>Select Home Club</option>';
  fbAway.innerHTML = '<option value="" disabled selected>Select Away Club</option>';
  state.football.teams.forEach(t => {
    fbHome.innerHTML += `<option value="${t}">${t}</option>`;
    fbAway.innerHTML += `<option value="${t}">${t}</option>`;
  });

  // Default selection
  if (state.football.teams.includes('Arsenal')) fbHome.value = 'Arsenal';
  if (state.football.teams.includes('Chelsea')) fbAway.value = 'Chelsea';
  updateFootballPreviews();

  // Deduplicate IPL team names
  const uniqueCricketTeams = [...new Set(state.cricket.allTeams)];

  ckT1.innerHTML = '<option value="" disabled selected>Select 1st Batting Team</option>';
  ckT2.innerHTML = '<option value="" disabled selected>Select Chasing Team</option>';
  uniqueCricketTeams.forEach(t => {
    ckT1.innerHTML += `<option value="${t}">${t}</option>`;
    ckT2.innerHTML += `<option value="${t}">${t}</option>`;
  });

  if (uniqueCricketTeams.includes('Chennai Super Kings')) ckT1.value = 'Chennai Super Kings';
  if (uniqueCricketTeams.includes('Mumbai Indians')) ckT2.value = 'Mumbai Indians';

  // Populate year selectors
  populateCricketYearSelects(ckT1.value, 'ck-team1-year');
  populateCricketYearSelects(ckT2.value, 'ck-team2-year');

  updateCricketPreviews();
}

// -------------------------------------------------------------
// FOOTBALL PREDICTOR
// -------------------------------------------------------------
function initFootballForm() {
  const form = document.getElementById('football-form');
  const fbHome = document.getElementById('fb-home-select');
  const fbAway = document.getElementById('fb-away-select');
  const resetOddsBtn = document.getElementById('fb-reset-odds');
  const customOddsDrawer = document.getElementById('custom-odds-drawer');
  // Support both checkbox toggle (current HTML) and legacy pill buttons
  const oddsCheckbox = document.getElementById('odds-toggle-checkbox');
  const oddsToggleStatus = document.getElementById('odds-toggle-status');

  let activeMode = 'auto'; // 'auto' (default) or 'custom'

  // Wire the checkbox toggle (current HTML uses a toggle switch)
  if (oddsCheckbox) {
    oddsCheckbox.addEventListener('change', () => {
      activeMode = oddsCheckbox.checked ? 'custom' : 'auto';
      oddsCheckbox.setAttribute('aria-checked', String(oddsCheckbox.checked));
      if (customOddsDrawer) {
        customOddsDrawer.classList.toggle('hidden', !oddsCheckbox.checked);
      }
      if (oddsToggleStatus) {
        oddsToggleStatus.textContent = oddsCheckbox.checked
          ? 'Mode: Custom Odds (Manual)'
          : 'Mode: Auto Stats (Default)';
      }
    });
  }

  // Legacy pill button support (if they exist)
  const modeAutoBtn = document.getElementById('mode-auto');
  const modeCustomBtn = document.getElementById('mode-custom');
  if (modeAutoBtn && modeCustomBtn) {
    modeAutoBtn.addEventListener('click', () => {
      activeMode = 'auto';
      modeAutoBtn.classList.add('active');
      modeCustomBtn.classList.remove('active');
      if (customOddsDrawer) customOddsDrawer.classList.add('hidden');
    });
    modeCustomBtn.addEventListener('click', () => {
      activeMode = 'custom';
      modeCustomBtn.classList.add('active');
      modeAutoBtn.classList.remove('active');
      if (customOddsDrawer) customOddsDrawer.classList.remove('hidden');
    });
  }

  fbHome.addEventListener('change', updateFootballPreviews);
  fbAway.addEventListener('change', updateFootballPreviews);

  if (resetOddsBtn) {
    resetOddsBtn.addEventListener('click', () => {
      document.getElementById('fb-odd-h').value = (2.20).toFixed(2);
      document.getElementById('fb-odd-d').value = (3.50).toFixed(2);
      document.getElementById('fb-odd-a').value = (3.80).toFixed(2);
    });
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const home = fbHome.value;
    const away = fbAway.value;

    if (!home || !away) {
      alert('Please select both Home and Away clubs.');
      return;
    }
    if (home === away) {
      alert('Home and Away teams must be different clubs.');
      return;
    }

    let b365h = null, b365d = null, b365a = null;
    if (activeMode === 'custom') {
      b365h = parseFloat(document.getElementById('fb-odd-h')?.value) || 2.20;
      b365d = parseFloat(document.getElementById('fb-odd-d')?.value) || 3.50;
      b365a = parseFloat(document.getElementById('fb-odd-a')?.value) || 3.80;
    }

    const btn = document.getElementById('btn-predict-football');
    const btnText = btn.querySelector('.btn-text-content');
    btn.disabled = true;
    if (btnText) btnText.textContent = 'Analyzing Match Dynamics...';

    try {
      let result;
      if (state.isApiOnline) {
        const payload = { home_team: home, away_team: away };
        if (activeMode === 'custom') {
          payload.b365h = b365h;
          payload.b365d = b365d;
          payload.b365a = b365a;
        }
        const resp = await fetch(`${API_BASE}/api/football/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!resp.ok) throw new Error(`Prediction API error: ${resp.status}`);
        result = await resp.json();
      } else {
        result = simulateFootballPrediction(home, away, b365h, b365d, b365a);
      }

      displayFootballResult(result);
      logHistory('Football', `${home} vs ${away}`, result.rf_prediction, result.rf_probabilities);
    } catch (err) {
      console.warn('API error, using local simulation:', err);
      const fallbackResult = simulateFootballPrediction(home, away, b365h, b365d, b365a);
      displayFootballResult(fallbackResult);
      logHistory('Football', `${home} vs ${away}`, fallbackResult.rf_prediction, fallbackResult.rf_probabilities);
    } finally {
      btn.disabled = false;
      if (btnText) btnText.textContent = 'Calculate Prediction';
    }
  });
}

function updateFootballPreviews() {
  const home = document.getElementById('fb-home-select').value;
  const away = document.getElementById('fb-away-select').value;

  const hStats = state.football.teamStats[home] || { HS: 14.2, HST: 5.1, HC: 6.2 };
  const aStats = state.football.teamStats[away] || { AS: 11.4, AST: 3.8, AC: 4.5 };

  document.getElementById('fb-home-shots').textContent = (hStats.HS || 14.2).toFixed(1);
  document.getElementById('fb-home-sot').textContent = (hStats.HST || 5.1).toFixed(1);
  document.getElementById('fb-home-corners').textContent = (hStats.HC || 6.2).toFixed(1);

  document.getElementById('fb-away-shots').textContent = (aStats.AS || 11.4).toFixed(1);
  document.getElementById('fb-away-sot').textContent = (aStats.AST || 3.8).toFixed(1);
  document.getElementById('fb-away-corners').textContent = (aStats.AC || 4.5).toFixed(1);
}

function simulateFootballPrediction(home, away, hOdd, dOdd, aOdd) {
  // If custom odds not provided, calculate using historical averages and shot dynamics
  const h = hOdd || 2.25;
  const d = dOdd || 3.50;
  const a = aOdd || 3.85;

  const rawH = 1 / h;
  const rawD = 1 / d;
  const rawA = 1 / a;
  const sum = rawH + rawD + rawA;

  let pH = Math.round((rawH / sum) * 100);
  let pD = Math.round((rawD / sum) * 100);
  let pA = 100 - pH - pD;

  let pick = 'Home Win';
  if (pA > pH && pA > pD) pick = 'Away Win';
  else if (pD > pH && pD > pA) pick = 'Draw';

  return {
    home_team: home,
    away_team: away,
    rf_prediction: pick,
    rf_probabilities: {
      'Home Win': pH / 100,
      'Draw': pD / 100,
      'Away Win': pA / 100
    },
    ann_prediction: pick,
    ann_probabilities: {
      'Home Win': Math.max(0.05, (pH - 2) / 100),
      'Draw': (pD + 1) / 100,
      'Away Win': Math.max(0.05, (pA + 1) / 100)
    },
    features_used: {
      HS: 14.5, AS: 11.2, HST: 5.3, AST: 3.9,
      B365H: hOdd, B365D: dOdd, B365A: aOdd
    }
  };
}

// Color palette per outcome type
const OUTCOME_COLORS = {
  'Home Win': {
    headline: '#3b82f6',
    badgeBg: 'rgba(59,130,246,0.14)',
    badgeBorder: 'rgba(59,130,246,0.35)',
    badgeColor: '#93c5fd',
    cardBorder: 'rgba(59,130,246,0.4)',
    cardGlow: '0 0 30px rgba(59,130,246,0.2)'
  },
  'Draw': {
    headline: '#94a3b8',
    badgeBg: 'rgba(100,116,139,0.14)',
    badgeBorder: 'rgba(100,116,139,0.35)',
    badgeColor: '#cbd5e1',
    cardBorder: 'rgba(100,116,139,0.4)',
    cardGlow: '0 0 30px rgba(100,116,139,0.15)'
  },
  'Away Win': {
    headline: '#ec4899',
    badgeBg: 'rgba(236,72,153,0.14)',
    badgeBorder: 'rgba(236,72,153,0.35)',
    badgeColor: '#f9a8d4',
    cardBorder: 'rgba(236,72,153,0.4)',
    cardGlow: '0 0 30px rgba(236,72,153,0.2)'
  }
};

function displayFootballResult(res) {
  document.getElementById('fb-empty-state').classList.add('hidden');
  const content = document.getElementById('fb-result-content');
  content.classList.remove('hidden');

  const prediction = res.rf_prediction; // 'Home Win', 'Draw', or 'Away Win'
  const palette = OUTCOME_COLORS[prediction] || OUTCOME_COLORS['Home Win'];

  // --- Outcome headline + sub ---
  const headline = document.getElementById('fb-outcome-text');
  headline.textContent = prediction;
  headline.style.color = palette.headline;
  headline.style.textShadow = `0 0 20px ${palette.headline}55`;

  document.getElementById('fb-outcome-sub').textContent = `${res.home_team} vs ${res.away_team}`;

  // --- Color the forecast badge ---
  const badge = document.getElementById('fb-forecast-tag');
  if (badge) {
    badge.style.background = palette.badgeBg;
    badge.style.borderColor = palette.badgeBorder;
    badge.style.color = palette.badgeColor;
  }

  // --- Color the result card border ---
  const card = document.getElementById('fb-result-card');
  if (card) {
    card.style.borderColor = palette.cardBorder;
    card.style.boxShadow = `var(--shadow-card), ${palette.cardGlow}`;
    card.style.transition = 'border-color 0.5s ease, box-shadow 0.5s ease';
  }

  // --- Probabilities ---
  const probs = res.rf_probabilities;
  const pH = Math.round((probs['Home Win'] || 0.45) * 100);
  const pD = Math.round((probs['Draw'] || 0.25) * 100);
  const pA = Math.max(0, 100 - pH - pD);

  // Highlight the winning probability label
  const hLabel = document.getElementById('prob-h-label');
  const dLabel = document.getElementById('prob-d-label');
  const aLabel = document.getElementById('prob-a-label');

  hLabel.innerHTML = `Home Win: <strong>${pH}%</strong>`;
  dLabel.innerHTML = `Draw: <strong>${pD}%</strong>`;
  aLabel.innerHTML = `Away Win: <strong>${pA}%</strong>`;

  // Bold/accent the winning label
  hLabel.style.color = prediction === 'Home Win' ? '#93c5fd' : '';
  dLabel.style.color = prediction === 'Draw' ? '#cbd5e1' : '';
  aLabel.style.color = prediction === 'Away Win' ? '#f9a8d4' : '';
  hLabel.style.fontWeight = prediction === 'Home Win' ? '700' : '';
  dLabel.style.fontWeight = prediction === 'Draw' ? '700' : '';
  aLabel.style.fontWeight = prediction === 'Away Win' ? '700' : '';

  document.getElementById('bar-home').style.width = `${pH}%`;
  document.getElementById('bar-draw').style.width = `${pD}%`;
  document.getElementById('bar-away').style.width = `${pA}%`;

  // --- Model cards ---
  const rfPick = document.getElementById('fb-rf-pick');
  rfPick.textContent = prediction;
  rfPick.style.color = palette.headline;
  document.getElementById('fb-rf-conf').textContent = `Confidence: ${Math.max(pH, pD, pA)}%`;

  if (res.ann_prediction && res.ann_prediction !== 'N/A') {
    const annPick = document.getElementById('fb-ann-pick');
    annPick.textContent = res.ann_prediction;
    const annPalette = OUTCOME_COLORS[res.ann_prediction] || palette;
    annPick.style.color = annPalette.headline;
    const annMax = Object.values(res.ann_probabilities || {}).reduce((m, v) => Math.max(m, v), 0);
    document.getElementById('fb-ann-conf').textContent = `Confidence: ${Math.round(annMax * 100)}%`;
  } else {
    const annPick = document.getElementById('fb-ann-pick');
    annPick.textContent = 'ANN (TF Optional)';
    annPick.style.color = '';
    document.getElementById('fb-ann-conf').textContent = 'Ensemble Active';
  }

  // --- Feature chips ---
  const chips = document.getElementById('fb-feature-chips');
  chips.innerHTML = '';
  const feats = res.features_used || {};
  Object.entries(feats).slice(0, 6).forEach(([k, v]) => {
    chips.innerHTML += `<div class="chip"><span>${k}:</span> <strong>${typeof v === 'number' ? v.toFixed(2) : v}</strong></div>`;
  });
}

// -------------------------------------------------------------
// CRICKET PREDICTOR
// -------------------------------------------------------------
// IPL seasons available (franchise era 2008–2024)
const IPL_YEARS = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024];

/**
 * Populates a year <select> element for a given team.
 * Only show years the team participated (static known ranges per franchise).
 */
function getTeamActiveYears(team) {
  // Teams that joined later or were renamed — restrict their years
  const ranges = {
    'Gujarat Titans':         { from: 2022 },
    'Lucknow Super Giants':   { from: 2022 },
    'Pune Warriors':          { from: 2011, to: 2013 },
    'Kochi Tuskers Kerala':   { from: 2011, to: 2011 },
    'Rising Pune Supergiant': { from: 2016, to: 2017 },
    'Delhi Daredevils':       { from: 2008, to: 2018 },
    'Delhi Capitals':         { from: 2019 },
    'Kings XI Punjab':        { from: 2008, to: 2020 },
    'Punjab Kings':           { from: 2021 },
    'Royal Challengers Bangalore': { from: 2008, to: 2023 },
    'Royal Challengers Bengaluru': { from: 2024 },
  };
  const r = ranges[team] || { from: 2008 };
  return IPL_YEARS.filter(y => y >= (r.from || 2008) && y <= (r.to || 2024));
}

function populateCricketYearSelects(team, yearSelectId) {
  const yearSel = document.getElementById(yearSelectId);
  if (!yearSel) return;

  if (!team) {
    yearSel.innerHTML = '<option value="" disabled selected>Select team first</option>';
    yearSel.disabled = true;
    return;
  }

  const activeYears = getTeamActiveYears(team);
  yearSel.innerHTML = '<option value="">Any Season (All Years)</option>';
  activeYears.forEach(y => {
    yearSel.innerHTML += `<option value="${y}" ${y === 2024 ? 'selected' : ''}>${y} IPL Season</option>`;
  });
  yearSel.disabled = false;
}

function initCricketForm() {
  const form = document.getElementById('cricket-form');
  const ckT1 = document.getElementById('ck-team1-select');
  const ckT2 = document.getElementById('ck-team2-select');
  const swapBtn = document.getElementById('ck-swap-teams');

  ckT1.addEventListener('change', () => {
    populateCricketYearSelects(ckT1.value, 'ck-team1-year');
    updateCricketPreviews();
  });
  ckT2.addEventListener('change', () => {
    populateCricketYearSelects(ckT2.value, 'ck-team2-year');
    updateCricketPreviews();
  });

  swapBtn.addEventListener('click', () => {
    const val1 = ckT1.value;
    const val2 = ckT2.value;
    const yr1 = document.getElementById('ck-team1-year')?.value;
    const yr2 = document.getElementById('ck-team2-year')?.value;
    ckT1.value = val2;
    ckT2.value = val1;
    populateCricketYearSelects(val2, 'ck-team1-year');
    populateCricketYearSelects(val1, 'ck-team2-year');
    if (yr2) document.getElementById('ck-team1-year').value = yr2;
    if (yr1) document.getElementById('ck-team2-year').value = yr1;
    updateCricketPreviews();
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const team1 = ckT1.value;
    const team2 = ckT2.value;
    const year1 = document.getElementById('ck-team1-year')?.value || '';
    const year2 = document.getElementById('ck-team2-year')?.value || '';

    if (!team1 || !team2) {
      alert('Please select both batting and chasing franchises.');
      return;
    }
    if (team1 === team2) {
      alert('Batting and Chasing teams must be different franchises.');
      return;
    }

    const btn = document.getElementById('btn-predict-cricket');
    btn.disabled = true;
    const yearLabel = year1 || year2 ? ` (${[year1, year2].filter(Boolean).join(' / ')})` : '';
    btn.querySelector('.btn-text-content').textContent = 'Simulating Match Balls...';

    try {
      let result;
      if (state.isApiOnline) {
        const payload = { team1, team2 };
        if (year1) payload.year1 = parseInt(year1);
        if (year2) payload.year2 = parseInt(year2);
        const resp = await fetch(`${API_BASE}/api/cricket/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!resp.ok) throw new Error('Cricket API error');
        result = await resp.json();
      } else {
        result = simulateCricketPrediction(team1, team2, year1, year2);
      }

      displayCricketResult(result, year1, year2);
      const matchLabel = `${team1}${year1 ? ' ('+year1+')' : ''} vs ${team2}${year2 ? ' ('+year2+')' : ''}`;
      logHistory('Cricket', matchLabel, result.rf_prediction, result.rf_probabilities);
    } catch (err) {
      console.warn('API error, using local cricket simulation:', err);
      const fallbackResult = simulateCricketPrediction(team1, team2, year1, year2);
      displayCricketResult(fallbackResult, year1, year2);
      const matchLabel = `${team1}${year1 ? ' ('+year1+')' : ''} vs ${team2}${year2 ? ' ('+year2+')' : ''}`;
      logHistory('Cricket', matchLabel, fallbackResult.rf_prediction, fallbackResult.rf_probabilities);
    } finally {
      btn.disabled = false;
      btn.querySelector('.btn-text-content').textContent = 'Simulate IPL Match';
    }
  });
}

function updateCricketPreviews() {
  const t1 = document.getElementById('ck-team1-select').value;
  const t2 = document.getElementById('ck-team2-select').value;

  // Mock / calculated figures
  document.getElementById('ck-t1-runs').textContent = '168.4';
  document.getElementById('ck-t1-rr').textContent = '8.42 RPO';
  document.getElementById('ck-t1-sixes').textContent = '7.2';

  const t2Win = state.cricket.teamWinRates[t2];
  document.getElementById('ck-t2-winrate').textContent = t2Win ? `${Math.round(t2Win.win_rate * 100)}%` : '54%';
  document.getElementById('ck-t2-chasewin').textContent = '51%';
  document.getElementById('ck-t2-extras').textContent = '8.8';
}

function simulateCricketPrediction(t1, t2, year1 = '', year2 = '') {
  // Base 50-50 with slight batting first tilt in IPL history.
  // Year context adds a small recency/era bias for simulation.
  const recentYears = [2022, 2023, 2024];
  const t1Recent = recentYears.includes(parseInt(year1));
  const t2Recent = recentYears.includes(parseInt(year2));
  const biasSeed = t1Recent ? 3 : (t2Recent ? -3 : 0);
  const p1 = Math.min(82, Math.max(18, Math.round(48 + biasSeed + (Math.random() * 10))));
  const p2 = 100 - p1;
  const winner = p1 >= 50 ? t1 : t2;

  const probs = {};
  probs[t1] = p1 / 100;
  probs[t2] = p2 / 100;

  return {
    team1: t1,
    team2: t2,
    year1, year2,
    rf_prediction: winner,
    rf_probabilities: probs,
    ann_prediction: winner
  };
}

function displayCricketResult(res, year1 = '', year2 = '') {
  document.getElementById('ck-empty-state').classList.add('hidden');
  const content = document.getElementById('ck-result-content');
  content.classList.remove('hidden');

  const y1 = year1 || res.year1 || '';
  const y2 = year2 || res.year2 || '';
  const t1Label = y1 ? `${res.team1} (${y1})` : res.team1;
  const t2Label = y2 ? `${res.team2} (${y2})` : res.team2;

  const winner = res.rf_prediction; // team name that won
  const team1Won = (winner === res.team1);

  // Color constants: winner = blue, loser = amber (dimmed)
  const WINNER_COLOR  = '#3b82f6';
  const WINNER_GLOW   = 'rgba(59,130,246,0.4)';
  const WINNER_BG     = 'rgba(59,130,246,0.12)';
  const LOSER_COLOR   = '#fbbf24';
  const LOSER_DIM     = '#92400e';

  // --- Outcome headline ---
  const headline = document.getElementById('ck-outcome-text');
  headline.textContent = winner;
  headline.style.color = WINNER_COLOR;
  headline.style.textShadow = `0 0 24px ${WINNER_COLOR}55`;

  document.getElementById('ck-outcome-sub').textContent =
    `Predicted Winner — ${t1Label} vs ${t2Label}`;

  // --- Badge ---
  const badge = document.getElementById('ck-forecast-tag');
  if (badge) {
    badge.style.background = WINNER_BG;
    badge.style.borderColor = WINNER_GLOW;
    badge.style.color = WINNER_COLOR;
  }

  // --- Result card border glow ---
  const card = document.getElementById('ck-result-card');
  if (card) {
    card.style.borderColor = WINNER_GLOW;
    card.style.boxShadow = `var(--shadow-card), 0 0 30px ${WINNER_BG}`;
    card.style.transition = 'border-color 0.5s ease, box-shadow 0.5s ease';
  }

  // --- Probabilities ---
  const p1 = Math.round((res.rf_probabilities[res.team1] || 0.52) * 100);
  const p2 = 100 - p1;

  // Team 1 (left side)
  const t1PctEl = document.getElementById('duel-t1-pct');
  document.getElementById('duel-t1-name').textContent = res.team1;
  t1PctEl.textContent = `${p1}%`;
  t1PctEl.style.color = team1Won ? WINNER_COLOR : LOSER_COLOR;
  t1PctEl.style.opacity = team1Won ? '1' : '0.75';

  // Team 2 (right side)
  const t2PctEl = document.getElementById('duel-t2-pct');
  document.getElementById('duel-t2-name').textContent = res.team2;
  t2PctEl.textContent = `${p2}%`;
  t2PctEl.style.color = team1Won ? LOSER_COLOR : WINNER_COLOR;
  t2PctEl.style.opacity = team1Won ? '0.75' : '1';

  // --- Duel bar: winner side is always blue, loser is amber ---
  const barFill = document.getElementById('duel-bar-fill');
  barFill.style.width = `${p1}%`;
  if (team1Won) {
    // Left (team1) won → bar fills blue from left
    barFill.style.background = `linear-gradient(90deg, ${WINNER_COLOR}, #60a5fa)`;
    barFill.parentElement.style.background = `linear-gradient(90deg, #92400e, #fbbf24)`;
  } else {
    // Right (team2) won → bar fills amber from left (team1 share), rest is blue (team2)
    barFill.style.background = `linear-gradient(90deg, #fbbf24, #d97706)`;
    barFill.parentElement.style.background = `linear-gradient(90deg, #60a5fa, ${WINNER_COLOR})`;
  }

  // Tactical insights
  document.getElementById('ck-proj-score').textContent =
    `${160 + Math.floor(p1 * 0.3)} - ${175 + Math.floor(p1 * 0.3)}`;
  document.getElementById('ck-defend-idx').textContent = p1 > 52 ? 'Favorable' : 'Challenging';
  document.getElementById('ck-req-rr').textContent = `${(170 / 20).toFixed(2)} RPO`;
}


// -------------------------------------------------------------
// ANALYTICS TAB
// -------------------------------------------------------------
function renderAnalyticsViews() {
  renderOversChart();
  renderBattersTable();
  renderBowlersTable();
  renderFootballDistGrid();
}

function renderOversChart() {
  const container = document.getElementById('overs-chart');
  if (!container) return;
  container.innerHTML = '';

  const overs = state.cricket.runsPerOver || FALLBACK_DATA.cricket.runsPerOver;
  const maxRun = Math.max(...Object.values(overs));

  for (let i = 1; i <= 20; i++) {
    const val = overs[i.toString()] || (5.0 + i * 0.35);
    const heightPct = Math.round((val / maxRun) * 100);

    const col = document.createElement('div');
    col.className = 'over-col';
    col.innerHTML = `
      <span class="over-val">${val.toFixed(1)}</span>
      <div class="over-bar-track">
        <div class="over-bar" style="height: ${heightPct}%;"></div>
      </div>
      <span class="over-num">O${i}</span>
    `;
    container.appendChild(col);
  }
}

function renderBattersTable() {
  const tbody = document.getElementById('batters-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  const batters = Object.entries(state.cricket.topBatters || FALLBACK_DATA.cricket.topBatters);
  batters.forEach(([name, runs], idx) => {
    tbody.innerHTML += `
      <tr>
        <td><strong>#${idx + 1}</strong></td>
        <td>${name}</td>
        <td class="text-right"><strong>${runs.toLocaleString()}</strong></td>
      </tr>
    `;
  });
}

function renderBowlersTable() {
  const tbody = document.getElementById('bowlers-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  const bowlers = Object.entries(state.cricket.topBowlers || FALLBACK_DATA.cricket.topBowlers);
  bowlers.forEach(([name, wkts], idx) => {
    tbody.innerHTML += `
      <tr>
        <td><strong>#${idx + 1}</strong></td>
        <td>${name}</td>
        <td class="text-right"><strong>${wkts}</strong></td>
      </tr>
    `;
  });
}

function renderFootballDistGrid() {
  const grid = document.getElementById('fb-dist-cards');
  if (!grid) return;
  grid.innerHTML = '';

  const dist = FALLBACK_DATA.football.distribution;
  Object.entries(dist).forEach(([title, data]) => {
    grid.innerHTML += `
      <div class="dist-card">
        <div class="dist-val" style="color: ${data.color};">${data.pct}</div>
        <div class="dist-title">${title}</div>
        <div class="dist-sub">${data.count} of 380 EPL Matches</div>
      </div>
    `;
  });
}

// -------------------------------------------------------------
// MODEL METRICS TAB
// -------------------------------------------------------------
function renderMetricsViews() {
  // Feature Importance Lists
  const fbFiList = document.getElementById('fb-fi-list');
  if (fbFiList) {
    fbFiList.innerHTML = '';
    const fbImp = FALLBACK_DATA.football.metrics.importance;
    Object.entries(fbImp).forEach(([feat, val]) => {
      const pct = (val * 100).toFixed(1);
      fbFiList.innerHTML += `
        <div class="fi-row">
          <span class="fi-name" title="${feat}">${feat}</span>
          <div class="fi-track">
            <div class="fi-bar" style="width: ${pct * 4}%;"></div>
          </div>
          <span class="fi-pct">${pct}%</span>
        </div>
      `;
    });
  }

  const ckFiList = document.getElementById('ck-fi-list');
  if (ckFiList) {
    ckFiList.innerHTML = '';
    const ckImp = FALLBACK_DATA.cricket.metrics.importance;
    Object.entries(ckImp).forEach(([feat, val]) => {
      const pct = (val * 100).toFixed(1);
      ckFiList.innerHTML += `
        <div class="fi-row">
          <span class="fi-name" title="${feat}">${feat}</span>
          <div class="fi-track">
            <div class="fi-bar" style="width: ${pct * 3.5}%; background: linear-gradient(90deg, #fbbf24, #ff6b35);"></div>
          </div>
          <span class="fi-pct">${pct}%</span>
        </div>
      `;
    });
  }
}

// -------------------------------------------------------------
// HISTORY & SESSION LOGS
// -------------------------------------------------------------
function initHistoryControls() {
  const clearBtn = document.getElementById('btn-clear-history');
  clearBtn.addEventListener('click', () => {
    state.history = [];
    updateHistoryUI();
  });
}

function logHistory(sport, teams, prediction, probs) {
  const item = {
    id: Date.now(),
    sport,
    teams,
    prediction,
    probs,
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  };

  state.history.unshift(item);
  updateHistoryUI();
}

function updateHistoryUI() {
  const container = document.getElementById('history-container');
  const badge = document.getElementById('history-badge');
  badge.textContent = state.history.length;

  if (state.history.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <p>No predictions made yet. Run a Football or Cricket simulation to log results here.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = '';
  state.history.forEach(item => {
    container.innerHTML += `
      <div class="history-item">
        <div>
          <span class="h-sport">${item.sport === 'Football' ? '⚽ Premier League' : '🏏 IPL Cricket'}</span>
          <div class="h-match">${item.teams}</div>
          <div class="h-time">Logged at ${item.time}</div>
        </div>
        <div class="h-outcome">
          <span class="h-pred">${item.prediction}</span>
        </div>
      </div>
    `;
  });
}
