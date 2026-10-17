// ChurnPredict — Frontend Logic

(function () {
  'use strict';

  var form = document.getElementById('predict-form');
  var errorBox = document.getElementById('error-message');

  // Insight panel elements
  var placeholder = document.getElementById('insights-placeholder');
  var resultPane = document.getElementById('insights-result');
  var resultHeader = document.getElementById('result-header');
  var resultProb = document.getElementById('result-probability');
  var resultBadge = document.getElementById('result-badge');
  var gaugeFill = document.getElementById('gauge-fill');
  var assessmentBox = document.getElementById('assessment-box');
  var assessmentText = document.getElementById('assessment-text');
  var whatifBox = document.getElementById('whatif-box');
  var whatifText = document.getElementById('whatif-text');
  var factorsGrid = document.getElementById('factors-grid');

  // Profile cards
  var profileCards = document.querySelectorAll('.profile-card');

  // Debounce timer
  var debounceTimer = null;

  // ── Validation rules ──
  var rules = {
    CreditScore:   { min: 300, max: 850, label: 'Credit Score' },
    Age:           { min: 18,  max: 100, label: 'Age' },
    Tenure:        { min: 0,   max: 10,  label: 'Tenure' },
    NumOfProducts: { min: 1,   max: 4,   label: 'Num. Products' }
  };

  // ── Risk thresholds for key factors ──
  function ageRisk(v)      { return v < 35 ? 'low' : v <= 50 ? 'med' : 'high'; }
  function productsRisk(v)  { return v === 2 ? 'low' : (v === 1 ? 'med' : 'high'); }
  function activeRisk(v)    { return v === 1 ? 'low' : 'high'; }
  function tenureRisk(v)    { return v > 5 ? 'low' : v >= 2 ? 'med' : 'high'; }

  var riskLabels = {
    low:  'Low risk \u2193',
    med:  'Med risk \u2192',
    high: 'High risk \u2191'
  };

  // ── Helpers ──
  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.add('visible');
  }

  function hideError() { errorBox.classList.remove('visible'); }

  function riskLevel(prob) {
    if (prob < 0.35) return 'low';
    if (prob < 0.65) return 'med';
    return 'high';
  }

  function validate(d) {
    for (var key in rules) {
      if (!rules.hasOwnProperty(key)) continue;
      var r = rules[key];
      var v = d[key];
      if (v === '' || v === null || v === undefined || isNaN(v)) return r.label + ' is required.';
      if (v < r.min || v > r.max) return r.label + ' must be between ' + r.min + ' and ' + r.max + '.';
    }
    if (isNaN(d.Balance) || d.Balance === '') return 'Account Balance is required.';
    if (isNaN(d.EstimatedSalary) || d.EstimatedSalary === '') return 'Estimated Salary is required.';
    return null;
  }

  function clearInvalid() {
    document.querySelectorAll('.invalid').forEach(function (el) { el.classList.remove('invalid'); });
  }

  // ── Currency formatting ──
  function fmtCurrency(v) {
    return '$' + Number(v).toLocaleString('en-US');
  }

  function parseCurrency(str) {
    return parseFloat(String(str).replace(/[^0-9.\-]/g, '')) || 0;
  }

  // ── Slider ↔ Number input sync ──
  var sliderFields = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'];
  var currencyFields = ['Balance', 'EstimatedSalary'];

  function isCurrency(id) {
    return currencyFields.indexOf(id) !== -1;
  }

  function syncNumberFromSlider(id) {
    var slider = document.getElementById(id);
    var numInput = document.getElementById(id + '-num');
    if (!slider || !numInput) return;
    if (isCurrency(id)) {
      numInput.value = fmtCurrency(slider.value);
    } else {
      numInput.value = slider.value;
    }
  }

  function syncSliderFromNumber(id) {
    var slider = document.getElementById(id);
    var numInput = document.getElementById(id + '-num');
    if (!slider || !numInput) return;
    var val;
    if (isCurrency(id)) {
      val = parseCurrency(numInput.value);
    } else {
      val = parseFloat(numInput.value);
    }
    if (!isNaN(val)) {
      var min = parseFloat(slider.min);
      var max = parseFloat(slider.max);
      val = Math.max(min, Math.min(max, val));
      slider.value = val;
    }
  }

  // Initialize slider ↔ number sync and event listeners
  sliderFields.forEach(function(id) {
    var slider = document.getElementById(id);
    var numInput = document.getElementById(id + '-num');
    if (!slider || !numInput) return;

    // Initial sync
    syncNumberFromSlider(id);

    // Slider → number input + auto-predict
    slider.addEventListener('input', function() {
      syncNumberFromSlider(id);
      schedulePredict();
    });

    // Number input → slider + auto-predict
    numInput.addEventListener('input', function() {
      syncSliderFromNumber(id);
      schedulePredict();
    });

    // Reformat on blur and clamp to valid range
    if (isCurrency(id)) {
      numInput.addEventListener('blur', function() {
        var val = parseCurrency(numInput.value);
        var min = parseFloat(slider.min);
        var max = parseFloat(slider.max);
        val = Math.max(min, Math.min(max, val));
        slider.value = val;
        numInput.value = fmtCurrency(parseFloat(slider.value));
      });
      numInput.addEventListener('focus', function() {
        var val = parseCurrency(numInput.value);
        numInput.value = val;
        numInput.select();
      });
    } else {
      numInput.addEventListener('blur', function() {
        var val = parseFloat(numInput.value);
        if (!isNaN(val)) {
          var min = parseFloat(slider.min);
          var max = parseFloat(slider.max);
          val = Math.max(min, Math.min(max, val));
          slider.value = val;
        }
        // Read back snapped value from slider
        numInput.value = slider.value;
      });
    }
  });

  // ── Toggle accessibility: update aria-checked on change ──
  ['HasCrCard', 'IsActiveMember'].forEach(function(id) {
    var toggle = document.getElementById(id);
    if (!toggle) return;
    toggle.addEventListener('change', function() {
      toggle.setAttribute('aria-checked', toggle.checked ? 'true' : 'false');
      schedulePredict();
    });
  });

  // ── Gather form values from sliders (source of truth) ──
  function gatherPayload() {
    return {
      CreditScore:     parseInt(document.getElementById('CreditScore').value, 10),
      Age:             parseInt(document.getElementById('Age').value, 10),
      Tenure:          parseInt(document.getElementById('Tenure').value, 10),
      Balance:         parseFloat(document.getElementById('Balance').value),
      NumOfProducts:   parseInt(document.getElementById('NumOfProducts').value, 10),
      HasCrCard:       document.getElementById('HasCrCard').checked ? 1 : 0,
      IsActiveMember:  document.getElementById('IsActiveMember').checked ? 1 : 0,
      EstimatedSalary: parseFloat(document.getElementById('EstimatedSalary').value)
    };
  }

  // ── Render factor cards sorted by risk level ──
  function renderFactors(payload) {
    var factors = [
      { name: 'Age',           value: payload.Age,             display: String(payload.Age),                              riskFn: ageRisk,      order: 0 },
      { name: 'Products',      value: payload.NumOfProducts,   display: String(payload.NumOfProducts),                    riskFn: productsRisk, order: 1 },
      { name: 'Active Member', value: payload.IsActiveMember,  display: payload.IsActiveMember === 1 ? 'Yes' : 'No',     riskFn: activeRisk,   order: 2 },
      { name: 'Tenure',        value: payload.Tenure,          display: payload.Tenure + ' yr',                           riskFn: tenureRisk,   order: 3 }
    ];

    // Compute risk level per factor
    factors.forEach(function(f) { f.level = f.riskFn(f.value); });

    // Sort: highest risk first, preserve original order within same level
    var levelOrder = { high: 0, med: 1, low: 2 };
    factors.sort(function(a, b) {
      if (levelOrder[a.level] !== levelOrder[b.level]) return levelOrder[a.level] - levelOrder[b.level];
      return a.order - b.order;
    });

    // Render into grid
    factorsGrid.innerHTML = '';
    factors.forEach(function(f) {
      var card = document.createElement('div');
      card.className = 'factor-card ' + f.level;
      card.innerHTML =
        '<span class="factor-name">' + f.name + '</span>' +
        '<span class="factor-value">' + f.display + '</span>' +
        '<span class="factor-risk ' + f.level + '">' + riskLabels[f.level] + '</span>';
      factorsGrid.appendChild(card);
    });
  }

  // ── Render prediction result ──
  function renderResult(data, payload) {
    var prob = data.churn_probability;
    var pct = Math.round(prob * 100);
    var level = riskLevel(prob);

    // Hide placeholder, show result
    placeholder.style.display = 'none';
    resultPane.classList.add('visible');

    // Result header
    resultHeader.className = 'result-header ' + level;
    resultProb.textContent = pct + '%';

    // Badge
    if (level === 'low') {
      resultBadge.textContent = '\u2713 Low';
      resultBadge.className = 'result-badge low';
    } else if (level === 'med') {
      resultBadge.textContent = '\u26A0 Medium';
      resultBadge.className = 'result-badge med';
    } else {
      resultBadge.textContent = '\u26A0 High';
      resultBadge.className = 'result-badge high';
    }

    // Gauge fill
    gaugeFill.style.width = pct + '%';
    gaugeFill.className = 'gauge-fill ' + level;

    // Assessment text
    assessmentBox.className = 'assessment-box ' + level;
    if (prob < 0.35) {
      assessmentText.innerHTML = 'Strong loyalty signals \u2014 long tenure, active status, and multiple products. No immediate action needed.';
    } else if (prob < 0.65) {
      assessmentText.innerHTML = 'This customer shows some risk factors. Monitor engagement and consider proactive outreach.';
    } else {
      assessmentText.innerHTML = '<div class="action-heading">RECOMMENDED ACTION</div>This customer is at high risk of leaving. Consider proactive retention \u2014 loyalty discount, rate improvement, or relationship manager outreach.';
    }

    // Factor cards (sorted by risk)
    renderFactors(payload);

    // What-if guidance (parallel requests)
    computeWhatIf(payload, prob);
  }

  // ── What-if guidance ──
  function computeWhatIf(payload, currentProb) {
    var scenarios = [];

    if (payload.Age > 30) {
      scenarios.push({ factor: 'Age', displayValue: '30', payload: assign(payload, { Age: 30 }) });
    }
    if (payload.NumOfProducts !== 2) {
      scenarios.push({ factor: 'Products', displayValue: '2', payload: assign(payload, { NumOfProducts: 2 }) });
    }
    if (payload.IsActiveMember === 0) {
      scenarios.push({ factor: 'Active Member', displayValue: 'Yes', payload: assign(payload, { IsActiveMember: 1 }) });
    }
    if (payload.Tenure < 5) {
      scenarios.push({ factor: 'Tenure', displayValue: '5 years', payload: assign(payload, { Tenure: 5 }) });
    }

    if (scenarios.length === 0) {
      whatifBox.classList.remove('visible');
      return;
    }

    // Fire all what-if requests in parallel
    Promise.all(scenarios.map(function(s) {
      return fetch('/predict_single', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(s.payload)
      })
      .then(function(r) { return r.json(); })
      .then(function(data) {
        s.newProb = data.churn_probability;
        return s;
      })
      .catch(function() {
        s.newProb = currentProb; // on error, treat as no improvement
        return s;
      });
    })).then(function(results) {
      // Find the scenario with the lowest new probability
      var best = results.reduce(function(prev, curr) {
        return curr.newProb < prev.newProb ? curr : prev;
      });

      if (best.newProb < currentProb) {
        var newPct = Math.round(best.newProb * 100);
        whatifText.textContent = '\uD83D\uDCA1 If ' + best.factor + ' were ' + best.displayValue + ', risk drops to ' + newPct + '%';
        whatifBox.classList.add('visible');
      } else {
        whatifBox.classList.remove('visible');
      }
    });
  }

  // Simple object merge helper
  function assign(target, overrides) {
    var result = {};
    for (var key in target) {
      if (target.hasOwnProperty(key)) result[key] = target[key];
    }
    for (var key2 in overrides) {
      if (overrides.hasOwnProperty(key2)) result[key2] = overrides[key2];
    }
    return result;
  }

  // ── Submit prediction ──
  function submitPrediction(payload) {
    hideError();
    clearInvalid();

    var err = validate(payload);
    if (err) {
      showError(err);
      return;
    }

    fetch('/predict_single', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(function (res) {
      if (!res.ok) {
        return res.json().then(function (body) {
          throw new Error(body.detail || 'Server error (' + res.status + ')');
        });
      }
      return res.json();
    })
    .then(function (data) {
      renderResult(data, payload);
    })
    .catch(function (error) {
      showError('Prediction failed: ' + error.message);
    });
  }

  // ── Debounced auto-submit (300ms) ──
  function schedulePredict() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function() {
      submitPrediction(gatherPayload());
    }, 300);
  }

  // ── Prevent default form submit (no button, but handle Enter key) ──
  form.addEventListener('submit', function (e) {
    e.preventDefault();
    submitPrediction(gatherPayload());
  });

  // ── Profile card click — fill form and submit immediately ──
  profileCards.forEach(function (card) {
    card.addEventListener('click', function () {
      // Highlight selected profile
      profileCards.forEach(function (c) { c.classList.remove('selected'); });
      card.classList.add('selected');

      // Fill slider values
      document.getElementById('CreditScore').value     = card.dataset.creditscore;
      document.getElementById('Age').value              = card.dataset.age;
      document.getElementById('Tenure').value           = card.dataset.tenure;
      document.getElementById('Balance').value          = card.dataset.balance;
      document.getElementById('NumOfProducts').value    = card.dataset.numofproducts;
      document.getElementById('HasCrCard').checked      = card.dataset.hascrcard === '1';
      document.getElementById('IsActiveMember').checked = card.dataset.isactivemember === '1';
      document.getElementById('EstimatedSalary').value  = card.dataset.estimatedsalary;

      // Update aria-checked on toggles
      document.getElementById('HasCrCard').setAttribute('aria-checked', card.dataset.hascrcard === '1' ? 'true' : 'false');
      document.getElementById('IsActiveMember').setAttribute('aria-checked', card.dataset.isactivemember === '1' ? 'true' : 'false');

      // Sync number inputs from sliders
      sliderFields.forEach(function(id) { syncNumberFromSlider(id); });

      // Immediate submit (cancel any pending debounce)
      if (debounceTimer) clearTimeout(debounceTimer);
      submitPrediction({
        CreditScore:     parseInt(card.dataset.creditscore, 10),
        Age:             parseInt(card.dataset.age, 10),
        Tenure:          parseInt(card.dataset.tenure, 10),
        Balance:         parseFloat(card.dataset.balance),
        NumOfProducts:   parseInt(card.dataset.numofproducts, 10),
        HasCrCard:       parseInt(card.dataset.hascrcard, 10),
        IsActiveMember:  parseInt(card.dataset.isactivemember, 10),
        EstimatedSalary: parseFloat(card.dataset.estimatedsalary)
      });
    });
  });
})();
