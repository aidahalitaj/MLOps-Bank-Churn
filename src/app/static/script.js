// ChurnPredict — Frontend Logic

(function () {
  'use strict';

  var form = document.getElementById('predict-form');
  var btn = document.getElementById('predict-btn');
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

  // Profile cards
  var profileCards = document.querySelectorAll('.profile-card');

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

  function setLoading(on) {
    btn.disabled = on;
    btn.classList.toggle('loading', on);
  }

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

  function markInvalid(payload) {
    for (var key in rules) {
      var el = document.getElementById(key);
      if (!el) continue;
      var r = rules[key];
      var v = payload[key];
      if (isNaN(v) || v < r.min || v > r.max) {
        el.classList.add('invalid');
      } else {
        el.classList.remove('invalid');
      }
    }
    // Balance & Salary — mark the wrapper
    ['Balance', 'EstimatedSalary'].forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      var wrap = el.closest('.input-prefix-wrap');
      if (isNaN(payload[id]) || payload[id] === '') {
        if (wrap) wrap.classList.add('invalid');
        else el.classList.add('invalid');
      } else {
        if (wrap) wrap.classList.remove('invalid');
        else el.classList.remove('invalid');
      }
    });
  }

  function clearInvalid() {
    document.querySelectorAll('.invalid').forEach(function (el) { el.classList.remove('invalid'); });
  }

  // ── Gather form values ──
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

  // ── Update factor card ──
  function setFactor(id, value, displayValue, riskFn) {
    var level = riskFn(value);
    var card = document.getElementById('factor-' + id);
    var valEl = document.getElementById('factor-' + id + '-val');
    var riskEl = document.getElementById('factor-' + id + '-risk');
    card.className = 'factor-card ' + level;
    valEl.textContent = displayValue;
    riskEl.textContent = riskLabels[level];
    riskEl.className = 'factor-risk ' + level;
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

    // Gauge
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

    // Factor cards
    setFactor('age', payload.Age, payload.Age, ageRisk);
    setFactor('products', payload.NumOfProducts, payload.NumOfProducts, productsRisk);
    setFactor('active', payload.IsActiveMember, payload.IsActiveMember === 1 ? 'Yes' : 'No', activeRisk);
    setFactor('tenure', payload.Tenure, payload.Tenure + ' yr', tenureRisk);
  }

  // ── Submit prediction ──
  function submitPrediction(payload) {
    hideError();
    clearInvalid();

    var err = validate(payload);
    if (err) {
      showError(err);
      markInvalid(payload);
      return;
    }

    setLoading(true);

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
    })
    .finally(function () {
      setLoading(false);
    });
  }

  // ── Form submit ──
  form.addEventListener('submit', function (e) {
    e.preventDefault();
    submitPrediction(gatherPayload());
  });

  // ── Profile card click ──
  profileCards.forEach(function (card) {
    card.addEventListener('click', function () {
      // Highlight selected profile
      profileCards.forEach(function (c) { c.classList.remove('selected'); });
      card.classList.add('selected');

      // Fill form values
      document.getElementById('CreditScore').value     = card.dataset.creditscore;
      document.getElementById('Age').value              = card.dataset.age;
      document.getElementById('Tenure').value           = card.dataset.tenure;
      document.getElementById('Balance').value          = card.dataset.balance;
      document.getElementById('NumOfProducts').value    = card.dataset.numofproducts;
      document.getElementById('HasCrCard').checked      = card.dataset.hascrcard === '1';
      document.getElementById('IsActiveMember').checked = card.dataset.isactivemember === '1';
      document.getElementById('EstimatedSalary').value  = card.dataset.estimatedsalary;

      // Auto-submit
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
