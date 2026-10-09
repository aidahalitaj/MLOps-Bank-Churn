// Bank Customer Churn Predictor — Frontend Logic

(function () {
  'use strict';

  const form = document.getElementById('predict-form');
  const btn = document.getElementById('predict-btn');
  const resultsCard = document.getElementById('results-card');
  const probDisplay = document.getElementById('result-probability');
  const labelDisplay = document.getElementById('result-label');
  const gaugeFill = document.getElementById('gauge-fill');
  const errorBox = document.getElementById('error-message');

  // Validation rules
  const rules = {
    CreditScore: { min: 300, max: 850, label: 'Credit Score' },
    Age:         { min: 18,  max: 100, label: 'Age' },
    Tenure:      { min: 0,   max: 10,  label: 'Tenure' },
    NumOfProducts: { min: 1, max: 4,   label: 'Number of Products' }
  };

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.add('visible');
    resultsCard.classList.remove('visible');
  }

  function hideError() {
    errorBox.classList.remove('visible');
  }

  function validate(data) {
    for (var key in rules) {
      if (!rules.hasOwnProperty(key)) continue;
      var r = rules[key];
      var v = data[key];
      if (v === '' || v === null || v === undefined || isNaN(v)) {
        return r.label + ' is required.';
      }
      if (v < r.min || v > r.max) {
        return r.label + ' must be between ' + r.min + ' and ' + r.max + '.';
      }
    }
    if (isNaN(data.Balance) || data.Balance === '') return 'Balance is required.';
    if (isNaN(data.EstimatedSalary) || data.EstimatedSalary === '') return 'Estimated Salary is required.';
    return null;
  }

  function setLoading(on) {
    if (on) {
      btn.disabled = true;
      btn.classList.add('loading');
    } else {
      btn.disabled = false;
      btn.classList.remove('loading');
    }
  }

  function gaugeClass(prob) {
    if (prob < 0.35) return 'low';
    if (prob < 0.65) return 'medium';
    return 'high';
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    hideError();

    // Gather values
    var payload = {
      CreditScore:     parseInt(document.getElementById('CreditScore').value, 10),
      Age:             parseInt(document.getElementById('Age').value, 10),
      Tenure:          parseInt(document.getElementById('Tenure').value, 10),
      Balance:         parseFloat(document.getElementById('Balance').value),
      NumOfProducts:   parseInt(document.getElementById('NumOfProducts').value, 10),
      HasCrCard:       document.getElementById('HasCrCard').checked ? 1 : 0,
      IsActiveMember:  document.getElementById('IsActiveMember').checked ? 1 : 0,
      EstimatedSalary: parseFloat(document.getElementById('EstimatedSalary').value)
    };

    // Validate
    var err = validate(payload);
    if (err) {
      showError(err);
      // Mark invalid fields
      for (var key in rules) {
        var el = document.getElementById(key);
        if (el) {
          var r = rules[key];
          var v = payload[key];
          if (isNaN(v) || v < r.min || v > r.max) {
            el.classList.add('invalid');
          } else {
            el.classList.remove('invalid');
          }
        }
      }
      return;
    }

    // Clear invalid markers
    document.querySelectorAll('.invalid').forEach(function (el) {
      el.classList.remove('invalid');
    });

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
        var prob = data.churn_probability;
        var pct = (prob * 100).toFixed(1);
        var isChurn = data.prediction === 'Churn';

        probDisplay.textContent = pct + '%';

        if (isChurn) {
          labelDisplay.textContent = '\u26A0 High Churn Risk';
          labelDisplay.className = 'result-label churn';
        } else {
          labelDisplay.textContent = '\u2713 Low Churn Risk';
          labelDisplay.className = 'result-label safe';
        }

        // Gauge bar
        gaugeFill.style.width = pct + '%';
        gaugeFill.className = 'gauge-fill ' + gaugeClass(prob);

        resultsCard.classList.add('visible');
      })
      .catch(function (error) {
        showError('Prediction failed: ' + error.message);
      })
      .finally(function () {
        setLoading(false);
      });
  });
})();
