"""
Flask API for Medical Insurance Cost Prediction.
Serves dashboard UI + async JSON endpoints for skeleton loaders.
"""

import os
import json
import pickle
import time
import re
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Resolve paths relative to repository root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
DATASET_PATH = os.path.join(ROOT_DIR, 'dataset', 'insurance.csv')
USD_TO_INR = float(os.environ.get('USD_TO_INR', 83.5))
LOG_DIR = os.path.join(ROOT_DIR, 'user_logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Global model references
rf_model = None
lr_model = None
dt_model = None
log_model = None
comparison_data = None


def load_models():
    """Load serialized ML models and comparison data if present."""
    global rf_model, lr_model, dt_model, log_model, comparison_data
    rf_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
    lr_path = os.path.join(MODEL_DIR, 'linear_model.pkl')
    dt_path = os.path.join(MODEL_DIR, 'dt_model.pkl')
    log_path = os.path.join(MODEL_DIR, 'log_model.pkl')
    comp_path = os.path.join(MODEL_DIR, 'comparison.json')

    # Load Random Forest (preferred)
    try:
        if os.path.exists(rf_path):
            with open(rf_path, 'rb') as f:
                rf_model = pickle.load(f)
            app.logger.info('Random Forest model loaded from %s', rf_path)
        elif os.path.exists(os.path.join(MODEL_DIR, 'model.pkl')):
            # backward compatibility
            with open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb') as f:
                rf_model = pickle.load(f)
            app.logger.info('Default model loaded from model.pkl')
    except Exception as e:
        app.logger.exception('Failed to load Random Forest model: %s', e)

    # Load Linear Regression model
    try:
        if os.path.exists(lr_path):
            with open(lr_path, 'rb') as f:
                lr_model = pickle.load(f)
            app.logger.info('Linear Regression model loaded from %s', lr_path)
    except Exception as e:
        app.logger.exception('Failed to load Linear Regression model: %s', e)

    # Load Decision Tree model
    try:
        if os.path.exists(dt_path):
            with open(dt_path, 'rb') as f:
                dt_model = pickle.load(f)
            app.logger.info('Decision Tree model loaded from %s', dt_path)
    except Exception as e:
        app.logger.exception('Failed to load Decision Tree model: %s', e)

    # Load Logistic Regression model
    try:
        if os.path.exists(log_path):
            with open(log_path, 'rb') as f:
                log_model = pickle.load(f)
            app.logger.info('Logistic Regression model loaded from %s', log_path)
    except Exception as e:
        app.logger.exception('Failed to load Logistic Regression model: %s', e)

    # Comparison data
    try:
        if os.path.exists(comp_path):
            with open(comp_path, 'r') as f:
                comparison_data = json.load(f)
            app.logger.info('Comparison data loaded from %s', comp_path)
    except Exception as e:
        app.logger.exception('Failed to load comparison.json: %s', e)


@app.route('/')
def home():
    return render_template('form.html')


@app.route('/result/<log_id>')
def result_page(log_id: str):
    return render_template('result.html', log_id=log_id)


def _validate_and_parse_submission(data: dict) -> tuple[list[str], dict]:
    """Validate user inputs for the log + prediction submission."""
    errors: list[str] = []

    # Patient fields (stored for transparency; not used as ML features)
    person_name = (data.get('person_name') or '').strip()
    person_mail = (data.get('person_mail') or '').strip()
    person_mobile_raw = (data.get('person_mobile') or '').strip()
    user_address = (data.get('user_address') or '').strip()
    has_disease_raw = (data.get('has_disease') or '').strip().lower()
    diseases = (data.get('diseases') or '').strip()

    if not person_name:
        errors.append('Person name is required.')
    if not person_mail or '@' not in person_mail:
        errors.append('Valid person mail id is required.')
    person_mobile = re.sub(r'\D', '', person_mobile_raw)
    if not person_mobile or not re.match(r'^[0-9]{10,15}$', person_mobile):
        errors.append('Valid mobile number is required (digits only, e.g. 9876543210).')
    if not user_address:
        errors.append('User address is required.')

    if has_disease_raw not in ['yes', 'no']:
        errors.append('Please select whether you have any diseases now (yes/no).')
    if has_disease_raw == 'yes' and not diseases:
        errors.append('Please enter your diseases.')

    # ML input fields
    try:
        age = int(data.get('age', 0))
        sex = int(data.get('sex', 0))
        bmi = float(data.get('bmi', 0))
        children = int(data.get('children', 0))
        smoker = int(data.get('smoker', 0))
        region = int(data.get('region', 0))
    except Exception:
        return ['Invalid numeric inputs.'], {}

    if age < 1 or age > 100:
        errors.append('Age must be between 1 and 100.')
    if sex not in [0, 1]:
        errors.append('Invalid gender selection.')
    if bmi < 10 or bmi > 60:
        errors.append('BMI must be between 10 and 60.')
    if children < 0 or children > 10:
        errors.append('Children must be between 0 and 10.')
    if smoker not in [0, 1]:
        errors.append('Invalid smoking status.')
    if region not in range(9):
        errors.append('Invalid region.')

    patient = {
        'name': person_name,
        'email': person_mail,
        'mobile': person_mobile,
        'address': user_address,
        'has_disease': has_disease_raw == 'yes',
        'diseases': diseases if has_disease_raw == 'yes' else ''
    }

    model_input = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
    return errors, {'patient': patient, 'model_input': model_input}


@app.route('/api/stats')
def api_stats():
    """Async endpoint for dataset statistics (skeleton loader)."""
    try:
        df = pd.read_csv(DATASET_PATH)
        stats = {
            'total_records': int(len(df)),
            'avg_charge': round(df['charges'].mean() * USD_TO_INR, 2),
            'max_charge': round(df['charges'].max() * USD_TO_INR, 2),
            'min_charge': round(df['charges'].min() * USD_TO_INR, 2),
            'median_charge': round(df['charges'].median() * USD_TO_INR, 2),
            'smoker_avg': round(df[df['smoker'] == 'yes']['charges'].mean() * USD_TO_INR, 2),
            'nonsmoker_avg': round(df[df['smoker'] == 'no']['charges'].mean() * USD_TO_INR, 2),
            'avg_age': round(df['age'].mean(), 1),
            'avg_bmi': round(df['bmi'].mean(), 1),
            'smoker_count': int((df['smoker'] == 'yes').sum()),
            'nonsmoker_count': int((df['smoker'] == 'no').sum()),
            'male_count': int((df['sex'] == 'male').sum()),
            'female_count': int((df['sex'] == 'female').sum()),
            'age_distribution': {
                '18-25': int(((df['age'] >= 18) & (df['age'] <= 25)).sum()),
                '26-35': int(((df['age'] >= 26) & (df['age'] <= 35)).sum()),
                '36-45': int(((df['age'] >= 36) & (df['age'] <= 45)).sum()),
                '46-55': int(((df['age'] >= 46) & (df['age'] <= 55)).sum()),
                '56-65': int(((df['age'] >= 56) & (df['age'] <= 65)).sum()),
            },
            'region_distribution': df['region'].value_counts().to_dict(),
        }
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dataset-preview')
def api_dataset_preview():
    """Async endpoint for dataset table preview."""
    try:
        df = pd.read_csv(DATASET_PATH)
        rows = df.head(12).to_dict('records')
        for row in rows:
            row['charges_inr'] = round(row['charges'] * USD_TO_INR, 2)
        return jsonify({'success': True, 'data': rows})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model-comparison')
def api_model_comparison():
    """Async endpoint for model comparison metrics."""
    if comparison_data:
        # Convert metrics to INR
        comp = json.loads(json.dumps(comparison_data))
        for key in ['linear_regression', 'decision_tree', 'random_forest']:
            if key in comp:
                comp[key]['mae_inr'] = round(comp[key]['mae'] * USD_TO_INR, 2)
                comp[key]['rmse_inr'] = round(comp[key]['rmse'] * USD_TO_INR, 2)
        return jsonify({'success': True, 'data': comp})
    return jsonify({'success': False, 'error': 'No comparison data'}), 404


@app.route('/api/insights')
def api_insights():
    """Chart.js data: smoker vs non-smoker avg + BMI vs charges scatter."""
    try:
        df = pd.read_csv(DATASET_PATH)
        smoker_avg = round(df[df['smoker'] == 'yes']['charges'].mean() * USD_TO_INR, 2)
        nonsmoker_avg = round(df[df['smoker'] == 'no']['charges'].mean() * USD_TO_INR, 2)

        # Sample 150 points for scatter plot performance
        sample = df.sample(n=min(250, len(df)), random_state=42)
        bmi_list = sample['bmi'].round(2).tolist()
        charges_list = (sample['charges'] * USD_TO_INR).round(2).tolist()
        smoker_flags = sample['smoker'].map({'yes': 1, 'no': 0}).tolist()

        # Age vs charges for line chart
        age_groups = df.groupby(pd.cut(df['age'], bins=[17, 25, 35, 45, 55, 65]))['charges'].mean()
        age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65']
        age_avg_charges = [round(v * USD_TO_INR, 2) for v in age_groups.values]

        # Region-wise averages
        region_avg = df.groupby('region')['charges'].mean().to_dict()
        region_data = {k: round(v * USD_TO_INR, 2) for k, v in region_avg.items()}

        return jsonify({
            'success': True,
            'data': {
                'smoker_avg': smoker_avg,
                'non_smoker_avg': nonsmoker_avg,
                'bmi': bmi_list,
                'charges': charges_list,
                'smoker_flags': smoker_flags,
                'age_labels': age_labels,
                'age_avg_charges': age_avg_charges,
                'region_data': region_data,
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def generate_explanation(age, sex, bmi, children, smoker, region, prediction_inr):
    """AI Explanation Engine: dynamically generate reasons for the prediction."""
    reasons = []
    severity_score = 0

    # Smoking analysis
    if smoker == 1:
        reasons.append({
            'icon': '🚬', 'factor': 'Smoking Status',
            'impact': 'Critical',
            'detail': 'Smoking is the #1 cost driver, increasing premiums by 3-4x compared to non-smokers.',
            'color': 'red'
        })
        severity_score += 40
    else:
        reasons.append({
            'icon': '🫁', 'factor': 'Non-Smoker',
            'impact': 'Positive',
            'detail': 'Non-smoking status significantly reduces your insurance premium.',
            'color': 'emerald'
        })

    # BMI analysis
    if bmi > 35:
        reasons.append({
            'icon': '⚖️', 'factor': 'High BMI (Obese)',
            'impact': 'High',
            'detail': f'BMI of {bmi} indicates obesity (Class II+). This significantly increases health risks and costs.',
            'color': 'red'
        })
        severity_score += 25
    elif bmi > 30:
        reasons.append({
            'icon': '⚖️', 'factor': 'Elevated BMI',
            'impact': 'Moderate',
            'detail': f'BMI of {bmi} is in the obese range (30+). Weight management can reduce premiums.',
            'color': 'amber'
        })
        severity_score += 15
    elif bmi >= 25:
        reasons.append({
            'icon': '⚖️', 'factor': 'Overweight BMI',
            'impact': 'Low',
            'detail': f'BMI of {bmi} is slightly above normal (25-30). Minor impact on costs.',
            'color': 'amber'
        })
        severity_score += 5
    else:
        reasons.append({
            'icon': '💪', 'factor': 'Healthy BMI',
            'impact': 'Positive',
            'detail': f'BMI of {bmi} is within the healthy range. Great for lower premiums!',
            'color': 'emerald'
        })

    # Age analysis
    if age > 55:
        reasons.append({
            'icon': '👴', 'factor': 'Senior Age Group',
            'impact': 'High',
            'detail': f'Age {age} falls in the highest premium bracket. Medical expenses typically rise after 55.',
            'color': 'red'
        })
        severity_score += 20
    elif age > 45:
        reasons.append({
            'icon': '🩺', 'factor': 'Middle-Age Premium',
            'impact': 'Moderate',
            'detail': f'Age {age} enters the higher risk bracket. Regular health screenings recommended.',
            'color': 'amber'
        })
        severity_score += 12
    elif age > 35:
        reasons.append({
            'icon': '📅', 'factor': 'Age Factor',
            'impact': 'Low',
            'detail': f'Age {age} has a moderate impact on premium calculation.',
            'color': 'blue'
        })
        severity_score += 5
    else:
        reasons.append({
            'icon': '🌟', 'factor': 'Young Age Advantage',
            'impact': 'Positive',
            'detail': f'Age {age} qualifies for lower age-based premiums.',
            'color': 'emerald'
        })

    # Children analysis
    if children > 3:
        reasons.append({
            'icon': '👨‍👩‍👧‍👦', 'factor': 'Large Family',
            'impact': 'Moderate',
            'detail': f'{children} dependents increase coverage requirements and premium costs.',
            'color': 'amber'
        })
        severity_score += 8
    elif children > 0:
        reasons.append({
            'icon': '👶', 'factor': 'Family Coverage',
            'impact': 'Low',
            'detail': f'{children} dependent(s) add a small amount to the base premium.',
            'color': 'blue'
        })

    # Summary
    if severity_score >= 50:
        summary = 'Your profile indicates high medical expense risk. Consider comprehensive coverage.'
    elif severity_score >= 25:
        summary = 'Your profile shows moderate risk factors. Standard coverage should be adequate.'
    else:
        summary = 'Your profile looks healthy! You may qualify for lower premium rates.'

    return {'reasons': reasons, 'severity_score': severity_score, 'summary': summary}


def generate_recommendation(age, bmi, smoker, prediction_inr):
    """Insurance Plan Recommendation Engine."""
    if smoker == 1 and bmi > 30:
        return {
            'plan': 'Recommended Plan',
            'tier': 'premium',
            'icon': '🏥',
            'reason': 'High risk due to smoking and elevated BMI. Comprehensive coverage recommended with critical illness rider.',
            'features': ['Full hospitalization', 'Critical illness cover', 'Annual health check-up', 'No-claim bonus', 'Cashless treatment'],
            'est_premium': f"₹{prediction_inr * 0.08:,.0f} - ₹{prediction_inr * 0.12:,.0f}/year"
        }
    elif smoker == 1 or bmi > 30:
        return {
            'plan': 'Recommended Plan',
            'tier': 'premium',
            'icon': '🏥',
            'reason': f"{'Smoking habit' if smoker == 1 else 'High BMI'} increases health risk. Enhanced coverage advisable.",
            'features': ['Full hospitalization', 'Critical illness cover', 'Annual health check-up', 'No-claim bonus'],
            'est_premium': f"₹{prediction_inr * 0.06:,.0f} - ₹{prediction_inr * 0.10:,.0f}/year"
        }
    elif age > 40:
        return {
            'plan': 'Recommended Plan',
            'tier': 'standard',
            'icon': '🩺',
            'reason': 'Age-appropriate coverage with good balance of protection and affordability.',
            'features': ['Hospitalization cover', 'Day-care procedures', 'Annual check-up', 'No-claim bonus'],
            'est_premium': f"₹{prediction_inr * 0.04:,.0f} - ₹{prediction_inr * 0.07:,.0f}/year"
        }
    else:
        return {
            'plan': 'Recommended Plan',
            'tier': 'basic',
            'icon': '💚',
            'reason': 'Low risk profile. Basic coverage with essential benefits is sufficient.',
            'features': ['Hospitalization cover', 'Day-care procedures', 'No-claim bonus'],
            'est_premium': f"₹{prediction_inr * 0.03:,.0f} - ₹{prediction_inr * 0.05:,.0f}/year"
        }


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Async prediction endpoint returning JSON."""
    try:
        data = request.get_json()
        age = int(data.get('age', 0))
        sex = int(data.get('sex', 0))
        bmi = float(data.get('bmi', 0))
        children = int(data.get('children', 0))
        smoker = int(data.get('smoker', 0))
        region = int(data.get('region', 0))

        # Validate
        errors = []
        if age < 1 or age > 100:
            errors.append("Age must be between 1 and 100.")
        if sex not in [0, 1]:
            errors.append("Invalid gender selection.")
        if bmi < 10 or bmi > 60:
            errors.append("BMI must be between 10 and 60.")
        if children < 0 or children > 10:
            errors.append("Children must be between 0 and 10.")
        if smoker not in [0, 1]:
            errors.append("Invalid smoking status.")
        if region not in range(9):
            errors.append("Invalid region.")
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400

        features = np.array([[age, sex, bmi, children, smoker, region]])

        # Predict with both models
        results = {}
        if rf_model:
            pred_rf = float(rf_model.predict(features)[0])
            results['random_forest'] = {
                'inr': round(pred_rf * USD_TO_INR, 2),
            }
        if lr_model:
            pred_lr = float(lr_model.predict(features)[0])
            results['linear_regression'] = {
                'inr': round(pred_lr * USD_TO_INR, 2),
            }
        if dt_model:
            pred_dt = float(dt_model.predict(features)[0])
            results['decision_tree'] = {
                'inr': round(pred_dt * USD_TO_INR, 2),
            }
        if log_model:
            pred_log = int(log_model.predict(features)[0])
            results['logistic_regression'] = {
                'classification': 'High (> Median)' if pred_log == 1 else 'Low (<= Median)',
                'class_val': pred_log
            }

        # Primary prediction (Random Forest)
        primary = results.get('random_forest', results.get('linear_regression'))

        # Risk assessment
        risk_level = 'Low'
        risk_color = 'emerald'
        if primary['inr'] > 2500000:
            risk_level = 'High'
            risk_color = 'red'
        elif primary['inr'] > 800000:
            risk_level = 'Medium'
            risk_color = 'amber'

        # Health tips
        tips = []
        if smoker == 1:
            tips.append({
                'icon': '🚬', 'type': 'warning',
                'title': 'Smoking Impact',
                'text': 'Smoking increases insurance cost by 3-4x. Quitting could save you ₹' +
                        f'{(primary["inr"] * 0.65):,.0f} annually.'
            })
        if bmi > 30:
            tips.append({
                'icon': '⚖️', 'type': 'info',
                'title': 'BMI Advisory',
                'text': f'Your BMI ({bmi}) is above normal range. Maintaining a healthy weight can reduce costs significantly.'
            })
        if age > 50:
            tips.append({
                'icon': '🩺', 'type': 'info',
                'title': 'Age Factor',
                'text': 'Regular health check-ups are recommended. Preventive care can help manage insurance costs.'
            })
        if children > 3:
            tips.append({
                'icon': '👨‍👩‍👧‍👦', 'type': 'info',
                'title': 'Family Coverage',
                'text': f'With {children} dependents, consider a family floater policy for better value.'
            })
        if not tips:
            tips.append({
                'icon': '✅', 'type': 'success',
                'title': 'Good Health Profile',
                'text': 'Your health profile looks good! Maintain your current lifestyle for optimal insurance rates.'
            })

        # Generate AI explanation and recommendation
        explanation = generate_explanation(age, sex, bmi, children, smoker, region, primary['inr'])
        recommendation = generate_recommendation(age, bmi, smoker, primary['inr'])

        return jsonify({
            'success': True,
            'primary': primary,
            'models': results,
            'risk': {'level': risk_level, 'color': risk_color},
            'tips': tips,
            'explanation': explanation,
            'recommendation': recommendation,
            'is_smoker': smoker == 1,
            'input': {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region},
        })

    except Exception as e:
        return jsonify({'success': False, 'errors': [str(e)]}), 500


@app.route('/api/submit', methods=['POST'])
def api_submit():
    """Validate, run prediction, save user logs, and return a redirect id."""
    try:
        payload = request.get_json(silent=True) or {}
        errors, parsed = _validate_and_parse_submission(payload)
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400

        patient = parsed['patient']
        age = parsed['model_input']['age']
        sex = parsed['model_input']['sex']
        bmi = parsed['model_input']['bmi']
        children = parsed['model_input']['children']
        smoker = parsed['model_input']['smoker']
        region = parsed['model_input']['region']

        features = np.array([[age, sex, bmi, children, smoker, region]])

        # Predict with all available models (INR only)
        results = {}
        if rf_model:
            pred_rf = float(rf_model.predict(features)[0])
            results['random_forest'] = {'inr': round(pred_rf * USD_TO_INR, 2)}
        if lr_model:
            pred_lr = float(lr_model.predict(features)[0])
            results['linear_regression'] = {'inr': round(pred_lr * USD_TO_INR, 2)}
        if dt_model:
            pred_dt = float(dt_model.predict(features)[0])
            results['decision_tree'] = {'inr': round(pred_dt * USD_TO_INR, 2)}
        if log_model:
            pred_log = int(log_model.predict(features)[0])
            results['logistic_regression'] = {
                'classification': 'High (> Median)' if pred_log == 1 else 'Low (<= Median)',
                'class_val': pred_log
            }

        primary = results.get('random_forest', results.get('linear_regression'))
        if not primary:
            return jsonify({'success': False, 'errors': ['Prediction models are not available.']}), 500

        risk_level = 'Low'
        risk_color = 'emerald'
        if primary['inr'] > 2500000:
            risk_level = 'High'
            risk_color = 'red'
        elif primary['inr'] > 800000:
            risk_level = 'Medium'
            risk_color = 'amber'

        # Health tips
        tips = []
        if smoker == 1:
            tips.append({
                'icon': '🚬', 'type': 'warning',
                'title': 'Smoking Impact',
                'text': 'Smoking increases insurance cost by 3-4x. Quitting could save you ₹' +
                        f'{(primary["inr"] * 0.65):,.0f} annually.'
            })
        if bmi > 30:
            tips.append({
                'icon': '⚖️', 'type': 'info',
                'title': 'BMI Advisory',
                'text': f'Your BMI ({bmi}) is above normal range. Maintaining a healthy weight can reduce costs significantly.'
            })
        if age > 50:
            tips.append({
                'icon': '🩺', 'type': 'info',
                'title': 'Age Factor',
                'text': 'Regular health check-ups are recommended. Preventive care can help manage insurance costs.'
            })
        if children > 3:
            tips.append({
                'icon': '👨‍👩‍👧‍👦', 'type': 'info',
                'title': 'Family Coverage',
                'text': f'With {children} dependents, consider a family floater policy for better value.'
            })
        if not tips:
            tips.append({
                'icon': '✅', 'type': 'success',
                'title': 'Good Health Profile',
                'text': 'Your health profile looks good! Maintain your current lifestyle for optimal insurance rates.'
            })

        explanation = generate_explanation(age, sex, bmi, children, smoker, region, primary['inr'])
        recommendation = generate_recommendation(age, bmi, smoker, primary['inr'])

        entry = {
            'created_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'patient': patient,
            'primary': primary,
            'models': results,
            'risk': {'level': risk_level, 'color': risk_color},
            'tips': tips,
            'explanation': explanation,
            'recommendation': recommendation,
            'is_smoker': smoker == 1,
            'input': {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
        }

        log_id = str(uuid.uuid4())
        log_path = os.path.join(LOG_DIR, f'{log_id}.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False)

        return jsonify({'success': True, 'id': log_id})
    except Exception as e:
        return jsonify({'success': False, 'errors': [str(e)]}), 500


@app.route('/api/result/<log_id>')
def api_result(log_id: str):
    """Fetch a previously saved submission."""
    try:
        try:
            uuid.UUID(log_id)
        except Exception:
            return jsonify({'success': False, 'error': 'Invalid id.'}), 400

        log_path = os.path.join(LOG_DIR, f'{log_id}.json')
        if not os.path.exists(log_path):
            return jsonify({'success': False, 'error': 'Result not found.'}), 404

        with open(log_path, 'r', encoding='utf-8') as f:
            entry = json.load(f)

        return jsonify({'success': True, 'data': entry})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Some environments may not expose Flask's before_first_request decorator.
# To keep model loading robust across platforms, load models at import time.
try:
    load_models()
except Exception:
    # Defer to runtime load if import-time load fails
    app.logger.exception('Import-time model load failed; models will be attempted at runtime')


def _get_port():
    try:
        return int(os.environ.get('PORT', 5000))
    except Exception:
        return 5000


if __name__ == '__main__':
    # Helpful defaults for local development
    load_models()
    port = _get_port()
    app.run(host='0.0.0.0', port=port, debug=True)
