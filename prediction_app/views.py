from django.shortcuts import render, redirect
from django.contrib import messages
from .ml_predictor import EffluentPredictor
from .standards import NEMA_STANDARDS
import tempfile
import os
def home(request):
    return render(request, 'index.html')
def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')
predictor = EffluentPredictor()

def _get_unit(param):
    """Get unit for parameter (moved outside class)"""
    units = {
        'COD': 'mg/L',
        'TSS': 'mg/L',
        'TDS': 'mg/L',
        'Conductivity': 'μS/cm',
        'pH': ''
    }
    return units.get(param.replace('Effluent_', ''), '')


def train(request):
    context = {}

    if request.method == 'POST':
        # Handle influent upload
        if 'influent_file' in request.FILES:
            try:
                # Save influent file temporarily
                influent_file = request.FILES['influent_file']
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                    for chunk in influent_file.chunks():
                        tmp.write(chunk)
                    request.session['influent_path'] = tmp.name

                messages.success(request, "Influent data uploaded. Now upload effluent data.")
                return redirect('train')

            except Exception as e:
                messages.error(request, f"Error uploading influent file: {str(e)}")

        # Handle effluent upload and training
        elif 'effluent_file' in request.FILES and 'influent_path' in request.session:
            try:
                # Save effluent file temporarily
                effluent_file = request.FILES['effluent_file']
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                    for chunk in effluent_file.chunks():
                        tmp.write(chunk)
                    effluent_path = tmp.name

                # Train model
                if predictor.train_from_excel(request.session['influent_path'], effluent_path):
                    messages.success(request, "Model trained successfully!")
                else:
                    messages.error(request, "Training failed")

                # Cleanup
                os.unlink(request.session['influent_path'])
                os.unlink(effluent_path)
                del request.session['influent_path']

            except Exception as e:
                messages.error(request, f"Training error: {str(e)}")
                if 'influent_path' in request.session:
                    os.unlink(request.session['influent_path'])
                if 'effluent_path' in locals():
                    os.unlink(effluent_path)

    context['training_done'] = predictor.is_trained
    return render(request, 'train.html', context)

def predict(request):
    context = {'standards': NEMA_STANDARDS}

    if request.method == 'POST':
        try:
            # Get form data
            form_data = {
                'COD': float(request.POST.get('COD')),
                'pH': float(request.POST.get('pH')),
                'TSS': float(request.POST.get('TSS')),
                'TDS': float(request.POST.get('TDS')),
                'Conductivity': float(request.POST.get('Conductivity'))
            }

            # Get predictions
            predictions = predictor.predict(form_data)

            # Format results
            results = []
            for param, value in predictions.items():
                clean_param = param.replace('Effluent_', '')
                standard = NEMA_STANDARDS.get(clean_param)

                if isinstance(standard, tuple):  # pH range
                    within_limits = standard[0] <= value <= standard[1]
                    limit_str = f"{standard[0]} - {standard[1]}"
                else:
                    within_limits = value <= standard
                    limit_str = f"≤ {standard}"

                results.append({
                    'parameter': clean_param,
                    'predicted_value': round(value, 2),
                    'standard_limit': limit_str,
                    'within_limits': within_limits,
                    'unit': _get_unit(param)  # Fixed: Using standalone function now
                })

            context.update({
                'input_data': form_data,
                'results': results,
                'prediction_success': True
            })

        except ValueError as e:
            messages.error(request, f"Invalid input: {str(e)}")
        except Exception as e:
            messages.error(request, f"Prediction error: {str(e)}")

    context['model_ready'] = predictor.is_trained or predictor._check_models_exist()
    return render(request, 'predict.html', context)