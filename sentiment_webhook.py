
"""
Sentiment Analysis Webhook para Botmaker
Hotel Concierge MVP - Marriott Singapore
Equipo 24

Este webhook recibe mensajes del chatbot y retorna un score de sentimiento
que determina si se debe escalar la conversación a un agente humano.

Regla de handoff: SENTIMENT ≤ -0.6 → derivar a agente humano
"""

from flask import Flask, request, jsonify
from textblob import TextBlob
from typing import Dict, Tuple
import logging
from datetime import datetime
import os
from functools import lru_cache

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuración
SENTIMENT_HANDOFF_THRESHOLD = -0.6
CONFIDENCE_HANDOFF_THRESHOLD = 0.55

# Palabras clave negativas por idioma para mejorar detección
NEGATIVE_KEYWORDS = {
    'es': [
        'terrible', 'horrible', 'pésimo', 'mal', 'malo', 'disgustado',
        'enojado', 'furioso', 'inaceptable', 'decepcionado', 'frustrado',
        'molesto', 'insatisfecho', 'quejas', 'problema', 'urgente'
    ],
    'en': [
        'terrible', 'horrible', 'awful', 'bad', 'poor', 'disgusted',
        'angry', 'furious', 'unacceptable', 'disappointed', 'frustrated',
        'annoyed', 'unsatisfied', 'complaints', 'problem', 'urgent'
    ]
}

# Palabras clave positivas para balancear
POSITIVE_KEYWORDS = {
    'es': [
        'excelente', 'perfecto', 'genial', 'fantástico', 'maravilloso',
        'increíble', 'bueno', 'gracias', 'agradecido', 'contento',
        'satisfecho', 'feliz', 'encantado'
    ],
    'en': [
        'excellent', 'perfect', 'great', 'fantastic', 'wonderful',
        'amazing', 'good', 'thanks', 'grateful', 'happy',
        'satisfied', 'pleased', 'delighted'
    ]
}


@lru_cache(maxsize=1000)
def analyze_sentiment_textblob(text: str, lang: str) -> float:
    """
    Analiza el sentimiento usando TextBlob.
    
    Args:
        text: Texto a analizar
        lang: Código de idioma (es/en)
    
    Returns:
        float: Score de sentimiento entre -1 (muy negativo) y 1 (muy positivo)
    """
    try:
        blob = TextBlob(text)
        # TextBlob retorna polarity entre -1 y 1
        polarity = blob.sentiment.polarity
        return polarity
    except Exception as e:
        logger.error(f"Error en análisis TextBlob: {str(e)}")
        return 0.0


def adjust_sentiment_with_keywords(text: str, base_sentiment: float, lang: str) -> float:
    """
    Ajusta el sentimiento base considerando palabras clave específicas.
    
    Args:
        text: Texto original
        base_sentiment: Sentimiento base de TextBlob
        lang: Código de idioma
    
    Returns:
        float: Sentimiento ajustado
    """
    text_lower = text.lower()
    
    # Contar palabras negativas
    negative_count = sum(1 for word in NEGATIVE_KEYWORDS.get(lang, []) 
                        if word in text_lower)
    
    # Contar palabras positivas
    positive_count = sum(1 for word in POSITIVE_KEYWORDS.get(lang, []) 
                        if word in text_lower)
    
    # Ajustar sentimiento
    adjustment = (positive_count * 0.1) - (negative_count * 0.15)
    adjusted_sentiment = max(-1.0, min(1.0, base_sentiment + adjustment))
    
    logger.info(f"Sentimiento base: {base_sentiment:.2f}, "
               f"Positivas: {positive_count}, Negativas: {negative_count}, "
               f"Ajustado: {adjusted_sentiment:.2f}")
    
    return adjusted_sentiment


def determine_handoff(sentiment_score: float, intent_confidence: float = None) -> Tuple[bool, str]:
    """
    Determina si se debe transferir a agente humano.
    
    Args:
        sentiment_score: Score de sentimiento
        intent_confidence: Confianza del intent detectado (opcional)
    
    Returns:
        Tuple[bool, str]: (requiere_handoff, razón)
    """
    reasons = []
    
    if sentiment_score <= SENTIMENT_HANDOFF_THRESHOLD:
        reasons.append(f"sentimiento negativo ({sentiment_score:.2f})")
    
    if intent_confidence is not None and intent_confidence < CONFIDENCE_HANDOFF_THRESHOLD:
        reasons.append(f"baja confianza intent ({intent_confidence:.2f})")
    
    requires_handoff = len(reasons) > 0
    reason = " y ".join(reasons) if reasons else "ninguna"
    
    return requires_handoff, reason


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check."""
    return jsonify({
        'status': 'healthy',
        'service': 'sentiment-webhook',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200


@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    """
    Endpoint principal para análisis de sentimiento.
    
    Request JSON:
    {
        "text": "El servicio es terrible",
        "lang": "es",
        "intent_confidence": 0.85 (opcional),
        "session_id": "abc123" (opcional),
        "user_id": "user456" (opcional)
    }
    
    Response JSON:
    {
        "sentiment_score": -0.75,
        "sentiment_label": "negative",
        "requires_handoff": true,
        "handoff_reason": "sentimiento negativo (-0.75)",
        "confidence": 0.90,
        "timestamp": "2025-11-10T12:00:00"
    }
    """
    try:
        # Validar request
        if not request.is_json:
            return jsonify({'error': 'Content-Type debe ser application/json'}), 400
        
        data = request.get_json()
        
        # Validar campos requeridos
        text = data.get('text', '').strip()
        lang = data.get('lang', 'es').lower()
        
        if not text:
            return jsonify({'error': 'Campo "text" es requerido y no puede estar vacío'}), 400
        
        if lang not in ['es', 'en']:
            return jsonify({'error': 'Campo "lang" debe ser "es" o "en"'}), 400
        
        # Campos opcionales
        intent_confidence = data.get('intent_confidence')
        session_id = data.get('session_id', '')
        user_id = data.get('user_id', '')
        
        logger.info(f"Analizando sentimiento - Session: {session_id}, User: {user_id}, Lang: {lang}")
        logger.info(f"Texto: {text[:100]}...")
        
        # Análisis de sentimiento
        base_sentiment = analyze_sentiment_textblob(text, lang)
        sentiment_score = adjust_sentiment_with_keywords(text, base_sentiment, lang)
        
        # Determinar label
        if sentiment_score > 0.3:
            sentiment_label = 'positive'
        elif sentiment_score < -0.3:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        # Determinar handoff
        requires_handoff, handoff_reason = determine_handoff(
            sentiment_score, 
            intent_confidence
        )
        
        # Calcular confidence del análisis
        confidence = min(0.95, abs(sentiment_score) + 0.2)
        
        # Respuesta
        response = {
            'sentiment_score': round(sentiment_score, 2),
            'sentiment_label': sentiment_label,
            'requires_handoff': requires_handoff,
            'handoff_reason': handoff_reason,
            'confidence': round(confidence, 2),
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': {
                'session_id': session_id,
                'user_id': user_id,
                'language': lang,
                'intent_confidence': intent_confidence
            }
        }
        
        logger.info(f"Resultado: score={sentiment_score:.2f}, "
                   f"handoff={requires_handoff}, reason={handoff_reason}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error en análisis de sentimiento: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500


@app.route('/batch', methods=['POST'])
def analyze_batch():
    """
    Endpoint para análisis de sentimiento en lote.
    
    Request JSON:
    {
        "messages": [
            {"text": "Excelente servicio", "lang": "es"},
            {"text": "Terrible experience", "lang": "en"}
        ]
    }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type debe ser application/json'}), 400
        
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages or not isinstance(messages, list):
            return jsonify({'error': 'Campo "messages" debe ser una lista no vacía'}), 400
        
        results = []
        for idx, msg in enumerate(messages):
            text = msg.get('text', '').strip()
            lang = msg.get('lang', 'es').lower()
            
            if not text:
                results.append({
                    'index': idx,
                    'error': 'Texto vacío'
                })
                continue
            
            base_sentiment = analyze_sentiment_textblob(text, lang)
            sentiment_score = adjust_sentiment_with_keywords(text, base_sentiment, lang)
            
            if sentiment_score > 0.3:
                sentiment_label = 'positive'
            elif sentiment_score < -0.3:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            requires_handoff, handoff_reason = determine_handoff(sentiment_score)
            
            results.append({
                'index': idx,
                'text': text[:50] + '...' if len(text) > 50 else text,
                'sentiment_score': round(sentiment_score, 2),
                'sentiment_label': sentiment_label,
                'requires_handoff': requires_handoff
            })
        
        return jsonify({
            'results': results,
            'total': len(messages),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error en análisis batch: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Iniciando Sentiment Webhook en puerto {port}")
    logger.info(f"Umbral de handoff por sentimiento: {SENTIMENT_HANDOFF_THRESHOLD}")
    logger.info(f"Umbral de handoff por confianza: {CONFIDENCE_HANDOFF_THRESHOLD}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
