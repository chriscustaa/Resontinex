#!/usr/bin/env python3
"""
RESONTINEX Cross-Judge Fusion Evaluator
Implements dual-evaluator scoring with rule-based validation to eliminate evaluator bias.
"""

import os
import re
import ast
import json
import time
import statistics
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure monkeypatch targets exist
try:
    import openai  # noqa
except Exception:  # provide stub for tests
    class _OpenAI:
        def __init__(self): pass
    openai = _OpenAI()

class Prompt3Evaluator:
    """Production-grade evaluator implementing Prompt-3 methodology for response quality assessment."""
    
    def __init__(self):
        # Specificity indicators (higher scores for more specific responses)
        self.specificity_patterns = {
            'numeric_precision': re.compile(r'\b\d+\.?\d*\b'),
            'temporal_specificity': re.compile(r'\b(\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|within \d+|by \d+)\b'),
            'technical_terms': re.compile(r'\b(API|database|server|endpoint|configuration|implementation)\b', re.IGNORECASE),
            'actionable_verbs': re.compile(r'\b(implement|execute|configure|deploy|analyze|validate|test)\b', re.IGNORECASE),
            'quantified_metrics': re.compile(r'\b(\d+%|\d+\.\d+%|>\d+|<\d+|\d+MB|\d+GB|\d+ms|\d+s)\b')
        }
        
        # Rationale density indicators (reasoning and justification quality)
        self.rationale_patterns = {
            'causal_indicators': re.compile(r'\b(because|since|due to|caused by|results in|leads to)\b', re.IGNORECASE),
            'logical_connectors': re.compile(r'\b(therefore|thus|consequently|however|furthermore|additionally)\b', re.IGNORECASE),
            'evidence_markers': re.compile(r'\b(based on|according to|analysis shows|data indicates|evidence suggests)\b', re.IGNORECASE),
            'risk_assessment': re.compile(r'\b(risk|impact|consequence|mitigation|prevention|safeguard)\b', re.IGNORECASE),
            'comparison_analysis': re.compile(r'\b(versus|compared to|alternative|option|trade-off)\b', re.IGNORECASE)
        }
        
        # Operationality indicators (actionable implementation guidance)
        self.operationality_patterns = {
            'step_sequences': re.compile(r'\b(step \d+|first|next|then|finally|\d+\.)\b', re.IGNORECASE),
            'resource_specifications': re.compile(r'\b(team|developer|hour|day|budget|\$\d+|person)\b', re.IGNORECASE),
            'timeline_markers': re.compile(r'\b(immediate|within|by|deadline|schedule|timeline)\b', re.IGNORECASE),
            'dependency_indicators': re.compile(r'\b(requires|depends on|prerequisite|before|after)\b', re.IGNORECASE),
            'success_criteria': re.compile(r'\b(success|completion|validation|verification|testing)\b', re.IGNORECASE)
        }

    def evaluate_specificity(self, text: str) -> float:
        """Evaluate response specificity (0.0-1.0 scale)."""
        total_score = 0.0
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        for pattern_name, pattern in self.specificity_patterns.items():
            matches = len(pattern.findall(text))
            weights = {
                'numeric_precision': 0.25,
                'temporal_specificity': 0.30,
                'technical_terms': 0.15,
                'actionable_verbs': 0.20,
                'quantified_metrics': 0.35
            }
            normalized_score = min(matches / (word_count / 100), 1.0)
            total_score += normalized_score * weights.get(pattern_name, 0.2)
        
        return min(total_score, 1.0)

    def evaluate_rationale_density(self, text: str) -> float:
        """Evaluate reasoning and justification density (0.0-1.0 scale)."""
        total_score = 0.0
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        if sentence_count == 0:
            return 0.0
        
        for pattern_name, pattern in self.rationale_patterns.items():
            matches = len(pattern.findall(text))
            weights = {
                'causal_indicators': 0.30,
                'logical_connectors': 0.20,
                'evidence_markers': 0.25,
                'risk_assessment': 0.15,
                'comparison_analysis': 0.10
            }
            normalized_score = min(matches / sentence_count, 1.0)
            total_score += normalized_score * weights.get(pattern_name, 0.2)
        
        return min(total_score, 1.0)

    def evaluate_operationality(self, text: str) -> float:
        """Evaluate actionability and implementation guidance (0.0-1.0 scale)."""
        total_score = 0.0
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        if paragraph_count == 0:
            paragraph_count = 1
        
        for pattern_name, pattern in self.operationality_patterns.items():
            matches = len(pattern.findall(text))
            weights = {
                'step_sequences': 0.35,
                'resource_specifications': 0.20,
                'timeline_markers': 0.25,
                'dependency_indicators': 0.15,
                'success_criteria': 0.05
            }
            normalized_score = min(matches / paragraph_count, 1.0)
            total_score += normalized_score * weights.get(pattern_name, 0.2)
        
        return min(total_score, 1.0)

    def calculate_entropy_score(self, text: str) -> float:
        """Calculate entropy score for response consistency assessment."""
        if not text:
            return 1.0
        
        char_freq = {}
        for char in text.lower():
            if char.isalnum():
                char_freq[char] = char_freq.get(char, 0) + 1
        
        if not char_freq:
            return 1.0
        
        total_chars = sum(char_freq.values())
        entropy = 0.0
        
        for count in char_freq.values():
            probability = count / total_chars
            if probability > 0:
                import math
                entropy -= probability * math.log2(probability)
        
        max_entropy = 4.5
        return min(entropy / max_entropy, 1.0)

@dataclass
class EvaluationResult:
    """Evaluation result for a single scenario run."""
    scenario_id: str
    configuration: str
    specificity: float
    rationale_density: float
    operationality: float
    response_time_ms: int
    response_length: int
    entropy_score: float
    execution_timestamp: str


@dataclass
class RuleCheckResult:
    """Result from rule-based validation checks."""
    has_required_hooks: bool
    has_guardrails: bool
    code_compiles: bool
    key_phrases_present: bool
    technical_completeness: float
    rule_score: float


@dataclass
class CrossJudgeResult:
    """Result from cross-judge evaluation including rule validation."""
    scenario_id: str
    configuration: str
    primary_judge_scores: Dict[str, float]
    secondary_judge_scores: Dict[str, float]
    averaged_scores: Dict[str, float]
    rule_check: RuleCheckResult
    final_scores: Dict[str, float]
    execution_timestamp: str
    judges_used: List[str]


class RuleBasedValidator:
    """Rule-based validation engine for response quality assurance."""
    
    def __init__(self):
        # Define required patterns based on scenario categories
        self.required_hooks = [
            'woocommerce_scheduled_subscription_payment',
            'subscription_status_updated',
            'payment_complete',
            'order_status_changed',
            'user_register',
            'init',
            'wp_enqueue_scripts',
            'admin_menu'
        ]
        
        self.guardrail_keywords = [
            'idempotency', 'rollback', 'observability', 'retries',
            'validation', 'error_handling', 'logging', 'monitoring',
            'transaction', 'atomic', 'recovery', 'fallback'
        ]
        
        self.technical_completeness_patterns = {
            'implementation_steps': re.compile(r'\b(implement|execute|configure|deploy|create|setup)\s+\w+', re.IGNORECASE),
            'resource_specifications': re.compile(r'\b(\d+\s*(hours?|days?|weeks?)|team\s+of\s+\d+|\$\d+)', re.IGNORECASE),
            'technical_details': re.compile(r'\b(API|endpoint|database|table|function|class|method|variable)\s+\w+', re.IGNORECASE),
            'error_conditions': re.compile(r'\b(error|exception|failure|timeout|retry|fallback)\b', re.IGNORECASE),
            'success_criteria': re.compile(r'\b(success|complete|verify|validate|test|confirm)\b', re.IGNORECASE)
        }
        
        self.key_phrases_by_category = {
            'financial_operations': ['refund', 'payment', 'transaction', 'charge', 'billing'],
            'property_management': ['survey', 'boundary', 'property', 'legal', 'compliance'],
            'system_integration': ['webhook', 'endpoint', 'API', 'delivery', 'integration'],
            'compliance_management': ['GDPR', 'privacy', 'data', 'consent', 'audit'],
            'security_operations': ['security', 'breach', 'incident', 'containment', 'forensic']
        }

    def check_code_compilation(self, response: str) -> bool:
        """Check if code blocks in response are syntactically valid."""
        # Extract code blocks (simple pattern matching)
        code_patterns = [
            re.compile(r'```(?:python|php|javascript|js|sql)\s*(.*?)```', re.DOTALL | re.IGNORECASE),
            re.compile(r'`([^`\n]+)`', re.MULTILINE)
        ]
        
        for pattern in code_patterns:
            code_blocks = pattern.findall(response)
            for code in code_blocks:
                # Basic syntax validation for Python-like code
                try:
                    # Remove common non-Python elements that might be in the code
                    cleaned_code = re.sub(r'<\?php|<script>|</script>|\?>', '', code.strip())
                    cleaned_code = re.sub(r'\/\/.*$|\/\*.*?\*\/', '', cleaned_code, flags=re.MULTILINE | re.DOTALL)
                    
                    if cleaned_code and len(cleaned_code.strip()) > 5:
                        # Try to parse as Python AST for basic syntax validation
                        try:
                            ast.parse(cleaned_code)
                        except SyntaxError:
                            # If Python parsing fails, do basic bracket/quote matching
                            if not self._validate_bracket_matching(cleaned_code):
                                return False
                except Exception:
                    continue
                    
        return True

    def _validate_bracket_matching(self, code: str) -> bool:
        """Validate bracket and quote matching in code."""
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        in_string = False
        quote_char = None
        
        for char in code:
            if not in_string:
                if char in ['"', "'", '`']:
                    in_string = True
                    quote_char = char
                elif char in brackets:
                    stack.append(char)
                elif char in brackets.values():
                    if not stack:
                        return False
                    if brackets[stack.pop()] != char:
                        return False
            else:
                if char == quote_char and (len(code) == 1 or code[code.index(char)-1] != '\\'):
                    in_string = False
                    quote_char = None
        
        return len(stack) == 0 and not in_string

    def validate_response(self, response: str, scenario_category: str) -> RuleCheckResult:
        """Perform comprehensive rule-based validation."""
        # Check for required hooks (context-dependent)
        has_hooks = any(hook in response.lower() for hook in self.required_hooks)
        
        # Check for guardrail patterns
        guardrail_count = sum(1 for keyword in self.guardrail_keywords if keyword in response.lower())
        has_guardrails = guardrail_count >= 4
        
        # Check code compilation
        code_compiles = self.check_code_compilation(response)
        
        # Check for category-specific key phrases
        category_phrases = self.key_phrases_by_category.get(scenario_category, [])
        phrases_found = sum(1 for phrase in category_phrases if phrase.lower() in response.lower())
        key_phrases_present = phrases_found >= max(1, len(category_phrases) // 2)
        
        # Calculate technical completeness score
        completeness_score = 0.0
        word_count = len(response.split())
        
        for pattern_name, pattern in self.technical_completeness_patterns.items():
            matches = len(pattern.findall(response))
            normalized = min(matches / max(word_count / 100, 1), 1.0)
            
            weights = {
                'implementation_steps': 0.25,
                'resource_specifications': 0.20,
                'technical_details': 0.20,
                'error_conditions': 0.20,
                'success_criteria': 0.15
            }
            
            completeness_score += normalized * weights.get(pattern_name, 0.2)
        
        # Calculate overall rule score
        rule_components = [
            has_hooks * 0.2,
            has_guardrails * 0.3,
            code_compiles * 0.2,
            key_phrases_present * 0.15,
            min(completeness_score, 1.0) * 0.15
        ]
        
        rule_score = sum(rule_components)
        
        return RuleCheckResult(
            has_required_hooks=has_hooks,
            has_guardrails=has_guardrails,
            code_compiles=code_compiles,
            key_phrases_present=key_phrases_present,
            technical_completeness=min(completeness_score, 1.0),
            rule_score=min(rule_score, 1.0)
        )
    
    def validate_consistency(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Validate consistency of scoring across judges."""
        variance = self._calculate_variance(scores)
        outliers = self._detect_outliers(scores)
        confidence = self._calculate_confidence(scores)
        
        is_consistent = variance < 0.1 and len(outliers) == 0
        
        return {
            'is_consistent': is_consistent,
            'variance': variance,
            'outliers': outliers,
            'confidence': confidence,
            'recommendation': 'accept' if is_consistent else 'review'
        }
    
    def _calculate_variance(self, scores) -> float:
        """Calculate variance in scoring."""
        # Handle both dict and list inputs
        if isinstance(scores, dict):
            values = list(scores.values())
        elif isinstance(scores, list):
            values = scores
        else:
            return 0.0
        
        if not values or len(values) < 2:
            return 0.0
        
        mean_score = sum(values) / len(values)
        variance = sum((score - mean_score) ** 2 for score in values) / len(values)
        return variance
    
    def _detect_outliers(self, scores) -> List[str]:
        """Detect outlier scores using simple threshold."""
        # Handle both dict and list inputs
        if isinstance(scores, dict):
            values = list(scores.values())
            keys = list(scores.keys())
        elif isinstance(scores, list):
            values = scores
            keys = [f"item_{i}" for i in range(len(scores))]
        else:
            return []
        
        if not values or len(values) < 3:
            return []
        
        mean_score = sum(values) / len(values)
        std_dev = (sum((score - mean_score) ** 2 for score in values) / len(values)) ** 0.5
        
        outliers = []
        threshold = 2.0 * std_dev  # 2 standard deviations
        
        for i, score in enumerate(values):
            if abs(score - mean_score) > threshold:
                outliers.append(keys[i])
        
        return outliers
    
    def _calculate_confidence(self, scores) -> float:
        """Calculate confidence level based on score consistency."""
        if not scores:
            return 0.0
        
        variance = self._calculate_variance(scores)
        # Higher variance = lower confidence
        confidence = max(0.0, min(1.0, 1.0 - (variance * 10)))
        
        return confidence


class AlternativeJudgeEvaluator:
    """Alternative judge implementing different evaluation methodology for bias reduction."""
    
    def __init__(self):
        # Alternative patterns focusing on different aspects
        self.specificity_indicators = {
            'quantitative_measures': re.compile(r'\b(\d+\.?\d*\s*(hours?|days?|%|MB|GB|ms|seconds?)|within\s+\d+|\$\d+)\b'),
            'concrete_actions': re.compile(r'\b(create|modify|delete|update|configure|install|deploy|execute)\s+\w+', re.IGNORECASE),
            'system_components': re.compile(r'\b(server|database|API|endpoint|service|application|script)\s+\w+', re.IGNORECASE),
            'performance_metrics': re.compile(r'\b(\d+\.?\d*\s*(response\s+time|latency|throughput|utilization|availability))\b', re.IGNORECASE)
        }
        
        self.rationale_indicators = {
            'problem_analysis': re.compile(r'\b(analysis|assessment|evaluation|investigation|diagnosis)\b', re.IGNORECASE),
            'decision_reasoning': re.compile(r'\b(rationale|justification|reasoning|explanation|because|since)\b', re.IGNORECASE),
            'trade_off_discussion': re.compile(r'\b(trade-off|alternative|versus|compared|option|choice)\b', re.IGNORECASE),
            'impact_assessment': re.compile(r'\b(impact|effect|consequence|result|outcome)\b', re.IGNORECASE)
        }
        
        self.operationality_indicators = {
            'sequential_steps': re.compile(r'\b(step\s+\d+|phase\s+\d+|stage\s+\d+|\d+\.|first|second|third|next|then|finally)\b', re.IGNORECASE),
            'role_assignments': re.compile(r'\b(developer|admin|user|team|engineer|analyst)\s+(should|must|will|needs to)\b', re.IGNORECASE),
            'deliverables': re.compile(r'\b(deliverable|output|result|artifact|documentation|report)\b', re.IGNORECASE),
            'validation_methods': re.compile(r'\b(test|verify|validate|confirm|check|review|audit)\b', re.IGNORECASE)
        }

    def evaluate_specificity(self, text: str) -> float:
        """Alternative specificity evaluation focusing on concrete details."""
        total_score = 0.0
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        if sentence_count == 0:
            return 0.0
        
        for category, pattern in self.specificity_indicators.items():
            matches = len(pattern.findall(text))
            
            weights = {
                'quantitative_measures': 0.35,
                'concrete_actions': 0.25,
                'system_components': 0.25,
                'performance_metrics': 0.15
            }
            
            normalized = min(matches / sentence_count, 1.0)
            total_score += normalized * weights.get(category, 0.25)
        
        return min(total_score, 1.0)

    def evaluate_rationale_density(self, text: str) -> float:
        """Alternative rationale evaluation focusing on analytical depth."""
        total_score = 0.0
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        if paragraph_count == 0:
            paragraph_count = 1
        
        for category, pattern in self.rationale_indicators.items():
            matches = len(pattern.findall(text))
            
            weights = {
                'problem_analysis': 0.30,
                'decision_reasoning': 0.35,
                'trade_off_discussion': 0.20,
                'impact_assessment': 0.15
            }
            
            normalized = min(matches / paragraph_count, 1.0)
            total_score += normalized * weights.get(category, 0.25)
        
        return min(total_score, 1.0)

    def evaluate_operationality(self, text: str) -> float:
        """Alternative operationality evaluation focusing on implementation clarity."""
        total_score = 0.0
        line_count = len([l for l in text.split('\n') if l.strip()])
        
        if line_count == 0:
            return 0.0
        
        for category, pattern in self.operationality_indicators.items():
            matches = len(pattern.findall(text))
            
            weights = {
                'sequential_steps': 0.40,
                'role_assignments': 0.25,
                'deliverables': 0.20,
                'validation_methods': 0.15
            }
            
            normalized = min(matches / line_count, 1.0)
            total_score += normalized * weights.get(category, 0.25)
        
        return min(total_score, 1.0)


class CrossJudgeEvaluator:
    """Cross-judge evaluation system with rule-based validation."""
    
    def __init__(self):
        self.primary_judge = Prompt3Evaluator()
        self.secondary_judge = AlternativeJudgeEvaluator()
        self.validator = RuleBasedValidator()  # Alias for tests
        self.rule_validator = self.validator
        
    def evaluate_with_cross_judges(self, response: str, scenario: Dict[str, Any]) -> CrossJudgeResult:
        """Perform cross-judge evaluation with rule validation."""
        scenario_id = scenario.get('id', 'unknown')
        scenario_category = scenario.get('category', 'general')
        
        # Primary judge evaluation
        primary_scores = {
            'specificity': self.primary_judge.evaluate_specificity(response),
            'rationale_density': self.primary_judge.evaluate_rationale_density(response),
            'operationality': self.primary_judge.evaluate_operationality(response),
            'entropy': self.primary_judge.calculate_entropy_score(response)
        }
        
        # Secondary judge evaluation
        secondary_scores = {
            'specificity': self.secondary_judge.evaluate_specificity(response),
            'rationale_density': self.secondary_judge.evaluate_rationale_density(response),
            'operationality': self.secondary_judge.evaluate_operationality(response),
            'entropy': primary_scores['entropy']  # Use same entropy calculation
        }
        
        # Average the judge scores
        averaged_scores = {}
        for metric in ['specificity', 'rationale_density', 'operationality', 'entropy']:
            averaged_scores[metric] = (primary_scores[metric] + secondary_scores[metric]) / 2.0
        
        # Rule-based validation
        rule_check = self.rule_validator.validate_response(response, scenario_category)
        
        # Final scores incorporating rule validation
        final_scores = {}
        for metric in ['specificity', 'rationale_density', 'operationality']:
            # Weight: 80% judge average, 20% rule validation
            judge_component = averaged_scores[metric] * 0.80
            rule_component = rule_check.rule_score * 0.20
            final_scores[metric] = min(judge_component + rule_component, 1.0)
        
        final_scores['entropy'] = averaged_scores['entropy']
        final_scores['rules_ok'] = rule_check.rule_score >= 0.75
        
        return CrossJudgeResult(
            scenario_id=scenario_id,
            configuration='cross_judge',
            primary_judge_scores=primary_scores,
            secondary_judge_scores=secondary_scores,
            averaged_scores=averaged_scores,
            rule_check=rule_check,
            final_scores=final_scores,
            execution_timestamp=datetime.now(timezone.utc).isoformat(),
            judges_used=['prompt3', 'alternative']
        )
    
    def run_evaluation_comparison(self, scenario: Dict[str, Any], baseline_response: str, overlay_response: str) -> Tuple[CrossJudgeResult, CrossJudgeResult]:
        """Run cross-judge evaluation for baseline vs overlay comparison."""
        baseline_result = self.evaluate_with_cross_judges(baseline_response, scenario)
        baseline_result.configuration = 'baseline'
        
        overlay_result = self.evaluate_with_cross_judges(overlay_response, scenario)
        overlay_result.configuration = 'overlay'
        
        return baseline_result, overlay_result
    
    def calculate_improvement_metrics(self, baseline: CrossJudgeResult, overlay: CrossJudgeResult) -> Dict[str, float]:
        """Calculate improvement metrics between baseline and overlay."""
        improvements = {}
        
        for metric in ['specificity', 'rationale_density', 'operationality']:
            improvements[f'{metric}_improvement'] = overlay.final_scores[metric] - baseline.final_scores[metric]
        
        # Overall improvement (weighted)
        weights = {'specificity': 0.35, 'rationale_density': 0.30, 'operationality': 0.35}
        overall = sum(improvements[f'{metric}_improvement'] * weight for metric, weight in weights.items())
        improvements['overall_improvement'] = overall
        
        # Rule validation improvement
        improvements['rule_validation_improvement'] = overlay.rule_check.rule_score - baseline.rule_check.rule_score
        
        # Judge consensus (lower variance = higher consensus)
        baseline_variance = statistics.variance([
            abs(baseline.primary_judge_scores['specificity'] - baseline.secondary_judge_scores['specificity']),
            abs(baseline.primary_judge_scores['rationale_density'] - baseline.secondary_judge_scores['rationale_density']),
            abs(baseline.primary_judge_scores['operationality'] - baseline.secondary_judge_scores['operationality'])
        ])
        
        overlay_variance = statistics.variance([
            abs(overlay.primary_judge_scores['specificity'] - overlay.secondary_judge_scores['specificity']),
            abs(overlay.primary_judge_scores['rationale_density'] - overlay.secondary_judge_scores['rationale_density']),
            abs(overlay.primary_judge_scores['operationality'] - overlay.secondary_judge_scores['operationality'])
        ])
        
        improvements['judge_consensus_improvement'] = baseline_variance - overlay_variance
        
        return improvements
    
    def evaluate_cross_judge(self, scenario: Dict[str, Any], baseline_response: str, overlay_response: str) -> Dict[str, Any]:
        """Evaluate cross-judge process for test compatibility."""
        baseline_result = self.evaluate_with_cross_judges(baseline_response, scenario)
        baseline_result.configuration = 'baseline'
        
        overlay_result = self.evaluate_with_cross_judges(overlay_response, scenario)
        overlay_result.configuration = 'overlay'
        
        improvements = self.calculate_improvement_metrics(baseline_result, overlay_result)
        
        return {
            'primary_score': overlay_result.final_scores,
            'validation_result': {
                'is_consistent': overlay_result.rule_check.rule_score > 0.75,
                'rule_score': overlay_result.rule_check.rule_score
            },
            'final_recommendation': 'overlay' if improvements['overall_improvement'] > 0 else 'baseline',
            'baseline_result': baseline_result,
            'overlay_result': overlay_result,
            'improvements': improvements
        }


def main():
    """Demo of cross-judge evaluation system."""
    evaluator = CrossJudgeEvaluator()
    
    # Example scenario
    test_scenario = {
        'id': 'test_refund_processing',
        'category': 'financial_operations',
        'description': 'Customer refund processing with idempotency checks',
        'context': 'Customer requests refund for duplicate charge on subscription'
    }
    
    # Example responses
    baseline_response = "Process the refund request. Check if duplicate exists. Update records accordingly. Send confirmation to customer."
    
    overlay_response = """
    ## Refund Processing Analysis
    
    Based on the duplicate charge scenario, implement the following:
    
    1. **Idempotency Check**: Verify transaction ID hasn't been processed
    2. **Financial Validation**: Confirm charge amount ($299.99) against account records
    3. **Rollback Mechanism**: Execute atomic refund with observability logging
    4. **Customer Communication**: Send automated confirmation within 2 hours
    
    ### Implementation Steps:
    ```php
    function process_refund($transaction_id, $amount) {
        add_action('woocommerce_scheduled_subscription_payment', 'validate_refund');
        
        if (is_duplicate_processed($transaction_id)) {
            return error_log('Duplicate refund attempt detected');
        }
        
        try {
            $result = execute_refund($transaction_id, $amount);
            log_transaction('refund_success', $result);
            return $result;
        } catch (Exception $e) {
            rollback_transaction($transaction_id);
            throw $e;
        }
    }
    ```
    
    **Success Criteria**: 99.5% refund processing accuracy, <2 hour response time, complete audit trail.
    """
    
    print("Running Cross-Judge Evaluation Demo...")
    print("=" * 50)
    
    baseline_result, overlay_result = evaluator.run_evaluation_comparison(
        test_scenario, baseline_response, overlay_response
    )
    
    improvements = evaluator.calculate_improvement_metrics(baseline_result, overlay_result)
    
    print(f"Baseline Scores:")
    print(f"  Specificity: {baseline_result.final_scores['specificity']:.3f}")
    print(f"  Rationale: {baseline_result.final_scores['rationale_density']:.3f}")
    print(f"  Operations: {baseline_result.final_scores['operationality']:.3f}")
    print(f"  Rules OK: {baseline_result.final_scores['rules_ok']}")
    
    print(f"\nOverlay Scores:")
    print(f"  Specificity: {overlay_result.final_scores['specificity']:.3f}")
    print(f"  Rationale: {overlay_result.final_scores['rationale_density']:.3f}")
    print(f"  Operations: {overlay_result.final_scores['operationality']:.3f}")
    print(f"  Rules OK: {overlay_result.final_scores['rules_ok']}")
    
    print(f"\nImprovements:")
    for metric, value in improvements.items():
        print(f"  {metric}: {value:+.3f}")
    
    print(f"\nRule Validation Details:")
    print(f"  Baseline Rule Score: {baseline_result.rule_check.rule_score:.3f}")
    print(f"  Overlay Rule Score: {overlay_result.rule_check.rule_score:.3f}")
    print(f"  Code Compiles: {overlay_result.rule_check.code_compiles}")
    print(f"  Has Guardrails: {overlay_result.rule_check.has_guardrails}")


# Export classes for tests
__all__ = ["RuleBasedValidator", "CrossJudgeEvaluator", "Prompt3Evaluator", "openai"]

if __name__ == "__main__":
    main()