"""
Response Quality Checker and Reprompting System
Validates AI responses and triggers reprompting for better quality
"""
#
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import openai
from loguru import logger

class QualityIssue(Enum):
    TOO_SHORT = "Response is too short and lacks detail"
    TOO_LONG = "Response is too long and verbose"
    TOO_GENERIC = "Response is too generic and not personalized"
    LACKS_EMPATHY = "Response lacks empathy and warmth"
    NO_FOLLOW_UP = "Response doesn't include follow-up question"
    UNCLEAR_ADVICE = "Advice is unclear or not actionable"
    MISSING_REASONING = "Missing step-by-step reasoning"
    INAPPROPRIATE_TONE = "Tone is not professional or caring, or contains meta-commentary"
    FACTUAL_CONCERNS = "Potential factual accuracy concerns"
    NO_DISCLAIMER = "Missing appropriate medical disclaimers"

@dataclass
class QualityScore:
    overall_score: float  # 0-100
    issues: List[QualityIssue]
    strengths: List[str]
    suggestions: List[str]

class ResponseQualityChecker:
    """Comprehensive quality assessment for dental chatbot responses"""
    
    def __init__(self, openai_client):
        self.client = openai_client
        self.min_response_length = 50
        self.max_response_length = 600  # Allow longer for detailed medical info
        self.quality_threshold = 75.0
        
    def check_response_quality(self, user_question: str, ai_response: str, context: str = "") -> QualityScore:
        """Comprehensive quality assessment of AI response"""
        
        issues = []
        strengths = []
        suggestions = []
        
        # Smart length check based on question type
        medical_keywords = ["disease", "procedure", "treatment", "process", "symptoms", "causes", "prevention", "how does", "what is", "explain"]
        is_detailed_query = any(keyword in user_question.lower() for keyword in medical_keywords)

        if len(ai_response) < self.min_response_length:
            issues.append(QualityIssue.TOO_SHORT)
            suggestions.append("Provide a complete but concise answer")
        elif len(ai_response) > self.max_response_length and not is_detailed_query:
            issues.append(QualityIssue.TOO_LONG)
            suggestions.append("Keep response shorter and more focused - 2-3 sentences maximum")
        elif len(ai_response) > 800:  # Hard limit even for detailed queries
            issues.append(QualityIssue.TOO_LONG)
            suggestions.append("Response is too long even for detailed medical information")
        
        # Empathy and warmth check
        empathy_indicators = [
            "i understand", "i'm sorry", "i know", "i can imagine", 
            "that sounds", "i'm here to help", "let me help"
        ]
        if not any(indicator in ai_response.lower() for indicator in empathy_indicators):
            issues.append(QualityIssue.LACKS_EMPATHY)
            suggestions.append("Add more empathetic and caring language")
        else:
            strengths.append("Shows empathy and understanding")
        
        # Follow-up question check
        question_patterns = [r'\?', r'would you like', r'can you tell me', r'how long', r'when did']
        if not any(re.search(pattern, ai_response.lower()) for pattern in question_patterns):
            issues.append(QualityIssue.NO_FOLLOW_UP)
            suggestions.append("Include a follow-up question to engage the patient")
        else:
            strengths.append("Includes engaging follow-up question")
        
        # Professional tone check - should refer to Dr. Tomar in third person
        professional_indicators = [
            "dr. tomar recommends", "dr. tomar performs", "dr. tomar does", "dr. tomar specializes",
            "based on", "clinically", "professional", "dr. tomar's experience"
        ]
        # Check for inappropriate first person usage and meta-commentary
        first_person_issues = ["i recommend", "i perform", "i do", "i've seen", "in my experience"]
        meta_commentary = [
            "this response has been adjusted", "here's a revised response", "certainly! here", 
            "response that incorporates", "adjusted to reflect", "more engaging tone", 
            "while maintaining professionalism", "while ensuring clarity"
        ]
        has_first_person = any(phrase in ai_response.lower() for phrase in first_person_issues)
        has_meta_commentary = any(phrase in ai_response.lower() for phrase in meta_commentary)
        
        if has_first_person:
            issues.append(QualityIssue.INAPPROPRIATE_TONE)
            suggestions.append("Refer to Dr. Tomar in third person, not first person - you are the assistant")
        elif has_meta_commentary:
            issues.append(QualityIssue.INAPPROPRIATE_TONE)
            suggestions.append("Remove meta-commentary about response adjustments - respond directly as assistant")
        elif not any(indicator in ai_response.lower() for indicator in professional_indicators):
            issues.append(QualityIssue.INAPPROPRIATE_TONE)
            suggestions.append("Use proper references to Dr. Tomar and professional dental terminology")
        else:
            strengths.append("Properly refers to Dr. Tomar in third person")
        
        # Reasoning structure check
        reasoning_indicators = [
            "first", "second", "step", "because", "therefore", "this is why",
            "the reason", "let me explain", "here's what happens"
        ]
        if not any(indicator in ai_response.lower() for indicator in reasoning_indicators):
            issues.append(QualityIssue.MISSING_REASONING)
            suggestions.append("Include step-by-step reasoning and explanation")
        else:
            strengths.append("Provides clear reasoning and explanation")
        
        # Actionable advice check
        action_indicators = [
            "you should", "dr. tomar recommends", "try", "use", "avoid", "schedule",
            "call", "visit", "apply", "rinse", "brush"
        ]
        if not any(indicator in ai_response.lower() for indicator in action_indicators):
            issues.append(QualityIssue.UNCLEAR_ADVICE)
            suggestions.append("Provide more specific and actionable recommendations")
        else:
            strengths.append("Provides actionable advice")
        
        # Generic response check
        generic_phrases = [
            "it depends", "varies from person to person", "consult your dentist",
            "see a professional", "it's hard to say"
        ]
        generic_count = sum(1 for phrase in generic_phrases if phrase in ai_response.lower())
        if generic_count > 2:
            issues.append(QualityIssue.TOO_GENERIC)
            suggestions.append("Provide more specific, personalized advice")
        
        # Medical disclaimer check for serious conditions
        serious_keywords = ["pain", "infection", "swelling", "emergency", "severe"]
        if any(keyword in user_question.lower() for keyword in serious_keywords):
            disclaimer_indicators = [
                "see a dentist", "professional evaluation", "in-person examination",
                "consult", "visit", "appointment"
            ]
            if not any(indicator in ai_response.lower() for indicator in disclaimer_indicators):
                issues.append(QualityIssue.NO_DISCLAIMER)
                suggestions.append("Include recommendation for professional consultation")
        
        # Calculate overall score
        total_checks = 8
        passed_checks = total_checks - len(issues)
        overall_score = (passed_checks / total_checks) * 100
        
        return QualityScore(
            overall_score=overall_score,
            issues=issues,
            strengths=strengths,
            suggestions=suggestions
        )
    
    def needs_reprompting(self, quality_score: QualityScore) -> bool:
        """Determine if response needs reprompting based on quality score"""
        return quality_score.overall_score < self.quality_threshold
    
    def generate_improvement_feedback(self, quality_score: QualityScore) -> str:
        """Generate detailed feedback for response improvement"""
        
        feedback = "RESPONSE QUALITY ASSESSMENT:\n\n"
        feedback += f"Overall Score: {quality_score.overall_score:.1f}/100\n\n"
        
        if quality_score.strengths:
            feedback += "STRENGTHS:\n"
            for strength in quality_score.strengths:
                feedback += f"✓ {strength}\n"
            feedback += "\n"
        
        if quality_score.issues:
            feedback += "AREAS FOR IMPROVEMENT:\n"
            for issue in quality_score.issues:
                feedback += f"✗ {issue.value}\n"
            feedback += "\n"
        
        if quality_score.suggestions:
            feedback += "SPECIFIC SUGGESTIONS:\n"
            for suggestion in quality_score.suggestions:
                feedback += f"• {suggestion}\n"
        
        return feedback

class RepromptingSystem:
    """Handles automatic reprompting for quality improvement"""
    
    def __init__(self, openai_client, max_attempts: int = 3):
        self.client = openai_client
        self.max_attempts = max_attempts
        self.quality_checker = ResponseQualityChecker(openai_client)
        
    def improve_response_with_reprompting(
        self, 
        original_prompt: str, 
        user_question: str, 
        initial_response: str,
        context: str = ""
    ) -> Tuple[str, int, List[QualityScore]]:
        """
        Iteratively improve response through reprompting
        Returns: (final_response, attempts_used, quality_scores)
        """
        
        current_response = initial_response
        quality_scores = []
        attempts = 1
        
        while attempts <= self.max_attempts:
            # Check quality of current response
            quality_score = self.quality_checker.check_response_quality(
                user_question, current_response, context
            )
            quality_scores.append(quality_score)
            
            logger.info(f"Attempt {attempts}: Quality score {quality_score.overall_score:.1f}")
            
            # If quality is good enough, return current response
            if not self.quality_checker.needs_reprompting(quality_score):
                logger.info(f"Quality threshold met after {attempts} attempts")
                return current_response, attempts, quality_scores
            
            # If max attempts reached, return best response so far
            if attempts >= self.max_attempts:
                logger.warning(f"Max attempts ({self.max_attempts}) reached")
                return current_response, attempts, quality_scores
            
            # Generate improvement prompt
            improvement_feedback = self.quality_checker.generate_improvement_feedback(quality_score)
            
            reprompt = f"""{original_prompt}

PREVIOUS RESPONSE QUALITY FEEDBACK:
{improvement_feedback}

PREVIOUS RESPONSE:
{current_response}

CRITICAL INSTRUCTIONS:
- Provide ONLY the improved response content - NO meta-commentary
- Do NOT explain what you changed or how you improved it
- Do NOT include phrases like "This response has been adjusted" or "Here's a revised response"
- Do NOT mention empathy, warmth, tone, or professionalism in your response
- Respond DIRECTLY as Dr. Tomar's assistant would to the patient
- Address the quality issues silently without mentioning them

IMPROVED RESPONSE:"""
            
            # Get improved response
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": reprompt}],
                    temperature=0.7,
                    max_tokens=1500
                )
                current_response = response.choices[0].message.content
                # Clean meta-commentary immediately after generation
                current_response = self._remove_meta_commentary(current_response)
                attempts += 1
                
            except Exception as e:
                logger.error(f"Error during reprompting: {e}")
                break
        
        # Final cleanup - remove any remaining meta-commentary
        current_response = self._remove_meta_commentary(current_response)
        
        return current_response, attempts, quality_scores
    
    def _remove_meta_commentary(self, response: str) -> str:
        """Remove meta-commentary from response"""
        
        # Patterns to remove
        meta_patterns = [
            r"^.*?this response has been adjusted.*?\n",
            r"^.*?here's a revised response.*?\n",
            r"^.*?certainly! here.*?\n",
            r"^.*?response that incorporates.*?\n",
            r"^.*?adjusted to reflect.*?\n",
            r"^.*?more engaging tone.*?\n",
            r"^.*?while maintaining professionalism.*?\n",
            r"^.*?while ensuring clarity.*?\n",
            r"^.*?empathy.*?warmth.*?engaging.*?\n",
            r"^.*?incorporates empathy.*?\n"
        ]
        
        cleaned_response = response
        
        for pattern in meta_patterns:
            cleaned_response = re.sub(pattern, "", cleaned_response, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove empty lines at the beginning
        cleaned_response = cleaned_response.lstrip("\n\r ")
        
        return cleaned_response

# Example usage
if __name__ == "__main__":
    # Test the quality checker
    checker = ResponseQualityChecker(None)
    
    test_response = "You should see a dentist about that pain."
    test_question = "My tooth hurts when I drink cold water"
    
    quality_score = checker.check_response_quality(test_question, test_response)
    
    print(f"Quality Score: {quality_score.overall_score}")
    print(f"Issues: {[issue.value for issue in quality_score.issues]}")
    print(f"Suggestions: {quality_score.suggestions}")
