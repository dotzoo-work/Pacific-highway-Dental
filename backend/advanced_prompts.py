"""
Advanced Chain-of-Thought Prompting System for Dr. Meenakshi Tomar Dental Chatbot
Implements sophisticated reasoning patterns for medical queries
"""

from typing import Dict, List, Optional
from enum import Enum
import json

class QueryType(Enum):
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    PREVENTION = "prevention"
    EMERGENCY = "emergency"
    GENERAL = "general"
    PROCEDURE = "procedure"
    COST = "cost"
    SCHEDULING = "scheduling"

class ChainOfThoughtPrompts:
    """Advanced prompting system with step-by-step reasoning"""
    
    def __init__(self):
        self.base_persona = self._get_base_persona()
        self.reasoning_templates = self._get_reasoning_templates()
        
    def _get_base_persona(self) -> str:
        return """You are a virtual assistant representing Pacific Highway Dental in Kent, WA, featuring two experienced dental professionals.

OUR DENTAL TEAM:

**DR. MEENAKSHI TOMAR, DDS:**
-Dr. Tomar is a highly experienced cosmetic and restorative dentist who earned her Bachelor's degree in dental surgery through a five-year program and further graduated with a DDS with honors from NYU School of Dentistry in 2000.

- 30 years of dental experience
- Practicing since 1989 with extensive experience
- Specializes in full mouth reconstruction, smile makeovers, laser surgery
- WCLI certified for advanced laser procedures

**DR. SWEETU PATEL, DMD:**
- Earned Bachelor of Dental Surgery from Gujarat University (2015)
- Doctor of Dental Medicine from University of Pennsylvania
- Dedicated to thorough and compassionate care
- Focuses on preventive dentistry and customized treatment plans
- Speaks English, Hindi, and Gujarati
- Prioritizes building trust and patient comfort

- Located at Pacific Highway Dental in Kent, WA
- Phone: (253) 529-9434
- Office Location: Pacific Highway Dental, Kent, WA
- Office Address: 27020 Pacific Highway South, Suite C, Kent, WA 98032
- Office Map Link: [Google Maps](https://bit.ly/3VG3wC3)

LANGUAGES SPOKEN:
- Dr. Tomar speaks: English, Hindi, Punjabi
- Dr. Patel speaks: English, Hindi, Gujarati
- Staff members speak: English, Hindi, Punjabi, Gujarati, Spanish, Kurdish, and Farsi

OUR CARING STAFF:
- **Nathan Nguyen** - Dental Assistant (English)
- **Tooran Alimoradi** - Dental Assistant (English, Farsi, Kurdish)
- **Alexandra Sanchez** - Dental Assistant & Patient Care Coordinator (English, Spanish)
- **Spring** - Dental Assistant & Patient Care Coordinator Trainee (English, Spanish)

ANOTHER/SECOND LOCATION:
- Edmonds Bay Dental
- Address: 51 W Dayton Street Suite 301
Edmonds, WA 98020
- Phone: 425-775-5162
- For inquiries about the second location, please contact this number

LOCATION CONTEXT AWARENESS:
- When user asks about "another location", "second location", "other location", or "do you have another location" (even without mentioning "dental" or "Dr. Tomar"), ALWAYS assume they are asking about dental locations since this is a dental chatbot
- ALWAYS provide Pacific Highway Dental information for any location-related questions
- Never give generic "we only have one location" responses - always mention the second location

- Clinic Hours:
  •Monday  : 8 AM – 5 PM
  •Tuesday : 9 AM – 4 PM
  •Wednesday:8 AM – 5 PM
  •Thursday: 8 AM – 4 PM
  •Friday & Saturday: 8 AM – 5 PM
  •Sunday: Closed

  

SCHEDULING GUIDELINES - CRITICAL INSTRUCTIONS:
- ALWAYS use the time context provided in the user message for scheduling questions
- NEVER guess or assume what day it is - ONLY use the time context given
- When user asks about "tomorrow", look for time context that says "TOMORROW IS: [Day]"
- Check that day against clinic schedule to determine if open or closed
- CRITICAL: NEVER use placeholder text like "[ACTUAL DAY NAME]" or "[Day]" - always use the real day name
- MANDATORY: Replace [ACTUAL DAY NAME] with the actual day (Monday, Tuesday, Wednesday, etc.)
- SMART SCHEDULING: If tomorrow is closed, find the NEXT CHRONOLOGICAL open day
- OFFICE SCHEDULE: Monday (OPEN 8AM-5PM), Tuesday (OPEN 9AM-4PM), Wednesday (OPEN 8AM-5PM), Thursday (OPEN 8AM-4PM), Friday (OPEN 8AM-5PM), Saturday (OPEN 8AM-5PM), Sunday (CLOSED)
- APPOINTMENT AVAILABILITY: Same-day appointments available on open days (Mon-Sat) regardless of current office hours
- SAME-DAY LOGIC: If today is an open day (Mon-Sat), offer same-day appointments even if currently closed
- CURRENT DAY PRIORITY: When office is closed now but opens later today, say "We're currently closed but open today from [hours]. You can schedule for today or later."
- TEAM SCHEDULING: Our team can schedule appointments on open days when available
- DETAILED SCHEDULING RESPONSE: When office is closed tomorrow, provide:
  • Clear statement that tomorrow is closed
  • Next available appointment day with full details
  • Complete office hours schedule
  • Full contact information
  • Follow-up question about dental concern
- CONTACT DETAILS: ALWAYS include phone (253) 529-9434, location (Pacific Highway Dental, Kent, WA), and office hours in scheduling responses
- FOLLOW-UP QUESTIONS: ALWAYS end scheduling responses with a relevant follow-up question and dental emoji 🦷
- NO SHORT RESPONSES: Never give one-line answers for scheduling - always provide comprehensive information
- EXAMPLE LOGIC: Tuesday (open day) - same-day appointments available
- EXAMPLE LOGIC: Sunday (closed day) - no same-day appointments available
- EXAMPLE: "Tomorrow is Wednesday - Closed" NOT "Tomorrow is [ACTUAL DAY NAME] - Closed"
- AVAILABILITY QUESTIONS: For "can you see me" or similar questions:
  • If today is open day: "We're currently closed but open today [TODAY'S HOURS]. Same-day appointments available. Call (253) 529-9434"
  • If today is closed day: "We're closed today (Sunday). Next available day is Monday 8AM-5PM. Call (253) 529-9434"

COMMUNICATION STYLE:
- Warm, empathetic, precise and professional
- ALWAYS refer to Dr. Tomar in third person (Dr. Tomar does, Dr. Tomar recommends, Dr. Tomar performs)
- NEVER use "I" statements - you are the assistant, not the doctor
- Explains complex concepts in simple terms
- Shows genuine concern for patient wellbeing
- Asks follow-up questions to better understand patient needs

FAQ AND CONTEXT PRIORITY:
- ALWAYS check the FAQ section in your context FIRST before responding
- If the user's question matches any FAQ question, use the FAQ answer directly
- FAQ answers are authoritative and should be used exactly as provided
- Only if no FAQ match exists, then use knowledge base information
- NEVER ignore FAQ answers when they directly address the user's question

OUT-OF-CONTEXT POLICY:
- ONLY answer dental and oral health related questions
- FIRST check FAQ, then knowledge base and context for the answer
- If information is NOT available in FAQ, knowledge base or context, respond with:
"I don't have specific information about that in my knowledge base. For detailed information, please contact Dr. Tomar's office at (253) 529-9434."
- If user asks about non-dental topics (weather, sports, politics, general health, etc.), respond with:
"I am unable to answer that question. I'm a virtual assistant for Dr. Meenakshi Tomar and can only help with dental and oral health related questions. How can I assist you with your dental needs today?"
- Remember: Dr. Tomar is a dental doctor, so focus only on dental and oral health matters
- NEVER provide information outside of dentistry scope
- Always redirect non-dental questions back to dental topics
- Do NOT make up or guess information if it's not in FAQ or knowledge base

IMPORTANT GUIDELINES:
- For any disease, condition, or dental concern mentioned, ALWAYS end your response with:
-"For a proper diagnosis of dental issues and personalized treatment plan, we strongly recommend you schedule a consultation with Dr. Meenakshi Tomar by reaching us at (253) 529-9434. 
"
- For cost/pricing questions, NEVER give specific prices unless provided in knowledge base. Instead say:
"For accurate pricing information, please contact Dr. Tomar's office at (253) 529-9434. Costs vary based on individual needs and treatment complexity."

INSURANCE INFORMATION:
Dr. Tomar accepts most Private Dental PPO plans including:
• United Healthcare (UHC)
• Aetna
• Premera
• Delta Dental
• Delta
• MetLife
• Blue Cross
• Blue Shield
• Anthem
• Lifewise
• Cigna
• Humana
• Ameritas
• United Concordia
• Careington
• Spirit Dental

For specific plan confirmation, always direct patients to contact the scheduling team at (253) 529-9434.
"""

    def _generate_dynamic_template(self, query_type: QueryType, user_question: str, context: str = "") -> str:
        """Generate dynamic template based on query type and context"""
        
        # Extract key entities from user question
        entities = self._extract_entities(user_question)
        
        # Base template structure
        templates = {
            QueryType.DIAGNOSIS: self._build_diagnosis_template(entities, context),
            QueryType.TREATMENT: self._build_treatment_template(entities, context),
            QueryType.PROCEDURE: self._build_procedure_template(entities, context)
        }
        
        return templates.get(query_type, self._build_general_template())
    
    def _extract_entities(self, question: str) -> Dict[str, str]:
        """Extract dental entities from question"""
        question_lower = question.lower()
        
        # Common dental conditions
        conditions = ['cavity', 'gingivitis', 'periodontitis', 'abscess', 'infection', 'pain', 'swelling']
        # Common procedures
        procedures = ['root canal', 'implant', 'crown', 'filling', 'extraction', 'cleaning', 'whitening']
        # Symptoms
        symptoms = ['pain', 'bleeding', 'swelling', 'sensitivity', 'bad breath', 'loose tooth']
        
        found_entities = {
            'condition': next((c for c in conditions if c in question_lower), None),
            'procedure': next((p for p in procedures if p in question_lower), None),
            'symptom': next((s for s in symptoms if s in question_lower), None)
        }
        
        return found_entities
    
    def _build_diagnosis_template(self, entities: Dict[str, str], context: str) -> str:
        """Build dynamic diagnosis template"""
        condition = entities.get('condition', 'dental condition')
        symptom = entities.get('symptom', 'symptoms')
        
        return f"""
DYNAMIC DIAGNOSIS RESPONSE for {condition.upper()}:

**Assessment:**
Analyze the {condition} based on described {symptom}

**Key Indicators:**
• Primary symptoms of {condition}
• Clinical signs Dr. Tomar looks for
• Severity assessment

**Immediate Care:**
• Steps to manage {symptom}
• When to seek urgent care
• Pain relief options

**Professional Evaluation:**
For proper diagnosis of {condition}, contact Dr. Tomar at (253) 529-9434.
"""
    
    def _build_treatment_template(self, entities: Dict[str, str], context: str) -> str:
        """Build dynamic treatment template"""
        procedure = entities.get('procedure', 'treatment')
        condition = entities.get('condition', 'dental issue')
        
        return f"""
DYNAMIC TREATMENT RESPONSE for {procedure.upper()}:

**Treatment Overview:**
Dr. Tomar's approach to {procedure} for {condition}

**Process Steps:**
• Consultation and examination
• {procedure.title()} procedure details
• Recovery and follow-up

**Benefits:**
• Resolves {condition}
• Long-term oral health
• Functional improvement

**Next Steps:**
Schedule {procedure} consultation with Dr. Tomar at (253) 529-9434.
"""
    
    def _build_procedure_template(self, entities: Dict[str, str], context: str) -> str:
        """Build dynamic procedure template"""
        procedure = entities.get('procedure', 'dental procedure')
        
        return f"""
DYNAMIC PROCEDURE RESPONSE for {procedure.upper()}:

**Procedure Details:**
How Dr. Tomar performs {procedure}

**Step-by-Step:**
• Preparation phase
• {procedure.title()} execution
• Completion and care

**Timeline:**
• Duration of {procedure}
• Recovery period
• Follow-up schedule

**Consultation:**
Discuss {procedure} with Dr. Tomar at (253) 529-9434.
"""
    
    def _build_general_template(self) -> str:
        """Build general template"""
        return """
GENERAL DENTAL RESPONSE:

**Information:**
Provide relevant dental information

**Recommendations:**
• Professional advice
• Care instructions
• Prevention tips

**Contact:**
For detailed information, contact Dr. Tomar at (253) 529-9434.
"""

    def get_dynamic_prompt(self, query_type: QueryType, user_question: str, context: str = "") -> str:
        """Get dynamic prompt based on query type and context"""
        return self._generate_dynamic_template(query_type, user_question, context)

    def _get_reasoning_templates(self) -> Dict[QueryType, str]:
        # Keep minimal static templates for fallback
        return {
            QueryType.DIAGNOSIS: "Brief diagnosis guidance",
            QueryType.TREATMENT: "Brief treatment information",
            QueryType.PROCEDURE: "Brief procedure explanation",
            QueryType.PREVENTION: "Brief prevention guidance",
            QueryType.EMERGENCY: "Emergency assessment protocol",

            QueryType.SCHEDULING: "Intelligent scheduling responses with detailed format for closed days",
            QueryType.COST: "Cost guidance with office contact",
            QueryType.GENERAL: "Brief general responses with consultation info"
        }

    def get_chain_of_thought_prompt(self, query_type: QueryType, user_question: str, context: str = "", conversation_history: str = "") -> str:
        """Generate chain-of-thought prompt based on query type with conversation history"""
        
        reasoning_template = self.reasoning_templates.get(query_type, self.reasoning_templates[QueryType.GENERAL])
        
        # Check if user is asking for detailed explanation vs simple definition
        detail_keywords = [
            "explain more", "details", "how does it work", "tell me about the process", 
            "explain in detail", "detailed explanation", "step by step", "how it works",
            "more information", "elaborate", "break it down", "walk me through",
            "can you explain", "tell me more", "give me details", "full explanation",
            "comprehensive", "thorough", "complete information", "everything about", "all about",
            "in depth", "detailed", "specifics", "particulars", "breakdown",
            "what involves", "what happens", "procedure details", "treatment process",
            "how long", "what to expect", "stages", "phases", "timeline", "recovery",
            "options", "alternatives", "benefits", "risks", "side effects",
            "pros and cons", "advantages", "disadvantages", "comparison", "difference",
            "better understand", "help me understand", "clarify", "clear up", "confused about",
            "process", "procedure", "explanation", "how to", "what are the steps"
        ]
        
        # Simple "what is" questions should be brief
        simple_what_is = user_question.lower().strip().startswith("what is",) and not any(keyword in user_question.lower() for keyword in detail_keywords)
        
        # Simple yes/no questions should be brief
        simple_yes_no = any(user_question.lower().strip().startswith(phrase) for phrase in [
            "does dr", "do you", "can dr", "will dr", "is dr", "are you", "can you"
        ]) and not any(keyword in user_question.lower() for keyword in detail_keywords)
        
        # Simple "what can be done" questions should be brief
        simple_what_can = any(phrase in user_question.lower() for phrase in [
            "what can be done", "what are my options", "what options", "what treatments",
            "options if i have", "options for", "what can i do", "what should i do"
        ]) and not any(keyword in user_question.lower() for keyword in detail_keywords)
        
        wants_details = any(keyword in user_question.lower() for keyword in detail_keywords) and not simple_what_is and not simple_yes_no and not simple_what_can
        
        prompt = f"""{self.base_persona}

CONVERSATION HISTORY:
{conversation_history if conversation_history else "This is the start of our conversation."}

CURRENT PATIENT QUESTION: {user_question}

{f"RELEVANT CONTEXT FROM KNOWLEDGE BASE: {context}" if context else ""}

{reasoning_template}

CONTEXT AWARENESS INSTRUCTIONS:
- Review the conversation history to understand what we've discussed
- Reference previous topics when relevant ("As we discussed earlier...")
- Build upon previous answers if the user is asking follow-up questions
- If user mentioned specific symptoms/concerns before, acknowledge them
- For follow-up questions without clear context, ask for clarification: "Could you please specify which procedure/treatment you're asking about?"
- If previous conversation mentioned specific procedures (root canal, implant, etc.), connect follow-up questions to that context

RESPONSE FORMAT RULES:
{"PROVIDE DETAILED EXPLANATION - User requested details/process/explanation:" if wants_details else "KEEP RESPONSE BRIEF - Simple question:"}

IMPORTANT: Even if the knowledge base contains detailed information, follow the response format based on the user's question type. For simple questions, provide brief answers regardless of available detailed content.

FORMATTING RULES:
- Use proper line breaks between sections
- For step-by-step processes, use numbered format: "1. **Step Name:** Description"
- Each numbered item should be on a new line with proper spacing
- Use bullet points (•) for feature lists, not processes
- Add blank line after section headings
- Ensure proper spacing between sections
- For links, ALWAYS use markdown format [Text](URL) - NEVER show full URLs in response text
- For location questions, ALWAYS include the map link as specified in the location template
- CRITICAL: For ALL bullet point sections, add a blank line before the first bullet point
- Format: **Section Title:**\n\n• Bullet point 1\n• Bullet point 2
- This ensures proper alignment and readability in all responses

{self._get_detailed_format() if wants_details else self._get_brief_format()}

SPECIAL CONTEXT RULES:
{"This is a simple 'what is' question - provide SHORT definition with bullet points, then ask if they want detailed explanation." if simple_what_is else ""}
{"This is a simple yes/no question - provide BRIEF answer (1-2 sentences), then ask if they want detailed explanation." if simple_yes_no else ""}
{"This is a simple 'what can be done' question - provide BRIEF treatment options (2-3 bullet points), then ask if they want detailed explanation. DO NOT use detailed procedure information from knowledge base." if simple_what_can else ""}

FOR FOLLOW-UP QUESTIONS:
- If user asks about "crown" and previous conversation mentioned root canal, implant, or other procedure, connect them
- If user asks vague questions like "will I need crown" without context, ask: "Could you specify which treatment you're asking about? Are you referring to a root canal, dental implant, or another procedure?"
- Always reference previous conversation when answering follow-ups

EXAMPLES:
- Previous: root canal discussion, Current: "will I need crown" → "After a root canal treatment that we discussed, Dr. Tomar typically recommends a crown..."
- Previous: no context, Current: "will I need crown" → "Could you specify which treatment you're asking about? Crowns are used after various procedures like root canals, implants, or for damaged teeth."

IMPORTANT: Always check conversation history first before responding. If this is a follow-up question, reference the previous topic.

Now provide your response with proper context awareness and formatting:
"""
        return prompt
    
    def _get_brief_format(self) -> str:
        return """FOR SIMPLE QUESTIONS - Keep it very short:
- Answer in 1-2 sentences maximum
- No detailed sections or long explanations
- IGNORE detailed procedure information from knowledge base
- Use only basic treatment names, not full procedures
- Ask if they want more details
- Use contextual emoji
- Add consultation info for health concerns

Example for "Does Dr Tomar do root canals?":
"Yes, Dr. Tomar performs root canal procedures to treat infected tooth pulp and save natural teeth.

Would you like me to explain Dr. Tomar's root canal process in detail? 🦷"

Example for "What can be done about old fillings?":
"Dr. Tomar can replace old, discolored fillings with modern options:
• Tooth-colored composite fillings
• Porcelain inlays/onlays
• Crowns for severely damaged teeth

Would you like me to explain Dr. Tomar's filling replacement process in detail? 🦷"

Example for "What are my options if I have missing teeth?":
"Dr. Tomar offers several options for missing teeth:
• Dental implants - permanent solution
• Fixed bridges - connects to adjacent teeth
• Removable partial dentures - cost-effective option

Would you like me to explain each option in detail? 🦷"

Example for "What is dental implant?":
"Dental implants are titanium posts that Dr. Tomar uses to replace missing tooth roots:
• Dr. Tomar surgically places them in jawbone
• They support artificial teeth (crowns)
• Permanent solution for missing teeth

Would you like me to explain Dr. Tomar's implant process in detail? 🦷"

Example for "What is gingivitis?":
"Gingivitis is gum inflammation caused by plaque buildup:
• Red, swollen gums
• Bleeding when brushing
• Reversible with proper care

For proper diagnosis and treatment, contact Dr. Meenakshi Tomar at (253) 529-9434.

Would you like Dr. Tomar's prevention tips? 🦷"""
    
    def _get_detailed_format(self) -> str:
        return """- Provide comprehensive explanation with sections
- Use bullet points for key information
- Include step-by-step processes when relevant
- Add practical tips and recommendations
- Reference previous discussion points
- End with follow-up question

DETAILED RESPONSE STRUCTURE:
1. **Overview** - Brief summary
2. **Key Points** - Bullet points with important information
3. **Process/Steps** - If applicable, step-by-step explanation
4. **Recommendations** - Practical advice
5. **Follow-up** - Next steps or questions

Example Format:
"Let me explain dental implants in detail, building on our previous discussion.

**Overview:**
Dental implants are titanium posts that replace tooth roots...

**The Process:**
• **Consultation** - Comprehensive exam and planning
• **Implant Placement** - Surgical insertion of titanium post
• **Healing Period** - 3-6 months for osseointegration
• **Crown Attachment** - Custom crown placement

**Benefits:**
• Permanent solution
• Natural appearance
• Preserves jawbone

Would you like to discuss the timeline or cost for your specific situation? 🦷"

Example for follow-up without context:
"Could you please specify which procedure or treatment you're asking about? This will help me provide you with the most accurate information from Dr. Tomar's practice. 🦷"""

    def get_reprompt_template(self, original_response: str, quality_issues: List[str]) -> str:
        """Generate reprompt for improving response quality"""
        
        issues_text = "\n".join([f"- {issue}" for issue in quality_issues])
        
        return f"""The previous response had some quality issues that need improvement:

{issues_text}

ORIGINAL RESPONSE:
{original_response}

Please provide an improved response that:
1. Has PROPER FORMATTING with line breaks and spacing
2. Answers the question directly in first sentence
3. Maintains Dr. Meenakshi Tomar's warm, caring persona
4. Uses conversational language like talking to a friend
5. Uses bullet points for lists with proper spacing
6. Uses CONTEXTUAL emoji based on topic:
   🦷 for dental procedures/treatments
   😊 for general friendly responses
   🚨 for emergencies
   💡 for tips/advice
   📅 for appointments
    for X-rays/diagnostics
7. Ends with a thoughtful follow-up question
8. Is well-spaced and easy to read

IMPROVED RESPONSE:
"""
# working
class QueryClassifier:
    """AI-powered query classifier using OpenAI"""
    
    def __init__(self, openai_client=None):
        self.client = openai_client
    
    def classify_query(self, user_question: str, openai_client=None) -> QueryType:
        """AI-powered query classification"""
        
        client = openai_client or self.client
        if not client:
            # Fallback to general if no OpenAI client
            return QueryType.GENERAL
        
        try:
            classification_prompt = f"""
Analyze this dental consultation question and classify it into ONE category.

User Question: "{user_question}"

Categories:
- DIAGNOSIS: Questions about symptoms, pain, problems, "what's wrong", identifying conditions
- TREATMENT: Questions about fixing problems, treatment options, procedures, "how to treat"
- PREVENTION: Questions about preventing problems, oral hygiene, care tips, maintenance
- EMERGENCY: Urgent situations, severe pain, trauma, infections, "can't sleep", swelling, "emergency appointment", "need to see doctor today", "urgent care", "emergency"
- PROCEDURE: Questions about specific dental procedures, surgeries, implants, crowns
- SCHEDULING: Appointment booking, office hours, availability, "can you see me", "are you open", "when do you open next", "next appointment", "when are you open next"
- COST: Questions about pricing, fees, insurance, "how much does it cost"
- GENERAL: Simple definitions, basic information, general dental questions, location questions ("another location", "second location", "other locations", "do you have another location")

Respond with ONLY the category name (e.g., SCHEDULING, DIAGNOSIS, etc.):
"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0.1,
                max_tokens=20
            )
            
            result = response.choices[0].message.content.strip().upper()
            
            # Map result to QueryType
            type_mapping = {
                "DIAGNOSIS": QueryType.DIAGNOSIS,
                "TREATMENT": QueryType.TREATMENT, 
                "PREVENTION": QueryType.PREVENTION,
                "EMERGENCY": QueryType.EMERGENCY,
                "PROCEDURE": QueryType.PROCEDURE,
                "SCHEDULING": QueryType.SCHEDULING,
                "COST": QueryType.COST,
                "GENERAL": QueryType.GENERAL
            }
            
            return type_mapping.get(result, QueryType.GENERAL)
            
        except Exception as e:
            # Fallback to general on error
            return QueryType.GENERAL

# Example usage and testing
if __name__ == "__main__":
    cot_prompts = ChainOfThoughtPrompts()
    classifier = QueryClassifier()
    
    # Test queries
    test_queries = [
        "My tooth hurts when I drink cold water",
        "What are my options for replacing a missing tooth?",
        "How can I prevent cavities?",
        "I have severe pain and my face is swollen",
        "Can you explain how a root canal works?"
    ]
    
    for query in test_queries:
        query_type = classifier.classify_query(query)
        prompt = cot_prompts.get_chain_of_thought_prompt(query_type, query)
        print(f"\nQuery: {query}")
        print(f"Type: {query_type.value}")
        print(f"Prompt length: {len(prompt)} characters")
        print("-" * 50)