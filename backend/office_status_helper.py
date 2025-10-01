"""
Office Status Helper - Centralized logic for checking office availability
"""

from datetime import datetime, timedelta
from typing import Dict
import pytz

def get_office_hours(day: str) -> Dict[str, str]:
    """Get office hours for specific day"""
    hours_schedule = {
        'Monday': {'start': 8, 'end': 17, 'display': '8 AM - 5 PM'},
        'Tuesday': {'start': 9, 'end': 16, 'display': '9 AM - 4 PM'},
        'Wednesday': {'start': 8, 'end': 17, 'display': '8 AM - 5 PM'},
        'Thursday': {'start': 8, 'end': 16, 'display': '8 AM - 4 PM'},
        'Friday': {'start': 8, 'end': 17, 'display': '8 AM - 5 PM'},
        'Saturday': {'start': 8, 'end': 17, 'display': '8 AM - 5 PM'},
        'Sunday': {'start': 0, 'end': 0, 'display': 'Closed'}
    }
    return hours_schedule.get(day, {'start': 0, 'end': 0, 'display': 'Closed'})

def get_next_open_day() -> str:
    """Get the next open day from today"""
    pacific_tz = pytz.timezone('America/Los_Angeles')
    now = datetime.now(pacific_tz)
    open_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'}
    
    # Check next 7 days
    for i in range(1, 8):
        next_date = now + timedelta(days=i)
        next_day = next_date.strftime('%A')
        if next_day in open_days:
            return next_day
    
    return 'Monday'  # fallback

def get_dynamic_followup_question() -> str:
    """Generate dynamic follow-up questions for scheduling responses"""
    import random
    
    followup_questions = [
        "What type of dental concern would you like to address during your visit? 🦷",
        "Is this for a routine cleaning or do you have a specific dental concern? 🦷",
        "Are you experiencing any dental pain or discomfort? 🦷",
        "Would you like to schedule a consultation or cleaning appointment? 🦷",
        "Do you have any specific dental issues you'd like Dr. Tomar to examine? 🦷",
        "Is this for preventive care or do you need treatment for a dental problem? 🦷",
        "What brings you to our dental office today? 🦷",
        "Are you looking for a routine checkup or addressing a dental concern? 🦷"
    ]
    
    return random.choice(followup_questions)

def check_office_status(day: str) -> Dict[str, any]:
    """
    3-level check for office status:
    1. Check if it's an open day (Mon-Sat, closed Sunday)
    2. If open day, check current time
    3. Return appropriate status message
    """
    
    pacific_tz = pytz.timezone('America/Los_Angeles')
    now = datetime.now(pacific_tz)
    current_day = now.strftime('%A')
    
    # Get office hours for the day
    day_hours = get_office_hours(day)
    
    # Level 1: Check if it's an open day
    if day == 'Sunday' or day_hours['start'] == 0:
        # Level 3: Not an open day
        next_open = get_next_open_day()
        next_hours = get_office_hours(next_open)
        return {
            'is_open': False,
            'hours': 'Closed',
            'day': day,
            'status_message': f'Closed, next open {next_open} {next_hours["display"]}.'
        }
    
    # Level 2: It's an open day, check time (only for today)
    if day == current_day:
        current_hour = now.hour
        start_hour = day_hours['start']
        end_hour = day_hours['end']
        
        if current_hour < start_hour:
            # Before business hours
            return {
                'is_open': False,
                'hours': 'Currently closed',
                'day': day,
                'status_message': f'Currently closed, today we open {day_hours["display"]}.'
            }
        elif start_hour <= current_hour < end_hour:
            # Within business hours
            end_time = '5 PM' if end_hour == 17 else '4 PM'
            return {
                'is_open': True,
                'hours': f'Open until {end_time}',
                'day': day,
                'status_message': f'Open until {end_time}.'
            }
        else:
            # After business hours
            next_open = get_next_open_day()
            next_hours = get_office_hours(next_open)
            return {
                'is_open': False,
                'hours': 'Currently closed',
                'day': day,
                'status_message': f'Currently closed, next open {next_open} {next_hours["display"]}.'
            }
    else:
        # For future open days
        return {
            'is_open': True,
            'hours': day_hours['display'],
            'day': day,
            'status_message': f'{day} is open {day_hours["display"]}'
        }