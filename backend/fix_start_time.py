# The issue is that {start_time} is not being formatted as an f-string
# Find this line in the schedule_request section and add 'f' prefix:

# WRONG:
"Unfortunately,We're currently closed but will open today at {start_time}. "

# CORRECT:
f"Unfortunately, we're currently closed but will open today at {start_time}. "

# The fix is to add 'f' before the string to make it an f-string literal